"""
ConvexAdam 2D: Coupled Convex Optimization + Adam Instance Optimization for 2D registration.

Reference: "Fast 3D Registration with Accurate and Robust Optimisation"
Authors: Hanna Siebert, Lasse Hansen, Mattias P. Heinrich
Adapted for 2D cross-modal registration (CT→MR).

This is an optimization-based method - no training required.
Uses MIND-SSC (Modality Independent Neighbourhood Descriptor) features
for cross-modal similarity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MINDSSC2D:
    """2D MIND-SSC feature descriptor for modality-independent registration.

    Uses 6 offsets for self-similarity context in 2D, producing 6 descriptor channels.
    """

    def __init__(self, radius=2, dilation=2, device='cuda'):
        self.radius = radius
        self.dilation = dilation
        self.device = device

        # 6-point neighbourhood for 2D MIND-SSC
        # These offsets produce 6 descriptor channels
        # Using offsets similar to the 3D 6-neighbourhood pattern adapted for 2D
        six_neighbourhood = torch.Tensor([
            [0, 1], [1, 0], [2, 1], [1, 2], [2, 3], [3, 2]
        ]).long()

        # Compute pairwise squared distances
        dist = self._pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # Find pairs with distance == 2
        n_pts = six_neighbourhood.shape[0]
        x_idx, y_idx = torch.meshgrid(torch.arange(n_pts), torch.arange(n_pts), indexing='ij')
        mask = ((x_idx > y_idx).view(-1) & (dist == 2).view(-1))

        n_pairs = mask.sum().item()
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, n_pts, 1).view(-1, 2)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(n_pts, 1, 1).view(-1, 2)[mask, :]

        # Build convolution kernels for each shift pair
        mshift1 = torch.zeros(n_pairs, 1, 5, 5, device=device)
        mshift1.view(-1)[torch.arange(n_pairs, device=device) * 25 +
                          idx_shift1[:, 0].to(device) * 5 + idx_shift1[:, 1].to(device)] = 1
        mshift2 = torch.zeros(n_pairs, 1, 5, 5, device=device)
        mshift2.view(-1)[torch.arange(n_pairs, device=device) * 25 +
                          idx_shift2[:, 0].to(device) * 5 + idx_shift2[:, 1].to(device)] = 1

        self.mshift1 = mshift1
        self.mshift2 = mshift2
        self.n_ch = n_pairs

    @staticmethod
    def _pdist_squared(x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def __call__(self, img):
        """Extract MIND-SSC features from a 2D image.

        Args:
            img: [B, 1, H, W] grayscale image

        Returns:
            mind: [B, n_ch, H, W] MIND descriptor
        """
        kernel_size = self.radius * 2 + 1
        rpad1 = nn.ReplicationPad2d(self.dilation + 1)  # pad for 5x5 kernel
        rpad2 = nn.ReplicationPad2d(self.radius)

        ssd = F.avg_pool2d(
            rpad2(
                (F.conv2d(rpad1(img), self.mshift1, dilation=self.dilation) -
                 F.conv2d(rpad1(img), self.mshift2, dilation=self.dilation)) ** 2
            ),
            kernel_size, stride=1
        )

        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)
        return mind


def correlate2d(fixed_feat, moving_feat, disp_hw, grid_sp):
    """Compute SSD cost volume for 2D registration."""
    H, W = fixed_feat.shape[2], fixed_feat.shape[3]
    H_d = H // grid_sp
    W_d = W // grid_sp

    # Downsample features
    fixed_small = F.avg_pool2d(fixed_feat, grid_sp)
    moving_small = F.avg_pool2d(moving_feat, grid_sp)

    disp_range = torch.arange(-disp_hw, disp_hw + 1, device=fixed_feat.device, dtype=torch.float32)
    disp_y, disp_x = torch.meshgrid(disp_range, disp_range, indexing='ij')
    disp_y = disp_y.reshape(-1)
    disp_x = disp_x.reshape(-1)
    n_disp = disp_y.shape[0]

    # Sample moving features at all displacements
    cost_vol = torch.zeros(1, n_disp, H_d, W_d, device=fixed_feat.device)
    for i in range(n_disp):
        shifted = torch.roll(moving_small, shifts=(int(disp_y[i].item()), int(disp_x[i].item())), dims=(2, 3))
        cost_vol[0, i] = ((fixed_small - shifted) ** 2).mean(1)

    return cost_vol, disp_y, disp_x, H_d, W_d


def coupled_convex2d(cost_vol, disp_y, disp_x, H_d, W_d, coeffs):
    """Coupled convex optimization for 2D displacement field."""
    n_disp = disp_y.shape[0]

    # Initialize with argmin of cost volume
    idx = cost_vol.argmin(1)  # [1, H_d, W_d]
    disp_soft = torch.zeros(1, 2, H_d, W_d, device=cost_vol.device)
    for i in range(n_disp):
        mask = (idx[0] == i)
        disp_soft[0, 0][mask] = disp_y[i]
        disp_soft[0, 1][mask] = disp_x[i]

    for coeff in coeffs:
        cost_reg = cost_vol.clone()
        for i in range(n_disp):
            cost_reg[0, i] += coeff * ((disp_y[i] - disp_soft[0, 0]) ** 2 +
                                        (disp_x[i] - disp_soft[0, 1]) ** 2)

        # Soft argmin
        cost_softmax = F.softmax(-cost_reg * 10, dim=1)
        disp_soft = torch.zeros(1, 2, H_d, W_d, device=cost_vol.device)
        for i in range(n_disp):
            w = cost_softmax[0, i]
            disp_soft[0, 0] += w * disp_y[i]
            disp_soft[0, 1] += w * disp_x[i]

    return disp_soft


def adam_instance_opt_2d(fixed_feat, moving_feat, disp_init, grid_sp, lambda_weight=1.25,
                         n_iter=80, lr=1.0):
    """Adam instance optimization for 2D displacement field refinement."""
    H, W = fixed_feat.shape[2], fixed_feat.shape[3]

    # disp_init: [1, 2, H_d, W_d]
    disp = disp_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([disp], lr=lr)

    for it in range(n_iter):
        optimizer.zero_grad()

        # Upsample displacement to feature resolution and scale
        disp_full = F.interpolate(disp, size=(H, W), mode='bilinear', align_corners=True)
        disp_full = disp_full * grid_sp

        # Sample moving features at displaced positions using grid_sample
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=disp.device, dtype=torch.float32),
            torch.arange(0, W, device=disp.device, dtype=torch.float32), indexing='ij')
        sample_y = (grid_y + disp_full[0, 1]).clamp(0, H - 1)
        sample_x = (grid_x + disp_full[0, 0]).clamp(0, W - 1)

        norm_y = 2 * sample_y / (H - 1) - 1
        norm_x = 2 * sample_x / (W - 1) - 1
        grid_sample = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

        warped_moving = F.grid_sample(moving_feat, grid_sample, align_corners=True, mode='bilinear')

        # SSD loss
        n_ch = fixed_feat.shape[1]
        sim_loss = ((fixed_feat - warped_moving) ** 2).mean(1).mean() * n_ch

        # Diffusion regularization on downsampled displacement
        reg_loss = lambda_weight * (
            ((disp[0, 0, 1:, :] - disp[0, 0, :-1, :]) ** 2).mean() +
            ((disp[0, 1, :, 1:] - disp[0, 1, :, :-1]) ** 2).mean()
        )

        loss = sim_loss + reg_loss
        loss.backward()
        optimizer.step()

    # Upsample to full image resolution
    disp_final = F.interpolate(disp.detach(), size=(H, W), mode='bilinear', align_corners=True)
    disp_final = disp_final * grid_sp
    return disp_final


def spatial_transform_2d(source, flow):
    """Apply 2D displacement field to source image."""
    B, C, H, W = source.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, device=source.device, dtype=torch.float32),
        torch.arange(0, W, device=source.device, dtype=torch.float32), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    new_locs = grid + flow.permute(0, 2, 3, 1)
    new_locs[..., 0] = 2 * new_locs[..., 0] / (W - 1) - 1
    new_locs[..., 1] = 2 * new_locs[..., 1] / (H - 1) - 1
    return F.grid_sample(source, new_locs, align_corners=True, mode='bilinear')


def convex_adam_2d(fixed_img, moving_img, grid_sp=4, disp_hw=3, lambda_weight=1.25,
                   n_iter_adam=80, lr_adam=1.0):
    """
    Full ConvexAdam 2D registration pipeline.

    Args:
        fixed_img: Fixed (target) image [1, 1, H, W]
        moving_img: Moving (source) image [1, 1, H, W]
        grid_sp: Grid spacing for convex stage
        disp_hw: Displacement half-width for cost volume
        lambda_weight: Diffusion regularization weight
        n_iter_adam: Number of Adam optimization iterations
        lr_adam: Adam learning rate

    Returns:
        warped: Warped moving image [1, 1, H, W]
        disp_field: Displacement field [1, 2, H, W]
    """
    device = fixed_img.device
    mind_desc = MINDSSC2D(device=device)

    # Extract MIND features
    fixed_feat = mind_desc(fixed_img)
    moving_feat = mind_desc(moving_img)

    # Stage 1: Coupled convex optimization
    cost_vol, disp_y, disp_x, H_d, W_d = correlate2d(fixed_feat, moving_feat, disp_hw, grid_sp)
    coeffs = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    disp_init = coupled_convex2d(cost_vol, disp_y, disp_x, H_d, W_d, coeffs)

    # Stage 2: Adam instance optimization
    disp_field = adam_instance_opt_2d(fixed_feat, moving_feat, disp_init, grid_sp,
                                       lambda_weight=lambda_weight, n_iter=n_iter_adam, lr=lr_adam)

    # Resize displacement field to match source image size
    img_H, img_W = moving_img.shape[2], moving_img.shape[3]
    disp_H, disp_W = disp_field.shape[2], disp_field.shape[3]
    if disp_H != img_H or disp_W != img_W:
        disp_field = F.interpolate(disp_field, size=(img_H, img_W), mode='bilinear', align_corners=True)

    # Apply displacement to get warped image
    warped = spatial_transform_2d(moving_img, disp_field)

    return warped, disp_field
