"""
3D Spatial Transformer and loss functions for volumetric registration.

PyTorch F.grid_sample natively supports 5D input (3D volumes):
  - input: [B, C, D, H, W]
  - grid:  [B, D, H, W, 3] with coordinates in (x/W, y/H, z/D) order
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer3D(nn.Module):
    """3D spatial transformer using F.grid_sample with 5D tensors.

    Flow format: [B, 3, D, H, W] where channels are (x/w, y/h, z/d) offsets
    added to normalized identity grid.
    Uses align_corners=True for correct identity mapping.
    """

    def __init__(self):
        super(SpatialTransformer3D, self).__init__()

    def forward(self, src, flow):
        """
        Args:
            src: [B, C, D, H, W] source volume
            flow: [B, 3, D, H, W] deformation field (ch0=x/W, ch1=y/H, ch2=z/D)
        Returns:
            warped: [B, C, D, H, W]
        """
        if src.dim() == 4:
            src = src.unsqueeze(0)
            flow = flow.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, C, D, H, W = src.shape

        grid_d, grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, D, device=src.device, dtype=src.dtype),
            torch.linspace(-1, 1, H, device=src.device, dtype=src.dtype),
            torch.linspace(-1, 1, W, device=src.device, dtype=src.dtype),
            indexing='ij'
        )

        # F.grid_sample expects grid in (x, y, z) order where
        # x indexes the last dim (W), y indexes the middle dim (H), z indexes the first dim (D)
        grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)  # [D, H, W, 3]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, D, H, W, 3]

        # Add flow offsets (flow channels: 0=x/W, 1=y/H, 2=z/D)
        new_grid = grid + flow.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

        result = F.grid_sample(src, new_grid, mode='bilinear', padding_mode='border',
                               align_corners=True)
        if squeeze:
            result = result.squeeze(0)
        return result


class NormalizedMutualInformationLoss3D(nn.Module):
    """NMI loss for 3D volumes. Same algorithm as 2D but flattens spatial dims."""

    def __init__(self, num_bins=32):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, moving, fixed):
        """
        Args:
            moving: [B, 1, D, H, W] warped moving image
            fixed: [B, 1, D, H, W] fixed target image
        """
        # MI loss requires fp32 for stable histogram computation
        moving = moving.float()
        fixed = fixed.float()

        b = moving.shape[0]
        moving_flat = moving.reshape(b, -1)
        fixed_flat = fixed.reshape(b, -1)

        # Per-image normalization to [0, 1]
        moving_norm = (moving_flat - moving_flat.min(dim=1, keepdim=True)[0]) / \
                      (moving_flat.max(dim=1, keepdim=True)[0] - moving_flat.min(dim=1, keepdim=True)[0] + 1e-10)
        fixed_norm = (fixed_flat - fixed_flat.min(dim=1, keepdim=True)[0]) / \
                     (fixed_flat.max(dim=1, keepdim=True)[0] - fixed_flat.min(dim=1, keepdim=True)[0] + 1e-10)

        # Discretize
        moving_bins = torch.clamp(torch.floor(moving_norm * (self.num_bins - 1)).long(), 0, self.num_bins - 1)
        fixed_bins = torch.clamp(torch.floor(fixed_norm * (self.num_bins - 1)).long(), 0, self.num_bins - 1)

        # Joint histogram via one-hot outer product
        moving_onehot = F.one_hot(moving_bins, self.num_bins).float().transpose(1, 2)
        fixed_onehot = F.one_hot(fixed_bins, self.num_bins).float().transpose(1, 2)

        joint_hist = torch.bmm(moving_onehot, fixed_onehot.transpose(1, 2))
        joint_hist = joint_hist / (joint_hist.sum(dim=(1, 2), keepdim=True) + 1e-10)

        # Marginals and entropies
        p_moving = joint_hist.sum(dim=2, keepdim=True)
        p_fixed = joint_hist.sum(dim=1, keepdim=True)

        eps = 1e-10
        h_joint = -torch.sum(joint_hist * torch.log2(joint_hist + eps), dim=(1, 2))
        h_moving = -torch.sum(p_moving * torch.log2(p_moving + eps), dim=(1, 2))
        h_fixed = -torch.sum(p_fixed * torch.log2(p_fixed + eps), dim=(1, 2))

        mi = h_moving + h_fixed - h_joint
        nmi = 2.0 * mi / (h_moving + h_fixed + eps)
        return -nmi.mean()


class SmoothnessLoss3D(nn.Module):
    """Smoothness regularization for 3D deformation fields."""

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, flow):
        """
        Args:
            flow: [B, 3, D, H, W] deformation field
        """
        # Spatial gradients along each axis
        df_dx = torch.abs(flow[:, :, :, :, :-1] - flow[:, :, :, :, 1:])
        df_dy = torch.abs(flow[:, :, :, :-1, :] - flow[:, :, :, 1:, :])
        df_dz = torch.abs(flow[:, :, :-1, :, :] - flow[:, :, 1:, :, :])

        smoothness = torch.mean(df_dx) + torch.mean(df_dy) + torch.mean(df_dz)
        return self.weight * smoothness


def smoothness_loss_3d(deformation, img=None, alpha=0.0):
    """3D smoothness loss with optional bilateral filtering.

    Extends the 2D smoothness_loss to 3D with 6-neighbor differences.
    """
    # Axis-aligned differences (6 neighbors)
    diff_d = torch.abs(deformation[:, :, 1::, :, :] - deformation[:, :, 0:-1, :, :])
    diff_h = torch.abs(deformation[:, :, :, 1::, :] - deformation[:, :, :, 0:-1, :])
    diff_w = torch.abs(deformation[:, :, :, :, 1::] - deformation[:, :, :, :, 0:-1])

    # Diagonal differences (for additional smoothness)
    diff_dh = torch.abs(deformation[:, :, 0:-1, 0:-1, :] - deformation[:, :, 1::, 1::, :])
    diff_dw = torch.abs(deformation[:, :, 0:-1, :, 0:-1] - deformation[:, :, 1::, :, 1::])
    diff_hw = torch.abs(deformation[:, :, :, 0:-1, 0:-1] - deformation[:, :, :, 1::, 1::])

    if img is not None and alpha > 0.0:
        mask = img
        w_d = torch.exp(-alpha * torch.abs(mask[:, :, 1::, :, :] - mask[:, :, 0:-1, :, :]))
        w_d = torch.mean(w_d, dim=1, keepdim=True).repeat(1, 3, 1, 1, 1)
        w_h = torch.exp(-alpha * torch.abs(mask[:, :, :, 1::, :] - mask[:, :, :, 0:-1, :]))
        w_h = torch.mean(w_h, dim=1, keepdim=True).repeat(1, 3, 1, 1, 1)
        w_w = torch.exp(-alpha * torch.abs(mask[:, :, :, :, 1::] - mask[:, :, :, :, 0:-1]))
        w_w = torch.mean(w_w, dim=1, keepdim=True).repeat(1, 3, 1, 1, 1)
        w_dh = torch.exp(-alpha * torch.abs(mask[:, :, 0:-1, 0:-1, :] - mask[:, :, 1::, 1::, :]))
        w_dh = torch.mean(w_dh, dim=1, keepdim=True).repeat(1, 3, 1, 1, 1)
        w_dw = torch.exp(-alpha * torch.abs(mask[:, :, 0:-1, :, 0:-1] - mask[:, :, 1::, :, 1::]))
        w_dw = torch.mean(w_dw, dim=1, keepdim=True).repeat(1, 3, 1, 1, 1)
        w_hw = torch.exp(-alpha * torch.abs(mask[:, :, :, 0:-1, 0:-1] - mask[:, :, :, 1::, 1::]))
        w_hw = torch.mean(w_hw, dim=1, keepdim=True).repeat(1, 3, 1, 1, 1)
    else:
        w_d = w_h = w_w = w_dh = w_dw = w_hw = 1.0

    loss = (torch.mean(w_d * diff_d) + torch.mean(w_h * diff_h) + torch.mean(w_w * diff_w) +
            torch.mean(w_dh * diff_dh) + torch.mean(w_dw * diff_dw) + torch.mean(w_hw * diff_hw))
    return loss


class GradLoss3D(nn.Module):
    """Gradient diffusion loss for 3D deformation fields.
    L2 penalty on spatial gradients of the flow field.
    """

    def forward(self, flow):
        """
        Args:
            flow: [B, 3, D, H, W]
        """
        dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
        dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        return (dx ** 2).mean() + (dy ** 2).mean() + (dz ** 2).mean()


class MINDLoss3D(nn.Module):
    """3D MIND (Modality Independent Neighbourhood Descriptor) loss.

    Computes MIND feature descriptors for both images and measures L2 distance.
    MIND is invariant to intensity scaling and contrast, making it suitable for
    cross-modal registration.
    """

    def __init__(self, win=3):
        super().__init__()
        self.win = win

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, D, H, W] predicted (warped) image
            target: [B, 1, D, H, W] target (fixed) image
        """
        mind_pred = self._mind_descriptor(pred)
        mind_target = self._mind_descriptor(target)
        return F.mse_loss(mind_pred, mind_target)

    def _mind_descriptor(self, x):
        """Compute MIND descriptor for a 3D volume."""
        B, C, D, H, W = x.shape
        pad = self.win // 2

        # 6-neighborhood in 3D
        x_pad = F.pad(x, [pad]*6, mode='reflect')

        # Compute local mean and variance for 6 neighbors
        neighbors = []
        for dd, hh, ww in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            shifted = x_pad[:, :,
                     pad + dd:pad + dd + D,
                     pad + hh:pad + hh + H,
                     pad + ww:pad + ww + W]
            neighbors.append(shifted)

        # Stack neighbors and compute variance
        stacked = torch.cat(neighbors, dim=1)  # [B, 6, D, H, W]
        mean = stacked.mean(dim=1, keepdim=True)
        var = ((stacked - mean) ** 2).mean(dim=1, keepdim=True) + 1e-6

        # MIND descriptor: exp(-D/r) where D is normalized distance
        mind = torch.exp(-(stacked - mean) ** 2 / (var + 1e-6))
        return mind


def integrate_svf_3d(velocity, n_steps=7):
    """Integrate stationary velocity field via scaling & squaring.

    Computes exp(v) ≈ compose 2^n times the scaled velocity,
    yielding a diffeomorphic (smooth + invertible) deformation field.

    Args:
        velocity: [B, 3, D, H, W] velocity field (channels: x/W, y/H, z/D)
        n_steps: number of squaring steps (2^n_steps compositions)

    Returns:
        deformation: [B, 3, D, H, W] diffeomorphic deformation field
    """
    B, C, D, H, W = velocity.shape
    deformation = velocity / (2 ** n_steps)

    for _ in range(n_steps):
        # Warp deformation by itself (compose)
        # Build grid from current deformation
        grid_d, grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, D, device=velocity.device, dtype=velocity.dtype),
            torch.linspace(-1, 1, H, device=velocity.device, dtype=velocity.dtype),
            torch.linspace(-1, 1, W, device=velocity.device, dtype=velocity.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0)
        grid = grid.expand(B, -1, -1, -1, -1)  # [B, D, H, W, 3]

        # Grid for warping: identity + current deformation
        warp_grid = grid + deformation.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

        # Warp each channel of the deformation field
        d_x = F.grid_sample(deformation[:, 0:1], warp_grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
        d_y = F.grid_sample(deformation[:, 1:2], warp_grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
        d_z = F.grid_sample(deformation[:, 2:3], warp_grid, mode='bilinear',
                            padding_mode='border', align_corners=True)

        deformation = deformation + torch.cat([d_x, d_y, d_z], dim=1)

    return deformation
