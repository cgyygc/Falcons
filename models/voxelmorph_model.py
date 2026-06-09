"""
VoxelMorph-MI: A deep learning approach for multi-modal medical image registration.

Uses Mutual Information loss instead of NCC for cross-modal registration (e.g., CT→MR).

Reference:
- Balakrishnan et al., "VoxelMorph: A Learning Framework for Deformable Medical Image Registration"
  IEEE TMI 2019
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


class SpatialTransformer(nn.Module):
    """Spatial transformer network for applying deformation fields."""

    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, x, flow):
        B, C, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=x.device),
            torch.arange(0, W, dtype=torch.float32, device=x.device),
            indexing='ij'
        )
        grid_y = grid_y.contiguous()  # [H, W]
        grid_x = grid_x.contiguous()  # [H, W]

        # Normalize to [-1, 1]
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        grid_x = 2.0 * grid_x / (W - 1) - 1.0

        # Add flow to grid
        grid_y = grid_y.unsqueeze(0) + flow[:, 1, :, :]  # [B, H, W]
        grid_x = grid_x.unsqueeze(0) + flow[:, 0, :, :]  # [B, H, W]

        grid = torch.stack([grid_x, grid_y], dim=3)  # [B, H, W, 2]
        warped = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return warped


class Unet(nn.Module):
    """
    UNet architecture for VoxelMorph.

    Uses max pooling for downsampling and transposed convolution for upsampling.
    Includes skip connections between encoder and decoder.
    """

    def __init__(self, in_channels=2, out_channels=2, num_features=[32, 64, 128, 256], use_dropout=False):
        super(Unet, self).__init__()

        # Encoder (with pooling for downsampling)
        self.encoder1 = self._conv_block(in_channels, num_features[0], use_dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self._conv_block(num_features[0], num_features[1], use_dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self._conv_block(num_features[1], num_features[2], use_dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self._conv_block(num_features[2], num_features[3], use_dropout)

        # Decoder (with transposed conv for upsampling)
        self.up3 = nn.ConvTranspose2d(num_features[3], num_features[2], kernel_size=2, stride=2)
        self.decoder3 = self._conv_block(num_features[2] + num_features[2], num_features[2], use_dropout)

        self.up2 = nn.ConvTranspose2d(num_features[2], num_features[1], kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(num_features[1] + num_features[1], num_features[1], use_dropout)

        self.up1 = nn.ConvTranspose2d(num_features[1], num_features[0], kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(num_features[0] + num_features[0], num_features[0], use_dropout)

        # Final output
        self.output = nn.Conv2d(num_features[0], out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels, use_dropout):
        """Convolution block with normalization."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass with skip connections.

        Args:
            x: Input image [B, 2, H, W] (concatenated moving and fixed)

        Returns:
            flow: Predicted deformation field [B, 2, H, W]
        """
        # Encoder
        e1 = self.encoder1(x)      # [B, 32, H, W]
        e2 = self.encoder2(self.pool1(e1))  # [B, 64, H/2, W/2]
        e3 = self.encoder3(self.pool2(e2))  # [B, 128, H/4, W/4]
        e4 = self.encoder4(self.pool3(e3))  # [B, 256, H/8, W/8]

        # Decoder with skip connections
        d3 = self.decoder3(torch.cat([self.up3(e4), e3], dim=1))  # [B, 128, H/4, W/4]
        d2 = self.decoder2(torch.cat([self.up2(d3), e2], dim=1))  # [B, 64, H/2, W/2]
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))  # [B, 32, H, W]

        # Output flow field
        flow = self.output(d1)  # [B, 2, H, W]

        return flow


class NormalizedMutualInformationLoss(nn.Module):
    """Normalized Mutual Information loss for multi-modal registration."""

    def __init__(self, num_bins=64):
        super(NormalizedMutualInformationLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, moving, fixed):
        """Compute negative NMI (to be minimized).

        Args:
            moving: Warped moving image [B, 1, H, W]
            fixed: Fixed target image [B, 1, H, W]
        """
        b, _, h, w = moving.shape
        moving_flat = moving.view(b, -1)
        fixed_flat = fixed.view(b, -1)

        # Per-image normalization to [0, 1]
        moving_norm = (moving_flat - moving_flat.min(dim=1, keepdim=True)[0]) / \
                      (moving_flat.max(dim=1, keepdim=True)[0] - moving_flat.min(dim=1, keepdim=True)[0] + 1e-10)
        fixed_norm = (fixed_flat - fixed_flat.min(dim=1, keepdim=True)[0]) / \
                     (fixed_flat.max(dim=1, keepdim=True)[0] - fixed_flat.min(dim=1, keepdim=True)[0] + 1e-10)

        # Discretize into bins
        moving_bins = torch.clamp(torch.floor(moving_norm * (self.num_bins - 1)).long(), 0, self.num_bins - 1)
        fixed_bins = torch.clamp(torch.floor(fixed_norm * (self.num_bins - 1)).long(), 0, self.num_bins - 1)

        # Joint histogram via one-hot outer product
        moving_onehot = F.one_hot(moving_bins, self.num_bins).float().transpose(1, 2)  # [B, bins, N]
        fixed_onehot = F.one_hot(fixed_bins, self.num_bins).float().transpose(1, 2)    # [B, bins, N]

        joint_hist = torch.bmm(moving_onehot, fixed_onehot.transpose(1, 2))  # [B, bins, bins]
        joint_hist = joint_hist / (joint_hist.sum(dim=(1, 2), keepdim=True) + 1e-10)

        # Marginals
        p_moving = joint_hist.sum(dim=2, keepdim=True)
        p_fixed = joint_hist.sum(dim=1, keepdim=True)

        # Entropies
        eps = 1e-10
        h_joint = -torch.sum(joint_hist * torch.log2(joint_hist + eps), dim=(1, 2))
        h_moving = -torch.sum(p_moving * torch.log2(p_moving + eps), dim=(1, 2))
        h_fixed = -torch.sum(p_fixed * torch.log2(p_fixed + eps), dim=(1, 2))

        mi = h_moving + h_fixed - h_joint
        nmi = 2.0 * mi / (h_moving + h_fixed + eps)
        return -nmi.mean()


class NCCLoss(nn.Module):
    """Normalized Cross-Correlation loss (for same-modality registration)."""

    def __init__(self, win=9):
        super(NCCLoss, self).__init__()
        self.win = win

    def forward(self, pred, target):
        B, _, H, W = pred.shape
        win = self.win
        pad = win // 2

        pred_pad = F.pad(pred, (pad, pad, pad, pad), mode='reflect')
        target_pad = F.pad(target, (pad, pad, pad, pad), mode='reflect')

        patches_pred = pred_pad.unfold(2, win, 1)
        patches_target = target_pad.unfold(2, win, 1)

        mean_pred = patches_pred.mean(dim=4, keepdim=True)
        mean_target = patches_target.mean(dim=4, keepdim=True)

        pred_centered = patches_pred - mean_pred
        target_centered = patches_target - mean_target

        ncc_num = (pred_centered * target_centered).sum(dim=4)
        ncc_den = torch.sqrt((pred_centered ** 2).sum(dim=4) *
                           (target_centered ** 2).sum(dim=4) + 1e-10)

        ncc = ncc_num / ncc_den
        return 1.0 - ncc.mean()


class SmoothnessLoss(nn.Module):
    """
    Smoothness regularization for deformation fields.
    Penalizes spatial gradients to ensure smooth deformations.
    """

    def __init__(self, weight=0.01):
        super(SmoothnessLoss, self).__init__()
        self.weight = weight

    def forward(self, flow):
        """
        Compute smoothness loss on deformation field.

        Args:
            flow: Deformation field [B, 2, H, W]

        Returns:
            Smoothness loss
        """
        # Spatial gradients
        df_dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        df_dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])

        # Total smoothness
        smoothness = torch.mean(df_dx) + torch.mean(df_dy)

        return self.weight * smoothness


class VoxelMorphModel(BaseModel):
    """
    VoxelMorph-MI: Deep Learning based Multi-Modal Image Registration.

    Uses Mutual Information loss for cross-modal registration (CT→MR).
    Also supports NCC and MSE for same-modality registration.

    Reference:
    - Balakrishnan et al., "VoxelMorph: A Learning Framework for Deformable Medical Image Registration",
      IEEE TMI 2019
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add VoxelMorph specific options."""
        parser = BaseOptions.modify_commandline_options(parser, is_train)

        parser.add_argument('--vm_num_features', type=int, default=[32, 64, 128, 256],
                            nargs='+', help='UNet feature channels')
        parser.add_argument('--vm_use_dropout', action='store_true',
                            help='Use dropout in UNet')
        parser.add_argument('--vm_loss_type', type=str, default='mi',
                            choices=['mi', 'ncc', 'mse'],
                            help='Loss type: mi (multi-modal), ncc (same-modality), mse')
        parser.add_argument('--vm_smoothness_weight', type=float, default=1.0,
                            help='Smoothness regularization weight')
        parser.add_argument('--vm_ncc_window', type=int, default=9,
                            help='NCC window size (only for ncc loss)')
        parser.add_argument('--vm_mi_bins', type=int, default=64,
                            help='Number of bins for MI histogram')

        if is_train:
            parser.add_argument('--vm_lr', type=float, default=1e-4,
                                help='Learning rate')
            parser.add_argument('--vm_weight_decay', type=float, default=0.0,
                                help='Weight decay')
            parser.add_argument('--vm_niter', type=int, default=200,
                                help='Number of training epochs')
            parser.add_argument('--vm_batch_size', type=int, default=4,
                                help='Batch size')

        return parser

    def __init__(self, opt):
        """Initialize VoxelMorph Model."""
        BaseModel.__init__(self, opt)

        self.model_names = ['V']

        self.loss_type = getattr(opt, 'vm_loss_type', 'mi')
        self.loss_names = ['total', 'similarity', 'smoothness']

        self.visual_names = ['moving', 'fixed', 'warped', 'flow_vis']

        self.num_features = getattr(opt, 'vm_num_features', [32, 64, 128, 256])
        self.use_dropout = getattr(opt, 'vm_use_dropout', False)
        self.smoothness_weight = getattr(opt, 'vm_smoothness_weight', 1.0)

        self.netV = Unet(
            in_channels=2,
            out_channels=2,
            num_features=self.num_features,
            use_dropout=self.use_dropout
        )

        self.spatial_transform = SpatialTransformer()

        # Loss functions
        self.criterionMI = NormalizedMutualInformationLoss(
            num_bins=getattr(opt, 'vm_mi_bins', 64)
        )
        self.criterionNCC = NCCLoss(win=getattr(opt, 'vm_ncc_window', 9))
        self.criterionMSE = nn.MSELoss()
        self.criterionSmooth = SmoothnessLoss(weight=1.0)

        if self.isTrain:
            lr = getattr(opt, 'vm_lr', 1e-4)
            weight_decay = getattr(opt, 'vm_weight_decay', 0.0)
            self.optimizer = torch.optim.Adam(
                self.netV.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        self.lr_scheduler = None

    def set_input(self, input):
        """Unpack input data from dataloader."""
        self.moving = input['A'].to(self.device)  # [B, 1, H, W]
        self.fixed = input['B'].to(self.device)    # [B, 1, H, W]
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""
        x = torch.cat([self.moving, self.fixed], dim=1)  # [B, 2, H, W]
        flow = self.netV(x)  # [B, 2, H, W]
        self.warped = self.spatial_transform(self.moving, flow)
        self.flow_vis = torch.cat([flow[:, 0:1, :, :], flow[:, 1:2, :, :]], dim=1)

        # Similarity loss
        if self.loss_type == 'mi':
            self.loss_similarity = self.criterionMI(self.warped, self.fixed)
        elif self.loss_type == 'ncc':
            self.loss_similarity = self.criterionNCC(self.warped, self.fixed)
        else:  # mse
            self.loss_similarity = self.criterionMSE(self.warped, self.fixed)

        self.loss_smoothness = self.criterionSmooth(flow)
        self.loss_total = self.loss_similarity + self.smoothness_weight * self.loss_smoothness

    def optimize_parameters(self):
        """Update network weights."""
        self.optimizer.zero_grad()
        self.loss_total.backward()
        self.optimizer.step()

    def get_current_visuals(self):
        return {
            'moving': self.moving,
            'fixed': self.fixed,
            'warped': self.warped,
            'flow_vis': self.flow_vis
        }

    def get_current_losses(self):
        return {
            'total': self.loss_total.item(),
            'similarity': self.loss_similarity.item(),
            'smoothness': self.loss_smoothness.item()
        }

    def save_networks(self, epoch):
        self.save_network(self.netV, 'V', epoch)

    def load_networks(self, epoch):
        self.load_network(self.netV, 'V', epoch)
