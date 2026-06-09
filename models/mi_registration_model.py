"""
Mutual Information based Registration Model.

This is a traditional image registration method that maximizes the mutual information
between the moving and fixed images. It serves as a baseline comparison for NEMAR.

Reference:
- Pluim et al., "Mutual information matching in multiresolution contexts",
  Image and Vision Computing, 2001.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS

from .base_model import BaseModel
from . import networks


class NormalizedMutualInformationLoss(nn.Module):
    """
    Normalized Mutual Information (NMI) Loss for multi-modal image registration.

    NMI = MI / sqrt(H(X) * H(Y))
    where H(X) is the entropy.

    This is more stable than raw MI for multi-modal registration.
    """

    def __init__(self, num_bins=64, sigma=2.0):
        super(NormalizedMutualInformationLoss, self).__init__()
        self.num_bins = num_bins
        self.sigma = sigma

    def forward(self, moving, fixed):
        """
        Compute normalized mutual information between moving and fixed images.

        Args:
            moving: Registered moving image [B, 1, H, W]
            fixed: Fixed target image [B, 1, H, W]

        Returns:
            NMI loss (negative NMI, to be minimized)
        """
        b, _, h, w = moving.shape

        # Normalize images to [0, 1] per image
        moving_flat = moving.view(b, -1)
        fixed_flat = fixed.view(b, -1)

        # Per-image normalization (min-max scaling)
        moving_min = moving_flat.min(dim=1, keepdim=True)[0]
        moving_max = moving_flat.max(dim=1, keepdim=True)[0]
        moving_norm = (moving_flat - moving_min) / (moving_max - moving_min + 1e-10)

        fixed_min = fixed_flat.min(dim=1, keepdim=True)[0]
        fixed_max = fixed_flat.max(dim=1, keepdim=True)[0]
        fixed_norm = (fixed_flat - fixed_min) / (fixed_max - fixed_min + 1e-10)

        # Discretize into bins
        moving_bins = torch.floor(moving_norm * (self.num_bins - 1)).long()
        fixed_bins = torch.floor(fixed_norm * (self.num_bins - 1)).long()

        moving_bins = torch.clamp(moving_bins, 0, self.num_bins - 1)
        fixed_bins = torch.clamp(fixed_bins, 0, self.num_bins - 1)

        # Compute joint histogram efficiently using one-hot encoding
        # This is much faster than looping through pixels
        moving_onehot = F.one_hot(moving_bins, self.num_bins).float()  # [B, N, num_bins]
        fixed_onehot = F.one_hot(fixed_bins, self.num_bins).float()   # [B, N, num_bins]

        # Reshape for outer product
        moving_onehot = moving_onehot.transpose(1, 2)  # [B, num_bins, N]
        fixed_onehot = fixed_onehot.transpose(1, 2)    # [B, num_bins, N]

        # Joint histogram: [B, num_bins, num_bins]
        joint_hist = torch.bmm(
            moving_onehot,  # [B, num_bins, N]
            fixed_onehot.transpose(1, 2)  # [B, N, num_bins]
        )

        # Normalize joint histogram
        joint_hist = joint_hist / (joint_hist.sum(dim=(1, 2), keepdim=True) + 1e-10)

        # Compute marginal histograms
        p_moving = joint_hist.sum(dim=2, keepdim=True)  # [B, num_bins, 1]
        p_fixed = joint_hist.sum(dim=1, keepdim=True)  # [B, 1, num_bins]

        # Compute entropies
        epsilon = 1e-10
        h_joint = -torch.sum(joint_hist * torch.log2(joint_hist + epsilon), dim=(1, 2))
        h_moving = -torch.sum(p_moving * torch.log2(p_moving + epsilon), dim=(1, 2))
        h_fixed = -torch.sum(p_fixed * torch.log2(p_fixed + epsilon), dim=(1, 2))

        # Compute MI
        mi = h_joint - h_moving - h_fixed

        # Normalized MI
        nmi = 2.0 * mi / (h_moving + h_fixed + epsilon)

        return -nmi.mean()  # Return negative NMI (to be minimized)


class GridSampler(nn.Module):
    """
    Differentiable grid sampler for affine transformations.
    """

    def __init__(self):
        super(GridSampler, self).__init__()

    def forward(self, image, theta, size=None):
        """
        Apply affine transformation to image.

        Args:
            image: Input image [B, 1, H, W]
            theta: Affine transformation parameters [B, 6]
                   [scale_x, scale_y, shear, rotation, translate_x, translate_y]
            size: Output size (H, W)

        Returns:
            Transformed image
        """
        b, _, h, w = image.shape
        if size is None:
            size = (h, w)

        # Create affine matrix
        cos_theta = torch.cos(theta[:, 3])
        sin_theta = torch.sin(theta[:, 3])

        # Affine matrix: [scale_x * cos, -sin, tx]
        #                 [sin, scale_y * cos, ty]
        affine_matrix = torch.zeros(b, 2, 3, device=image.device)
        affine_matrix[:, 0, 0] = theta[:, 0] * cos_theta  # scale_x * cos
        affine_matrix[:, 0, 1] = -sin_theta + theta[:, 2]  # -sin + shear
        affine_matrix[:, 0, 2] = theta[:, 4]  # translate_x
        affine_matrix[:, 1, 0] = sin_theta
        affine_matrix[:, 1, 1] = theta[:, 1] * cos_theta  # scale_y * cos
        affine_matrix[:, 1, 2] = theta[:, 5]  # translate_y

        # Create sampling grid
        grid = F.affine_grid(affine_matrix, [b, 1, size[0], size[1]], align_corners=False)

        # Sample from image
        transformed = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)

        return transformed


class SmoothnessLoss(nn.Module):
    """
    Smoothness regularization for transformation parameters.
    """

    def __init__(self, lambda_smooth=0.1):
        super(SmoothnessLoss, self).__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self, theta):
        """
        Compute smoothness loss on transformation parameters.

        Args:
            theta: Transformation parameters [B, 6]

        Returns:
            Smoothness loss
        """
        # Penalize deviations from identity transform
        # Identity: [1, 1, 0, 0, 0, 0]
        identity = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=theta.device)
        smoothness = torch.mean(theta[:, :3] ** 2) + 10.0 * torch.mean(theta[:, 4:6] ** 2)
        return self.lambda_smooth * smoothness


class MIRegistrationModel(BaseModel):
    """
    Mutual Information based Image Registration Model.

    This model performs registration by maximizing mutual information between
    the registered moving image and fixed image. It uses traditional optimization
    (L-BFGS) rather than deep learning.

    Architecture:
        - Moving image (A) -> Affine transformation -> Registered A
        - Registered A and Fixed image (B) -> MI loss -> Optimize theta

    This serves as a baseline comparison for NEMAR.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add MI registration specific options."""
        parser = BaseOptions.modify_commandline_options(parser, is_train)

        # Optimization parameters
        parser.add_argument('--mi_num_bins', type=int, default=64,
                            help='Number of bins for MI histogram')
        parser.add_argument('--mi_sigma', type=float, default=2.0,
                            help='Sigma for Parzen window estimation')
        parser.add_argument('--lambda_mi_smooth', type=float, default=0.1,
                            help='Smoothness regularization weight')
        parser.add_argument('--optim_lr', type=float, default=0.1,
                            help='Learning rate for L-BFGS optimizer')
        parser.add_argument('--optim_max_iter', type=int, default=100,
                            help='Maximum iterations per image')
        parser.add_argument('--optim_tolerance', type=float, default=1e-5,
                            help='Tolerance for optimizer convergence')
        parser.add_argument('--optim_history_size', type=int, default=100,
                            help='History size for L-BFGS')
        parser.add_argument('--optim_line_search', type=str, default='strong_wolfe',
                            help='Line search for L-BFGS')

        # Transformation type
        parser.add_argument('--transform_type', type=str, default='affine',
                            choices=['affine', 'similarity', 'rigid'],
                            help='Type of transformation')

        return parser

    def __init__(self, opt):
        """Initialize MI Registration Model."""
        BaseModel.__init__(self, opt)

        # Model names for saving
        self.model_names = []

        # Loss names
        self.loss_names = ['MI', 'smoothness', 'total']

        # Visual names
        self.visual_names = ['real_A', 'real_B', 'registered_A', 'error_map']

        # Define losses
        self.criterionMI = NormalizedMutualInformationLoss(
            num_bins=opt.mi_num_bins,
            sigma=opt.mi_sigma
        )
        self.criterionSmooth = SmoothnessLoss(lambda_smooth=opt.lambda_mi_smooth)

        # Grid sampler for transformations
        self.grid_sampler = GridSampler()

        # Store optimizer options
        self.transform_type = opt.transform_type
        self.optim_lr = getattr(opt, 'optim_lr', 0.1)
        self.optim_max_iter = getattr(opt, 'optim_max_iter', 100)
        self.optim_tolerance = getattr(opt, 'optim_tolerance', 1e-5)
        self.optim_history_size = getattr(opt, 'optim_history_size', 100)
        self.optim_line_search = getattr(opt, 'optim_line_search', 'strong_wolfe')

        # Register buffer for theta
        self.register_buffer('theta', torch.zeros(1, 6))

    def initialize_transform(self, batch_size):
        """
        Initialize transformation parameters based on type.

        Args:
            batch_size: Batch size
        """
        theta = torch.zeros(batch_size, 6, device=self.device)

        if self.transform_type == 'rigid':
            # Rigid: scale_x = scale_y = 1, shear = 0
            theta[:, 0] = 0.0  # log scale
            theta[:, 1] = 0.0  # log scale
            theta[:, 2] = 0.0  # shear
        elif self.transform_type == 'similarity':
            # Similarity: scale_x = scale_y, shear = 0
            theta[:, 0] = 0.0  # log scale
            theta[:, 1] = 0.0  # log scale
            theta[:, 2] = 0.0  # shear
        elif self.transform_type == 'affine':
            # Affine: all parameters free
            theta[:, 0] = 0.0  # log scale
            theta[:, 1] = 0.0  # log scale
            theta[:, 2] = 0.0  # shear
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")

        # Identity transform: rotation=0, translation=0
        theta[:, 3] = 0.0  # rotation
        theta[:, 4] = 0.0  # translate_x
        theta[:, 5] = 0.0  # translate_y

        # Ensure parameters require gradient
        theta.requires_grad_(True)

        return theta

    def optimize_registration(self, moving, fixed, theta):
        """
        Optimize transformation parameters using L-BFGS.

        Args:
            moving: Moving image [B, 1, H, W]
            fixed: Fixed image [B, 1, H, W]
            theta: Initial transformation parameters [B, 6]

        Returns:
            Optimal theta and registered image
        """
        b = moving.shape[0]

        def closure():
            optimizer.zero_grad()

            # Apply transformation
            registered = self.grid_sampler(moving, theta, size=(fixed.shape[2], fixed.shape[3]))

            # Compute losses
            mi_loss = self.criterionMI(registered, fixed)
            smoothness_loss = self.criterionSmooth(theta)
            total_loss = mi_loss + smoothness_loss

            # Backward
            total_loss.backward()

            # Store losses
            self.loss_MI = mi_loss.item()
            self.loss_smoothness = smoothness_loss.item()
            self.loss_total = total_loss.item()

            return total_loss

        # L-BFGS optimizer
        optimizer = LBFGS(
            [theta],
            lr=self.optim_lr,
            max_iter=self.optim_max_iter,
            tolerance_grad=self.optim_tolerance,
            tolerance_change=self.optim_tolerance,
            history_size=self.optim_history_size,
            line_search_fn=self.optim_line_search
        )

        # Optimize
        optimizer.step(closure)

        # Get final registration
        registered = self.grid_sampler(moving, theta, size=(fixed.shape[2], fixed.shape[3]))

        return theta.detach(), registered

    def set_input(self, input):
        """Unpack input data from the dataloader."""
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass - optimize registration for current batch."""
        # Initialize transformation
        theta = self.initialize_transform(self.real_A.shape[0])

        # Optimize registration
        self.theta, self.registered_A = self.optimize_registration(
            self.real_A, self.real_B, theta
        )

        # Compute error map for visualization
        self.error_map = torch.abs(self.registered_A - self.real_B)

    def optimize_parameters(self):
        """No optimization in test mode - forward does everything."""
        pass

    def test(self):
        """Test mode - run registration."""
        with torch.no_grad():
            self.forward()

    def get_current_visuals(self):
        """Return visualization images."""
        return {
            'real_A': self.real_A,
            'real_B': self.real_B,
            'registered_A': self.registered_A,
            'error_map': self.error_map
        }

    def get_current_losses(self):
        """Return current losses."""
        return {
            'MI': self.loss_MI,
            'smoothness': self.loss_smoothness,
            'total': self.loss_total
        }

    def save_networks(self, epoch):
        """MI registration doesn't save networks (only transformation parameters)."""
        pass

    def load_networks(self, epoch):
        """MI registration doesn't load networks."""
        pass
