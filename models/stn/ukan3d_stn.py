"""
3D UKAN-STN: Spatial Transformer Network with UKAN3D backbone for 3D registration.

Extends the 2D UKANSTN to 3D volumetric registration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ukan3d import UKAN3D_Backbone
from .spatial_transformer_3d import smoothness_loss_3d, integrate_svf_3d

sampling_align_corners = True


class UKAN3DSTN(nn.Module):
    """3D UKAN-based Spatial Transformer Network.

    Predicts a 3D velocity field [B, 3, D, H, W] and integrates it
    via scaling & squaring to produce a diffeomorphic deformation field.
    """

    def __init__(self, img_size=(192, 160, 192), in_channels=2, out_channels=3,
                 encoder_channels=(8, 16, 32, 64),
                 kan_embed_dims=(32, 64, 128),
                 kan_depths=(1, 1, 1),
                 use_svf=True, svf_steps=7):
        super().__init__()

        self.oh, self.ow, self.od = img_size  # H, W, D
        self.use_svf = use_svf
        self.svf_steps = svf_steps
        self.offset_map = UKAN3D_Backbone(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_channels=encoder_channels,
            kan_embed_dims=kan_embed_dims,
            kan_depths=kan_depths,
        )
        self.identity_grid = self.get_identity_grid()

    def get_identity_grid(self):
        """Create 3D identity grid in [-1, 1]."""
        z = torch.linspace(-1.0, 1.0, self.od)
        y = torch.linspace(-1.0, 1.0, self.oh)
        x = torch.linspace(-1.0, 1.0, self.ow)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        # Grid order: (x/W, y/H, z/D) as expected by F.grid_sample
        identity = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)  # [1, 3, D, H, W]
        return identity

    def forward(self, img_a, img_b, apply_on=None):
        """Forward pass: predict deformation and warp image.

        Args:
            img_a: [B, 1, D, H, W] moving image
            img_b: [B, 1, D, H, W] fixed image
            apply_on: optional tensor to warp instead of img_a

        Returns:
            list of warped images (one per image to warp)
        """
        if img_a.is_cuda and not self.identity_grid.is_cuda:
            self.identity_grid = self.identity_grid.to(img_a.device)

        b_size = img_a.size(0)
        x = torch.cat([img_a, img_b], dim=1)
        velocity = self.offset_map(x)

        # Handle size mismatch via interpolation
        if velocity.size(2) != self.od or velocity.size(3) != self.oh or velocity.size(4) != self.ow:
            velocity = F.interpolate(
                velocity, (self.od, self.oh, self.ow),
                mode='trilinear', align_corners=sampling_align_corners
            )

        # Integrate velocity field to get diffeomorphic deformation
        if self.use_svf:
            deformation = integrate_svf_3d(velocity, n_steps=self.svf_steps)
        else:
            deformation = velocity

        resampling_grid = (self.identity_grid.repeat(b_size, 1, 1, 1, 1) + deformation)
        resampling_grid = resampling_grid.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

        # Warp images
        target = apply_on if apply_on is not None else img_a
        if isinstance(target, (list, tuple)):
            warped_images = []
            for t in target:
                warped_images.append(
                    F.grid_sample(t, resampling_grid, align_corners=sampling_align_corners,
                                  mode='bilinear', padding_mode='border')
                )
        else:
            warped_images = [
                F.grid_sample(target, resampling_grid, align_corners=sampling_align_corners,
                              mode='bilinear', padding_mode='border')
            ]

        return warped_images, deformation

    def get_grid(self, img_a, img_b, return_offsets_only=False):
        """Return the predicted sampling grid / deformation field."""
        if img_a.is_cuda and not self.identity_grid.is_cuda:
            self.identity_grid = self.identity_grid.to(img_a.device)

        b_size = img_a.size(0)
        x = torch.cat([img_a, img_b], dim=1)
        velocity = self.offset_map(x)

        if velocity.size(2) != self.od or velocity.size(3) != self.oh or velocity.size(4) != self.ow:
            velocity = F.interpolate(
                velocity, (self.od, self.oh, self.ow),
                mode='trilinear', align_corners=sampling_align_corners
            )

        if self.use_svf:
            deformation = integrate_svf_3d(velocity, n_steps=self.svf_steps)
        else:
            deformation = velocity

        if return_offsets_only:
            return deformation.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

        resampling_grid = (self.identity_grid.repeat(b_size, 1, 1, 1, 1) + deformation)
        return resampling_grid.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

    def get_encoder_features(self, img_a, img_b):
        """Extract encoder features for contrastive learning."""
        x = torch.cat([img_a, img_b], dim=1)

        e1 = self.offset_map.enc1(x)
        e2 = self.offset_map.enc2(self.offset_map.pool1(e1))
        e3 = self.offset_map.enc3(self.offset_map.pool2(e2))
        e4 = self.offset_map.enc4(self.offset_map.pool3(e3))

        return [e2, e3, e4]

    def reg_term(self, img=None, alpha=0.0):
        """Compute smoothness regularization on the current deformation."""
        # This needs to be called after forward() to have access to deformation
        # For now, return 0 - the loss will be computed externally
        return 0.0
