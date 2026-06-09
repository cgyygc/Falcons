"""
Contrastive Learning enhanced Spatial Transformer Network.

This module provides a wrapper around STN networks that adds contrastive learning
capabilities to the encoder for better feature representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .contrastive_head import (
    ContrastiveHead,
    InfoNCELoss,
    ByolLoss,
    get_encoder_channels
)
from .unet_stn import UnetSTN, ResUnet
from .ukan_stn import UKANSTN, UKAN_Backbone


class ContrastiveSTNWrapper(nn.Module):
    """
    Wrapper that adds contrastive learning to STN networks.

    This wrapper:
    1. Extracts features from the encoder
    2. Projects them using contrastive head
    3. Computes contrastive loss between source and target features

    Usage:
        # Create wrapper around existing STN
        stn = UnetSTN(...)
        contrastive_stn = ContrastiveSTNWrapper(
            stn=stn,
            stn_type='unet',
            proj_dim=128,
            temperature=0.07
        )

        # Forward pass with contrastive loss
        warped_images, reg_term, contrastive_loss = contrastive_stn(
            img_a, img_b, return_contrastive_loss=True
        )
    """

    def __init__(
        self,
        stn,
        stn_type='unet',
        proj_dim=128,
        hidden_dim=None,
        num_proj_layers=2,
        temperature=0.07,
        contrastive_loss_type='infonce',
        num_contrastive_stages=None,
        loss_weight=0.1,
        cfg='A'
    ):
        """
        Args:
            stn: The base STN network (UnetSTN or UKANSTN)
            stn_type: Type of STN ('unet' or 'ukan')
            proj_dim: Projection dimension for contrastive learning
            hidden_dim: Hidden dimension for projection head
            num_proj_layers: Number of layers in projection head
            temperature: Temperature for InfoNCE loss
            contrastive_loss_type: Type of contrastive loss ('infonce' or 'byol')
            num_contrastive_stages: Number of encoder stages to use for contrastive
            loss_weight: Weight for contrastive loss in total loss
            cfg: Configuration name for encoder channels
        """
        super(ContrastiveSTNWrapper, self).__init__()

        self.stn = stn
        self.stn_type = stn_type
        self.loss_weight = loss_weight
        self.cfg = cfg

        # Get encoder channel dimensions
        encoder_channels = get_encoder_channels(stn_type, cfg=cfg)

        # Create contrastive head
        self.contrastive_head = ContrastiveHead(
            encoder_channels=encoder_channels,
            hidden_dims=hidden_dim,
            proj_dim=proj_dim,
            num_layers=num_proj_layers,
            num_stages=num_contrastive_stages
        )

        # Create contrastive loss
        if contrastive_loss_type == 'infonce':
            self.contrastive_loss_fn = InfoNCELoss(temperature=temperature)
        elif contrastive_loss_type == 'byol':
            self.contrastive_loss_fn = ByolLoss()
        else:
            raise ValueError(f"Unknown contrastive loss type: {contrastive_loss_type}")

    def get_contrastive_features(self, img_a, img_b):
        """
        Extract contrastive features from encoder.

        Args:
            img_a: Source image [B, C, H, W]
            img_b: Target image [B, C, H, W]

        Returns:
            Dictionary containing contrastive features from both images
        """
        # Get encoder features from backbone
        if hasattr(self.stn, 'offset_map'):
            # For UnetSTN and UKANSTN
            result = self.stn.offset_map.get_encoder_features(img_a, img_b)
            # Check if it returns separate features for a and b
            if isinstance(result, tuple) and len(result) == 2:
                features_a, features_b = result
            else:
                # Fallback for old API
                features_a = result
                features_b = result
        else:
            # Fallback
            features_a = []
            features_b = []

        # Project features using contrastive head
        if len(features_a) > 0:
            contrastive_a = self.contrastive_head(features_a)
        else:
            contrastive_a = None

        if len(features_b) > 0:
            contrastive_b = self.contrastive_head(features_b)
        else:
            contrastive_b = None

        return contrastive_a, contrastive_b

    def compute_contrastive_loss(self, contrastive_a, contrastive_b):
        """
        Compute contrastive loss between source and target features.

        Args:
            contrastive_a: Contrastive features from source image
            contrastive_b: Contrastive features from target image

        Returns:
            Contrastive loss value
        """
        if contrastive_a is None or contrastive_b is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Use the combined feature for contrastive loss
        feat_a = contrastive_a.get('combined_feature', None)
        feat_b = contrastive_b.get('combined_feature', None)

        if feat_a is None or feat_b is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        return self.contrastive_loss_fn(feat_a, feat_b)

    def forward(self, img_a, img_b, apply_on=None, return_contrastive_loss=False):
        """
        Forward pass with optional contrastive loss computation.

        Args:
            img_a: Source image [B, C, H, W]
            img_b: Target image [B, C, H, W]
            apply_on: List of tensors to apply transformation to
            return_contrastive_loss: Whether to compute and return contrastive loss

        Returns:
            If return_contrastive_loss=False:
                warped_images, reg_term
            If return_contrastive_loss=True:
                warped_images, reg_term, contrastive_loss
        """
        # Standard STN forward pass
        warped_images, reg_term = self.stn(img_a, img_b, apply_on=apply_on)

        contrastive_loss = None
        if return_contrastive_loss and self.training:
            # Extract contrastive features separately for source and target
            contrastive_a, contrastive_b = self.get_contrastive_features(img_a, img_b)

            # Compute contrastive loss
            contrastive_loss = self.compute_contrastive_loss(contrastive_a, contrastive_b)

            if contrastive_loss is not None:
                contrastive_loss = self.loss_weight * contrastive_loss

        if return_contrastive_loss:
            return warped_images, reg_term, contrastive_loss
        else:
            return warped_images, reg_term


class ContrastiveUnetSTN(nn.Module):
    """
    UnetSTN with integrated contrastive learning.

    This is a more integrated version that combines the contrastive head
    directly into the UnetSTN architecture.
    """

    def __init__(
        self,
        in_channels_a,
        in_channels_b,
        height,
        width,
        cfg='A',
        init_func='normal',
        stn_bilateral_alpha=0.01,
        init_to_identity=True,
        multi_resolution_regularization=1,
        proj_dim=128,
        temperature=0.07,
        use_contrastive=True
    ):
        super(ContrastiveUnetSTN, self).__init__()

        self.use_contrastive = use_contrastive
        self.oh = height
        self.ow = width
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base STN
        self.offset_map = ResUnet(in_channels_a, in_channels_b, cfg, init_func, init_to_identity).to(self.device)
        self.identity_grid = self._get_identity_grid()
        self.alpha = stn_bilateral_alpha
        self.multi_resolution_regularization = multi_resolution_regularization

        if use_contrastive:
            # Get encoder channels
            from .unet_stn import ndf
            encoder_channels = ndf[cfg]

            # Create contrastive head
            self.contrastive_head = ContrastiveHead(
                encoder_channels=encoder_channels,
                proj_dim=proj_dim,
                num_layers=2
            )

            # Create contrastive loss
            self.contrastive_loss_fn = InfoNCELoss(temperature=temperature)

    def _get_identity_grid(self):
        """Returns a sampling-grid that represents the identity transformation."""
        x = torch.linspace(-1.0, 1.0, self.ow)
        y = torch.linspace(-1.0, 1.0, self.oh)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return identity

    def get_encoder_features(self, img_a, img_b):
        """Extract encoder features for contrastive learning."""
        result = self.offset_map.get_encoder_features(img_a, img_b)
        # Handle both old API (single list) and new API (tuple of two lists)
        if isinstance(result, tuple) and len(result) == 2:
            return result  # (features_a, features_b)
        else:
            return result, result  # For backward compatibility

    def forward(self, img_a, img_b, apply_on=None, return_contrastive_loss=False):
        """
        Forward pass.

        Args:
            img_a: Source image [B, C, H, W]
            img_b: Target image [B, C, H, W]
            apply_on: List of tensors to apply transformation to
            return_contrastive_loss: Whether to compute and return contrastive loss

        Returns:
            warped_images, reg_term, [contrastive_loss]
        """
        if img_a.is_cuda and not self.identity_grid.is_cuda:
            self.identity_grid = self.identity_grid.to(img_a.device)

        # Get deformation field
        b_size = img_a.size(0)
        deformation = self.offset_map(img_a, img_b)
        deformation_upsampled = deformation

        if deformation.size(2) != self.oh or deformation.size(3) != self.ow:
            deformation_upsampled = F.interpolate(
                deformation, (self.oh, self.ow),
                mode='bilinear'
            )

        resampling_grid = (self.identity_grid.repeat(b_size, 1, 1, 1) + deformation_upsampled).permute([0, 2, 3, 1])

        # Warp images
        if apply_on is None:
            apply_on = [img_a]

        warped_images = []
        for img in apply_on:
            warped_images.append(F.grid_sample(
                img, resampling_grid, mode='bilinear',
                padding_mode='zeros', align_corners=False
            ))

        # Calculate STN regularization term
        reg_term = self._calculate_regularization_term(deformation, warped_images[0])

        # Calculate contrastive loss if requested
        contrastive_loss = None
        if return_contrastive_loss and self.use_contrastive and self.training:
            # Extract separate features for source and target images
            features_a, features_b = self.get_encoder_features(img_a, img_b)
            
            # Project features using contrastive head
            contrastive_a = self.contrastive_head(features_a)
            contrastive_b = self.contrastive_head(features_b)

            # Use combined features for contrastive loss
            feat_a = contrastive_a['combined_feature']
            feat_b = contrastive_b['combined_feature']

            # InfoNCE loss with query from source, positive key from target
            # Batch negatives are automatically handled by InfoNCELoss
            contrastive_loss = self.contrastive_loss_fn(feat_a, feat_b)

        if return_contrastive_loss:
            return warped_images, reg_term, contrastive_loss
        else:
            return warped_images, reg_term

    def _calculate_regularization_term(self, deformation, img):
        """Calculate the regularization term of the predicted deformation."""
        from .stn_losses import smoothness_loss

        dh, dw = deformation.size(2), deformation.size(3)
        img = None if img is None else img.detach()
        reg = torch.tensor(0.0, device=deformation.device, dtype=deformation.dtype)
        factor = 1.0

        for i in range(self.multi_resolution_regularization):
            if i != 0:
                deformation_resized = F.interpolate(
                    deformation, (dh // (2 ** i), dw // (2 ** i)), mode='bilinear'
                )
                img_resized = F.interpolate(
                    img, (dh // (2 ** i), dw // (2 ** i)), mode='bilinear'
                )
            elif deformation.size()[2::] != img.size()[2::]:
                deformation_resized = deformation
                img_resized = F.interpolate(
                    img, deformation.size()[2::], mode='bilinear'
                )
            else:
                deformation_resized = deformation
                img_resized = img

            reg += factor * smoothness_loss(deformation_resized, img_resized, alpha=self.alpha)
            factor /= 2.0

        return reg


def create_contrastive_stn(stn_type, **kwargs):
    """
    Factory function to create contrastive-enhanced STN.

    Args:
        stn_type: Type of STN ('unet' or 'ukan')
        **kwargs: Arguments passed to STN constructor

    Returns:
        ContrastiveSTNWrapper instance
    """
    if stn_type == 'unet':
        # Create base UnetSTN
        base_stn = UnetSTN(**kwargs)
    elif stn_type == 'ukan':
        # Create base UKANSTN
        base_stn = UKANSTN(**kwargs)
    else:
        raise ValueError(f"Unknown STN type: {stn_type}")

    # Wrap with contrastive learning
    return ContrastiveSTNWrapper(
        stn=base_stn,
        stn_type=stn_type,
        proj_dim=kwargs.get('proj_dim', 128),
        temperature=kwargs.get('temperature', 0.07),
        loss_weight=kwargs.get('contrastive_weight', 0.1)
    )
