"""
DINO-Reg 2D: DINOv2 feature-based registration with ConvexAdam optimization.

Reference: "DINO-Reg: General Purpose Image Encoder for Training-Free
Multi-modal Deformable Medical Image Registration" (MICCAI 2024)

This is a training-free method that uses DINOv2 ViT features + ConvexAdam optimization.
Adapted for 2D cross-modal registration (CT→MR).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .convexadam_2d import (
    coupled_convex2d, adam_instance_opt_2d, spatial_transform_2d, correlate2d
)


class DINOv2FeatureExtractor:
    """Extract DINOv2 features for 2D images."""

    def __init__(self, model_name='dinov2_vits14', feat_size=(36, 36), device='cuda'):
        self.device = device
        self.feat_size = feat_size
        self.patch_size = 14

        # Load from local cache to avoid network issues
        self.model = self._load_model(model_name, device)
        self.embed_dim = self.model.embed_dim

        self.target_h = feat_size[0] * self.patch_size
        self.target_w = feat_size[1] * self.patch_size

    @staticmethod
    def _load_model(model_name, device):
        """Load DINOv2 model from local cache."""
        import sys
        hub_dir = '/root/.cache/torch/hub/facebookresearch_dinov2_main'
        ckpt_dir = '/root/.cache/torch/hub/checkpoints'

        sys.path.insert(0, hub_dir)
        from dinov2.models.vision_transformer import DinoVisionTransformer

        configs = {
            'dinov2_vits14': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
            'dinov2_vitb14': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
            'dinov2_vitl14': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
            'dinov2_vitg14': {'embed_dim': 1536, 'depth': 40, 'num_heads': 24},
        }

        cfg = configs.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown model: {model_name}")

        model = DinoVisionTransformer(
            img_size=518, patch_size=14,
            embed_dim=cfg['embed_dim'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            ffn_layer='mlp',
            block_chunks=0,
        )

        ckpt_path = os.path.join(ckpt_dir, f'{model_name}_pretrain.pth')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt, strict=False)
        else:
            # Fall back to torch.hub
            model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)

        return model.to(device).eval()

    @torch.no_grad()
    def extract_features(self, img):
        """
        Extract DINOv2 features from a 2D image.

        Args:
            img: [1, 1, H, W] grayscale image in [-1, 1]

        Returns:
            features: [1, C, feat_h, feat_w] feature maps
        """
        # Convert to 3-channel and resize
        img_3ch = img.repeat(1, 3, 1, 1)
        img_3ch = (img_3ch + 1) / 2.0  # [-1,1] -> [0,1]
        img_resized = F.interpolate(img_3ch, size=(self.target_h, self.target_w), mode='bilinear', align_corners=True)

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img_normalized = (img_resized - mean) / std

        # Forward pass
        outputs = self.model.forward_features(img_normalized)
        patch_tokens = outputs['x_norm_patchtokens']  # [1, N, C]

        # Reshape to spatial feature map
        feat_h = self.feat_size[0]
        feat_w = self.feat_size[1]
        features = patch_tokens.reshape(1, feat_h, feat_w, -1).permute(0, 3, 1, 2)

        return features


def pca_reduce(features, n_components=24):
    """PCA dimensionality reduction on feature maps."""
    B, C, H, W = features.shape
    feat_flat = features.reshape(B, C, -1).permute(0, 2, 1)

    mean = feat_flat.mean(1, keepdim=True)
    feat_centered = feat_flat - mean

    # PCA via SVD
    U, S, Vh = torch.linalg.svd(feat_centered, full_matrices=False)
    V = Vh[:, :min(n_components, C), :]
    reduced = torch.bmm(feat_centered, V.transpose(1, 2))
    reduced = reduced.permute(0, 2, 1).reshape(B, -1, H, W)

    return reduced


def dino_reg_2d(fixed_img, moving_img, feat_size=(36, 36), reg_feature_dim=24,
                grid_sp=2, disp_hw=3, lambda_weight=2.0, n_iter_adam=200, lr_adam=3.0,
                extractor=None):
    """
    DINO-Reg 2D registration pipeline.

    Args:
        fixed_img: Fixed (target) image [1, 1, H, W] in [-1, 1]
        moving_img: Moving (source) image [1, 1, H, W] in [-1, 1]
        feat_size: Feature map size (h, w)
        reg_feature_dim: PCA output dimensions
        grid_sp: Grid spacing for convex stage
        disp_hw: Displacement half-width
        lambda_weight: Diffusion regularization weight
        n_iter_adam: Adam optimization iterations
        lr_adam: Adam learning rate
        extractor: Pre-loaded DINOv2FeatureExtractor (to avoid reloading)

    Returns:
        warped: Warped moving image [1, 1, H, W]
        disp_field: Displacement field [1, 2, H, W]
    """
    device = fixed_img.device

    if extractor is None:
        extractor = DINOv2FeatureExtractor(feat_size=feat_size, device=device)

    # Extract DINOv2 features
    with torch.no_grad():
        fixed_feat = extractor.extract_features(fixed_img)
        moving_feat = extractor.extract_features(moving_img)

    # PCA dimensionality reduction
    fixed_feat = pca_reduce(fixed_feat, reg_feature_dim)
    moving_feat = pca_reduce(moving_feat, reg_feature_dim)

    # Normalize features
    fixed_feat = fixed_feat / (fixed_feat.norm(dim=1, keepdim=True) + 1e-8)
    moving_feat = moving_feat / (moving_feat.norm(dim=1, keepdim=True) + 1e-8)

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

    # Apply displacement
    warped = spatial_transform_2d(moving_img, disp_field)

    return warped, disp_field
