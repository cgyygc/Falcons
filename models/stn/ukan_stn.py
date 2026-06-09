"""
UKAN-based Spatial Transformer Network for image registration.
Replaces the standard U-Net backbone with U-KAN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ukan import (
    KANBlock, PatchEmbed, ConvLayer, D_ConvLayer
)
from .stn_losses import smoothness_loss

sampling_align_corners = False
sampling_mode = 'bilinear'


class UKAN_Backbone(nn.Module):
    """
    U-KAN backbone for predicting dense deformation fields.

    This network uses KAN (Kolmogorov-Arnold Networks) blocks instead of
    standard convolutional blocks for better feature representation.
    """

    def __init__(self, nc_a, nc_b, cfg='A', init_func='normal', init_to_identity=False,
                 img_size=256, embed_dims=[64, 128, 256], depths=[1, 1, 1], single_mode=False):
        """
        Args:
            nc_a: Number of channels in image A (source)
            nc_b: Number of channels in image B (target)
            cfg: Configuration name
            init_func: Weight initialization function
            init_to_identity: Whether to initialize output to identity transformation
            img_size: Input image size
            embed_dims: Embedding dimensions for each stage
            depths: Number of KAN blocks for each stage
            single_mode: If True, network processes single images (for contrastive learning)
        """
        super(UKAN_Backbone, self).__init__()

        self.nc_a = nc_a
        self.nc_b = nc_b
        self.single_mode = single_mode
        # For single mode, use only source channels, otherwise concat both
        in_chans = nc_a if single_mode else nc_a + nc_b
        super(UKAN_Backbone, self).__init__()

        self.nc_a = nc_a
        self.nc_b = nc_b
        in_chans = nc_a + nc_b

        # Embedding dimensions
        self.embed_dims = embed_dims
        kan_input_dim = embed_dims[0]

        # Encoder - Convolutional stages
        self.encoder1 = ConvLayer(in_chans, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)

        # Normalization layers
        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])
        self.dnorm3 = nn.LayerNorm(embed_dims[1])
        self.dnorm4 = nn.LayerNorm(embed_dims[0])

        # Dropout path rates
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # KAN blocks for encoder (bottleneck)
        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], drop=0., drop_path=dpr[0], norm_layer=nn.LayerNorm
        )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2], drop=0., drop_path=dpr[1], norm_layer=nn.LayerNorm
        )])

        # KAN blocks for decoder
        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], drop=0., drop_path=dpr[0], norm_layer=nn.LayerNorm
        )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], drop=0., drop_path=dpr[1], norm_layer=nn.LayerNorm
        )])

        # Patch embedding for tokenization
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )

        # Decoder
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        # Output layer - 2 channels for deformation field (x, y)
        self.final = nn.Conv2d(embed_dims[0] // 8, 2, kernel_size=1)

        # Initialize output to zero (identity transformation when added to grid)
        if init_to_identity:
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)

    def forward(self, img_a, img_b):
        """
        Forward pass to predict deformation field.

        Args:
            img_a: Source image [B, nc_a, H, W]
            img_b: Target image [B, nc_b, H, W]

        Returns:
            deformation: Deformation field [B, 2, H, W]
        """
        B = img_a.shape[0]

        # Concatenate input images
        x = torch.cat([img_a, img_b], 1)

        # Encoder - Convolutional stages
        # Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        # Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        # Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        # Tokenized KAN Stage (Bottleneck)
        # Stage 4
        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        # Bottleneck
        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Decoder
        # Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        # Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        # Output deformation field
        deformation = self.final(out)
        return deformation

    def get_encoder_features(self, img_a, img_b):
        """
        Extract encoder features for contrastive learning.

        Args:
            img_a: Source image [B, nc_a, H, W]
            img_b: Target image [B, nc_b, H, W]

        Returns:
            Tuple of (features_a, features_b) where each is a list of
            encoder feature maps from each stage [t1, t2, t3, t4]
        """
        B = img_a.shape[0]

        # Extract features from source image (img_a) separately
        # Create dummy input with same shape as img_b but with zeros to process img_a
        img_b_dummy = torch.zeros_like(img_b)
        x_a = torch.cat([img_a, img_b_dummy], 1)

        out_a = F.relu(F.max_pool2d(self.encoder1(x_a), 2, 2))
        t1_a = out_a
        out_a = F.relu(F.max_pool2d(self.encoder2(out_a), 2, 2))
        t2_a = out_a
        out_a = F.relu(F.max_pool2d(self.encoder3(out_a), 2, 2))
        t3_a = out_a

        # Stage 4 for source
        out_a, H, W = self.patch_embed3(t3_a)
        for i, blk in enumerate(self.block1):
            out_a = blk(out_a, H, W)
        out_a = self.norm3(out_a)
        t4_a = out_a.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Extract features from target image (img_b) separately
        img_a_dummy = torch.zeros_like(img_a)
        x_b = torch.cat([img_a_dummy, img_b], 1)

        out_b = F.relu(F.max_pool2d(self.encoder1(x_b), 2, 2))
        t1_b = out_b
        out_b = F.relu(F.max_pool2d(self.encoder2(out_b), 2, 2))
        t2_b = out_b
        out_b = F.relu(F.max_pool2d(self.encoder3(out_b), 2, 2))
        t3_b = out_b

        # Stage 4 for target
        out_b, H, W = self.patch_embed3(t3_b)
        for i, blk in enumerate(self.block1):
            out_b = blk(out_b, H, W)
        out_b = self.norm3(out_b)
        t4_b = out_b.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return [t1_a, t2_a, t3_a, t4_a], [t1_b, t2_b, t3_b, t4_b]


class UKANSTN(nn.Module):
    """
    U-KAN Spatial Transformer Network.

    This class generates and applies deformable transformations on input images
    using the U-KAN backbone for deformation field prediction.
    """

    def __init__(self, in_channels_a, in_channels_b, height, width, cfg='A',
                 init_func='normal', stn_bilateral_alpha=0.01, init_to_identity=True,
                 multi_resolution_regularization=1, img_size=256,
                 embed_dims=[64, 128, 256], depths=[1, 1, 1]):
        """
        Args:
            in_channels_a: Number of channels in source images
            in_channels_b: Number of channels in target images
            height: Image height
            width: Image width
            cfg: Configuration name
            init_func: Weight initialization function
            stn_bilateral_alpha: Alpha parameter for bilateral regularization
            init_to_identity: Whether to initialize to identity transformation
            multi_resolution_regularization: Number of resolutions for regularization
            img_size: Input image size
            embed_dims: Embedding dimensions for UKAN stages
            depths: Number of KAN blocks for each stage
        """
        super(UKANSTN, self).__init__()
        self.oh, self.ow = height, width
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # UKAN backbone for deformation field prediction
        self.offset_map = UKAN_Backbone(
            nc_a=in_channels_a,
            nc_b=in_channels_b,
            cfg=cfg,
            init_func=init_func,
            init_to_identity=init_to_identity,
            img_size=img_size,
            embed_dims=embed_dims,
            depths=depths
        ).to(self.device)

        self.identity_grid = self.get_identity_grid()
        self.alpha = stn_bilateral_alpha
        self.multi_resolution_regularization = multi_resolution_regularization

    def get_identity_grid(self):
        """Returns a sampling-grid that represents the identity transformation."""
        x = torch.linspace(-1.0, 1.0, self.ow)
        y = torch.linspace(-1.0, 1.0, self.oh)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return identity

    def get_grid(self, img_a, img_b, return_offsets_only=False):
        """Return the predicted sampling grid that aligns img_a with img_b."""
        if img_a.is_cuda and not self.identity_grid.is_cuda:
            self.identity_grid = self.identity_grid.to(img_a.device)

        b_size = img_a.size(0)
        deformation = self.offset_map(img_a, img_b)
        deformation_upsampled = deformation

        if deformation.size(2) != self.oh and deformation.size(3) != self.ow:
            deformation_upsampled = F.interpolate(
                deformation, (self.oh, self.ow), mode=sampling_mode,
                align_corners=sampling_align_corners
            )

        if return_offsets_only:
            resampling_grid = deformation_upsampled.permute([0, 2, 3, 1])
        else:
            resampling_grid = (self.identity_grid.repeat(b_size, 1, 1, 1) + deformation_upsampled).permute([0, 2, 3, 1])

        return resampling_grid

    def forward(self, img_a, img_b, apply_on=None):
        """
        Predicts the spatial alignment needed to align img_a with img_b.
        The spatial transformation will be applied on the tensors passed by apply_on.

        Args:
            img_a: The source image
            img_b: The target image
            apply_on: The geometric transformation can be applied on different tensors.
                     If not set, the transformation will be applied on img_a.

        Returns:
            warped_images: List of warped images
            reg_term: Regularization term for the predicted transformation
        """
        if img_a.is_cuda and not self.identity_grid.is_cuda:
            self.identity_grid = self.identity_grid.to(img_a.device)

        b_size = img_a.size(0)
        deformation = self.offset_map(img_a, img_b)
        deformation_upsampled = deformation

        if deformation.size(2) != self.oh and deformation.size(3) != self.ow:
            deformation_upsampled = F.interpolate(
                deformation, (self.oh, self.ow), mode=sampling_mode
            )

        resampling_grid = (self.identity_grid.repeat(b_size, 1, 1, 1) + deformation_upsampled).permute([0, 2, 3, 1])

        # Wrap image with respect to the deformation field
        if apply_on is None:
            apply_on = [img_a]

        warped_images = []
        for img in apply_on:
            warped_images.append(F.grid_sample(
                img, resampling_grid, mode=sampling_mode, padding_mode='zeros',
                align_corners=sampling_align_corners
            ))

        # Calculate STN regularization term
        reg_term = self._calculate_regularization_term(deformation, warped_images[0])
        return warped_images, reg_term

    def _calculate_regularization_term(self, deformation, img):
        """Calculate the regularization term of the predicted deformation."""
        dh, dw = deformation.size(2), deformation.size(3)
        img = None if img is None else img.detach()
        # Start with Python 0, then add tensor losses. Python 0 + tensor = tensor (with gradients preserved)
        reg = 0
        factor = 1.0

        for i in range(self.multi_resolution_regularization):
            if i != 0:
                deformation_resized = F.interpolate(
                    deformation, (dh // (2 ** i), dw // (2 ** i)), mode=sampling_mode,
                    align_corners=sampling_align_corners
                )
                img_resized = F.interpolate(
                    img, (dh // (2 ** i), dw // (2 ** i)), mode=sampling_mode,
                    align_corners=sampling_align_corners
                )
            elif deformation.size()[2::] != img.size()[2::]:
                deformation_resized = deformation
                img_resized = F.interpolate(
                    img, deformation.size()[2::], mode=sampling_mode,
                    align_corners=sampling_align_corners
                )
            else:
                deformation_resized = deformation
                img_resized = img

            reg = reg + factor * smoothness_loss(deformation_resized, img_resized, alpha=self.alpha)
            factor /= 2.0

        return reg
