"""
3D KAN modules for UKAN3D-STN.

Extends 2D UKAN modules to 3D by replacing Conv2d->Conv3d, BN2d->BN3d.
KANLinear is dimension-agnostic and reused directly.
KANLayer3D and KANBlock3D are 3D-specific versions using 3D depthwise convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .ukan import KANLinear, DropPath, trunc_normal_


class Conv3dLayer(nn.Module):
    """3D convolution + batch norm + activation."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.InstanceNorm3d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class D_Conv3dLayer(nn.Module):
    """3D decoder convolution layer."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PatchEmbed3D(nn.Module):
    """3D patch embedding for KAN stages."""
    def __init__(self, in_ch, out_ch, patch_size=1):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=patch_size,
                              stride=patch_size, bias=False)
        self.bn = nn.InstanceNorm3d(out_ch)

    def forward(self, x):
        return self.bn(self.proj(x))


class DW_bn_relu_3d_tokens(nn.Module):
    """3D depthwise conv for KAN token processing (reshape tokens→3D→tokens)."""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.InstanceNorm3d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        x = self.relu(self.bn(self.dwconv(x)))
        x = x.flatten(2).transpose(1, 2)
        return x


class KANLayer3D(nn.Module):
    """3D KAN Layer with depthwise 3D convolutions."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        grid_size, spline_order = 5, 3
        base_activation = torch.nn.SiLU
        kw = dict(grid_size=grid_size, spline_order=spline_order, scale_noise=0.1,
                   scale_base=1.0, scale_spline=1.0, base_activation=base_activation,
                   grid_eps=0.02, grid_range=[-1, 1])

        self.fc1 = KANLinear(in_features, hidden_features, **kw)
        self.fc2 = KANLinear(hidden_features, out_features, **kw)
        self.fc3 = KANLinear(hidden_features, out_features, **kw)

        self.dwconv_1 = DW_bn_relu_3d_tokens(hidden_features)
        self.dwconv_2 = DW_bn_relu_3d_tokens(hidden_features)
        self.dwconv_3 = DW_bn_relu_3d_tokens(hidden_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C)).reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, D, H, W)
        x = self.fc2(x.reshape(B * N, C)).reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, D, H, W)
        x = self.fc3(x.reshape(B * N, C)).reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, D, H, W)
        return x


class KANBlock3D(nn.Module):
    """KAN block for 3D feature maps with residual connection."""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer = KANLayer3D(in_features=dim, hidden_features=dim)

    def forward(self, x, D, H, W):
        # x: [B, C, D, H, W]
        B, C, D_, H_, W_ = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, D*H*W, C]
        tokens = tokens + self.drop_path(self.layer(self.norm2(tokens), D, H, W))
        return tokens.transpose(1, 2).reshape(B, C, D, H, W)


class KANStage3D(nn.Module):
    """A KAN stage with multiple KANBlock3D blocks."""
    def __init__(self, dim, depth, drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            KANBlock3D(dim, drop_path=drop_path) for _ in range(depth)
        ])

    def forward(self, x):
        B, C, D, H, W = x.shape
        for blk in self.blocks:
            x = blk(x, D, H, W)
        return x


class UKAN3D_Backbone(nn.Module):
    """3D UKAN backbone for deformation field prediction.

    Encoder: Conv3d stages with MaxPool3d
    Bottleneck: KANStage3D stages with PatchEmbed3D
    Decoder: D_Conv3dLayer with trilinear upsampling + skip connections
    """

    def __init__(self, in_channels=2, out_channels=3,
                 encoder_channels=(8, 16, 32, 64),
                 kan_embed_dims=(32, 64, 128),
                 kan_depths=(1, 1, 1)):
        super().__init__()

        enc_ch = encoder_channels
        kan_dims = kan_embed_dims

        # Encoder
        self.enc1 = nn.Sequential(Conv3dLayer(in_channels, enc_ch[0]),
                                  Conv3dLayer(enc_ch[0], enc_ch[0]))
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(Conv3dLayer(enc_ch[0], enc_ch[1]),
                                  Conv3dLayer(enc_ch[1], enc_ch[1]))
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = nn.Sequential(Conv3dLayer(enc_ch[1], enc_ch[2]),
                                  Conv3dLayer(enc_ch[2], enc_ch[2]))
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = nn.Sequential(Conv3dLayer(enc_ch[2], enc_ch[3]),
                                  Conv3dLayer(enc_ch[3], enc_ch[3]))

        # KAN bottleneck stages
        self.patch_embed1 = PatchEmbed3D(enc_ch[3], kan_dims[0])
        self.kan_stage1 = KANStage3D(kan_dims[0], kan_depths[0])

        self.patch_embed2 = PatchEmbed3D(kan_dims[0], kan_dims[1], patch_size=2)
        self.kan_stage2 = KANStage3D(kan_dims[1], kan_depths[1])

        self.patch_embed3 = PatchEmbed3D(kan_dims[1], kan_dims[2], patch_size=2)
        self.kan_stage3 = KANStage3D(kan_dims[2], kan_depths[2])

        # Decoder (5 stages: 1/32 → 1/16 → 1/8 → 1/4 → 1/2 → full)
        # up5: k3(1/32) → 1/16, skip k2
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(kan_dims[2], kan_dims[1], 1))
        self.dec5 = D_Conv3dLayer(kan_dims[1] * 2, kan_dims[1])

        # up4: 1/16 → 1/8, skip k1
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(kan_dims[1], kan_dims[0], 1))
        self.dec4 = D_Conv3dLayer(kan_dims[0] * 2, kan_dims[0])

        # up3: 1/8 → 1/4, skip e3
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(kan_dims[0], enc_ch[2], 1))
        self.dec3 = D_Conv3dLayer(enc_ch[2] * 2, enc_ch[2])

        # up2: 1/4 → 1/2, skip e2
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(enc_ch[2], enc_ch[1], 1))
        self.dec2 = D_Conv3dLayer(enc_ch[1] * 2, enc_ch[1])

        # up1: 1/2 → full, skip e1
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(enc_ch[1], enc_ch[0], 1))
        self.dec1 = D_Conv3dLayer(enc_ch[0] * 2, enc_ch[0])

        # Output deformation field
        self.output = nn.Conv3d(enc_ch[0], out_channels, kernel_size=1)
        nn.init.normal_(self.output.weight, 0, 1e-5)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        k1 = self.kan_stage1(self.patch_embed1(e4))  # 1/8
        k2 = self.kan_stage2(self.patch_embed2(k1))   # 1/16
        k3 = self.kan_stage3(self.patch_embed3(k2))   # 1/32

        d5 = self.dec5(torch.cat([self.up5(k3), k2], dim=1))  # 1/32→1/16
        d4 = self.dec4(torch.cat([self.up4(d5), k1], dim=1))   # 1/16→1/8
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))   # 1/8→1/4
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))   # 1/4→1/2
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))   # 1/2→full

        return self.output(d1)
