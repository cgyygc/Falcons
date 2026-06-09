"""
TransMorph: Transformer-based Medical Image Registration
Reference: "TransMorph: Transformer for Unsupervised Medical Image Registration"
Author: Junyu Chen (jchen245@jhmi.edu)
Adapted for 2D cross-modal registration (CT→MR) in the NEMAR framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from torch.distributions.normal import Normal
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

from .base_model import BaseModel


# ============================================================
# Swin Transformer 2D Components (from RaFD/TransMorph2D)
# ============================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(8, 8), shift_size=(0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, rpe=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=(8, 8), mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, rpe=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=2, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class SwinTransformer2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=2, embed_dim=96,
                 depths=(2, 2, 4, 2), num_heads=(4, 4, 8, 8),
                 window_size=(8, 8), mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3,
                 ape=False, rpe=True, patch_norm=True,
                 out_indices=(0, 1, 2, 3), use_checkpoint=False):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, 128, 128))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                rpe=rpe,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = nn.LayerNorm(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs


# ============================================================
# Decoder and Registration Head
# ============================================================

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        relu = nn.LeakyReLU(inplace=True)
        nm = nn.InstanceNorm2d(out_channels) if not use_batchnorm else nn.BatchNorm2d(out_channels)
        super().__init__(conv, nm, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, 3, 1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, 3, 1, use_batchnorm=use_batchnorm)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels=2, kernel_size=3):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv2d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv2d.weight.shape))
        conv2d.bias = nn.Parameter(torch.zeros(conv2d.bias.shape))
        super().__init__(conv2d)


class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, flow):
        B, C, H, W = src.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=src.device, dtype=torch.float32),
            torch.arange(0, W, device=src.device, dtype=torch.float32), indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)
        grid = grid.unsqueeze(0)
        new_locs = grid + flow
        for i, size in enumerate([W, H]):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (size - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        return F.grid_sample(src, new_locs, align_corners=True, mode='bilinear')


# ============================================================
# TransMorph Network
# ============================================================

class TransMorphNet(nn.Module):
    def __init__(self, img_size=(512, 512), embed_dim=96, depths=(2, 2, 4, 2),
                 num_heads=(4, 4, 8, 8), window_size=(8, 8), reg_head_chan=16,
                 if_convskip=True, if_transskip=True):
        super().__init__()
        self.if_convskip = if_convskip
        self.if_transskip = if_transskip

        self.transformer = SwinTransformer2D(
            patch_size=4, in_chans=2, embed_dim=embed_dim,
            depths=depths, num_heads=num_heads,
            window_size=window_size, mlp_ratio=4.,
            qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.3, ape=False, rpe=True,
            patch_norm=True, out_indices=(0, 1, 2, 3))

        self.up0 = DecoderBlock(embed_dim * 8, embed_dim * 4, skip_channels=embed_dim * 4 if if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0, use_batchnorm=False)
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0, use_batchnorm=False)
        self.up4 = DecoderBlock(embed_dim // 2, reg_head_chan, skip_channels=reg_head_chan if if_convskip else 0, use_batchnorm=False)
        self.c1 = Conv2dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv2dReLU(2, reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(reg_head_chan, out_channels=2, kernel_size=3)
        self.spatial_trans = SpatialTransformer()
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        source = x[:, 0:1, :, :]
        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)
        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        flow = self.reg_head(x)
        out = self.spatial_trans(source, flow)
        return out, flow


# ============================================================
# Loss Functions
# ============================================================

class NCC(nn.Module):
    def __init__(self, win=9):
        super().__init__()
        self.win = win

    def forward(self, y_pred, y_true):
        I = y_true
        J = y_pred
        ndims = 2
        win = [self.win] * ndims
        sum_filt = torch.ones([1, 1, *win], device=I.device)
        pad_no = win[0] // 2
        stride = (1, 1)
        padding = (pad_no, pad_no)
        I2 = I * I
        J2 = J * J
        IJ = I * J
        I_sum = F.conv2d(I, sum_filt, stride=stride, padding=padding)
        J_sum = F.conv2d(J, sum_filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, sum_filt, stride=stride, padding=padding)
        win_size = self.win ** 2
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        cc = cross * cross / (I_var * J_var + 1e-5)
        return -torch.mean(cc)


class MINDLoss(nn.Module):
    def __init__(self, radius=2, dilation=2):
        super().__init__()
        self.radius = radius
        self.dilation = dilation

    def _pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def _mindssc2d(self, img):
        device = img.device
        kernel_size = self.radius * 2 + 1
        six_neighbourhood = torch.Tensor([
            [0, 1], [1, 0], [2, 1], [1, 2], [2, 3], [3, 2]
        ]).long()
        dist = self._pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
        n_pts = six_neighbourhood.shape[0]
        x_idx, y_idx = torch.meshgrid(torch.arange(n_pts), torch.arange(n_pts), indexing='ij')
        mask = ((x_idx > y_idx).view(-1) & (dist == 2).view(-1))
        n_pairs = mask.sum().item()
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, n_pts, 1).view(-1, 2)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(n_pts, 1, 1).view(-1, 2)[mask, :]
        mshift1 = torch.zeros(n_pairs, 1, 5, 5, device=device)
        mshift1.view(-1)[torch.arange(n_pairs, device=device) * 25 +
                          idx_shift1[:, 0].to(device) * 5 + idx_shift1[:, 1].to(device)] = 1
        mshift2 = torch.zeros(n_pairs, 1, 5, 5, device=device)
        mshift2.view(-1)[torch.arange(n_pairs, device=device) * 25 +
                          idx_shift2[:, 0].to(device) * 5 + idx_shift2[:, 1].to(device)] = 1
        rpad1 = nn.ReplicationPad2d(self.dilation + 1).to(device)
        rpad2 = nn.ReplicationPad2d(self.radius).to(device)
        ssd = F.avg_pool2d(rpad2(
            (F.conv2d(rpad1(img), mshift1, dilation=self.dilation) - F.conv2d(rpad1(img), mshift2, dilation=self.dilation)) ** 2),
            kernel_size, stride=1)
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)
        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self._mindssc2d(y_pred) - self._mindssc2d(y_true)) ** 2)


class GradLoss(nn.Module):
    def __init__(self, penalty='l2'):
        super().__init__()
        self.penalty = penalty

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0


# ============================================================
# BaseModel Wrapper
# ============================================================

class TransmorphModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = BaseModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--lambda_sim', type=float, default=1.0, help='weight for similarity loss')
        parser.add_argument('--lambda_reg', type=float, default=1.0, help='weight for regularization loss')
        parser.add_argument('--sim_loss', type=str, default='mind', choices=['ncc', 'mind'], help='similarity loss type')
        parser.add_argument('--embed_dim', type=int, default=96, help='TransMorph embedding dimension')
        parser.add_argument('--tm_depths', type=str, default='2,2,4,2', help='TransMorph transformer depths')
        parser.add_argument('--tm_num_heads', type=str, default='4,4,8,8', help='TransMorph attention heads')
        parser.add_argument('--tm_window_size', type=str, default='8,8', help='TransMorph window size')
        parser.add_argument('--reg_head_chan', type=int, default=16, help='registration head channels')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['sim', 'reg', 'total']
        self.visual_names = ['real_A', 'warped', 'real_B']
        self.model_names = ['TM']

        depths = tuple(int(x) for x in opt.tm_depths.split(','))
        num_heads = tuple(int(x) for x in opt.tm_num_heads.split(','))
        window_size = tuple(int(x) for x in opt.tm_window_size.split(','))

        self.netTM = TransMorphNet(
            img_size=(opt.img_height, opt.img_width),
            embed_dim=opt.embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            reg_head_chan=opt.reg_head_chan)

        # Move to GPU and wrap with DataParallel
        if len(self.gpu_ids) > 0:
            self.netTM.to(self.gpu_ids[0])
            self.netTM = torch.nn.DataParallel(self.netTM, self.gpu_ids)

        if self.isTrain:
            if opt.sim_loss == 'mind':
                self.criterionSim = MINDLoss().to(self.device)
            else:
                self.criterionSim = NCC().to(self.device)
            self.criterionReg = GradLoss(penalty='l2').to(self.device)
            self.optimizer = torch.optim.Adam(self.netTM.parameters(), lr=opt.lr, weight_decay=0, amsgrad=True)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input.get('A_paths', [''])

    def forward(self):
        x_in = torch.cat([self.real_A, self.real_B], dim=1)
        self.warped, self.flow = self.netTM(x_in)

    def backward(self):
        self.loss_sim = self.criterionSim(self.warped, self.real_B) * self.opt.lambda_sim
        self.loss_reg = self.criterionReg(self.flow, self.real_B) * self.opt.lambda_reg
        self.loss_total = self.loss_sim + self.loss_reg
        self.loss_total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
