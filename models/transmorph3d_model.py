"""
TransMorph3D: 3D hybrid CNN-Transformer registration model.

Uses a 3D UNet backbone with transformer attention at the bottleneck.
Much simpler than a full 3D Swin Transformer but captures long-range dependencies.

Reference:
- TransMorph: Transformer for Unsupervised Medical Image Registration (2022)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .stn.spatial_transformer_3d import (
    SpatialTransformer3D,
    MINDLoss3D,
    GradLoss3D,
    integrate_svf_3d,
)


class TransformerBlock3D(nn.Module):
    """Transformer block for bottleneck features in 3D.

    Flattens 3D spatial features into token sequences,
    applies standard self-attention, then reshapes back.
    """

    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: [B, C, D, H, W] → [B, C, D, H, W]"""
        B, C, D, H, W = x.shape
        # Flatten spatial dims to tokens
        tokens = x.flatten(2).transpose(1, 2)  # [B, D*H*W, C]

        # Self-attention
        residual = tokens
        tokens = self.norm1(tokens)
        tokens, _ = self.attn(tokens, tokens, tokens)
        tokens = residual + tokens

        # MLP
        tokens = tokens + self.mlp(self.norm2(tokens))

        # Reshape back
        return tokens.transpose(1, 2).reshape(B, C, D, H, W)


class Conv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TransMorph3DNet(nn.Module):
    """3D TransMorph with CNN encoder/decoder + transformer bottleneck."""

    def __init__(self, in_channels=2, out_channels=3, enc_channels=(48, 96, 192, 384),
                 num_transformer_blocks=2, num_heads=6):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(Conv3dBlock(in_channels, enc_channels[0]),
                                  Conv3dBlock(enc_channels[0], enc_channels[0]))
        self.enc2 = nn.Sequential(Conv3dBlock(enc_channels[0], enc_channels[1], stride=2),
                                  Conv3dBlock(enc_channels[1], enc_channels[1]))
        self.enc3 = nn.Sequential(Conv3dBlock(enc_channels[1], enc_channels[2], stride=2),
                                  Conv3dBlock(enc_channels[2], enc_channels[2]))
        self.enc4 = nn.Sequential(Conv3dBlock(enc_channels[2], enc_channels[3], stride=2),
                                  Conv3dBlock(enc_channels[3], enc_channels[3]))

        # Transformer bottleneck
        self.transformer = nn.Sequential(
            *[TransformerBlock3D(enc_channels[3], num_heads=num_heads)
              for _ in range(num_transformer_blocks)]
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(enc_channels[3], enc_channels[2], 2, stride=2),
            Conv3dBlock(enc_channels[2] * 2, enc_channels[2]),
            Conv3dBlock(enc_channels[2], enc_channels[2]),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(enc_channels[2], enc_channels[1], 2, stride=2),
            Conv3dBlock(enc_channels[1] * 2, enc_channels[1]),
            Conv3dBlock(enc_channels[1], enc_channels[1]),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(enc_channels[1], enc_channels[0], 2, stride=2),
            Conv3dBlock(enc_channels[0] * 2, enc_channels[0]),
            Conv3dBlock(enc_channels[0], enc_channels[0]),
        )

        # Registration head
        self.reg_head = nn.Sequential(
            nn.Conv3d(enc_channels[0], 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(16, out_channels, 3, padding=1),
        )

        # Small weight init for flow
        nn.init.normal_(self.reg_head[-1].weight, 0, 1e-5)
        nn.init.zeros_(self.reg_head[-1].bias)

    def forward(self, x):
        """Returns (warped_features, flow). Input x: [B, 2, D, H, W]"""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b4 = self.transformer(e4)

        d3 = self.dec3[0](b4)
        d3 = self.dec3[1:](torch.cat([d3, e3], dim=1))
        d2 = self.dec2[0](d3)
        d2 = self.dec2[1:](torch.cat([d2, e2], dim=1))
        d1 = self.dec1[0](d2)
        d1 = self.dec1[1:](torch.cat([d1, e1], dim=1))

        flow = self.reg_head(d1)
        return flow


class Transmorph3DModel(BaseModel):
    """3D TransMorph model for volumetric registration."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        from options.base_options import BaseOptions
        parser = BaseOptions.modify_commandline_options(parser, is_train)

        parser.add_argument('--tm3d_enc_channels', type=int, nargs='+', default=[48, 96, 192, 384])
        parser.add_argument('--tm3d_num_heads', type=int, default=6)
        parser.add_argument('--tm3d_num_transformer_blocks', type=int, default=2)
        parser.add_argument('--tm3d_sim_loss', type=str, default='mind', choices=['mind', 'mse'])
        parser.add_argument('--tm3d_reg_weight', type=float, default=1.0)
        parser.add_argument('--tm3d_lr', type=float, default=1e-4)
        parser.add_argument('--use_amp', action='store_true')
        parser.add_argument('--tm3d_use_svf', action='store_true', default=True)
        parser.add_argument('--tm3d_no_svf', action='store_true')
        parser.add_argument('--tm3d_svf_steps', type=int, default=7)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['TM3D']
        self.loss_names = ['sim', 'reg', 'total']

        enc_ch = tuple(getattr(opt, 'tm3d_enc_channels', [48, 96, 192, 384]))
        self.reg_weight = getattr(opt, 'tm3d_reg_weight', 1.0)
        self.sim_loss_type = getattr(opt, 'tm3d_sim_loss', 'mind')
        self.use_amp = getattr(opt, 'use_amp', False)
        self.use_svf = getattr(opt, 'tm3d_use_svf', True) and not getattr(opt, 'tm3d_no_svf', False)
        self.svf_steps = getattr(opt, 'tm3d_svf_steps', 7)

        self.netTM3D = TransMorph3DNet(
            in_channels=2, out_channels=3,
            enc_channels=enc_ch,
            num_transformer_blocks=getattr(opt, 'tm3d_num_transformer_blocks', 2),
            num_heads=getattr(opt, 'tm3d_num_heads', 6),
        )

        self.spatial_transform = SpatialTransformer3D()
        self.criterionMIND = MINDLoss3D()
        self.criterionMSE = nn.MSELoss()
        self.criterionGrad = GradLoss3D()

        if self.isTrain:
            self.optimizer = torch.optim.Adam(
                self.netTM3D.parameters(),
                lr=getattr(opt, 'tm3d_lr', 1e-4)
            )
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    def set_input(self, input):
        self.moving = input['A'].to(self.device)
        self.fixed = input['B'].to(self.device)
        self.image_paths = input.get('A_paths', '')

    def forward(self):
        x = torch.cat([self.moving, self.fixed], dim=1)
        velocity = self.netTM3D(x)
        if self.use_svf:
            self.flow = integrate_svf_3d(velocity, n_steps=self.svf_steps)
        else:
            self.flow = velocity
        self.warped = self.spatial_transform(self.moving, self.flow)

        with torch.amp.autocast('cuda', enabled=False):
            warped_f = self.warped.float()
            fixed_f = self.fixed.float()
            if self.sim_loss_type == 'mind':
                self.loss_sim = self.criterionMIND(warped_f, fixed_f)
            else:
                self.loss_sim = self.criterionMSE(warped_f, fixed_f)

        self.loss_reg = self.criterionGrad(self.flow)
        self.loss_total = self.loss_sim + self.reg_weight * self.loss_reg

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        if self.use_amp:
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                self.forward()
            self.scaler.scale(self.loss_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.forward()
            self.loss_total.backward()
            self.optimizer.step()

    def get_current_losses(self):
        return {
            'sim': self.loss_sim.item(),
            'reg': self.loss_reg.item(),
            'total': self.loss_total.item()
        }

    def save_networks(self, epoch):
        self.save_network(self.netTM3D, 'TM3D', epoch)

    def load_networks(self, epoch):
        self.load_network(self.netTM3D, 'TM3D', epoch)
