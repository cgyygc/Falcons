"""
VoxelMorph3D-MI: 3D deep learning volumetric registration with MI loss.

Extends the 2D VoxelMorph-MI to 3D volumetric registration.
Uses Conv3d-based UNet with 3-channel flow output and 3D spatial transformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .stn.spatial_transformer_3d import (
    SpatialTransformer3D,
    NormalizedMutualInformationLoss3D,
    MINDLoss3D,
    GradLoss3D,
    integrate_svf_3d,
)


class Unet3D(nn.Module):
    """3D UNet for VoxelMorph.

    Conv2d→Conv3d, MaxPool2d→MaxPool3d, ConvTranspose2d→ConvTranspose3d.
    Output: 3-channel flow [B, 3, D, H, W] for 3D deformation.
    """

    def __init__(self, in_channels=2, out_channels=3,
                 num_features=(32, 64, 128, 256), use_dropout=False):
        super().__init__()

        # Encoder
        self.encoder1 = self._conv_block(in_channels, num_features[0], use_dropout)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self._conv_block(num_features[0], num_features[1], use_dropout)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = self._conv_block(num_features[1], num_features[2], use_dropout)
        self.pool3 = nn.MaxPool3d(2)
        self.encoder4 = self._conv_block(num_features[2], num_features[3], use_dropout)

        # Decoder
        self.up3 = nn.ConvTranspose3d(num_features[3], num_features[2], kernel_size=2, stride=2)
        self.decoder3 = self._conv_block(num_features[2] + num_features[2], num_features[2], use_dropout)

        self.up2 = nn.ConvTranspose3d(num_features[2], num_features[1], kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(num_features[1] + num_features[1], num_features[1], use_dropout)

        self.up1 = nn.ConvTranspose3d(num_features[1], num_features[0], kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(num_features[0] + num_features[0], num_features[0], use_dropout)

        self.output = nn.Conv3d(num_features[0], out_channels, kernel_size=1)

        # Small weight init for flow prediction
        nn.init.normal_(self.output.weight, 0, 1e-3)
        nn.init.zeros_(self.output.bias)

    def _conv_block(self, in_ch, out_ch, use_dropout):
        layers = [
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout3d(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        d3 = self.decoder3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output(d1)


class VoxelMorph3DModel(BaseModel):
    """3D VoxelMorph-MI for volumetric cross-modal registration."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        from options.base_options import BaseOptions
        parser = BaseOptions.modify_commandline_options(parser, is_train)

        parser.add_argument('--vm3d_num_features', type=int, nargs='+', default=[32, 64, 128, 256])
        parser.add_argument('--vm3d_use_dropout', action='store_true')
        parser.add_argument('--vm3d_loss_type', type=str, default='mind', choices=['mi', 'mse', 'mind'])
        parser.add_argument('--vm3d_smoothness_weight', type=float, default=10.0)
        parser.add_argument('--vm3d_mi_bins', type=int, default=32)
        parser.add_argument('--vm3d_lr', type=float, default=1e-4)
        parser.add_argument('--vm3d_niter', type=int, default=500)
        parser.add_argument('--vm3d_use_svf', action='store_true', default=True)
        parser.add_argument('--vm3d_no_svf', action='store_true')
        parser.add_argument('--vm3d_svf_steps', type=int, default=7)
        parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['V']
        self.loss_names = ['total', 'similarity', 'smoothness']
        self.visual_names = ['moving', 'fixed', 'warped']

        num_features = tuple(getattr(opt, 'vm3d_num_features', [32, 64, 128, 256]))
        self.loss_type = getattr(opt, 'vm3d_loss_type', 'mind')
        self.smoothness_weight = getattr(opt, 'vm3d_smoothness_weight', 1.0)
        self.use_amp = getattr(opt, 'use_amp', False)
        self.use_svf = getattr(opt, 'vm3d_use_svf', True) and not getattr(opt, 'vm3d_no_svf', False)
        self.svf_steps = getattr(opt, 'vm3d_svf_steps', 7)

        self.netV = Unet3D(
            in_channels=2, out_channels=3,
            num_features=num_features,
            use_dropout=getattr(opt, 'vm3d_use_dropout', False)
        )

        self.spatial_transform = SpatialTransformer3D()

        self.criterionMI = NormalizedMutualInformationLoss3D(
            num_bins=getattr(opt, 'vm3d_mi_bins', 32)
        )
        self.criterionMSE = nn.MSELoss()
        self.criterionMIND = MINDLoss3D()
        self.criterionGrad = GradLoss3D()

        if self.isTrain:
            self.optimizer = torch.optim.Adam(
                self.netV.parameters(),
                lr=getattr(opt, 'vm3d_lr', 1e-4)
            )
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    def set_input(self, input):
        self.moving = input['A'].to(self.device)
        self.fixed = input['B'].to(self.device)
        self.image_paths = input.get('A_paths', '')

    def forward(self):
        x = torch.cat([self.moving, self.fixed], dim=1)
        velocity = self.netV(x)
        if self.use_svf:
            self.flow = integrate_svf_3d(velocity, n_steps=self.svf_steps)
        else:
            self.flow = velocity
        self.warped = self.spatial_transform(self.moving, self.flow)

        # Compute similarity loss in fp32 for stability
        with torch.amp.autocast('cuda', enabled=False):
            warped_f32 = self.warped.float()
            fixed_f32 = self.fixed.float()
            if self.loss_type == 'mi':
                self.loss_similarity = self.criterionMI(warped_f32, fixed_f32)
            elif self.loss_type == 'mind':
                self.loss_similarity = self.criterionMIND(warped_f32, fixed_f32)
            else:
                self.loss_similarity = self.criterionMSE(warped_f32, fixed_f32)

        self.loss_smoothness = self.criterionGrad(self.flow)
        self.loss_total = self.loss_similarity + self.smoothness_weight * self.loss_smoothness

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
            'total': self.loss_total.item(),
            'similarity': self.loss_similarity.item(),
            'smoothness': self.loss_smoothness.item()
        }

    def save_networks(self, epoch):
        self.save_network(self.netV, 'V', epoch)

    def load_networks(self, epoch):
        self.load_network(self.netV, 'V', epoch)
