"""
NEMAR3D (Falcon3D): 3D Neural Multimodal Adversarial Registration.

Key innovations over plain UNet (VM3D):
1. Dual-Path Registration: TR and RT paths provide complementary supervision
   for the registration network through modality translation
2. Direct MIND Registration Loss: R network receives direct cross-modal
   supervision via MIND(warped_A, B), bypassing translation quality dependency
3. Cycle Consistency: T1→T2→T1 cycle ensures anatomical preservation
4. Adversarial Discriminator: provides realistic texture supervision

Architecture: T (Translation) + R (Registration/UKAN3D) + D (Discriminator)
- L1 translation loss provides precise pixel-level supervision for T
- Direct MIND loss provides cross-modal structural supervision for R
- Together they complement: L1 = hard constraint (precision), MIND = soft constraint (cross-modal)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .stn.spatial_transformer_3d import (
    SpatialTransformer3D,
    NormalizedMutualInformationLoss3D,
    MINDLoss3D,
    SmoothnessLoss3D,
)
from .stn.ukan3d_stn import UKAN3DSTN


class ResnetBlock3D(nn.Module):
    """3D Residual block with InstanceNorm."""

    def __init__(self, dim, padding_mode='reflect', norm_layer=None, use_dropout=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d
        p = 1
        conv_block = [
            nn.ReplicationPad3d(p),
            nn.Conv3d(dim, dim, 3, padding=0, bias=False),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block.append(nn.Dropout3d(0.5))
        conv_block += [
            nn.ReplicationPad3d(p),
            nn.Conv3d(dim, dim, 3, padding=0, bias=False),
            norm_layer(dim),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator3D(nn.Module):
    """3D ResNet generator for modality translation."""

    def __init__(self, in_ch=1, out_ch=1, ngf=32, n_blocks=9, use_dropout=False):
        super().__init__()
        norm_layer = nn.InstanceNorm3d

        model = [
            nn.ReplicationPad3d(3),
            nn.Conv3d(in_ch, ngf, 7, padding=0, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_down = 3
        for i in range(n_down):
            mult = 2 ** i
            model += [
                nn.Conv3d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=False),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_down
        for _ in range(n_blocks):
            model += [ResnetBlock3D(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_down):
            mult = 2 ** (n_down - i)
            model += [
                nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2), 3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [
            nn.ReplicationPad3d(3),
            nn.Conv3d(ngf, out_ch, 7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator."""

    def __init__(self, in_ch=2, ndf=32, n_layers=3, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(in_ch, ndf, kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv3d(ndf * nf_mult, 1, kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class GANLoss3D(nn.Module):
    """GAN loss for 3D discriminator."""

    def __init__(self, gan_mode='vanilla', label_smoothing=0.0):
        super().__init__()
        self.gan_mode = gan_mode
        self.label_smoothing = label_smoothing
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_val = 1.0 - self.label_smoothing
        else:
            target_val = 0.0 + self.label_smoothing * 0.5
        return torch.full_like(prediction, target_val)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class NEMAR3DModel(BaseModel):
    """3D NEMAR (Falcon3D) with MIND-guided translation and direct registration loss."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        from options.base_options import BaseOptions
        parser = BaseOptions.modify_commandline_options(parser, is_train)

        parser.add_argument('--n3d_ngf', type=int, default=16)
        parser.add_argument('--n3d_ndf', type=int, default=16)
        parser.add_argument('--n3d_n_blocks', type=int, default=6)
        parser.add_argument('--n3d_kan_embed_dims', type=int, nargs='+', default=[32, 64, 128])
        parser.add_argument('--n3d_kan_depths', type=int, nargs='+', default=[1, 1, 1])
        parser.add_argument('--n3d_lambda_gan', type=float, default=1.0)
        parser.add_argument('--n3d_lambda_recon', type=float, default=10.0,
                            help='Weight for L1 translation loss (hard constraint)')
        parser.add_argument('--n3d_lambda_smooth', type=float, default=1.0)
        parser.add_argument('--n3d_lambda_direct', type=float, default=1.0,
                            help='Weight for direct MIND registration loss (soft cross-modal constraint)')
        parser.add_argument('--n3d_lambda_cycle', type=float, default=1.0,
                            help='Weight for L1 cycle consistency loss (T1→T2→T1)')
        parser.add_argument('--n3d_gan_mode', type=str, default='vanilla', choices=['vanilla', 'lsgan'])
        parser.add_argument('--n3d_lr', type=float, default=1e-4)
        parser.add_argument('--n3d_use_dropout', action='store_true')
        parser.add_argument('--n3d_warmup_epochs', type=int, default=0,
                            help='Epochs to train R only with MIND before joint training')
        parser.add_argument('--use_amp', action='store_true')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['T', 'R', 'D']
        self.loss_names = ['L1_TR', 'GAN_TR', 'L1_RT', 'GAN_RT', 'direct_MIND',
                           'cycle_L1', 'smoothness', 'D']

        ngf = getattr(opt, 'n3d_ngf', 16)
        ndf = getattr(opt, 'n3d_ndf', 16)
        n_blocks = getattr(opt, 'n3d_n_blocks', 6)
        kan_embed_dims = tuple(getattr(opt, 'n3d_kan_embed_dims', [32, 64, 128]))
        kan_depths = tuple(getattr(opt, 'n3d_kan_depths', [1, 1, 1]))
        self.lambda_gan = getattr(opt, 'n3d_lambda_gan', 1.0)
        self.lambda_recon = getattr(opt, 'n3d_lambda_recon', 10.0)
        self.lambda_smooth = getattr(opt, 'n3d_lambda_smooth', 1.0)
        self.lambda_direct = getattr(opt, 'n3d_lambda_direct', 1.0)
        self.lambda_cycle = getattr(opt, 'n3d_lambda_cycle', 1.0)
        self.use_amp = getattr(opt, 'use_amp', False)
        self.warmup_epochs = getattr(opt, 'n3d_warmup_epochs', 0)
        self.current_epoch = 0

        # Get image size from dataset
        self.vol_depth = getattr(opt, 'vol_depth', 192)
        self.vol_height = getattr(opt, 'img_height', 160)
        self.vol_width = getattr(opt, 'img_width', 192)
        img_size = (self.vol_height, self.vol_width, self.vol_depth)

        # Translation network (1ch→1ch modality translation)
        self.netT = ResnetGenerator3D(
            in_ch=1, out_ch=1, ngf=ngf, n_blocks=n_blocks,
            use_dropout=getattr(opt, 'n3d_use_dropout', False)
        )

        # Registration network (UKAN3D-STN)
        self.netR = UKAN3DSTN(
            img_size=img_size,
            in_channels=2, out_channels=3,
            kan_embed_dims=kan_embed_dims,
            kan_depths=kan_depths,
        )

        # Discriminator (2ch input: real_A + fake/real_B)
        self.netD = NLayerDiscriminator3D(in_ch=2, ndf=ndf)

        # Spatial transformer for warping
        self.spatial_transform = SpatialTransformer3D()

        # Loss functions
        self.criterionGAN = GANLoss3D(
            gan_mode=getattr(opt, 'n3d_gan_mode', 'vanilla'),
            label_smoothing=0.1
        )
        self.criterionL1 = nn.L1Loss()
        self.criterionSmooth = SmoothnessLoss3D()
        self.criterionMIND = MINDLoss3D()

        if self.isTrain:
            self.optimizer_T = torch.optim.Adam(self.netT.parameters(), lr=getattr(opt, 'n3d_lr', 1e-4))
            self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=getattr(opt, 'n3d_lr', 1e-4))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=getattr(opt, 'n3d_lr', 1e-4))
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input.get('A_paths', '')

    def forward(self):
        # Translation: real_A → fake_B (A modality → B modality)
        self.fake_B = self.netT(self.real_A)

        # Registration with TR/RT paths
        warped_images, self.deformation = self.netR(self.real_A, self.real_B,
                                                     apply_on=[self.real_A, self.fake_B])
        self.registered_real_A = warped_images[0]
        self.fake_TR_B = self.netT(self.registered_real_A)
        self.fake_RT_B = warped_images[1]

        # Cycle consistency: T2→T1 (real_B → T^{-1} → recon_A)
        self.recon_A = self.netT(self.real_B)

    def backward_T_and_R(self):
        # === Warmup mode: only R with direct MIND loss ===
        if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
            with torch.amp.autocast('cuda', enabled=False):
                self.loss_direct_reg = self.lambda_direct * self.criterionMIND(
                    self.registered_real_A.float(), self.real_B.float())
            self.loss_smoothness = self.lambda_smooth * self.criterionSmooth(self.deformation)
            self.loss_L1_TR = torch.tensor(0.0, device=self.device)
            self.loss_GAN_TR = torch.tensor(0.0, device=self.device)
            self.loss_L1_RT = torch.tensor(0.0, device=self.device)
            self.loss_GAN_RT = torch.tensor(0.0, device=self.device)
            self.loss_cycle = torch.tensor(0.0, device=self.device)
            loss = self.loss_direct_reg + self.loss_smoothness
            loss.backward()
            return

        # === Joint training mode ===

        # TR path: L1 translation loss (hard constraint for pixel-level precision)
        self.loss_L1_TR = self.lambda_recon * self.criterionL1(self.fake_TR_B, self.real_B)

        # RT path: L1 translation loss
        self.loss_L1_RT = self.lambda_recon * self.criterionL1(self.fake_RT_B, self.real_B)

        # GAN losses (adversarial supervision for realistic translation)
        fake_AB = torch.cat([self.real_A, self.fake_TR_B], dim=1)
        pred_fake = self.netD(fake_AB)
        self.loss_GAN_TR = self.lambda_gan * self.criterionGAN(pred_fake, True)

        fake_AB = torch.cat([self.real_A, self.fake_RT_B], dim=1)
        pred_fake = self.netD(fake_AB)
        self.loss_GAN_RT = self.lambda_gan * self.criterionGAN(pred_fake, True)

        # Direct MIND registration loss: cross-modal structural supervision for R
        # Key innovation: R gets direct cross-modal signal independent of T quality
        with torch.amp.autocast('cuda', enabled=False):
            self.loss_direct_reg = self.lambda_direct * self.criterionMIND(
                self.registered_real_A.float(), self.real_B.float())

        # Cycle consistency: L1 ensures pixel-level anatomical preservation
        self.loss_cycle = self.lambda_cycle * self.criterionL1(self.recon_A, self.real_A)

        # Smoothness
        self.loss_smoothness = self.lambda_smooth * self.criterionSmooth(self.deformation)

        loss = (self.loss_L1_TR + self.loss_L1_RT +
                self.loss_GAN_TR + self.loss_GAN_RT +
                self.loss_direct_reg + self.loss_cycle +
                self.loss_smoothness)
        loss.backward()

    def backward_D(self):
        # Warmup: skip D
        if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
            self.loss_D = torch.tensor(0.0, device=self.device)
            return

        # Real
        real_AB = torch.cat([self.real_A, self.real_B], dim=1)
        pred_real = self.netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Fake TR
        fake_AB = torch.cat([self.real_A, self.fake_TR_B.detach()], dim=1)
        pred_fake = self.netD(fake_AB)
        loss_D_fake_TR = self.criterionGAN(pred_fake, False)

        # Fake RT
        fake_AB = torch.cat([self.real_A, self.fake_RT_B.detach()], dim=1)
        pred_fake = self.netD(fake_AB)
        loss_D_fake_RT = self.criterionGAN(pred_fake, False)

        self.loss_D = 0.5 * self.lambda_gan * (loss_D_real + loss_D_fake_TR + loss_D_fake_RT)
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        # Update D
        self.set_requires_grad([self.netT, self.netR], False)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad([self.netT, self.netR], True)

        # Update T and R
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.optimizer_T.zero_grad()
        self.optimizer_R.zero_grad()
        self.backward_T_and_R()
        self.optimizer_T.step()
        self.optimizer_R.step()
        self.set_requires_grad([self.netD], True)

    def get_current_losses(self):
        return {
            'L1_TR': self.loss_L1_TR.item() if self.loss_L1_TR.requires_grad else self.loss_L1_TR,
            'GAN_TR': self.loss_GAN_TR.item() if isinstance(self.loss_GAN_TR, torch.Tensor) and self.loss_GAN_TR.requires_grad else (self.loss_GAN_TR if isinstance(self.loss_GAN_TR, float) else self.loss_GAN_TR.item()),
            'L1_RT': self.loss_L1_RT.item() if self.loss_L1_RT.requires_grad else self.loss_L1_RT,
            'GAN_RT': self.loss_GAN_RT.item() if isinstance(self.loss_GAN_RT, torch.Tensor) and self.loss_GAN_RT.requires_grad else (self.loss_GAN_RT if isinstance(self.loss_GAN_RT, float) else self.loss_GAN_RT.item()),
            'direct_MIND': self.loss_direct_reg.item() if self.loss_direct_reg.requires_grad else self.loss_direct_reg,
            'cycle_L1': self.loss_cycle.item() if self.loss_cycle.requires_grad else self.loss_cycle,
            'smoothness': self.loss_smoothness.item(),
            'D': self.loss_D.item() if isinstance(self.loss_D, torch.Tensor) and self.loss_D.requires_grad else self.loss_D,
        }

    def save_networks(self, epoch):
        self.save_network(self.netT, 'T', epoch)
        self.save_network(self.netR, 'R', epoch)
        self.save_network(self.netD, 'D', epoch)

    def load_networks(self, epoch):
        self.load_network(self.netT, 'T', epoch)
        self.load_network(self.netR, 'R', epoch)
        self.load_network(self.netD, 'D', epoch)
