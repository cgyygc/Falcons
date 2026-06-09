import torch

from .affine_stn import AffineSTN
from .unet_stn import UnetSTN
from .ukan_stn import UKANSTN
from .contrastive_stn import (
    ContrastiveSTNWrapper,
    ContrastiveUnetSTN,
    create_contrastive_stn
)
from .contrastive_head import (
    ContrastiveHead,
    InfoNCELoss,
    ByolLoss
)
from .ukan3d import UKAN3D_Backbone
from .ukan3d_stn import UKAN3DSTN

sampling_align_corners = False
sampling_mode = 'bilinear'


def modify_commandline_options(parser, is_train=True):
    parser.add_argument('--stn_cfg', type=str, default='A', help='Set the configuration used to build the STN.')
    parser.add_argument('--stn_type', type=str, default='affine',
                        help='The type of STN to use. Currently supported are [unet, affine, ukan, unet_contrastive, ukan_contrastive]')
    if is_train:
        parser.add_argument('--stn_bilateral_alpha', type=float, default=0.0,
                            help='The bilateral filtering coefficient used in the the smoothness loss.'
                                 'This is relevant for unet stn only.')
        parser.add_argument('--stn_no_identity_init', action='store_true',
                            help='Whether to start the transformation from identity transformation or some random'
                                 'transformation. This is only relevant for unet stn (for affine the model'
                                 'doesn\'t converge).')
        parser.add_argument('--stn_multires_reg', type=int, default=1,
                            help='In multi-resolution smoothness, the regularization is applied on multiple resolution.'
                                 '(default : 1, means no multi-resolution)')
        # UKAN specific parameters
        parser.add_argument('--ukan_embed_dims', type=int, nargs='+', default=[64, 128, 256],
                            help='Embedding dimensions for UKAN stages.')
        parser.add_argument('--ukan_depths', type=int, nargs='+', default=[1, 1, 1],
                            help='Number of KAN blocks for each UKAN stage.')
        # Contrastive learning parameters
        parser.add_argument('--use_contrastive', action='store_true',
                            help='Enable contrastive learning in STN.')
        parser.add_argument('--contrastive_proj_dim', type=int, default=128,
                            help='Projection dimension for contrastive learning.')
        parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                            help='Temperature for InfoNCE loss.')
        parser.add_argument('--contrastive_loss_type', type=str, default='infonce',
                            choices=['infonce', 'byol'],
                            help='Type of contrastive loss to use.')
        parser.add_argument('--contrastive_weight', type=float, default=0.1,
                            help='Weight for contrastive loss in total loss.')
        parser.add_argument('--contrastive_num_stages', type=int, default=None,
                            help='Number of encoder stages to use for contrastive learning. None means use all stages.')
    return parser


def define_stn(opt, stn_type='affine'):
    """Create and return an STN model with the relevant configuration."""
    def wrap_multigpu(stn_module, opt):
        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            stn_module.to(opt.gpu_ids[0])
            stn_module = torch.nn.DataParallel(stn_module, opt.gpu_ids)  # multi-GPUs
        return stn_module

    nc_a = opt.input_nc if opt.direction == 'AtoB' else opt.output_nc
    nc_b = opt.output_nc if opt.direction == 'AtoB' else opt.input_nc
    height = opt.img_height
    width = opt.img_width
    cfg = opt.stn_cfg
    use_contrastive = getattr(opt, 'use_contrastive', False)

    stn = None
    if stn_type == 'affine':
        stn = AffineSTN(nc_a, nc_b, height, width, cfg, opt.init_type)
    if stn_type == 'unet' or stn_type == 'unet_contrastive':
        if use_contrastive or stn_type == 'unet_contrastive':
            # Create contrastive UnetSTN
            stn = ContrastiveUnetSTN(
                in_channels_a=nc_a,
                in_channels_b=nc_b,
                height=height,
                width=width,
                cfg=cfg,
                init_func=opt.init_type,
                stn_bilateral_alpha=getattr(opt, 'stn_bilateral_alpha', 0.0),
                init_to_identity=(not getattr(opt, 'stn_no_identity_init', False)),
                multi_resolution_regularization=getattr(opt, 'stn_multires_reg', 1),
                proj_dim=getattr(opt, 'contrastive_proj_dim', 128),
                temperature=getattr(opt, 'contrastive_temperature', 0.07),
                use_contrastive=True
            )
        else:
            stn = UnetSTN(nc_a, nc_b, height, width, cfg, opt.init_type, opt.stn_bilateral_alpha,
                          (not opt.stn_no_identity_init), opt.stn_multires_reg)
    if stn_type == 'ukan' or stn_type == 'ukan_contrastive':
        img_size = max(height, width)
        base_stn = UKANSTN(nc_a, nc_b, height, width, cfg, opt.init_type, opt.stn_bilateral_alpha,
                           (not opt.stn_no_identity_init), opt.stn_multires_reg, img_size,
                           getattr(opt, 'ukan_embed_dims', [64, 128, 256]),
                           getattr(opt, 'ukan_depths', [1, 1, 1]))
        if use_contrastive or stn_type == 'ukan_contrastive':
            # Wrap UKANSTN with contrastive learning
            stn = ContrastiveSTNWrapper(
                stn=base_stn,
                stn_type='ukan',
                proj_dim=getattr(opt, 'contrastive_proj_dim', 128),
                temperature=getattr(opt, 'contrastive_temperature', 0.07),
                contrastive_loss_type=getattr(opt, 'contrastive_loss_type', 'infonce'),
                num_contrastive_stages=getattr(opt, 'contrastive_num_stages', None),
                loss_weight=getattr(opt, 'contrastive_weight', 0.1),
                cfg=cfg
            )
        else:
            stn = base_stn
    return wrap_multigpu(stn, opt)
