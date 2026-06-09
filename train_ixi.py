#!/usr/bin/env python3
"""
IXI T1↔T2 3D registration training script.

No segmentation available — evaluation via MSE/NCC/SSIM.
Supports multi-GPU training via DataParallel.

Usage:
    python train_ixi.py --model voxelmorph3d --name vm3d_ixi --niter 200 --use_amp
    python train_ixi.py --model nemar3d --name falcon3d_ixi --niter 200 --use_amp --gpu_ids 0 1 2 3
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description='IXI 3D Registration Training')
    parser.add_argument('--model', type=str, required=True,
                        choices=['voxelmorph3d', 'nemar3d', 'transmorph3d'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataroot', type=str, default='./datasets/IXI')
    parser.add_argument('--niter', type=int, default=200)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0],
                        help='GPU IDs for training (multi-GPU with DataParallel)')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument('--save_epoch_freq', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'linear', 'none'])

    # Model-specific
    parser.add_argument('--vm3d_lr', type=float, default=1e-4)
    parser.add_argument('--vm3d_loss_type', type=str, default='mind', choices=['mi', 'mse', 'mind'])
    parser.add_argument('--vm3d_smoothness_weight', type=float, default=1.0)
    parser.add_argument('--vm3d_num_features', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--vm3d_no_svf', action='store_true')
    parser.add_argument('--vm3d_svf_steps', type=int, default=7)

    parser.add_argument('--tm3d_enc_channels', type=int, nargs='+', default=[48, 96, 192, 384])
    parser.add_argument('--tm3d_num_heads', type=int, default=6)
    parser.add_argument('--tm3d_num_transformer_blocks', type=int, default=2)
    parser.add_argument('--tm3d_sim_loss', type=str, default='mind', choices=['mind', 'mse'])
    parser.add_argument('--tm3d_reg_weight', type=float, default=1.0)
    parser.add_argument('--tm3d_lr', type=float, default=1e-4)
    parser.add_argument('--tm3d_use_svf', action='store_true', default=True)
    parser.add_argument('--tm3d_no_svf', action='store_true')
    parser.add_argument('--tm3d_svf_steps', type=int, default=7)

    parser.add_argument('--n3d_ngf', type=int, default=16)
    parser.add_argument('--n3d_ndf', type=int, default=16)
    parser.add_argument('--n3d_n_blocks', type=int, default=6)
    parser.add_argument('--n3d_kan_embed_dims', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--n3d_kan_depths', type=int, nargs='+', default=[1, 1, 1])
    parser.add_argument('--n3d_lambda_gan', type=float, default=1.0)
    parser.add_argument('--n3d_lambda_recon', type=float, default=10.0)
    parser.add_argument('--n3d_lambda_smooth', type=float, default=1.0)
    parser.add_argument('--n3d_lambda_direct', type=float, default=2.0)
    parser.add_argument('--n3d_lambda_cycle', type=float, default=0.0)

    return parser.parse_args()


def create_model(opt):
    if opt.model == 'voxelmorph3d':
        from models.voxelmorph3d_model import VoxelMorph3DModel
        return VoxelMorph3DModel(opt)
    elif opt.model == 'nemar3d':
        from models.nemar3d_model import NEMAR3DModel
        return NEMAR3DModel(opt)
    elif opt.model == 'transmorph3d':
        from models.transmorph3d_model import Transmorph3DModel
        return Transmorph3DModel(opt)


def compute_ncc(pred, target):
    pred_f = pred.flatten().float()
    target_f = target.flatten().float()
    mean_p = pred_f.mean()
    mean_t = target_f.mean()
    pred_centered = pred_f - mean_p
    target_centered = target_f - mean_t
    ncc = (pred_centered * target_centered).sum() / (
        pred_centered.norm() * target_centered.norm() + 1e-10)
    return ncc.item()


def compute_ssim_3d(pred, target, win_size=7):
    pred_np = pred.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()
    mid = pred_np.shape[0] // 2
    p = pred_np[mid]
    t = target_np[mid]
    from scipy.ndimage import uniform_filter
    mu_p = uniform_filter(p, size=win_size)
    mu_t = uniform_filter(t, size=win_size)
    sigma_p = uniform_filter(p**2, size=win_size) - mu_p**2
    sigma_t = uniform_filter(t**2, size=win_size) - mu_t**2
    sigma_pt = uniform_filter(p*t, size=win_size) - mu_p*mu_t
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu_p*mu_t + C1) * (2*sigma_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (sigma_p + sigma_t + C2))
    return ssim_map.mean()


def evaluate_metrics(model, dataset, device, max_patients=20):
    mse_vals, ncc_vals, ssim_vals = [], [], []

    for i in range(min(max_patients, len(dataset))):
        sample = dataset[i]
        ct = sample['A'].unsqueeze(0).to(device)
        mr = sample['B'].unsqueeze(0).to(device)
        if ct.dim() == 4: ct = ct.unsqueeze(0)
        if mr.dim() == 4: mr = mr.unsqueeze(0)

        model.set_input({'A': ct, 'B': mr, 'A_paths': ''})
        with torch.no_grad():
            model.forward()

        warped = getattr(model, 'warped', None)
        if warped is None:
            warped = getattr(model, 'registered_real_A', None)
        if warped is None:
            continue

        mse_vals.append(torch.nn.functional.mse_loss(warped, mr).item())
        ncc_vals.append(compute_ncc(warped, mr))
        ssim_vals.append(compute_ssim_3d(warped, mr))

    return {
        'mse': np.mean(mse_vals) if mse_vals else 0,
        'ncc': np.mean(ncc_vals) if ncc_vals else 0,
        'ssim': np.mean(ssim_vals) if ssim_vals else 0,
    }


def main():
    opt = parse_args()
    opt.isTrain = True
    opt.phase = 'test'
    opt.input_nc = 1
    opt.output_nc = 1
    opt.num_threads = 0
    opt.serial_batches = False
    opt.max_dataset_size = float('inf')
    opt.load_seg = False
    opt.checkpoints_dir = './checkpoints'
    opt.epoch = 'latest'
    opt.load_iter = 0
    opt.verbose = False
    opt.preprocess = 'none'
    opt.no_flip = True
    opt.display_winsize = 256
    opt.suffix = ''
    opt.pool_size = 0
    opt.lr = 1e-4
    opt.lr_policy = 'linear'
    opt.beta1 = 0.5
    opt.niter_decay = 0
    opt.epoch_count = 1
    opt.augment_3d = True
    opt.crop_3d_size = 0
    opt.vol_depth = opt.target_shape[0]
    opt.vol_height = opt.target_shape[1]
    opt.vol_width = opt.target_shape[2]
    opt.img_height = opt.target_shape[1]
    opt.img_width = opt.target_shape[2]

    n_gpus = len(opt.gpu_ids)
    primary_gpu = opt.gpu_ids[0]
    opt.device = torch.device(f'cuda:{primary_gpu}' if torch.cuda.is_available() else 'cpu')
    opt.batch_size = n_gpus  # 1 sample per GPU

    print(f"Using GPUs: {opt.gpu_ids} ({n_gpus} GPUs)")
    print(f"Model: {opt.model}")
    print(f"AMP: {opt.use_amp}")
    print(f"Target shape: {opt.target_shape}")
    print(f"Batch size: {opt.batch_size} ({n_gpus} GPUs x 1 sample)")

    from data.ixi_3d_dataset import IXI3DDataset
    dataset = IXI3DDataset(opt)
    model = create_model(opt)

    # Move networks to primary GPU
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        if isinstance(net, nn.Module):
            net.to(opt.device)

    # Wrap with DataParallel for multi-GPU
    if n_gpus > 1:
        for name in model.model_names:
            net = getattr(model, 'net' + name)
            if isinstance(net, nn.Module):
                dp_net = nn.DataParallel(net, device_ids=opt.gpu_ids)
                setattr(model, 'net' + name, dp_net)
        print(f"DataParallel enabled on {n_gpus} GPUs")

    # Create DataLoader for efficient batched data loading
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=4 * n_gpus, pin_memory=True, drop_last=True
    )

    checkpoint_dir = f'./checkpoints/{opt.name}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(checkpoint_dir, 'train_opt.txt'), 'w') as f:
        for k, v in sorted(vars(opt).items()):
            f.write(f'{k}: {v}\n')

    if opt.continue_train:
        for name in model.model_names:
            ckpt = os.path.join(checkpoint_dir, f'latest_net_{name}.pth')
            if os.path.exists(ckpt):
                net = getattr(model, 'net' + name)
                state = torch.load(ckpt, map_location=opt.device, weights_only=True)
                # Handle DataParallel state dict
                if n_gpus > 1:
                    net.load_state_dict(state)
                else:
                    net.load_state_dict(state)
                print(f"Resumed {name}")

    # Collect optimizers
    optimizers = []
    for attr in dir(model):
        if attr.startswith('optimizer') and attr != 'optimizers':
            opt_obj = getattr(model, attr)
            if isinstance(opt_obj, torch.optim.Optimizer):
                optimizers.append(opt_obj)

    if not optimizers:
        all_params = []
        for name in model.model_names:
            net = getattr(model, 'net' + name)
            all_params.extend(list(net.parameters()))
        optimizers = [torch.optim.Adam(all_params, lr=opt.lr)]

    schedulers = []
    for opt_obj in optimizers:
        if opt.lr_scheduler == 'cosine':
            schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(opt_obj, T_max=opt.niter, eta_min=1e-6))
        elif opt.lr_scheduler == 'linear':
            schedulers.append(torch.optim.lr_scheduler.LinearLR(opt_obj, start_factor=1.0, end_factor=0.01, total_iters=opt.niter))
        else:
            schedulers.append(torch.optim.lr_scheduler.ConstantLR(opt_obj))

    print(f"\nTraining {opt.model} for {opt.niter} epochs...")
    print(f"Dataset: {len(dataset)} patients, {len(dataloader)} batches/epoch")

    for epoch in range(1, opt.niter + 1):
        if hasattr(model, 'current_epoch'):
            model.current_epoch = epoch

        epoch_losses = {}
        epoch_start = time.time()

        for batch in dataloader:
            # batch['A']: [B, 1, D, H, W], batch['B']: [B, 1, D, H, W]
            model.set_input(batch)
            model.optimize_parameters()

            for name in model.model_names:
                net = getattr(model, 'net' + name)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            losses = model.get_current_losses()
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v

        n = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= n

        for scheduler in schedulers:
            scheduler.step()

        elapsed = time.time() - epoch_start

        if epoch % opt.print_freq == 0 or epoch == 1:
            loss_str = '  '.join(f'{k}: {v:.6f}' for k, v in epoch_losses.items())
            lr = schedulers[0].get_last_lr()[0] if schedulers else opt.lr
            print(f'  Epoch [{epoch}/{opt.niter}] lr={lr:.2e} {loss_str}  ({elapsed:.1f}s)')

        if epoch % opt.save_epoch_freq == 0 or epoch == opt.niter:
            for name in model.model_names:
                net = getattr(model, 'net' + name)
                # Unwrap DataParallel for saving
                raw_net = net.module if isinstance(net, nn.DataParallel) else net
                state = raw_net.state_dict()
                torch.save(state, os.path.join(checkpoint_dir, f'{epoch}_net_{name}.pth'))
                torch.save(state, os.path.join(checkpoint_dir, f'latest_net_{name}.pth'))
            print(f'  Saved checkpoints')

        if epoch % opt.eval_freq == 0 and epoch > 0:
            for name in model.model_names:
                getattr(model, 'net' + name).eval()
            metrics = evaluate_metrics(model, dataset, opt.device)
            for name in model.model_names:
                getattr(model, 'net' + name).train()
            print(f'  Eval: MSE={metrics["mse"]:.6f}  NCC={metrics["ncc"]:.4f}  SSIM={metrics["ssim"]:.4f}')

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
