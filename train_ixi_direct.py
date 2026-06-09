#!/usr/bin/env python3
"""
Direct 3D registration training — R only (no T, no D).
Uses UKAN3D-STN with MIND loss for cross-modal registration.

Usage:
    python train_ixi_direct.py --name ukan_ixi --niter 200 --use_amp --gpu_ids 0 1 2 3
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description='Direct 3D Registration Training (R only)')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataroot', type=str, default='./datasets/IXI')
    parser.add_argument('--niter', type=int, default=200)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_mind', type=float, default=1.0)
    parser.add_argument('--lambda_smooth', type=float, default=1.0)
    parser.add_argument('--kan_embed_dims', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--kan_depths', type=int, nargs='+', default=[1, 1, 1])
    parser.add_argument('--save_epoch_freq', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'linear', 'none'])
    return parser.parse_args()


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
    from scipy.ndimage import uniform_filter
    pred_np = pred.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()
    mid = pred_np.shape[0] // 2
    p = pred_np[mid]
    t = target_np[mid]
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


def evaluate(netR, dataset, device, max_patients=20):
    netR.eval()
    mse_vals, ncc_vals, ssim_vals = [], [], []
    spatial_transform = None

    for i in range(min(max_patients, len(dataset))):
        sample = dataset[i]
        ct = sample['A'].unsqueeze(0).to(device)
        mr = sample['B'].unsqueeze(0).to(device)
        if ct.dim() == 4: ct = ct.unsqueeze(0)
        if mr.dim() == 4: mr = mr.unsqueeze(0)

        with torch.no_grad():
            warped_images, _ = netR(ct, mr, apply_on=[ct])
        warped = warped_images[0]

        mse_vals.append(F.mse_loss(warped, mr).item())
        ncc_vals.append(compute_ncc(warped, mr))
        ssim_vals.append(compute_ssim_3d(warped, mr))

    netR.train()
    return {
        'mse': np.mean(mse_vals) if mse_vals else 0,
        'ncc': np.mean(ncc_vals) if ncc_vals else 0,
        'ssim': np.mean(ssim_vals) if ssim_vals else 0,
    }


def main():
    opt = parse_args()
    n_gpus = len(opt.gpu_ids)
    primary_gpu = opt.gpu_ids[0]
    device = torch.device(f'cuda:{primary_gpu}')

    print(f"GPUs: {opt.gpu_ids} ({n_gpus})")
    print(f"LR: {opt.lr}, λ_mind: {opt.lambda_mind}, λ_smooth: {opt.lambda_smooth}")

    # Setup opt for dataset
    opt.isTrain = True
    opt.phase = 'test'
    opt.input_nc = 1
    opt.output_nc = 1
    opt.num_threads = 0
    opt.serial_batches = False
    opt.max_dataset_size = float('inf')
    opt.load_seg = False
    opt.checkpoints_dir = './checkpoints'
    opt.no_flip = True
    opt.preprocess = 'none'
    opt.augment_3d = True
    opt.vol_depth = opt.target_shape[0]
    opt.vol_height = opt.target_shape[1]
    opt.vol_width = opt.target_shape[2]
    opt.img_height = opt.target_shape[1]
    opt.img_width = opt.target_shape[2]

    from data.ixi_3d_dataset import IXI3DDataset
    dataset = IXI3DDataset(opt)

    # Create UKAN3D-STN
    from models.stn.ukan3d_stn import UKAN3DSTN
    from models.stn.spatial_transformer_3d import MINDLoss3D, SmoothnessLoss3D

    img_size = (opt.target_shape[1], opt.target_shape[2], opt.target_shape[0])
    netR = UKAN3DSTN(
        img_size=img_size,
        in_channels=2, out_channels=3,
        kan_embed_dims=tuple(opt.kan_embed_dims),
        kan_depths=tuple(opt.kan_depths),
    ).to(device)

    if n_gpus > 1:
        netR = nn.DataParallel(netR, device_ids=opt.gpu_ids)
        print(f"DataParallel on {n_gpus} GPUs")

    criterion_mind = MINDLoss3D()
    criterion_smooth = SmoothnessLoss3D()
    optimizer = torch.optim.Adam(netR.parameters(), lr=opt.lr)

    if opt.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=1e-6)
    elif opt.lr_scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=opt.niter)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    batch_size = n_gpus
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4 * n_gpus, pin_memory=True, drop_last=True)

    checkpoint_dir = f'./checkpoints/{opt.name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'train_opt.txt'), 'w') as f:
        for k, v in sorted(vars(opt).items()):
            f.write(f'{k}: {v}\n')

    if opt.continue_train:
        ckpt = os.path.join(checkpoint_dir, 'latest_net_R.pth')
        if os.path.exists(ckpt):
            raw_net = netR.module if isinstance(netR, nn.DataParallel) else netR
            raw_net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            print("Resumed from checkpoint")

    print(f"\nTraining UKAN3D (R only) for {opt.niter} epochs...")
    print(f"Dataset: {len(dataset)} patients, {len(dataloader)} batches/epoch")

    for epoch in range(1, opt.niter + 1):
        epoch_mind, epoch_smooth, n_batches = 0, 0, 0
        epoch_start = time.time()

        for batch in dataloader:
            A = batch['A'].to(device)
            B = batch['B'].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=opt.use_amp):
                warped_images, deformation = netR(A, B, apply_on=[A])
                warped_A = warped_images[0]

                with torch.amp.autocast('cuda', enabled=False):
                    loss_mind = opt.lambda_mind * criterion_mind(warped_A.float(), B.float())
                loss_smooth = opt.lambda_smooth * criterion_smooth(deformation)
                loss = loss_mind + loss_smooth

            if opt.use_amp:
                from torch.amp import GradScaler
                if not hasattr(main, '_scaler'):
                    main._scaler = GradScaler('cuda')
                main._scaler.scale(loss).backward()
                main._scaler.step(optimizer)
                main._scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(netR.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_mind += loss_mind.item()
            epoch_smooth += loss_smooth.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - epoch_start

        if epoch % opt.print_freq == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f'  Epoch [{epoch}/{opt.niter}] lr={lr:.2e} '
                  f'MIND: {epoch_mind/n_batches:.6f}  smooth: {epoch_smooth/n_batches:.6f}  ({elapsed:.1f}s)')

        if epoch % opt.save_epoch_freq == 0 or epoch == opt.niter:
            raw_net = netR.module if isinstance(netR, nn.DataParallel) else netR
            torch.save(raw_net.state_dict(), os.path.join(checkpoint_dir, f'{epoch}_net_R.pth'))
            torch.save(raw_net.state_dict(), os.path.join(checkpoint_dir, 'latest_net_R.pth'))
            print(f'  Saved checkpoint')

        if epoch % opt.eval_freq == 0 and epoch > 0:
            metrics = evaluate(netR, dataset, device)
            print(f'  Eval: MSE={metrics["mse"]:.6f}  NCC={metrics["ncc"]:.4f}  SSIM={metrics["ssim"]:.4f}')

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
