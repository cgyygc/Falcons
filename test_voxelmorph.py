#!/usr/bin/env python3
"""
Testing/training script for VoxelMorph-MI Registration Model.

Uses Mutual Information loss by default for cross-modal registration.

Usage:
    # Train with MI loss (default, for cross-modal CT→MR)
    python test_voxelmorph.py --dataroot ./datasets/RIRE_2d_paired --dataset_mode aligned_2d --train

    # Train with NCC loss (same-modality)
    python test_voxelmorph.py --dataroot ./datasets/RIRE_2d_paired --dataset_mode aligned_2d --train --vm_loss_type ncc

    # Test
    python test_voxelmorph.py --dataroot ./datasets/RIRE_2d_paired --dataset_mode aligned_2d --name voxelmorph_rire
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image

from models.voxelmorph_model import (
    VoxelMorphModel, Unet, SpatialTransformer,
    NormalizedMutualInformationLoss, NCCLoss, SmoothnessLoss
)
from data import create_dataset


def get_vm_parser():
    """Create parser for VoxelMorph testing."""
    parser = argparse.ArgumentParser(description='VoxelMorph-MI Registration Options')

    # Basic options
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--name', type=str, default='voxelmorph_rire', help='experiment name')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--model', type=str, default='voxelmorph')
    parser.add_argument('--dataset_mode', type=str, default='aligned_2d')
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--no_flip', action='store_true', default=True)
    parser.add_argument('--serial_batches', action='store_true', default=True)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--output_nc', type=int, default=1)
    parser.add_argument('--max_dataset_size', type=int, default=float('inf'))
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--direction', type=str, default='AtoB')
    parser.add_argument('--load_size', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--preprocess', type=str, default='none')
    parser.add_argument('--display_winsize', type=int, default=512)
    parser.add_argument('--epoch', type=str, default='latest')
    parser.add_argument('--load_iter', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    # VoxelMorph specific options
    parser.add_argument('--vm_num_features', type=int, default=[32, 64, 128, 256], nargs='+')
    parser.add_argument('--vm_use_dropout', action='store_true')
    parser.add_argument('--vm_loss_type', type=str, default='mi', choices=['mi', 'ncc', 'mse'])
    parser.add_argument('--vm_smoothness_weight', type=float, default=1.0)
    parser.add_argument('--vm_ncc_window', type=int, default=9)
    parser.add_argument('--vm_mi_bins', type=int, default=64)

    # Training options
    parser.add_argument('--vm_lr', type=float, default=1e-4)
    parser.add_argument('--vm_weight_decay', type=float, default=0.0)
    parser.add_argument('--vm_niter', type=int, default=200)

    # Training mode flag
    parser.add_argument('--train', action='store_true', help='Training mode')

    return parser


def tensor2im(input_image, imtype=np.uint8):
    """Convert tensor to numpy image."""
    if isinstance(input_image, torch.Tensor):
        image_numpy = input_image[0].cpu().float().numpy()
        if image_numpy.ndim == 3 and image_numpy.shape[0] == 1:
            image_numpy = image_numpy[0]
        image_numpy = np.clip((image_numpy + 1.0) / 2.0 * 255, 0, 255)
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def train_voxelmorph(opt, device):
    """Train VoxelMorph model."""
    opt.phase = 'train'
    dataset_loader = create_dataset(opt)

    netV = Unet(
        num_features=opt.vm_num_features,
        use_dropout=opt.vm_use_dropout
    ).to(device)

    spatial_transform = SpatialTransformer()

    # Loss functions based on type
    loss_type = getattr(opt, 'vm_loss_type', 'mi')
    if loss_type == 'mi':
        criterion_sim = NormalizedMutualInformationLoss(num_bins=opt.vm_mi_bins)
    elif loss_type == 'ncc':
        criterion_sim = NCCLoss(win=opt.vm_ncc_window)
    else:
        criterion_sim = torch.nn.MSELoss()

    criterion_smooth = SmoothnessLoss(weight=1.0)
    smoothness_weight = opt.vm_smoothness_weight

    optimizer = torch.optim.Adam(netV.parameters(), lr=opt.vm_lr)

    niter = opt.vm_niter
    print(f'\nTraining VoxelMorph-MI ({loss_type} loss) for {niter} epochs...')

    for iteration in range(niter):
        epoch_loss_sim = 0
        epoch_loss_smooth = 0
        epoch_loss_total = 0

        batch_count = 0
        for data in dataset_loader:
            moving = data['A'].to(device)   # [-1, 1]
            fixed = data['B'].to(device)    # [-1, 1]

            # Forward
            x = torch.cat([moving, fixed], dim=1)
            flow = netV(x)
            warped = spatial_transform(moving, flow)

            # Similarity loss
            loss_sim = criterion_sim(warped, fixed)
            loss_smooth = criterion_smooth(flow)
            loss_total = loss_sim + smoothness_weight * loss_smooth

            # Backward
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            epoch_loss_sim += loss_sim.item()
            epoch_loss_smooth += loss_smooth.item()
            epoch_loss_total += loss_total.item()
            batch_count += 1

        if batch_count > 0:
            epoch_loss_sim /= batch_count
            epoch_loss_smooth /= batch_count
            epoch_loss_total /= batch_count

        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f'  Epoch [{iteration+1}/{niter}] '
                  f'{loss_type.upper()}: {epoch_loss_sim:.6f}, '
                  f'Smooth: {epoch_loss_smooth:.6f}, '
                  f'Total: {epoch_loss_total:.6f}')

    # Save model
    save_path = os.path.join(opt.checkpoints_dir, opt.name, 'latest_net_V.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(netV.state_dict(), save_path)
    print(f'Model saved to: {save_path}')

    return netV


def test_voxelmorph(opt, device, netV=None):
    """Test VoxelMorph model."""
    opt.phase = 'test'
    dataset = create_dataset(opt)

    print(f'Testing {opt.num_test} images')

    if netV is None:
        netV = Unet(
            in_channels=2,
            out_channels=2,
            num_features=opt.vm_num_features,
            use_dropout=opt.vm_use_dropout
        ).to(device)

        checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, 'latest_net_V.pth')
        if os.path.exists(checkpoint_path):
            netV.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f'Loaded model from: {checkpoint_path}')
        else:
            print(f'Warning: No checkpoint found at {checkpoint_path}')
            print('Using randomly initialized model (results will be poor)')

    netV.eval()
    spatial_transform = SpatialTransformer()

    results_dir = os.path.join(opt.results_dir, opt.name, 'test_latest', 'images')
    os.makedirs(results_dir, exist_ok=True)

    from skimage.metrics import structural_similarity as ssim

    metrics = {'mse': [], 'mae': [], 'psnr': [], 'ssim': [], 'ncc': []}

    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= opt.num_test:
                break

            moving = data['A'].to(device)   # [-1, 1]
            fixed = data['B'].to(device)    # [-1, 1]

            x = torch.cat([moving, fixed], dim=1)
            flow = netV(x)
            warped = spatial_transform(moving, flow)

            # Convert to [0, 1] for metrics
            warped_np = ((warped[0, 0].cpu().numpy() + 1.0) / 2.0)
            fixed_np = ((fixed[0, 0].cpu().numpy() + 1.0) / 2.0)
            warped_np = np.clip(warped_np, 0, 1)
            fixed_np = np.clip(fixed_np, 0, 1)

            # Compute metrics
            mse = np.mean((warped_np - fixed_np) ** 2)
            mae = np.mean(np.abs(warped_np - fixed_np))
            psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10))

            warped_uint8 = (warped_np * 255).astype(np.uint8)
            fixed_uint8 = (fixed_np * 255).astype(np.uint8)
            ssim_val = ssim(warped_uint8, fixed_uint8, data_range=255)

            # NCC
            w_flat = warped_np.flatten()
            f_flat = fixed_np.flatten()
            w_mean = w_flat.mean()
            f_mean = f_flat.mean()
            ncc = np.sum((w_flat - w_mean) * (f_flat - f_mean)) / \
                  (np.sqrt(np.sum((w_flat - w_mean) ** 2) * np.sum((f_flat - f_mean) ** 2)) + 1e-10)

            metrics['mse'].append(mse)
            metrics['mae'].append(mae)
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim_val)
            metrics['ncc'].append(ncc)

            # Save images (convert from [-1,1] to uint8)
            moving_save = tensor2im(moving)
            fixed_save = tensor2im(fixed)
            warped_save = tensor2im(warped)

            Image.fromarray(moving_save).save(os.path.join(results_dir, f'moving_{i:04d}.png'))
            Image.fromarray(fixed_save).save(os.path.join(results_dir, f'fixed_{i:04d}.png'))
            Image.fromarray(warped_save).save(os.path.join(results_dir, f'registered_{i:04d}.png'))

            if (i + 1) % 10 == 0:
                print(f'Processed {i+1}/{opt.num_test}')

    print('\n' + '=' * 60)
    print(f'VoxelMorph-MI Evaluation Results ({getattr(opt, "vm_loss_type", "mi")} loss)')
    print('=' * 60)
    for key, values in metrics.items():
        if values:
            print(f'{key.upper():>6}: {np.mean(values):.6f} +/- {np.std(values):.6f}')
    print('=' * 60)

    return metrics


def main():
    parser = get_vm_parser()
    opt = parser.parse_args()

    str_ids = list(map(int, opt.gpu_ids.split(','))) if opt.gpu_ids else []
    if len(str_ids) > 0:
        torch.cuda.set_device(str_ids[0])
        device = torch.device(f'cuda:{str_ids[0]}')
        print(f'Using GPU: {str_ids[0]}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    if opt.train:
        netV = train_voxelmorph(opt, device)
        test_voxelmorph(opt, device, netV=netV)
    else:
        test_voxelmorph(opt, device)


if __name__ == '__main__':
    main()
