#!/usr/bin/env python3
"""
Testing script for CycleGAN Translation Model.

Evaluates translation quality (CT -> MR) by comparing the translated (fake MR)
images with real MR images using MSE, MAE, PSNR, SSIM, and NCC metrics.

Usage:
    python test_cyclegan.py --dataroot ./datasets/RIRE_2d --name cyclegan_rire --num_test 100
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from models.cycle_gan_model import CycleGANModel
from data import create_dataset


def get_parser():
    """Create parser for CycleGAN testing."""
    parser = argparse.ArgumentParser(description='CycleGAN Translation Options')

    # Basic options
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--name', type=str, default='cyclegan_rire', help='experiment name')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--model', type=str, default='cycle_gan')
    parser.add_argument('--dataset_mode', type=str, default='rire_2d')
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

    # CycleGAN specific options (must match training)
    parser.add_argument('--netG_A', type=str, default='resnet_9blocks')
    parser.add_argument('--netG_B', type=str, default='resnet_9blocks')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--netD', type=str, default='basic')
    parser.add_argument('--n_layers_D', type=int, default=3)
    parser.add_argument('--norm', type=str, default='instance')
    parser.add_argument('--init_type', type=str, default='normal')
    parser.add_argument('--init_gain', type=float, default=0.02)
    parser.add_argument('--no_dropout', action='store_true', default=True)
    parser.add_argument('--lambda_A', type=float, default=10.0)
    parser.add_argument('--lambda_B', type=float, default=10.0)
    parser.add_argument('--lambda_identity', type=float, default=0.5)
    parser.add_argument('--lambda_gan', type=float, default=1.0)
    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--gan_mode', type=str, default='vanilla')

    # Training-only options (needed for model init but not used)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--isTrain', action='store_true', default=False)

    return parser


def tensor2im(input_image, imtype=np.uint8):
    """Convert tensor to numpy image."""
    if isinstance(input_image, torch.Tensor):
        image_numpy = input_image[0].cpu().float().numpy()
        if image_numpy.ndim == 3 and image_numpy.shape[0] == 1:
            image_numpy = image_numpy[0]
        image_numpy = np.clip((image_numpy + 1) / 2.0 * 255, 0, 255)
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def compute_metrics(fake_np, real_np):
    """Compute MSE, MAE, PSNR, SSIM, NCC between two numpy arrays in [0,1]."""
    mse = np.mean((fake_np - real_np) ** 2)
    mae = np.mean(np.abs(fake_np - real_np))
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10))

    fake_uint8 = (fake_np * 255).astype(np.uint8)
    real_uint8 = (real_np * 255).astype(np.uint8)
    ssim_val = ssim(fake_uint8, real_uint8, data_range=255)

    # NCC
    f_flat = fake_np.flatten()
    r_flat = real_np.flatten()
    f_mean = f_flat.mean()
    r_mean = r_flat.mean()
    ncc = np.sum((f_flat - f_mean) * (r_flat - r_mean)) / \
          (np.sqrt(np.sum((f_flat - f_mean) ** 2) * np.sum((r_flat - r_mean) ** 2)) + 1e-10)

    return mse, mae, psnr, ssim_val, ncc


def main():
    parser = get_parser()
    opt = parser.parse_args()

    # Set device - convert gpu_ids string to list of ints (like BaseOptions.parse does)
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for id_str in str_ids:
        id_int = int(id_str)
        if id_int >= 0:
            opt.gpu_ids.append(id_int)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
        device = torch.device(f'cuda:{opt.gpu_ids[0]}')
        print(f'Using GPU: {opt.gpu_ids[0]}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # Create dataset
    opt.phase = 'test'
    opt.isTrain = False
    dataset = create_dataset(opt)

    # Load CycleGAN model
    model = CycleGANModel(opt)
    # In test mode, only load generator networks (D_A, D_B are not initialized when isTrain=False)
    model.model_names = ['G_A', 'G_B']
    model.setup(opt)
    model.eval()
    print(f'Loaded CycleGAN model from: {os.path.join(opt.checkpoints_dir, opt.name)}')

    # Results directory
    results_dir = os.path.join(opt.results_dir, opt.name, 'test_latest', 'images')
    os.makedirs(results_dir, exist_ok=True)

    # Metrics
    metrics = {'mse': [], 'mae': [], 'psnr': [], 'ssim': [], 'ncc': []}

    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= opt.num_test:
                break

            model.set_input(data)
            model.test()

            # Get outputs - model translates A(CT) -> B'(fake MR)
            real_A = model.real_A  # input CT
            fake_B = model.fake_B  # translated MR
            real_B = model.real_B  # real MR

            # Convert from [-1,1] to [0,1] for metrics
            fake_np = ((fake_B[0, 0].cpu().numpy() + 1) / 2.0)
            real_np = ((real_B[0, 0].cpu().numpy() + 1) / 2.0)

            mse, mae, psnr_val, ssim_val, ncc = compute_metrics(fake_np, real_np)

            metrics['mse'].append(mse)
            metrics['mae'].append(mae)
            metrics['psnr'].append(psnr_val)
            metrics['ssim'].append(ssim_val)
            metrics['ncc'].append(ncc)

            # Save images
            input_ct = tensor2im(real_A)
            fake_mr = tensor2im(fake_B)
            real_mr = tensor2im(real_B)

            Image.fromarray(input_ct).save(os.path.join(results_dir, f'input_ct_{i:04d}.png'))
            Image.fromarray(fake_mr).save(os.path.join(results_dir, f'fake_mr_{i:04d}.png'))
            Image.fromarray(real_mr).save(os.path.join(results_dir, f'real_mr_{i:04d}.png'))

            if (i + 1) % 10 == 0:
                print(f'Processed {i+1}/{opt.num_test} | '
                      f'PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | NCC: {ncc:.4f}')

    # Print results
    print('\n' + '=' * 60)
    print('CycleGAN Translation Evaluation Results')
    print('=' * 60)
    for key, values in metrics.items():
        if values:
            print(f'{key.upper():>6}: {np.mean(values):.6f} +/- {np.std(values):.6f}')
    print('=' * 60)

    # Save metrics to file
    metrics_path = os.path.join(opt.results_dir, opt.name, 'metrics.txt')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        f.write('CycleGAN Translation Evaluation Results\n')
        f.write('=' * 60 + '\n')
        for key, values in metrics.items():
            if values:
                f.write(f'{key.upper():>6}: {np.mean(values):.6f} +/- {np.std(values):.6f}\n')
        f.write('=' * 60 + '\n')
        f.write(f'\nNumber of test images: {len(metrics["mse"])}\n')
        f.write(f'Model: {opt.name}\n')
        f.write(f'Epoch: {opt.epoch}\n')

    print(f'Metrics saved to: {metrics_path}')


if __name__ == '__main__':
    main()
