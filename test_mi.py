#!/usr/bin/env python3
"""
Testing script specifically for MI Registration Model.

Since MI registration doesn't require training, this script runs
the registration optimization directly.

Usage:
    python test_mi.py --dataroot ./datasets/rire --dataset_mode rire_2d --num_test 10
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from models.mi_registration_model import MIRegistrationModel, NormalizedMutualInformationLoss, GridSampler, SmoothnessLoss
from data import create_dataset
from data.base_dataset import get_transform
from options.base_options import BaseOptions


def get_mi_parser():
    """Create parser for MI registration testing."""
    parser = argparse.ArgumentParser(description='MI Registration Options')

    # Basic options
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--name', type=str, default='mi_test', help='name of the experiment')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0')
    parser.add_argument('--model', type=str, default='mi_registration')
    parser.add_argument('--dataset_mode', type=str, default='rire_2d')
    parser.add_argument('--num_test', type=int, default=10)
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

    # MI-specific options
    parser.add_argument('--mi_num_bins', type=int, default=64)
    parser.add_argument('--mi_sigma', type=float, default=2.0)
    parser.add_argument('--lambda_mi_smooth', type=float, default=0.1)
    parser.add_argument('--optim_lr', type=float, default=0.1)
    parser.add_argument('--optim_max_iter', type=int, default=100)
    parser.add_argument('--optim_tolerance', type=float, default=1e-5)
    parser.add_argument('--optim_history_size', type=int, default=100)
    parser.add_argument('--optim_line_search', type=str, default='strong_wolfe')
    parser.add_argument('--transform_type', type=str, default='affine',
                       choices=['affine', 'similarity', 'rigid'])

    return parser


def tensor2im(input_image, imtype=np.uint8):
    """Convert a Tensor array into a numpy image array."""
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def main():
    parser = get_mi_parser()
    opt = parser.parse_args()

    # Set device
    opt.gpu_ids = list(map(int, opt.gpu_ids.split(','))) if opt.gpu_ids else []
    str_ids = opt.gpu_ids
    opt.gpu_ids = []
    if len(str_ids) > 0:
        gpu_id = str_ids[0]
        torch.cuda.set_device(gpu_id)
        opt.gpu_ids = [gpu_id]
        device = torch.device(f'cuda:{gpu_id}')
        print(f'Using GPU: {gpu_id}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # Create dataset
    print(f'Loading dataset from: {opt.dataroot}')
    opt.phase = 'test'
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'Dataset size: {dataset_size} images')

    # Limit number of test images
    num_test = min(opt.num_test, dataset_size)
    print(f'Testing {num_test} images')

    # Create results directory
    results_dir = os.path.join(opt.results_dir, opt.name, 'test_latest', 'images')
    os.makedirs(results_dir, exist_ok=True)

    # Initialize model components
    grid_sampler = GridSampler().to(device)
    criterion_mi = NormalizedMutualInformationLoss(
        num_bins=opt.mi_num_bins,
        sigma=opt.mi_sigma
    ).to(device)
    criterion_smooth = SmoothnessLoss(lambda_smooth=opt.lambda_mi_smooth)

    # Metrics accumulator
    metrics = {'mi': [], 'smoothness': [], 'total': []}

    print('\nStarting MI registration...')
    print('='*60)

    for i, data in enumerate(dataset):
        if i >= num_test:
            break

        # Get data
        real_A = data['A'].to(device)  # moving image
        real_B = data['B'].to(device)  # fixed image
        image_paths = data['A_paths']

        b, _, h, w = real_A.shape

        # Initialize transformation
        theta = torch.zeros(b, 6, device=device)

        if opt.transform_type == 'rigid':
            theta[:, 0] = 1.0  # scale_x
            theta[:, 1] = 1.0  # scale_y
            theta[:, 2] = 0.0  # shear
        elif opt.transform_type == 'similarity':
            theta[:, 0] = 1.0
            theta[:, 1] = 1.0
            theta[:, 2] = 0.0
        else:  # affine
            theta[:, 0] = 1.0
            theta[:, 1] = 1.0
            theta[:, 2] = 0.0

        # Small random initialization
        theta[:, 3] = torch.randn(b, device=device) * 0.01
        theta[:, 4] = torch.randn(b, device=device) * 0.01
        theta[:, 5] = torch.randn(b, device=device) * 0.01
        theta.requires_grad_(True)

        # L-BFGS optimization
        def closure():
            registered = grid_sampler(real_A, theta, size=(h, w))
            mi_loss = criterion_mi(registered, real_B)
            smooth_loss = criterion_smooth(theta)
            total_loss = mi_loss + smooth_loss
            optimizer.zero_grad()
            total_loss.backward()
            return total_loss

        optimizer = torch.optim.LBFGS(
            [theta],
            lr=opt.optim_lr,
            max_iter=opt.optim_max_iter,
            tolerance_grad=opt.optim_tolerance,
            tolerance_change=opt.optim_tolerance,
            history_size=opt.optim_history_size,
            line_search_fn='strong_wolfe'
        )

        optimizer.step(closure)

        # Get final registration
        with torch.no_grad():
            registered_A = grid_sampler(real_A, theta, size=(h, w))

        # Compute final metrics
        with torch.no_grad():
            mi_loss = criterion_mi(registered_A, real_B)
            smooth_loss = criterion_smooth(theta)

        metrics['mi'].append(mi_loss.item())
        metrics['smoothness'].append(smooth_loss.item())
        metrics['total'].append((mi_loss + smooth_loss).item())

        # Save images
        img_basename = os.path.basename(image_paths[0])
        img_prefix = os.path.splitext(img_basename)[0]

        # Save images
        for name, img in [('real_A', real_A), ('real_B', real_B),
                          ('registered_A', registered_A)]:
            img_np = tensor2im(img)
            if img_np.shape[-1] == 3:  # RGB
                img_np = img_np[:, :, 0]  # take first channel

            save_path = os.path.join(results_dir, f'{name}_{i:04d}.png')
            Image.fromarray(img_np).save(save_path)

        # Print progress
        if (i + 1) % 5 == 0 or (i + 1) == num_test:
            print(f'Processed {i + 1}/{num_test} images')
            print(f'  Avg MI Loss: {np.mean(metrics["mi"]):.6f}')
            print(f'  Avg Total Loss: {np.mean(metrics["total"]):.6f}')

    print('='*60)
    print(f'\nTesting completed!')
    print(f'Results saved to: {results_dir}')
    print(f'\nFinal Metrics:')
    print(f'  Mean MI Loss:     {np.mean(metrics["mi"]):.6f} ± {np.std(metrics["mi"]):.6f}')
    print(f'  Mean Smoothness:  {np.mean(metrics["smoothness"]):.6f} ± {np.std(metrics["smoothness"]):.6f}')
    print(f'  Mean Total Loss:  {np.mean(metrics["total"]):.6f} ± {np.std(metrics["total"]):.6f}')


if __name__ == '__main__':
    main()