#!/usr/bin/env python3
"""
Evaluation script for MI Registration Model.

Computes registration quality metrics:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- NCC (Normalized Cross-Correlation)
"""

import argparse
import os
import glob
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F


def load_image(path):
    """Load image as numpy array."""
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def compute_mse(pred, target):
    """Compute Mean Squared Error."""
    return np.mean((pred - target) ** 2)


def compute_mae(pred, target):
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(pred - target))


def compute_psnr(pred, target, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(pred, target):
    """Compute Structural Similarity Index."""
    return ssim(pred, target, data_range=pred.max() - pred.min())


def compute_ncc(pred, target):
    """Compute Normalized Cross-Correlation."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    pred_mean = np.mean(pred_flat)
    target_mean = np.mean(target_flat)

    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean

    numerator = np.sum(pred_centered * target_centered)
    denominator = np.sqrt(np.sum(pred_centered ** 2) * np.sum(target_centered ** 2))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_mi_registration(results_dir, num_samples=None):
    """
    Evaluate MI registration results.

    Args:
        results_dir: Directory containing registration results
        num_samples: Number of samples to evaluate (None = all)

    Returns:
        Dictionary of metrics
    """
    # Find registered images
    registered_dir = os.path.join(results_dir, 'test_latest', 'images')
    if not os.path.exists(registered_dir):
        registered_dir = os.path.join(results_dir, 'registered_A')

    registered_files = sorted(glob.glob(os.path.join(registered_dir, 'registered_A_*.png')))

    if num_samples is not None:
        registered_files = registered_files[:num_samples]

    if len(registered_files) == 0:
        print(f"Error: No registered images found in {registered_dir}")
        return None

    print(f"Found {len(registered_files)} registered images")

    # Initialize metric accumulators
    metrics = {
        'mse': [],
        'mae': [],
        'psnr': [],
        'ssim': [],
        'ncc': []
    }

    # Find corresponding target images (real_B)
    target_dir = registered_dir
    for registered_file in registered_files:
        filename = os.path.basename(registered_file)
        target_filename = filename.replace('registered_A', 'real_B')
        target_file = os.path.join(target_dir, target_filename)

        if not os.path.exists(target_file):
            print(f"Warning: Target image not found: {target_file}")
            continue

        # Load images
        pred = load_image(registered_file)
        target = load_image(target_file)

        # Compute metrics
        metrics['mse'].append(compute_mse(pred, target))
        metrics['mae'].append(compute_mae(pred, target))
        metrics['psnr'].append(compute_psnr(pred, target))
        metrics['ssim'].append(compute_ssim(pred, target))
        metrics['ncc'].append(compute_ncc(pred, target))

    # Compute statistics
    results = {}
    for metric_name, values in metrics.items():
        if values:
            results[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

    return results


def print_results(results):
    """Print evaluation results."""
    print("\n" + "="*60)
    print("MI Registration Evaluation Results")
    print("="*60)

    metric_names = {
        'mse': 'Mean Squared Error',
        'mae': 'Mean Absolute Error',
        'psnr': 'PSNR (dB)',
        'ssim': 'SSIM',
        'ncc': 'Normalized Cross-Correlation'
    }

    for key, name in metric_names.items():
        if key in results:
            mean = results[key]['mean']
            std = results[key]['std']
            print(f"\n{name}:")
            print(f"  Mean:   {mean:.6f} ± {std:.6f}")
            print(f"  Range:  [{results[key]['min']:.6f}, {results[key]['max']:.6f}]")

    print("\n" + "="*60)


def save_results(results, output_file):
    """Save results to file."""
    with open(output_file, 'w') as f:
        f.write("MI Registration Evaluation Results\n")
        f.write("="*60 + "\n\n")

        metric_names = {
            'mse': 'Mean Squared Error',
            'mae': 'Mean Absolute Error',
            'psnr': 'PSNR (dB)',
            'ssim': 'SSIM',
            'ncc': 'Normalized Cross-Correlation'
        }

        for key, name in metric_names.items():
            if key in results:
                mean = results[key]['mean']
                std = results[key]['std']
                f.write(f"{name}:\n")
                f.write(f"  Mean:   {mean:.6f} ± {std:.6f}\n")
                f.write(f"  Min:    {results[key]['min']:.6f}\n")
                f.write(f"  Max:    {results[key]['max']:.6f}\n\n")

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate MI Registration Model')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing registration results')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (default: results_dir/mi_eval_results.txt)')

    args = parser.parse_args()

    # Evaluate
    results = evaluate_mi_registration(args.results_dir, args.num_samples)

    if results is None:
        return

    # Print results
    print_results(results)

    # Save results
    if args.output is None:
        args.output = os.path.join(args.results_dir, 'mi_eval_results.txt')

    save_results(results, args.output)


if __name__ == '__main__':
    main()
