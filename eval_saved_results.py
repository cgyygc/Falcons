#!/usr/bin/env python3
"""
Evaluate saved registration results by comparing registered/translated images
with target images.

Usage:
    python eval_saved_results.py --results_dir ./results/mi_l2r/test_latest/images \
        --prefix_registered registered_A --prefix_target real_B --num_test 100
"""

import os
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import glob
import json


def mse_metric(pred, target):
    return np.mean((pred - target) ** 2)


def mae_metric(pred, target):
    return np.mean(np.abs(pred - target))


def psnr_metric(pred, target, max_val=1.0):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def ncc_metric(pred, target):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean
    numerator = (pred_centered * target_centered).sum()
    denominator = np.sqrt((pred_centered ** 2).sum()) * np.sqrt((target_centered ** 2).sum())
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True, help='Directory containing result images')
    parser.add_argument('--prefix_pred', default='registered_A', help='Prefix for predicted images')
    parser.add_argument('--prefix_target', default='real_B', help='Prefix for target images')
    parser.add_argument('--num_test', type=int, default=100, help='Number of images to evaluate')
    parser.add_argument('--output', type=str, default=None, help='Output file for results')
    args = parser.parse_args()

    # Find matching image pairs
    pred_files = sorted(glob.glob(os.path.join(args.results_dir, f'{args.prefix_pred}_*.png')))
    target_files = sorted(glob.glob(os.path.join(args.results_dir, f'{args.prefix_target}_*.png')))

    # Match by index
    pred_map = {}
    for f in pred_files:
        idx = os.path.basename(f).replace(args.prefix_pred + '_', '').replace('.png', '')
        pred_map[idx] = f

    target_map = {}
    for f in target_files:
        idx = os.path.basename(f).replace(args.prefix_target + '_', '').replace('.png', '')
        target_map[idx] = f

    common_indices = sorted(set(pred_map.keys()) & set(target_map.keys()))
    common_indices = common_indices[:args.num_test]

    print(f"Found {len(common_indices)} matching image pairs")

    metrics = {'mse': [], 'mae': [], 'psnr': [], 'ssim': [], 'ncc': []}

    for i, idx in enumerate(common_indices):
        pred = np.array(Image.open(pred_map[idx]).convert('L')).astype(np.float32) / 255.0
        target = np.array(Image.open(target_map[idx]).convert('L')).astype(np.float32) / 255.0

        metrics['mse'].append(mse_metric(pred, target))
        metrics['mae'].append(mae_metric(pred, target))
        metrics['psnr'].append(psnr_metric(pred, target))
        metrics['ssim'].append(ssim(pred, target, data_range=1.0))
        metrics['ncc'].append(ncc_metric(pred, target))

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(common_indices)} images")

    # Print results
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({args.prefix_pred} vs {args.prefix_target})")
    print("=" * 60)
    results = {}
    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        results[metric_name] = {'mean': mean_val, 'std': std_val}
        print(f"{metric_name.upper():6s}: {mean_val:.4f} +/- {std_val:.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
