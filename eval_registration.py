"""
Registration Evaluation Script

This script evaluates registration performance using various metrics:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- NCC (Normalized Cross-Correlation)

Usage:
    python eval_registration.py --dataroot ./datasets/rire --name rire2d_ukan_gbcm_contrastive_stable
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

# Import project modules
from options.test_options import TestOptions
from data import create_dataset
from models import create_model


def mse_metric(pred, target):
    """Mean Squared Error."""
    return torch.mean((pred - target) ** 2).item()


def mae_metric(pred, target):
    """Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target)).item()


def psnr_metric(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def ssim_metric(pred, target, window_size=11, max_val=1.0):
    """
    Structural Similarity Index (SSIM).
    Simplified implementation for 2D images.
    """
    from torch.nn.functional import avg_pool2d, conv2d

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Create Gaussian kernel
    def create_window(size):
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(size)
        ])
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)

    window = create_window(window_size).to(pred.device)

    # Extract channels
    if pred.dim() == 4:  # (B, C, H, W)
        pred = pred[:, 0:1, :, :]  # Use first channel
        target = target[:, 0:1, :, :]

    # Calculate means
    mu1 = conv2d(pred, window, padding=window_size // 2)
    mu2 = conv2d(target, window, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Calculate variances and covariance
    sigma1_sq = conv2d(pred * pred, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = conv2d(target * target, window, padding=window_size // 2) - mu2_sq
    sigma12 = conv2d(pred * target, window, padding=window_size // 2) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def ncc_metric(pred, target):
    """
    Normalized Cross-Correlation.
    Measures the linear correlation between two images.
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()

    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean

    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum()) * \
                  torch.sqrt((target_centered ** 2).sum())

    if denominator == 0:
        return 0.0

    return (numerator / denominator).item()


def compute_jacobian_folding(disp_field):
    """
    Compute folding rate of the deformation field using Jacobian determinant.
    Lower folding rate indicates more physically plausible deformation.

    Args:
        disp_field: Deformation/displacement field tensor

    Returns:
        folding_rate: Percentage of voxels with negative Jacobian determinant
    """
    # Compute spatial gradients
    if disp_field.dim() == 4:  # (B, C, H, W)
        # For 2D displacement field with 2 channels
        if disp_field.size(1) >= 2:
            # Gradient in y direction
            dy = disp_field[:, :, 1:, :] - disp_field[:, :, :-1, :]
            dy = torch.nn.functional.pad(dy, (0, 0, 0, 1), 'replicate')

            # Gradient in x direction
            dx = disp_field[:, :, :, 1:] - disp_field[:, :, :, :-1]
            dx = torch.nn.functional.pad(dx, (0, 1, 0, 0), 'replicate')

            # Jacobian determinant: det(I + grad(disp))
            # For 2D: det([[1+dx_x, dx_y], [dy_x, 1+dy_y]])
            if disp_field.size(1) == 2:
                # Assume channel 0 is y displacement, channel 1 is x displacement
                dxdy = dx[:, 0:1, :, :]  # dx/dy
                dxdx = dx[:, 1:2, :, :]  # dx/dx
                dydy = dy[:, 0:1, :, :]  # dy/dy
                dydx = dy[:, 1:2, :, :]  # dy/dx

                jac_det = (1 + dxdx) * (1 + dydy) - dxdy * dydx
                folding_rate = (jac_det < 0).float().mean().item()
                return folding_rate

    return 0.0


class RegistrationEvaluator:
    """Evaluator for registration performance."""

    def __init__(self, opt):
        self.opt = opt
        self.metrics = {
            'mse': [],
            'mae': [],
            'psnr': [],
            'ssim': [],
            'ncc': [],
        }
        self.metrics_registered = {
            'mse': [],
            'mae': [],
            'psnr': [],
            'ssim': [],
            'ncc': [],
        }
        self.jacobian_folding_rates = []

    def evaluate_batch(self, model, data):
        """
        Evaluate a single batch.

        Args:
            model: The registration model
            data: Dictionary containing 'A' (source) and 'B' (target)
        """
        model.set_input(data)
        model.test()

        # Get images
        real_A = model.real_A  # Source image
        real_B = model.real_B  # Target image

        # Get registered/translated results
        # fake_TR_B: Registration first, then translation
        # registered_real_A: Just the registered source (no translation)
        if hasattr(model, 'fake_TR_B'):
            fake_TR_B = model.fake_TR_B
        else:
            fake_TR_B = torch.zeros_like(real_B)

        if hasattr(model, 'registered_real_A'):
            registered_A = model.registered_real_A
        else:
            registered_A = torch.zeros_like(real_A)

        # Compute Jacobian folding rate
        # Get deformation field from STN
        try:
            # Access underlying STN module (may be wrapped in DataParallel)
            if hasattr(model.netR, 'module'):
                stn = model.netR.module
            else:
                stn = model.netR

            # Try to get deformation field using offset_map or get_grid
            if hasattr(stn, 'offset_map'):
                deformation_field = stn.offset_map(real_A, real_B)
                # If offset_map returns a grid, we need the displacement part
                if deformation_field.size(1) == 4:  # Full grid [x, y, 1-x, 1-y]
                    # Extract displacement (channels 2:3 = [1-x, 1-y])
                    disp_field = deformation_field[:, 2:, :, :]
                else:
                    disp_field = deformation_field
            elif hasattr(stn, 'get_grid'):
                deformation_field = stn.get_grid(real_A, real_B)
                # get_grid typically returns full grid, extract displacement
                if deformation_field.size(1) == 4:
                    disp_field = deformation_field[:, 2:, :, :]
                else:
                    disp_field = deformation_field
            else:
                disp_field = None

            if disp_field is not None:
                folding_rate = compute_jacobian_folding(disp_field)
                self.jacobian_folding_rates.append(folding_rate)
            else:
                print("Warning: Could not get deformation field for Jacobian analysis")
        except Exception as e:
            print(f"Warning: Error computing Jacobian folding rate: {e}")

        # Denormalize from [-1, 1] to [0, 1] for metrics
        def denorm(x):
            return (x + 1) / 2

        real_B_denorm = denorm(real_B)
        fake_TR_B_denorm = denorm(fake_TR_B)
        registered_A_denorm = denorm(registered_A)

        # Compute metrics for translated+registered image vs target
        self.metrics['mse'].append(mse_metric(fake_TR_B_denorm, real_B_denorm))
        self.metrics['mae'].append(mae_metric(fake_TR_B_denorm, real_B_denorm))
        self.metrics['psnr'].append(psnr_metric(fake_TR_B_denorm, real_B_denorm))
        self.metrics['ssim'].append(ssim_metric(fake_TR_B_denorm, real_B_denorm))
        self.metrics['ncc'].append(ncc_metric(fake_TR_B_denorm, real_B_denorm))

        # Compute metrics for registered source vs target (different modalities)
        # Note: These metrics may be less meaningful for cross-modal comparison
        self.metrics_registered['mse'].append(mse_metric(registered_A_denorm, real_B_denorm))
        self.metrics_registered['mae'].append(mae_metric(registered_A_denorm, real_B_denorm))
        self.metrics_registered['psnr'].append(psnr_metric(registered_A_denorm, real_B_denorm))
        self.metrics_registered['ssim'].append(ssim_metric(registered_A_denorm, real_B_denorm))
        self.metrics_registered['ncc'].append(ncc_metric(registered_A_denorm, real_B_denorm))

    def get_results(self):
        """Return average metrics."""
        results = {}

        print("\n" + "="*70)
        print("REGISTRATION EVALUATION RESULTS")
        print("="*70)

        print("\n--- Translated+Registered Image vs Target (Same Modality) ---")
        for metric_name, values in self.metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                results[f'{metric_name}_tr'] = mean_val
                print(f"{metric_name.upper():6s}: {mean_val:.4f} +/- {std_val:.4f}")

        print("\n--- Registered Source vs Target (Cross-Modality) ---")
        for metric_name, values in self.metrics_registered.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                results[f'{metric_name}_reg'] = mean_val
                print(f"{metric_name.upper():6s}: {mean_val:.4f} +/- {std_val:.4f}")

        # Print Jacobian folding rate
        if self.jacobian_folding_rates:
            jacob_mean = np.mean(self.jacobian_folding_rates)
            jacob_std = np.std(self.jacobian_folding_rates)
            results['jacobian_folding_rate'] = jacob_mean
            print("\n--- Deformation Field Analysis ---")
            print(f"JACOBIAN FOLDING RATE: {jacob_mean:.4%} +/- {jacob_std:.4%}")
            print(f"  (Percentage of voxels with negative Jacobian determinant)")
            print(f"  Lower is better - indicates topological preservation")
        else:
            print("\n--- Deformation Field Analysis ---")
            print("JACOBIAN FOLDING RATE: Not computed")

        print("\n" + "="*70)

        return results


def main():
    # Parse options
    opt = TestOptions().parse()

    # Hard-code some options for evaluation
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no vis.display

    # Set model-specific options that aren't in TestOptions
    # These need to match the training configuration
    if not hasattr(opt, 'use_contrastive'):
        opt.use_contrastive = False  # Will be overridden if needed based on stn_type

    # Enable contrastive if using ukan_contrastive STN type
    if opt.stn_type == 'ukan_contrastive':
        opt.use_contrastive = True
        # Set default contrastive parameters if not specified
        if not hasattr(opt, 'contrastive_weight'):
            opt.contrastive_weight = 0.1
        if not hasattr(opt, 'contrastive_temperature'):
            opt.contrastive_temperature = 0.07
        if not hasattr(opt, 'contrastive_loss_type'):
            opt.contrastive_loss_type = 'infonce'
        if not hasattr(opt, 'contrastive_proj_dim'):
            opt.contrastive_proj_dim = 128
        if not hasattr(opt, 'contrastive_num_stages'):
            opt.contrastive_num_stages = None

    # Set STN parameters
    if not hasattr(opt, 'stn_bilateral_alpha'):
        opt.stn_bilateral_alpha = 0.0
    if not hasattr(opt, 'stn_no_identity_init'):
        opt.stn_no_identity_init = False
    if not hasattr(opt, 'stn_multires_reg'):
        opt.stn_multires_reg = 1
    if not hasattr(opt, 'ukan_embed_dims'):
        opt.ukan_embed_dims = [64, 128, 256]
    if not hasattr(opt, 'ukan_depths'):
        opt.ukan_depths = [1, 1, 1]

    # Make sure these are boolean (not string)
    if isinstance(opt.no_flip, str):
        opt.no_flip = opt.no_flip.lower() == 'true'

    # Create dataset
    dataset = create_dataset(opt)
    print(f"Dataset created: {len(dataset)} images")

    # Create model
    model = create_model(opt)
    model.setup(opt)
    print(f"Model [{model.model_names}] created")

    # Skip loading checkpoint if load_iter is 0 (use latest which is already loaded by setup)
    if opt.load_iter != 0 and opt.load_iter != 'latest' and opt.load_iter != '0':
        model.load_networks(opt.load_iter)

    # Create evaluator
    evaluator = RegistrationEvaluator(opt)

    # Evaluate
    print("\nEvaluating...")
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        evaluator.evaluate_batch(model, data)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{min(opt.num_test, len(dataset))} images")

    # Get and print results
    results = evaluator.get_results()

    # Save results to file
    results_dir = os.path.join(opt.checkpoints_dir, opt.name, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, 'metrics.txt')
    with open(results_file, 'w') as f:
        f.write("REGISTRATION EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")

        f.write("--- Translated+Registered Image vs Target ---\n")
        for metric_name, values in evaluator.metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"{metric_name.upper()}: {mean_val:.4f} +/- {std_val:.4f}\n")

        f.write("\n--- Registered Source vs Target (Cross-Modality) ---\n")
        for metric_name, values in evaluator.metrics_registered.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"{metric_name.upper()}: {mean_val:.4f} +/- {std_val:.4f}\n")

        # Save Jacobian folding rate
        if evaluator.jacobian_folding_rates:
            jacob_mean = np.mean(evaluator.jacobian_folding_rates)
            jacob_std = np.std(evaluator.jacobian_folding_rates)
            f.write("\n--- Deformation Field Analysis ---\n")
            f.write(f"JACOBIAN FOLDING RATE: {jacob_mean:.4%} +/- {jacob_std:.4%}\n")
            f.write("  (Percentage of voxels with negative Jacobian determinant)\n")
            f.write("  Lower is better - indicates topological preservation\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
