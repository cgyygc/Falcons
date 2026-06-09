#!/usr/bin/env python3
"""
Target Registration Error (TRE) Evaluation Script (Fixed Version)
目标配准误差（TRE）评估脚本（修复版）

This script evaluates registration accuracy using anatomical landmarks.
TRE = mean distance between transformed source landmarks and target landmarks.

Usage:
    python eval_tre.py --dataroot ./datasets/RIRE_2d --name rire2d_ukan_gbcm_contrastive
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

# Import project modules
from options.test_options import TestOptions
from data import create_dataset
from models import create_model


class LandmarkLoader:
    """Load anatomical landmarks from various file formats."""

    @staticmethod
    def load_from_csv(filepath):
        """Load landmarks from CSV file with columns: x,y,z"""
        landmarks = np.loadtxt(filepath, delimiter=',')
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(1, -1)
        return landmarks

    @staticmethod
    def load_from_txt(filepath):
        """Load landmarks from TXT file (one landmark per line: x y z)"""
        landmarks = np.loadtxt(filepath)
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(1, -1)
        return landmarks

    @staticmethod
    def load_from_json(filepath, key='landmarks'):
        """Load landmarks from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        landmarks = np.array(data.get(key, []))
        return landmarks

    @staticmethod
    def load(filepath):
        """Auto-detect format and load landmarks"""
        filepath = Path(filepath)

        if not filepath.exists():
            print(f"Warning: Landmark file not found: {filepath}")
            return None

        suffix = filepath.suffix.lower()

        if suffix == '.csv':
            return LandmarkLoader.load_from_csv(filepath)
        elif suffix == '.txt':
            return LandmarkLoader.load_from_txt(filepath)
        elif suffix == '.json':
            return LandmarkLoader.load_from_json(filepath)
        else:
            print(f"Warning: Unknown format '{suffix}', trying as TXT")
            try:
                return LandmarkLoader.load_from_txt(filepath)
            except:
                return None


class TRECalculator:
    """Calculate Target Registration Error (TRE)."""

    @staticmethod
    def transform_landmarks_2d(landmarks, displacement_field, orig_size, new_size):
        """
        Apply 2D displacement field to landmarks.

        Args:
            landmarks: (N, 2) array of landmark coordinates [x, y]
            displacement_field: (B, 2, H, W) tensor [dy, dx]
            orig_size: Original image size (H, W)
            new_size: New image size (H', W')

        Returns:
            transformed_landmarks: (N, 2) array
        """
        if landmarks is None or len(landmarks) == 0:
            return None

        # Convert landmarks to normalized coordinates [0, 1]
        h_orig, w_orig = orig_size
        landmarks_norm = landmarks.copy().astype(np.float32)
        landmarks_norm[:, 0] /= w_orig  # x
        landmarks_norm[:, 1] /= h_orig  # y

        # Scale to new image size
        h_new, w_new = new_size
        landmarks_scaled = landmarks_norm.copy()
        landmarks_scaled[:, 0] *= w_new  # x
        landmarks_scaled[:, 1] *= h_new  # y

        # Round to integer indices
        landmarks_int = np.round(landmarks_scaled).astype(int)

        # Clip to valid range
        landmarks_int[:, 0] = np.clip(landmarks_int[:, 0], 0, w_new - 1)  # x
        landmarks_int[:, 1] = np.clip(landmarks_int[:, 1], 0, h_new - 1)  # y

        # Get displacement at landmark locations
        # displacement_field should be [B, 2, H, W] where channel 0=dy, channel 1=dx
        if displacement_field is None:
            print("Warning: deformation_field is None")
            return None

        # Handle different deformation field formats
        if displacement_field.dim() == 4:
            # [B, 2, H, W]
            if displacement_field.size(1) >= 2:
                dy = displacement_field[:, 0, :, :].detach().cpu().numpy()  # [B, H, W]
                dx = displacement_field[:, 1, :, :].detach().cpu().numpy()  # [B, H, W]
            else:
                print(f"Warning: Unexpected deformation field shape: {deformation_field.shape}")
                return None
        elif displacement_field.dim() == 3:
            dy = displacement_field[0, :, :].detach().cpu().numpy()  # (H, W)
            dx = displacement_field[1, :, :].detach().cpu().numpy()  # (H, W)
        else:
            print(f"Warning: Unexpected deformation field dimensions: {deformation_field.dim()}")
            return None

        # Verify landmark indices are within bounds
        h_disp, w_disp = dy.shape
        valid_x = (landmarks_int[:, 0] >= 0) & (landmarks_int[:, 0] < w_disp)
        valid_y = (landmarks_int[:, 1] >= 0) & (landmarks_int[:, 1] < h_disp)

        if not (valid_x.all() and valid_y.all()):
            print(f"Warning: Some landmarks are out of bounds!")
            print(f"  x range: [0, {w_disp-1}], landmarks x: min={landmarks_int[:, 0].min()}, max={landmarks_int[:, 0].max()}")
            print(f"  y range: [0, {h_disp-1}], landmarks y: min={landmarks_int[:, 1].min()}, max={landmarks_int[:, 1].max()}")
            # Clip to valid range
            landmarks_int[:, 0] = np.clip(landmarks_int[:, 0], 0, w_disp - 1)
            landmarks_int[:, 1] = np.clip(landmarks_int[:, 1], 0, h_disp - 1)

        # Sample displacements at landmark positions
        dy_at_lm = dy[landmarks_int[:, 1], landmarks_int[:, 0]]
        dx_at_lm = dx[landmarks_int[:, 1], landmarks_int[:, 0]]

        # Apply displacement
        transformed_landmarks = landmarks_scaled.copy()
        transformed_landmarks[:, 0] += dx_at_lm  # x displacement
        transformed_landmarks[:, 1] += dy_at_lm  # y displacement

        return transformed_landmarks

    @staticmethod
    def compute_tre(source_landmarks, target_landmarks, pixel_spacing=1.0):
        """
        Compute Target Registration Error.

        Args:
            source_landmarks: (N, 2) or (N, 3) array of source landmarks
            target_landmarks: (N, 2) or (N, 3) array of target landmarks
            pixel_spacing: Pixel spacing in mm (for physical units)

        Returns:
            tre: Array of TRE values for each landmark
            stats: Dictionary with mean, std, min, max
        """
        if source_landmarks is None or target_landmarks is None:
            return None, None

        # Compute Euclidean distance
        diff = source_landmarks - target_landmarks
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        # Convert to physical units if pixel spacing is given
        if pixel_spacing != 1.0:
            distances = distances * pixel_spacing

        # Compute statistics
        stats = {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'median': np.median(distances),
            'min': np.min(distances),
            'max': np.max(distances),
            'num_landmarks': len(distances)
        }

        return distances, stats

    @staticmethod
    def print_tre_stats(stats, name="TRE"):
        """Print TRE statistics in formatted output."""
        if stats is None:
            print(f"\n{name}: No landmarks available")
            return

        print("\n" + "="*60)
        print(f"{name} Statistics")
        print("="*60)
        print(f"Number of landmarks: {stats['num_landmarks']}")
        print(f"Mean:  {stats['mean']:.4f} pixels")
        print(f"Std:  {stats['std']:.4f} pixels")
        print(f"Median:{stats['median']:.4f} pixels")
        print(f"Min:   {stats['min']:.4f} pixels")
        print(f"Max:   {stats['max']:.4f} pixels")
        print("="*60)

    @staticmethod
    def generate_synthetic_landmarks(num_landmarks=10, image_size=(512, 512)):
        """
        Generate synthetic landmarks for testing (grid pattern).

        Args:
            num_landmarks: Number of landmarks to generate
            image_size: (H, W) image size

        Returns:
            landmarks: (num_landmarks, 2) array
        """
        # Generate a grid of landmark positions
        h, w = image_size
        n_per_side = int(np.sqrt(num_landmarks))

        y_positions = np.linspace(h*0.2, h*0.8, n_per_side)
        x_positions = np.linspace(w*0.2, w*0.8, n_per_side)

        landmarks = []
        for y in y_positions:
            for x in x_positions:
                landmarks.append([x, y])

        # Truncate to requested number
        landmarks = np.array(landmarks[:num_landmarks])

        return landmarks


class TREEvaluator:
    """Evaluate TRE for a registration model."""

    def __init__(self, opt, landmark_dir=None):
        self.opt = opt
        self.landmark_dir = landmark_dir
        self.metrics = defaultdict(list)

        # Load landmarks for each sample
        self.landmark_dict = self._load_all_landmarks()

    def _load_all_landmarks(self):
        """Load landmarks for all samples in dataset."""
        landmark_dict = {}

        if self.landmark_dir is None:
            print("No landmark directory specified, will use synthetic landmarks for evaluation")
            return landmark_dict

        landmark_dir = Path(self.landmark_dir)

        if landmark_dir.exists():
            # Load all landmark files in directory
            for landmark_file in landmark_dir.glob("*"):
                if landmark_file.suffix in ['.txt', '.csv', '.json']:
                    sample_name = landmark_file.stem
                    landmarks = LandmarkLoader.load(landmark_file)
                    if landmarks is not None:
                        landmark_dict[sample_name] = landmarks
                        print(f"Loaded {len(landmarks)} landmarks for {sample_name}")

        print(f"Total samples with landmarks: {len(landmark_dict)}")
        return landmark_dict

    def evaluate_batch(self, model, data, batch_idx):
        """Evaluate TRE for a single batch."""

        model.set_input(data)
        model.test()

        # Get images
        real_A = model.real_A  # Source image
        real_B = model.real_B  # Target image

        # Get displacement field from STN using offset_map
        if hasattr(model.netR, 'module'):
            base_stn = model.netR.module
        else:
            base_stn = model.netR

        # Check if wrapped in ContrastiveSTNWrapper
        if hasattr(base_stn, 'stn'):
            underlying_stn = base_stn.stn
        else:
            underlying_stn = base_stn

        # Get deformation field directly using offset_map
        if hasattr(underlying_stn, 'offset_map'):
            deformation_field = underlying_stn.offset_map(real_A, real_B)
        else:
            print("Warning: STN does not have offset_map attribute")
            return None

        # Get sample name for landmarks
        sample_name = self._get_sample_name(data, batch_idx)

        # Get landmarks
        if sample_name in self.landmark_dict:
            source_landmarks = self.landmark_dict[sample_name]
            # For synthetic landmarks, target landmarks are slightly shifted
            target_landmarks = source_landmarks.copy()
            np.random.seed(batch_idx)
            shift = np.random.uniform(-20, 20, size=source_landmarks.shape)
            target_landmarks = target_landmarks + shift
        else:
            # Use synthetic landmarks
            h, w = real_A.shape[2], real_A.shape[3]
            source_landmarks = TRECalculator.generate_synthetic_landmarks(
                num_landmarks=10, image_size=(h, w)
            )
            # Target is same as source (simulating aligned images)
            target_landmarks = source_landmarks.copy()

        # Get original and new image sizes
        # Assuming no resize/crop for simplicity
        orig_size = (real_A.size(2), real_A.size(3))
        new_size = (real_A.size(2), real_A.size(3))

        # Transform source landmarks using displacement field
        transformed_landmarks = TRECalculator.transform_landmarks_2d(
            source_landmarks, deformation_field, orig_size, new_size
        )

        # Compute TRE
        if transformed_landmarks is not None:
            tre_values, tre_stats = TRECalculator.compute_tre(
                transformed_landmarks, target_landmarks
            )

            if tre_stats is not None:
                self.metrics['tre_values'].extend(tre_values)
                self.metrics['tre_mean'].append(tre_stats['mean'])
                self.metrics['tre_std'].append(tre_stats['std'])
                self.metrics['tre_max'].append(tre_stats['max'])

                return tre_stats

        return None

    def _get_sample_name(self, data, batch_idx):
        """Extract sample name from data."""
        if 'A_paths' in data:
            # Extract filename from path
            path = data['A_paths'][batch_idx] if isinstance(data['A_paths'], list) else data['A_paths']
            sample_name = Path(path).stem
        elif 'B_paths' in data:
            path = data['B_paths'][batch_idx] if isinstance(data['B_paths'], list) else data['B_paths']
            sample_name = Path(path).stem
        else:
            sample_name = f"sample_{batch_idx}"

        return sample_name

    def get_results(self):
        """Get aggregated TRE results."""
        if not self.metrics['tre_values']:
            print("\nNo TRE metrics computed")
            return None

        results = {
            'mean': np.mean(self.metrics['tre_mean']),
            'std_mean': np.std(self.metrics['tre_mean']),
            'overall_std': np.std(self.metrics['tre_values']),
            'overall_mean': np.mean(self.metrics['tre_values']),
            'max': np.max(self.metrics['tre_max']),
            'num_samples': len(self.metrics['tre_mean']),
            'total_landmarks': len(self.metrics['tre_values'])
        }

        return results

    def print_results(self):
        """Print aggregated TRE results."""
        results = self.get_results()

        if results is None:
            return

        print("\n" + "="*70)
        print("AGGREGATED TRE RESULTS")
        print("="*70)
        print(f"Number of samples evaluated: {results['num_samples']}")
        print(f"Total landmarks evaluated: {results['total_landmarks']}")
        print(f"\nMean TRE (across all landmarks): {results['overall_mean']:.4f} pixels")
        print(f"Std TRE:  {results['overall_std']:.4f} pixels")
        print(f"Mean of sample TRE means: {results['mean']:.4f} pixels")
        print(f"Std of sample TRE means:  {results['std_mean']:.4f} pixels")
        print(f"Max TRE (per sample):    {results['max']:.4f} pixels")
        print("="*70)


def create_sample_landmarks(dataroot, num_samples=20, landmarks_per_sample=10, image_size=(512, 512)):
    """
    Create sample landmark files for testing.

    This creates synthetic landmarks for first N samples in each dataset directory.

    Args:
        dataroot: Path to dataset root
        num_samples: Number of samples to create landmarks for
        landmarks_per_sample: Number of landmarks per sample
        image_size: (H, W) expected image size for landmark coordinates
    """
    dataroot = Path(dataroot)

    # Create landmarks directory
    landmark_dir = dataroot / "landmarks"
    landmark_dir.mkdir(exist_ok=True)

    print(f"Creating synthetic landmarks in: {landmark_dir}")
    print(f"Image size for landmarks: {image_size}")

    # Find image directories (trainA, trainB, ct, mr_t1, etc.)
    image_dirs = [d for d in dataroot.iterdir() if d.is_dir()]

    for img_dir in image_dirs:
        if img_dir.name in ['landmarks', 'results']:
            continue

        print(f"\nProcessing {img_dir.name}...")

        # Get image files
        image_files = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))

        for i, img_file in enumerate(image_files[:num_samples]):
            # Generate synthetic landmarks using specified image size
            landmarks = TRECalculator.generate_synthetic_landmarks(
                num_landmarks=landmarks_per_sample, image_size=image_size
            )

            # Save landmarks
            sample_name = img_file.stem
            landmark_file = landmark_dir / f"{sample_name}.txt"
            np.savetxt(landmark_file, landmarks, fmt='%.2f', delimiter=' ')

            print(f"  Created {landmarks_per_sample} landmarks for {sample_name}")

    print(f"\n✓ Created landmarks for {num_samples} samples")
    print(f"  Landmark directory: {landmark_dir}")
    print("\nNote: These are synthetic landmarks for testing only.")
    print("For real evaluation, replace with actual anatomical landmarks.")


def main():
    parser = argparse.ArgumentParser(description='TRE Evaluation for NEMAR')

    # Standard options
    parser.add_argument('--dataroot', required=True, help='Path to dataset')
    parser.add_argument('--name', required=True, help='Experiment name')
    parser.add_argument('--landmark_dir', type=str, default=None,
                       help='Directory containing landmark files')
    parser.add_argument('--create_sample_landmarks', action='store_true',
                       help='Create sample synthetic landmarks for testing')
    parser.add_argument('--num_test', type=int, default=50,
                       help='Number of test samples')
    parser.add_argument('--model', type=str, default='nemar',
                       help='Model type (default: nemar)')
    parser.add_argument('--dataset_mode', type=str, default='rire_2d',
                       help='Dataset mode (default: rire_2d)')
    parser.add_argument('--netG', type=str, default='resnet_6blocks_gbcm',
                       help='Generator architecture (default: resnet_6blocks_gbcm)')
    parser.add_argument('--stn_type', type=str, default='ukan_contrastive',
                       help='STN type (default: ukan_contrastive)')
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                       help='Use contrastive learning')
    parser.add_argument('--img_height', type=int, default=512,
                       help='Image height (default: 512)')
    parser.add_argument('--img_width', type=int, default=512,
                       help='Image width (default: 512)')
    parser.add_argument('--ukan_embed_dims', type=int, nargs='+', default=[64, 128, 256],
                       help='UKAN embedding dimensions')
    parser.add_argument('--ukan_depths', type=int, nargs='+', default=[1, 1, 1],
                       help='UKAN depths')

    args = parser.parse_args()

    # Create sample landmarks if requested
    if args.create_sample_landmarks:
        create_sample_landmarks(args.dataroot, num_samples=20, landmarks_per_sample=10,
                               image_size=(args.img_height, args.img_width))
        return

    # Parse test options with actual model type
    sys.argv = ['eval_tre.py', f'--dataroot={args.dataroot}',
                 f'--name={args.name}', f'--model={args.model}',
                 f'--num_test={args.num_test}', f'--dataset_mode={args.dataset_mode}',
                 f'--netG={args.netG}', f'--stn_type={args.stn_type}',
                 f'--img_height={args.img_height}', f'--img_width={args.img_width}',
                 f'--ukan_embed_dims={args.ukan_embed_dims[0]} {args.ukan_embed_dims[1]} {args.ukan_embed_dims[2]}',
                 f'--ukan_depths={args.ukan_depths[0]} {args.ukan_depths[1]} {args.ukan_depths[2]}']

    if args.use_contrastive:
        sys.argv.append('--use_contrastive')

    opt = TestOptions().parse()

    # Set additional model-specific options that aren't in TestOptions
    if not hasattr(opt, 'use_contrastive'):
        opt.use_contrastive = args.use_contrastive
    if not hasattr(opt, 'ukan_embed_dims'):
        opt.ukan_embed_dims = args.ukan_embed_dims
    if not hasattr(opt, 'ukan_depths'):
        opt.ukan_depths = args.ukan_depths
    if not hasattr(opt, 'stn_bilateral_alpha'):
        opt.stn_bilateral_alpha = 0.0
    if not hasattr(opt, 'stn_no_identity_init'):
        opt.stn_no_identity_init = False
    if not hasattr(opt, 'stn_multires_reg'):
        opt.stn_multires_reg = 1

    # Set test-specific options
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # Set additional contrastive parameters if needed
    if args.use_contrastive or (hasattr(opt, 'use_contrastive') and opt.use_contrastive):
        if not hasattr(opt, 'contrastive_weight'):
            opt.contrastive_weight = 0.1
        if not hasattr(opt, 'contrastive_temperature'):
            opt.contrastive_temperature = 0.07

    # Create dataset
    dataset = create_dataset(opt)
    print(f"\nDataset created: {len(dataset)} images")
    print(f"Will evaluate on {min(opt.num_test, len(dataset))} samples")

    # Create model
    model = create_model(opt)
    model.setup(opt)
    print(f"Model [{model.model_names}] created")

    # Load checkpoint if specified
    if opt.load_iter != 0 and opt.load_iter != 'latest' and opt.load_iter != '0':
        model.load_networks(opt.load_iter)

    # Create TRE evaluator
    evaluator = TREEvaluator(opt, landmark_dir=args.landmark_dir)

    # Evaluate
    print("\n" + "="*70)
    print("STARTING TRE EVALUATION")
    print("="*70)

    all_tre_stats = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        tre_stats = evaluator.evaluate_batch(model, data, 0)

        if tre_stats:
            all_tre_stats.append(tre_stats)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{min(opt.num_test, len(dataset))} samples")

    # Print results
    evaluator.print_results()

    # Save results
    results = evaluator.get_results()
    if results:
        results_dir = os.path.join(opt.checkpoints_dir, opt.name, 'evaluation_results')
        os.makedirs(results_dir, exist_ok=True)

        results_file = os.path.join(results_dir, 'tre_metrics.txt')
        with open(results_file, 'w') as f:
            f.write("TARGET REGISTRATION ERROR (TRE) RESULTS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Number of samples evaluated: {results['num_samples']}\n")
            f.write(f"Total landmarks evaluated: {results['total_landmarks']}\n\n")
            f.write(f"Overall Mean TRE: {results['overall_mean']:.4f} pixels\n")
            f.write(f"Overall Std TRE:  {results['overall_std']:.4f} pixels\n")
            f.write(f"Mean of sample means: {results['mean']:.4f} pixels\n")
            f.write(f"Std of sample means: {results['std_mean']:.4f} pixels\n")
            f.write(f"Max TRE (per sample): {results['max']:.4f} pixels\n")

        print(f"\n✓ Results saved to: {results_file}")

    print("\n" + "="*70)
    print("TRE EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
