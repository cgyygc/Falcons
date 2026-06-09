"""Script to convert L2R 3D NIfTI files to 2D slice images.

L2R dataset format:
- Files are named: imgXXXX_tcia_CT.nii.gz, imgXXXX_tcia_MR.nii.gz
- Located in Train/ and Test/ directories

This script extracts 2D slices from 3D volumes for faster training.
"""
import os
import argparse
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import re


def load_nifti_volume(nii_path):
    """Load a NIfTI file and return the data array."""
    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata()
    return data


def normalize_volume(data):
    """Normalize volume data to [0, 255] range for saving as image."""
    data_min = np.min(data)
    data_max = np.max(data)

    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)

    return (data * 255).astype(np.uint8)


def save_slice(slice_data, output_path):
    """Save a 2D slice as a PNG image."""
    img = Image.fromarray(slice_data, mode='L')
    img.save(output_path)


def process_case(case_id, ct_path, mr_path, output_dir,
                slice_axis=2, skip_empty=True, intensity_threshold=10):
    """Process a single case (CT + MR pair).

    Parameters:
        case_id: Case identifier (e.g., '0002')
        ct_path: Path to CT NIfTI file
        mr_path: Path to MR NIfTI file
        output_dir: Output directory
        slice_axis: Axis to slice along (0=sagittal, 1=coronal, 2=axial)
        skip_empty: Skip slices with low intensity
        intensity_threshold: Threshold for non-empty slices

    Returns:
        dict: Statistics about processed slices
    """
    stats = {'ct': 0, 'mr': 0, 'matched': 0}

    # Create modality directories
    ct_output_dir = os.path.join(output_dir, 'ct')
    mr_output_dir = os.path.join(output_dir, 'mr')
    os.makedirs(ct_output_dir, exist_ok=True)
    os.makedirs(mr_output_dir, exist_ok=True)

    # Load CT
    if ct_path and os.path.exists(ct_path):
        ct_volume = load_nifti_volume(ct_path)
        ct_normalized = normalize_volume(ct_volume)
    else:
        ct_volume = None
        ct_normalized = None

    # Load MR
    if mr_path and os.path.exists(mr_path):
        mr_volume = load_nifti_volume(mr_path)
        mr_normalized = normalize_volume(mr_volume)
    else:
        mr_volume = None
        mr_normalized = None

    # Determine number of slices
    if ct_normalized is not None:
        num_slices = ct_normalized.shape[slice_axis]
    elif mr_normalized is not None:
        num_slices = mr_normalized.shape[slice_axis]
    else:
        return stats

    # Extract and save slices
    for slice_idx in range(num_slices):
        ct_slice = None
        mr_slice = None

        # Extract CT slice
        if ct_normalized is not None:
            if slice_axis == 0:
                ct_slice = ct_normalized[slice_idx, :, :]
            elif slice_axis == 1:
                ct_slice = ct_normalized[:, slice_idx, :]
            else:
                ct_slice = ct_normalized[:, :, slice_idx]

        # Extract MR slice
        if mr_normalized is not None:
            if slice_axis == 0:
                mr_slice = mr_normalized[slice_idx, :, :]
            elif slice_axis == 1:
                mr_slice = mr_normalized[:, slice_idx, :]
            else:
                mr_slice = mr_normalized[:, :, slice_idx]

        # Check if slices are non-empty
        ct_empty = ct_slice is not None and np.mean(ct_slice) < intensity_threshold
        mr_empty = mr_slice is not None and np.mean(mr_slice) < intensity_threshold

        if skip_empty and ct_empty and mr_empty:
            continue

        # Save slices
        slice_name = f'patient_{case_id}_slice_{slice_idx:04d}.png'

        if ct_slice is not None and not ct_empty:
            ct_path_out = os.path.join(ct_output_dir, slice_name)
            save_slice(ct_slice, ct_path_out)
            stats['ct'] += 1

        if mr_slice is not None and not mr_empty:
            mr_path_out = os.path.join(mr_output_dir, slice_name)
            save_slice(mr_slice, mr_path_out)
            stats['mr'] += 1

        if ct_slice is not None and mr_slice is not None and not ct_empty and not mr_empty:
            stats['matched'] += 1

    return stats


def find_l2r_cases(dataroot, split='Train'):
    """Find all CT/MR pairs in L2R dataset.

    Parameters:
        dataroot: L2R dataset root directory
        split: 'Train' or 'Test'

    Returns:
        list: List of (case_id, ct_path, mr_path) tuples
    """
    split_dir = os.path.join(dataroot, split)
    if not os.path.exists(split_dir):
        raise ValueError(f"Split directory not found: {split_dir}")

    cases = []
    files = os.listdir(split_dir)

    # Find all CT files
    ct_files = [f for f in files if f.endswith('_CT.nii.gz')]
    ct_ids = set()

    for ct_file in ct_files:
        # Extract case ID: imgXXXX_tcia_CT.nii.gz -> XXXX
        match = re.match(r'img(\d+)_tcia_CT\.nii\.gz', ct_file)
        if match:
            ct_ids.add(match.group(1))

    # For each CT, find corresponding MR
    for case_id in sorted(ct_ids):
        ct_path = os.path.join(split_dir, f'img{case_id}_tcia_CT.nii.gz')
        mr_path = os.path.join(split_dir, f'img{case_id}_tcia_MR.nii.gz')

        if os.path.exists(ct_path) or os.path.exists(mr_path):
            cases.append((case_id, ct_path, mr_path))

    return cases


def process_split(dataroot, output_dir, split='Train', num_workers=4, **kwargs):
    """Process a split (Train or Test) of L2R dataset.

    Parameters:
        dataroot: L2R dataset root
        output_dir: Output directory
        split: 'Train' or 'Test'
        num_workers: Number of parallel workers
        **kwargs: Additional arguments for process_case

    Returns:
        dict: Statistics
    """
    cases = find_l2r_cases(dataroot, split)
    print(f"Found {len(cases)} cases in {split}")

    if len(cases) == 0:
        return {}

    split_output_dir = os.path.join(output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)

    total_stats = {'ct': 0, 'mr': 0, 'matched': 0}

    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.starmap(
                process_case,
                [(case_id, ct_path, mr_path, split_output_dir) for case_id, ct_path, mr_path in cases]
            ),
            total=len(cases),
            desc=f"Processing {split}"
        ))

    for result in results:
        for k, v in result.items():
            total_stats[k] += v

    return total_stats


def reorganize_for_training(output_dir, split='Train'):
    """Reorganize into trainA/trainB structure.

    Parameters:
        output_dir: Directory containing split folders
        split: Which split to reorganize
    """
    split_dir = os.path.join(output_dir, split)

    # Create train directories
    train_a_dir = os.path.join(split_dir, 'trainA')
    train_b_dir = os.path.join(split_dir, 'trainB')
    os.makedirs(train_a_dir, exist_ok=True)
    os.makedirs(train_b_dir, exist_ok=True)

    ct_dir = os.path.join(split_dir, 'ct')
    mr_dir = os.path.join(split_dir, 'mr')

    if not os.path.exists(ct_dir) or not os.path.exists(mr_dir):
        print(f"Warning: ct/ or mr/ directory not found in {split_dir}")
        return

    ct_files = set(os.listdir(ct_dir))
    mr_files = set(os.listdir(mr_dir))

    # Match files (same filename in both folders)
    for ct_file in tqdm(ct_files, desc=f"Reorganizing {split}"):
        if ct_file in mr_files:
            src_ct = os.path.join(ct_dir, ct_file)
            src_mr = os.path.join(mr_dir, ct_file)
            dst_ct = os.path.join(train_a_dir, ct_file.replace('patient_', ''))
            dst_mr = os.path.join(train_b_dir, ct_file.replace('patient_', ''))

            try:
                os.symlink(src_ct, dst_ct)
                os.symlink(src_mr, dst_mr)
            except OSError:
                import shutil
                shutil.copy2(src_ct, dst_ct)
                shutil.copy2(src_mr, dst_mr)

    print(f"Reorganized {split}: {len(os.listdir(train_a_dir))} image pairs")


def main():
    parser = argparse.ArgumentParser(description='Convert L2R 3D NIfTI files to 2D slices')
    parser.add_argument('--dataroot', type=str, default='datasets/L2R',
                        help='Path to L2R dataset root directory')
    parser.add_argument('--output_dir', type=str, default='datasets/L2R_2d',
                        help='Output directory for 2D slices')
    parser.add_argument('--splits', type=str, nargs='+', default=['Train', 'Test'],
                        help='Which splits to process')
    parser.add_argument('--slice_axis', type=int, default=2,
                        choices=[0, 1, 2],
                        help='Axis to slice along (0=sagittal, 1=coronal, 2=axial)')
    parser.add_argument('--skip_empty', action='store_true', default=True,
                        help='Skip slices with low intensity')
    parser.add_argument('--intensity_threshold', type=int, default=10,
                        help='Threshold for non-empty slices (0-255)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--reorganize', action='store_true',
                        help='Reorganize into trainA/trainB structure')

    args = parser.parse_args()

    print("=" * 50)
    print("L2R 3D to 2D Conversion")
    print("=" * 50)
    print(f"Data root: {args.dataroot}")
    print(f"Output dir: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Slice axis: {args.slice_axis}")
    print("=" * 50)

    all_stats = {}

    for split in args.splits:
        stats = process_split(
            dataroot=args.dataroot,
            output_dir=args.output_dir,
            split=split,
            num_workers=args.num_workers,
            slice_axis=args.slice_axis,
            skip_empty=args.skip_empty,
            intensity_threshold=args.intensity_threshold
        )
        all_stats[split] = stats

    print("\n" + "=" * 50)
    print("Conversion Complete!")
    print("=" * 50)
    for split, stats in all_stats.items():
        print(f"\n{split}:")
        for k, v in stats.items():
            print(f"  {k}: {v} slices")

    # Reorganize if requested
    if args.reorganize:
        print("\nReorganizing into trainA/trainB structure...")
        for split in args.splits:
            reorganize_for_training(args.output_dir, split)
        print("Reorganization complete!")

    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
