"""Script to convert RIRE 3D NIfTI files to 2D slice images.

This script processes the RIRE dataset and extracts 2D slices from 3D volumes,
saving them as PNG images for faster data loading during training.
"""
import os
import argparse
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path


def load_nifti_volume(nii_path):
    """Load a NIfTI file and return the data array.

    Parameters:
        nii_path (str): Path to the .nii.gz file

    Returns:
        numpy array: The image data
    """
    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata()
    return data


def normalize_volume(data):
    """Normalize volume data to [0, 255] range for saving as image.

    Parameters:
        data (numpy array): Input volume data

    Returns:
        numpy array: Normalized data in [0, 255] range
    """
    data_min = np.min(data)
    data_max = np.max(data)

    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)

    return (data * 255).astype(np.uint8)


def save_slice(slice_data, output_path):
    """Save a 2D slice as a PNG image.

    Parameters:
        slice_data (numpy array): 2D slice data
        output_path (str): Path to save the image
    """
    img = Image.fromarray(slice_data, mode='L')
    img.save(output_path)


def process_patient(patient_dir, output_dir, modalities=['ct', 'mr_t1', 'mr_t2'],
                    slice_axis=2, skip_empty=True, intensity_threshold=10):
    """Process a single patient directory and extract 2D slices.

    Parameters:
        patient_dir (str): Path to patient directory
        output_dir (str): Path to output directory
        modalities (list): List of modality names to process
        slice_axis (int): Axis to slice along (0=sagittal, 1=coronal, 2=axial)
        skip_empty (bool): Skip slices with low intensity
        intensity_threshold (int): Threshold for considering a slice as non-empty

    Returns:
        dict: Statistics about processed slices
    """
    patient_name = os.path.basename(patient_dir)
    stats = {mod: 0 for mod in modalities}

    for modality in modalities:
        nii_path = os.path.join(patient_dir, f'{modality}.nii.gz')

        if not os.path.exists(nii_path):
            print(f"Warning: {nii_path} not found, skipping...")
            continue

        # Create output directory for this modality
        mod_output_dir = os.path.join(output_dir, modality)
        os.makedirs(mod_output_dir, exist_ok=True)

        # Load and normalize volume
        volume = load_nifti_volume(nii_path)
        volume_normalized = normalize_volume(volume)

        # Extract slices along the specified axis
        num_slices = volume.shape[slice_axis]

        for slice_idx in range(num_slices):
            # Extract slice based on axis
            if slice_axis == 0:
                slice_data = volume_normalized[slice_idx, :, :]
            elif slice_axis == 1:
                slice_data = volume_normalized[:, slice_idx, :]
            else:
                slice_data = volume_normalized[:, :, slice_idx]

            # Skip empty slices
            if skip_empty and np.mean(slice_data) < intensity_threshold:
                continue

            # Save slice
            output_filename = f'{patient_name}_{modality}_slice_{slice_idx:04d}.png'
            output_path = os.path.join(mod_output_dir, output_filename)
            save_slice(slice_data, output_path)
            stats[modality] += 1

    return stats


def process_all_patients(dataroot, output_dir, num_workers=4, **kwargs):
    """Process all patient directories in parallel.

    Parameters:
        dataroot (str): Root directory containing patient folders
        output_dir (str): Output directory for 2D slices
        num_workers (int): Number of parallel workers
        **kwargs: Additional arguments for process_patient

    Returns:
        dict: Total statistics
    """
    patient_dirs = []
    for item in sorted(os.listdir(dataroot)):
        item_path = os.path.join(dataroot, item)
        if os.path.isdir(item_path):
            patient_dirs.append(item_path)

    if len(patient_dirs) == 0:
        raise ValueError(f"No patient directories found in {dataroot}")

    print(f"Found {len(patient_dirs)} patient directories")
    print(f"Using {num_workers} workers for parallel processing")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process patients in parallel
    total_stats = {mod: 0 for mod in kwargs.get('modalities', ['ct', 'mr_t1', 'mr_t2'])}

    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.starmap(
                process_patient,
                [(pd, output_dir) for pd in patient_dirs]
            ),
            total=len(patient_dirs),
            desc="Processing patients"
        ))

    # Aggregate statistics
    for result in results:
        for mod, count in result.items():
            total_stats[mod] += count

    return total_stats


def reorganize_for_training(output_dir, modality_a='ct', modality_b='mr_t1'):
    """Reorganize slices into trainA/trainB structure for pix2pix/pix2pixHD style training.

    Parameters:
        output_dir (str): Directory containing modality folders
        modality_a (str): First modality name (domain A)
        modality_b (str): Second modality name (domain B)
    """
    # Create train directories
    train_a_dir = os.path.join(output_dir, 'trainA')
    train_b_dir = os.path.join(output_dir, 'trainB')
    os.makedirs(train_a_dir, exist_ok=True)
    os.makedirs(train_b_dir, exist_ok=True)

    # Get list of files
    mod_a_dir = os.path.join(output_dir, modality_a)
    mod_b_dir = os.path.join(output_dir, modality_b)

    if not os.path.exists(mod_a_dir) or not os.path.exists(mod_b_dir):
        print(f"Warning: Modality directories not found")
        return

    # Get common slice indices (patients with both modalities)
    mod_a_files = set(os.listdir(mod_a_dir))
    mod_b_files = set(os.listdir(mod_b_dir))

    # Match files by patient and slice index
    for file_a in mod_a_files:
        # Extract patient and slice info
        parts = file_a.replace(f'{modality_a}_', '').split('_slice_')
        if len(parts) != 2:
            continue

        patient_name = parts[0]
        slice_idx = parts[1].replace('.png', '')

        # Find corresponding B file
        file_b = f'{patient_name}_{modality_b}_slice_{slice_idx}.png'

        if file_b in mod_b_files:
            # Copy/link files to train directories
            src_a = os.path.join(mod_a_dir, file_a)
            src_b = os.path.join(mod_b_dir, file_b)
            dst_a = os.path.join(train_a_dir, file_a)
            dst_b = os.path.join(train_b_dir, file_b)

            # Create symlinks or copy
            try:
                os.symlink(src_a, dst_a)
                os.symlink(src_b, dst_b)
            except OSError:
                # Symlink not supported, copy instead
                import shutil
                shutil.copy2(src_a, dst_a)
                shutil.copy2(src_b, dst_b)


def main():
    parser = argparse.ArgumentParser(description='Convert RIRE 3D NIfTI files to 2D slices')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to RIRE dataset root directory')
    parser.add_argument('--output_dir', type=str, default='datasets/RIRE_2d',
                        help='Output directory for 2D slices')
    parser.add_argument('--modalities', type=str, nargs='+', default=['ct', 'mr_t1', 'mr_t2'],
                        help='Modalities to process')
    parser.add_argument('--slice_axis', type=int, default=2,
                        choices=[0, 1, 2],
                        help='Axis to slice along (0=sagittal, 1=coronal, 2=axial)')
    parser.add_argument('--skip_empty', action='store_true', default=True,
                        help='Skip slices with low intensity')
    parser.add_argument('--intensity_threshold', type=int, default=10,
                        help='Threshold for considering a slice as non-empty (0-255)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--reorganize', action='store_true',
                        help='Reorganize into trainA/trainB structure')
    parser.add_argument('--modality_a', type=str, default='ct',
                        help='First modality for training (domain A)')
    parser.add_argument('--modality_b', type=str, default='mr_t1',
                        help='Second modality for training (domain B)')

    args = parser.parse_args()

    print("=" * 50)
    print("RIRE 3D to 2D Conversion")
    print("=" * 50)
    print(f"Data root: {args.dataroot}")
    print(f"Output dir: {args.output_dir}")
    print(f"Modalities: {args.modalities}")
    print(f"Slice axis: {args.slice_axis}")
    print(f"Skip empty: {args.skip_empty}")
    print(f"Intensity threshold: {args.intensity_threshold}")
    print("=" * 50)

    # Process all patients
    stats = process_all_patients(
        dataroot=args.dataroot,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        modalities=args.modalities,
        slice_axis=args.slice_axis,
        skip_empty=args.skip_empty,
        intensity_threshold=args.intensity_threshold
    )

    print("\n" + "=" * 50)
    print("Conversion Complete!")
    print("=" * 50)
    for mod, count in stats.items():
        print(f"{mod}: {count} slices")

    # Reorganize if requested
    if args.reorganize:
        print("\nReorganizing into trainA/trainB structure...")
        reorganize_for_training(
            output_dir=args.output_dir,
            modality_a=args.modality_a,
            modality_b=args.modality_b
        )
        print("Reorganization complete!")

    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
