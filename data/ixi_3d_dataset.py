"""IXI 3D Dataset for T1↔T2 registration.

Loads paired T1/T2 brain MRI from the IXI dataset.
577 patients with both T1 and T2 scans.
No segmentation labels — evaluation via MSE/NCC/SSIM.

Dataset structure:
    dataroot/
        IXI-T1/
            IXI002-Guys-0828-T1.nii.gz
            ...
        IXI-T2/
            IXI002-Guys-0828-T2.nii.gz
            ...
"""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from data.base_dataset import BaseDataset


class IXI3DDataset(BaseDataset):
    """IXI 3D T1-T2 registration dataset."""

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.dataroot = opt.dataroot
        self.target_shape = getattr(opt, 'target_shape', (128, 128, 128))
        self.augment = getattr(opt, 'augment_3d', True) and opt.isTrain

        t1_dir = os.path.join(opt.dataroot, 'IXI-T1')
        t2_dir = os.path.join(opt.dataroot, 'IXI-T2')

        # Find paired T1/T2 by patient ID
        self.patients = []
        t1_files = {f.replace('-T1.nii.gz', ''): f for f in os.listdir(t1_dir) if f.endswith('-T1.nii.gz')}
        t2_files = {f.replace('-T2.nii.gz', ''): f for f in os.listdir(t2_dir) if f.endswith('-T2.nii.gz')}

        for pid in sorted(set(t1_files.keys()) & set(t2_files.keys())):
            self.patients.append({
                'pid': pid,
                't1': os.path.join(t1_dir, t1_files[pid]),
                't2': os.path.join(t2_dir, t2_files[pid]),
            })

        if len(self.patients) == 0:
            raise ValueError(f"No paired T1/T2 found in {opt.dataroot}")

        # Split train/test
        random.seed(42)
        indices = list(range(len(self.patients)))
        random.shuffle(indices)
        n_test = max(1, len(self.patients) // 10)
        if opt.isTrain:
            self.patients = [self.patients[i] for i in indices[n_test:]]
        else:
            self.patients = [self.patients[i] for i in indices[:n_test]]

        print(f"IXI 3D dataset: {len(self.patients)} patients ({'train' if opt.isTrain else 'test'})")

    @staticmethod
    def _normalize_vol(data):
        vmin, vmax = data.min(), data.max()
        if vmax - vmin < 1e-10:
            return np.zeros_like(data)
        return 2.0 * (data - vmin) / (vmax - vmin) - 1.0

    def _resize_volume(self, vol, target_shape):
        """Resize volume to target shape using trilinear interpolation.
        vol: [C, D, H, W] tensor
        """
        if vol.shape[1:] == target_shape:
            return vol
        return F.interpolate(
            vol.unsqueeze(0), size=target_shape,
            mode='trilinear', align_corners=True
        ).squeeze(0)

    def _augment_3d(self, t1, t2):
        """Apply consistent augmentation to T1 and T2."""
        if not self.augment:
            return t1, t2

        # Random flipping
        if random.random() < 0.5:
            flip_axes = []
            for ax in [1, 2, 3]:
                if random.random() < 0.5:
                    flip_axes.append(ax)
            if flip_axes:
                t1 = torch.flip(t1, flip_axes)
                t2 = torch.flip(t2, flip_axes)

        # Intensity augmentation
        if random.random() < 0.5:
            noise_std = random.uniform(0.005, 0.02)
            t1 = t1 + torch.randn_like(t1) * noise_std
            t2 = t2 + torch.randn_like(t2) * noise_std

        if random.random() < 0.3:
            shift = random.uniform(-0.05, 0.05)
            t1 = t1 + shift
            t2 = t2 + shift

        return t1, t2

    def __getitem__(self, index):
        patient = self.patients[index]

        t1_vol = nib.load(patient['t1']).get_fdata().astype(np.float32)
        t2_vol = nib.load(patient['t2']).get_fdata().astype(np.float32)

        t1_tensor = torch.from_numpy(self._normalize_vol(t1_vol)).unsqueeze(0)
        t2_tensor = torch.from_numpy(self._normalize_vol(t2_vol)).unsqueeze(0)

        # Resize to common target shape
        t1_tensor = self._resize_volume(t1_tensor, self.target_shape)
        t2_tensor = self._resize_volume(t2_tensor, self.target_shape)

        # Augmentation
        t1_tensor, t2_tensor = self._augment_3d(t1_tensor, t2_tensor)

        return {
            'A': t1_tensor,  # T1 as moving
            'B': t2_tensor,  # T2 as fixed
            'A_paths': patient['t1'],
            'B_paths': patient['t2'],
            'patient_id': patient['pid'],
            'has_seg': False,
        }

    def __len__(self):
        return len(self.patients)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=1, output_nc=1, batch_size=1, num_threads=0,
                            serial_batches=True)
        parser.add_argument('--target_shape', type=int, nargs=3, default=[128, 128, 128],
                            help='Target volume shape D H W after resampling')
        return parser
