"""L2R 3D Dataset Class for loading NIfTI volumes directly.

Supports loading from both Train/ and Test/ directories.
Unsupervised registration doesn't need segmentation for training,
so Test patients (no segmentation) are included in training.

Dataset structure:
    dataroot/
        Train/
            img0002_tcia_CT.nii.gz
            img0002_tcia_MR.nii.gz
            seg0002_tcia_CT.nii.gz
            seg0002_tcia_MR.nii.gz
            ...
        Test/
            img0001_tcia_CT.nii.gz
            img0001_tcia_MR.nii.gz
            ...
"""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate, affine_transform
from data.base_dataset import BaseDataset


class L2R3DDataset(BaseDataset):
    """Dataset class for loading L2R 3D NIfTI volumes."""

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.crop_size = getattr(opt, 'crop_3d_size', 0)
        self.load_seg = getattr(opt, 'load_seg', True)
        self.augment = getattr(opt, 'augment_3d', True) and opt.isTrain

        # Load patients from both Train and Test
        self.patients = []
        train_dir = os.path.join(opt.dataroot, 'Train')
        test_dir = os.path.join(opt.dataroot, 'Test')

        if os.path.exists(train_dir):
            self.patients.extend(self._find_patients(train_dir, has_seg=True))
        if os.path.exists(test_dir):
            self.patients.extend(self._find_patients(test_dir, has_seg=False))

        if len(self.patients) == 0:
            raise ValueError(f"No L2R patient data found in {opt.dataroot}")

        random.shuffle(self.patients)
        print(f"L2R 3D dataset: {len(self.patients)} patients loaded "
              f"({sum(1 for p in self.patients if p['has_seg'])} with segmentation)")

    def _find_patients(self, data_dir, has_seg=True):
        """Find all patients with paired CT and MR volumes."""
        patients = []
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('_tcia_CT.nii.gz') and f.startswith('img'):
                pid = f.replace('img', '').replace('_tcia_CT.nii.gz', '')
                ct_path = os.path.join(data_dir, f)
                mr_path = os.path.join(data_dir, f.replace('_CT.nii.gz', '_MR.nii.gz'))
                if not os.path.exists(mr_path):
                    continue
                entry = {'pid': pid, 'ct': ct_path, 'mr': mr_path, 'has_seg': has_seg}
                if self.load_seg and has_seg:
                    ct_seg = os.path.join(data_dir, f.replace('img', 'seg'))
                    mr_seg = os.path.join(data_dir, f.replace('img', 'seg').replace('_CT.nii.gz', '_MR.nii.gz'))
                    if os.path.exists(ct_seg) and os.path.exists(mr_seg):
                        entry['ct_seg'] = ct_seg
                        entry['mr_seg'] = mr_seg
                    else:
                        entry['has_seg'] = False
                patients.append(entry)
        return patients

    @staticmethod
    def _normalize_vol(data):
        """Normalize volume to [-1, 1]."""
        vmin, vmax = data.min(), data.max()
        if vmax - vmin < 1e-10:
            return np.zeros_like(data)
        return 2.0 * (data - vmin) / (vmax - vmin) - 1.0

    def _augment_3d(self, ct, mr, ct_seg=None, mr_seg=None):
        """Apply consistent 3D augmentation to CT, MR, and segmentation."""
        if not self.augment:
            return ct, mr, ct_seg, mr_seg

        # Random flipping along all 3 axes
        if random.random() < 0.5:
            flip_axes = []
            for ax in [1, 2, 3]:
                if random.random() < 0.5:
                    flip_axes.append(ax)
            if flip_axes:
                ct = torch.flip(ct, flip_axes)
                mr = torch.flip(mr, flip_axes)
                if ct_seg is not None:
                    ct_seg = torch.flip(ct_seg, flip_axes)
                if mr_seg is not None:
                    mr_seg = torch.flip(mr_seg, flip_axes)

        # Random rotation ±15° around each axis
        for axis in range(3):
            if random.random() < 0.3:
                angle = random.uniform(-15, 15)
                ct = self._rotate_volume(ct, angle, axis)
                mr = self._rotate_volume(mr, angle, axis)
                if ct_seg is not None:
                    ct_seg = self._rotate_volume(ct_seg, angle, axis, order=0)
                if mr_seg is not None:
                    mr_seg = self._rotate_volume(mr_seg, angle, axis, order=0)

        # Random scaling 0.9× to 1.1×
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            ct = self._scale_volume(ct, scale)
            mr = self._scale_volume(mr, scale)
            if ct_seg is not None:
                ct_seg = self._scale_volume(ct_seg, scale, order=0)
            if mr_seg is not None:
                mr_seg = self._scale_volume(mr_seg, scale, order=0)

        # Intensity augmentation
        if random.random() < 0.5:
            # Gaussian noise
            noise_std = random.uniform(0.005, 0.03)
            ct = ct + torch.randn_like(ct) * noise_std
            mr = mr + torch.randn_like(mr) * noise_std

        if random.random() < 0.3:
            # Brightness shift
            shift = random.uniform(-0.05, 0.05)
            ct = ct + shift
            mr = mr + shift

        if random.random() < 0.3:
            # Contrast adjustment
            gamma = random.uniform(0.9, 1.1)
            ct = torch.sign(ct) * torch.abs(ct) ** gamma
            mr = torch.sign(mr) * torch.abs(mr) ** gamma

        return ct, mr, ct_seg, mr_seg

    def _rotate_volume(self, vol, angle, axis, order=1):
        """Rotate 3D volume around given axis. vol: [C, D, H, W]"""
        arr = vol.numpy()
        C = arr.shape[0]
        results = []
        for c in range(C):
            rotated = rotate(arr[c], angle, axes=self._rotation_axes(axis),
                             reshape=False, order=order, mode='nearest')
            results.append(rotated)
        return torch.from_numpy(np.stack(results)).to(vol.dtype)

    def _rotation_axes(self, axis):
        """Get rotation axes for scipy rotate."""
        # axis 0=D, 1=H, 2=W → rotate in the plane perpendicular to axis
        if axis == 0:  # rotate in H-W plane
            return (1, 2)
        elif axis == 1:  # rotate in D-W plane
            return (0, 2)
        else:  # rotate in D-H plane
            return (0, 1)

    def _scale_volume(self, vol, scale, order=1):
        """Scale 3D volume uniformly. vol: [C, D, H, W]"""
        arr = vol.numpy()
        C, D, H, W = arr.shape
        results = []
        for c in range(C):
            # Use scipy zoom
            from scipy.ndimage import zoom as scipy_zoom
            scaled = scipy_zoom(arr[c], scale, order=order, mode='nearest')
            # Crop or pad back to original size
            scaled = self._crop_or_pad(scaled, D, H, W)
            results.append(scaled)
        return torch.from_numpy(np.stack(results)).to(vol.dtype)

    def _crop_or_pad(self, arr, target_d, target_h, target_w):
        """Crop or pad 3D array to target size."""
        d, h, w = arr.shape
        result = np.zeros((target_d, target_h, target_w), dtype=arr.dtype)

        # Compute slicing
        sd = max(0, (d - target_d) // 2)
        sh = max(0, (h - target_h) // 2)
        sw = max(0, (w - target_w) // 2)
        td = max(0, (target_d - d) // 2)
        th = max(0, (target_h - h) // 2)
        tw = max(0, (target_w - w) // 2)

        copy_d = min(d - sd, target_d - td)
        copy_h = min(h - sh, target_h - th)
        copy_w = min(w - sw, target_w - tw)

        result[td:td+copy_d, th:th+copy_h, tw:tw+copy_w] = \
            arr[sd:sd+copy_d, sh:sh+copy_h, sw:sw+copy_w]
        return result

    def __getitem__(self, index):
        patient = self.patients[index]

        import nibabel as nib
        ct_data = nib.load(patient['ct']).get_fdata().astype(np.float32)
        mr_data = nib.load(patient['mr']).get_fdata().astype(np.float32)

        ct_vol = self._normalize_vol(ct_data)
        mr_vol = self._normalize_vol(mr_data)

        ct_tensor = torch.from_numpy(ct_vol).unsqueeze(0)
        mr_tensor = torch.from_numpy(mr_vol).unsqueeze(0)

        # Optional random crop
        if self.crop_size > 0:
            ct_tensor, mr_tensor = self._random_crop(ct_tensor, mr_tensor)

        # Load segmentation if available
        ct_seg = mr_seg = None
        if 'ct_seg' in patient and 'mr_seg' in patient:
            ct_seg = nib.load(patient['ct_seg']).get_fdata().astype(np.int32)
            mr_seg = nib.load(patient['mr_seg']).get_fdata().astype(np.int32)
            ct_seg = torch.from_numpy(ct_seg).unsqueeze(0).long()
            mr_seg = torch.from_numpy(mr_seg).unsqueeze(0).long()

        # Apply augmentation
        ct_tensor, mr_tensor, ct_seg, mr_seg = self._augment_3d(
            ct_tensor, mr_tensor, ct_seg, mr_seg
        )

        result = {
            'A': ct_tensor,
            'B': mr_tensor,
            'A_paths': patient['ct'],
            'B_paths': patient['mr'],
            'patient_id': patient['pid'],
            'has_seg': patient['has_seg'],
        }

        if ct_seg is not None:
            result['A_seg'] = ct_seg
            result['B_seg'] = mr_seg

        return result

    def _random_crop(self, ct, mr):
        """Random crop volumes to crop_size^3."""
        s = self.crop_size
        _, d, h, w = ct.shape
        if d <= s or h <= s or w <= s:
            pad_d = max(0, s - d)
            pad_h = max(0, s - h)
            pad_w = max(0, s - w)
            ct = F.pad(ct, (0, pad_w, 0, pad_h, 0, pad_d))
            mr = F.pad(mr, (0, pad_w, 0, pad_h, 0, pad_d))
            _, d, h, w = ct.shape

        dd = random.randint(0, d - s)
        hh = random.randint(0, h - s)
        ww = random.randint(0, w - s)
        ct = ct[:, dd:dd+s, hh:hh+s, ww:ww+s]
        mr = mr[:, dd:dd+s, hh:hh+s, ww:ww+s]
        return ct, mr

    def __len__(self):
        return len(self.patients)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=1, output_nc=1, batch_size=1, num_threads=0,
                            serial_batches=True)
        parser.add_argument('--crop_3d_size', type=int, default=0,
                            help='Random crop size for 3D training (0=use full volume)')
        parser.add_argument('--load_seg', action='store_true', default=True,
                            help='Load segmentation masks')
        parser.add_argument('--augment_3d', action='store_true', default=True,
                            help='Enable aggressive 3D augmentation')
        parser.add_argument('--no_augment_3d', action='store_true',
                            help='Disable 3D augmentation')
        return parser
