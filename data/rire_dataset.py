"""This module implements the RIRE dataset class for loading CT and MR images.

The RIRE dataset contains multi-modal medical images (CT and MR) for registration tasks.
The dataset structure should be:
    dataroot/
        patient_001/
            ct.nii.gz
            mr_t1.nii.gz
            mr_t2.nii.gz
        patient_002/
            ...
"""
import os
import numpy as np
import torch
import torch.utils.data as data
from data.base_dataset import BaseDataset, get_transform
import nibabel as nib


class RIREDataset(BaseDataset):
    """A template dataset class for the RIRE dataset.

    This dataset class loads CT and MR image pairs from the RIRE dataset.
    The __getitem__ method returns a dictionary with 'A' and 'B' keys containing
    the image tensors for the two modalities.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # Get the list of patient directories
        self.dataroot = opt.dataroot
        self.patient_dirs = self._get_patient_dirs()

        # Modality selection: 'ct', 'mr_t1', or 'mr_t2'
        # Default: A=CT, B=MR_T1
        self.modality_a = getattr(opt, 'modality_a', 'ct')
        self.modality_b = getattr(opt, 'modality_b', 'mr_t1')

        # Load size for resizing images
        self.load_size = (opt.img_height, opt.img_width)

        print(f"RIRE dataset: {len(self.patient_dirs)} patients loaded")
        print(f"Modality A: {self.modality_a}, Modality B: {self.modality_b}")

    def _get_patient_dirs(self):
        """Return a list of patient directory paths."""
        patient_dirs = []
        if not os.path.exists(self.dataroot):
            raise ValueError(f"Data root directory does not exist: {self.dataroot}")

        for item in sorted(os.listdir(self.dataroot)):
            item_path = os.path.join(self.dataroot, item)
            if os.path.isdir(item_path):
                # Check if it contains the required files
                ct_path = os.path.join(item_path, 'ct.nii.gz')
                mr_t1_path = os.path.join(item_path, 'mr_t1.nii.gz')
                if os.path.exists(ct_path) and os.path.exists(mr_t1_path):
                    patient_dirs.append(item_path)

        if len(patient_dirs) == 0:
            raise ValueError(f"No valid patient directories found in {self.dataroot}")

        return patient_dirs

    def _load_nifti_image(self, path):
        """Load a NIfTI image file and return as numpy array.

        Parameters:
            path (str) -- path to the .nii.gz file

        Returns:
            numpy array of the image data
        """
        nii_img = nib.load(path)
        data = nii_img.get_fdata()

        # Normalize to [0, 1] range
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)

        return data

    def _process_slice(self, slice_data):
        """Process a 2D slice for model input.

        Parameters:
            slice_data (numpy array) -- 2D slice data

        Returns:
            processed tensor with shape (C, H, W)
        """
        # Convert to torch tensor and add channel dimension
        tensor = torch.from_numpy(slice_data).float()

        # Add channel dimension for grayscale: (H, W) -> (1, H, W)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        # Resize to target size if needed
        if tensor.shape[1:] != self.load_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=self.load_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Duplicate channel to match RGB format if needed (3 channels)
        if tensor.shape[0] == 1 and self.opt.input_nc == 3:
            tensor = tensor.repeat(3, 1, 1)

        # Normalize to [-1, 1] range
        tensor = tensor * 2 - 1

        return tensor

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It contains 'A' and 'B' keys with image tensors.
        """
        patient_dir = self.patient_dirs[index]

        # Load modality A image
        path_a = os.path.join(patient_dir, f'{self.modality_a}.nii.gz')
        img_a = self._load_nifti_image(path_a)

        # Load modality B image
        path_b = os.path.join(patient_dir, f'{self.modality_b}.nii.gz')
        img_b = self._load_nifti_image(path_b)

        # Select middle slice (assuming 3D volume)
        # You can modify this to return random slices or all slices
        slice_idx_a = img_a.shape[2] // 2
        slice_idx_b = img_b.shape[2] // 2

        slice_a = img_a[:, :, slice_idx_a]
        slice_b = img_b[:, :, slice_idx_b]

        # Process slices to tensors
        tensor_a = self._process_slice(slice_a)
        tensor_b = self._process_slice(slice_b)

        return {'A': tensor_a, 'B': tensor_b, 'A_paths': path_a, 'B_paths': path_b}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.patient_dirs)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(input_nc=3, output_nc=3)  # Default to 3 channels (RGB)
        parser.add_argument('--modality_a', type=str, default='ct',
                            help='modality for domain A (ct, mr_t1, mr_t2)')
        parser.add_argument('--modality_b', type=str, default='mr_t1',
                            help='modality for domain B (ct, mr_t1, mr_t2)')
        return parser
