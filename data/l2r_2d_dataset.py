"""L2R 2D Dataset Class for loading pre-converted slice images.

This dataset class works with 2D PNG images converted from 3D NIfTI files
from the L2R (Learn to Reg) dataset.

Dataset structure:
    dataroot/
        ct/              -- CT slice images (patient_XXXX_slice_YYYY.png)
        mr/              -- MR slice images (patient_XXXX_slice_YYYY.png)
        or
        trainA/          -- modality A images (e.g., CT slices)
        trainB/          -- modality B images (e.g., MR slices)

The dataset matches CT and MR slices by patient ID and slice index.
"""
import os
import random
from data.base_dataset import BaseDataset, get_params
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


class L2R2DDataset(BaseDataset):
    """Dataset class for loading L2R 2D slice images.

    This dataset loads pre-converted 2D PNG images from the L2R dataset.
    The __getitem__ method returns a dictionary with 'A' and 'B' keys containing
    the image tensors for the two modalities.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dataroot = opt.dataroot
        self.modality_a = getattr(opt, 'modality_a', 'ct')
        self.modality_b = getattr(opt, 'modality_b', 'mr')

        # Check if data is in trainA/trainB format or modality-specific folders
        train_a_dir = os.path.join(self.dataroot, 'trainA')
        train_b_dir = os.path.join(self.dataroot, 'trainB')

        if os.path.exists(train_a_dir) and os.path.exists(train_b_dir):
            # Use trainA/trainB format
            self.dir_A = train_a_dir
            self.dir_B = train_b_dir
            self.A_paths = sorted([os.path.join(self.dir_A, f)
                                   for f in os.listdir(self.dir_A)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
            self.B_paths = sorted([os.path.join(self.dir_B, f)
                                   for f in os.listdir(self.dir_B)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            # Use modality-specific folders
            self.dir_A = os.path.join(self.dataroot, self.modality_a)
            self.dir_B = os.path.join(self.dataroot, self.modality_b)

            if not os.path.exists(self.dir_A):
                raise ValueError(f"Modality A directory not found: {self.dir_A}")
            if not os.path.exists(self.dir_B):
                raise ValueError(f"Modality B directory not found: {self.dir_B}")

            self.A_paths = sorted([os.path.join(self.dir_A, f)
                                   for f in os.listdir(self.dir_A)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
            self.B_paths = sorted([os.path.join(self.dir_B, f)
                                   for f in os.listdir(self.dir_B)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Match files by patient and slice index
        self.matched_pairs = self._match_pairs()

        # Target size must match opt.img_height x opt.img_width for STN network
        self.target_size = (opt.img_width, opt.img_height)

        print(f"L2R 2D dataset: {len(self.matched_pairs)} image pairs loaded")
        print(f"Modality A: {self.modality_a}, Modality B: {self.modality_b}")
        print(f"Target size: {self.target_size}")

    def _match_pairs(self):
        """Match image pairs based on filename patterns.

        For L2R, the format is: patient_XXXX_slice_YYYY.png

        Returns:
            list: List of (path_A, path_B) tuples
        """
        pairs = []

        # Create a mapping from B filenames to their paths
        b_map = {}
        for b_path in self.B_paths:
            b_filename = os.path.basename(b_path)
            # Extract patient and slice info
            # Format: patient_0002_slice_0050.png
            parts = b_filename.replace('patient_', '').split('_slice_')
            if len(parts) == 2:
                patient = parts[0]
                slice_id = parts[1].replace('.png', '')
                key = f"{patient}_slice_{slice_id}"
                b_map[key] = b_path

        # Find matching A images
        for a_path in self.A_paths:
            a_filename = os.path.basename(a_path)
            parts = a_filename.replace('patient_', '').split('_slice_')
            if len(parts) == 2:
                patient = parts[0]
                slice_id = parts[1].replace('.png', '')
                key = f"{patient}_slice_{slice_id}"

                if key in b_map:
                    pairs.append((a_path, b_map[key]))

        if len(pairs) == 0:
            # If no matching pattern found, just pair by index
            min_len = min(len(self.A_paths), len(self.B_paths))
            pairs = [(self.A_paths[i], self.B_paths[i]) for i in range(min_len)]

        return pairs

    def _transform_image(self, img, params):
        """Transform image: resize, crop, flip, and normalize.

        Parameters:
            img (PIL Image): Input image
            params (dict): Transformation parameters

        Returns:
            torch.Tensor: Transformed image tensor
        """
        # Resize to target size
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.BICUBIC)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0

        # Apply flip if needed
        if params.get('flip', False):
            img_array = np.fliplr(img_array).copy()

        # Convert to tensor and add channel dimension
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)

        # Duplicate channels if needed
        if self.opt.input_nc == 3:
            img_tensor = img_tensor.repeat(3, 1, 1)

        # Normalize to [-1, 1]
        img_tensor = img_tensor * 2 - 1

        return img_tensor

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It contains 'A' and 'B' keys with image tensors.
        """
        path_A, path_B = self.matched_pairs[index]

        # Load images
        img_A = Image.open(path_A).convert('L')  # Load as grayscale
        img_B = Image.open(path_B).convert('L')

        # Get transform parameters (for flip, etc.)
        transform_params = get_params(self.opt, (img_A.size[0], img_A.size[1]))

        # Apply transforms
        img_A = self._transform_image(img_A, transform_params)
        img_B = self._transform_image(img_B, transform_params)

        return {'A': img_A, 'B': img_B, 'A_paths': path_A, 'B_paths': path_B}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.matched_pairs)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(input_nc=1, output_nc=1)  # Default to 1 channel (grayscale)
        parser.add_argument('--modality_a', type=str, default='ct',
                            help='modality for domain A (ct, mr)')
        parser.add_argument('--modality_b', type=str, default='mr',
                            help='modality for domain B (ct, mr)')
        return parser
