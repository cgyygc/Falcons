"""Aligned 2D Dataset for paired image-to-image translation (Pix2Pix).

Works with directories containing matched A/B image pairs:
    dataroot/
        trainA/  -- source images (e.g., CT)
        trainB/  -- target images (e.g., MR)
Images are matched by filename.
"""

import os
from data.base_dataset import BaseDataset, get_params
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


class Aligned2DDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dataroot = opt.dataroot
        dir_A = os.path.join(self.dataroot, 'trainA')
        dir_B = os.path.join(self.dataroot, 'trainB')

        self.A_paths = sorted([os.path.join(dir_A, f) for f in os.listdir(dir_A)
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.B_paths = sorted([os.path.join(dir_B, f) for f in os.listdir(dir_B)
                               if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Match by filename
        self.matched_pairs = []
        b_map = {os.path.basename(p): p for p in self.B_paths}
        for a_path in self.A_paths:
            a_name = os.path.basename(a_path)
            if a_name in b_map:
                self.matched_pairs.append((a_path, b_map[a_name]))

        self.target_size = (opt.img_width, opt.img_height)
        print(f"Aligned 2D dataset: {len(self.matched_pairs)} paired images loaded")

    def __getitem__(self, index):
        path_A, path_B = self.matched_pairs[index]
        img_A = Image.open(path_A).convert('L')
        img_B = Image.open(path_B).convert('L')

        transform_params = get_params(self.opt, (img_A.size[0], img_A.size[1]))

        # Same transform for both (aligned)
        def transform(img):
            if img.size != self.target_size:
                img = img.resize(self.target_size, Image.BICUBIC)
            arr = np.array(img).astype(np.float32) / 255.0
            if transform_params.get('flip', False):
                arr = np.fliplr(arr).copy()
            tensor = torch.from_numpy(arr).unsqueeze(0)
            if self.opt.input_nc == 3:
                tensor = tensor.repeat(3, 1, 1)
            return tensor * 2 - 1

        return {'A': transform(img_A), 'B': transform(img_B),
                'A_paths': path_A, 'B_paths': path_B}

    def __len__(self):
        return len(self.matched_pairs)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=1, output_nc=1)
        return parser
