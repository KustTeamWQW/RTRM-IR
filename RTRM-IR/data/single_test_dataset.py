import os.path

import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data.preprocess_config import (
    to_pseudo_rgb,
)


class SingleTestDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_I = os.path.join(opt.dataroot, opt.phase + 'A')
        self.I_paths = sorted(make_dataset(self.dir_I, opt.max_dataset_size))
        self.I_size = len(self.I_paths)
        crop_size = int(getattr(opt, 'crop_size', 300))
        self.crop_size = (crop_size, crop_size)
        self.do_crop = True

        grayscale = self.opt.input_nc == 1
        self.transform_real_I = get_transform(self.opt, params={}, grayscale=grayscale)
        self.transform_real_r = get_transform(self.opt, params={}, grayscale=grayscale)

    def _crop_pair(self, real_I_img, real_r_img):
        if not self.do_crop or not self.crop_size:
            return real_I_img, real_r_img
        crop_width, crop_height = self.crop_size
        min_width = min(real_I_img.width, real_r_img.width)
        min_height = min(real_I_img.height, real_r_img.height)
        if min_width < crop_width or min_height < crop_height:
            return real_I_img, real_r_img
        max_left = min_width - crop_width
        max_top = min_height - crop_height
        left = max_left // 2
        top = max_top // 2
        crop_box = (left, top, left + crop_width, top + crop_height)
        return real_I_img.crop(crop_box), real_r_img.crop(crop_box)

    def __getitem__(self, index):
        I_path = self.I_paths[index % self.I_size]
        I_img = Image.open(I_path).convert('L')

        real_I_img = I_img.copy()
        real_r_img = I_img.copy()
        real_I_img, real_r_img = self._crop_pair(real_I_img, real_r_img)

        return {
            'infrared': self.transform_real_I(to_pseudo_rgb(real_I_img)),
            'infrared-radiation': self.transform_real_r(to_pseudo_rgb(real_r_img)),
            'paths': I_path,
        }

    def __len__(self):
        return self.I_size
