import os.path

import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data.preprocess_config import (
    to_pseudo_rgb,
)


class SingleDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_I = os.path.join(opt.dataroot, opt.phase + 'A')
        self.save_dir = os.path.join(opt.dataroot, 'processed')
        crop_size = int(getattr(opt, 'crop_size', 300))
        self.crop_size = (crop_size, crop_size)
        self.do_crop = True

        if getattr(self.opt, 'save_processed_debug', False):
            os.makedirs(os.path.join(self.save_dir, 'infrared-radiation'), exist_ok=True)

        self.I_paths = sorted(make_dataset(self.dir_I, opt.max_dataset_size))
        self.I_size = len(self.I_paths)

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
        phase = str(getattr(self.opt, 'phase', 'train')).lower()
        if phase == 'train':
            left = 0 if max_left == 0 else np.random.randint(0, max_left + 1)
            top = 0 if max_top == 0 else np.random.randint(0, max_top + 1)
        else:
            left = max_left // 2
            top = max_top // 2
        crop_box = (left, top, left + crop_width, top + crop_height)
        return real_I_img.crop(crop_box), real_r_img.crop(crop_box)

    def __getitem__(self, index):
        I_path = self.I_paths[index % self.I_size]
        base_name = os.path.splitext(os.path.basename(I_path))[0]
        I_img = Image.open(I_path).convert('L')

        real_I_img = I_img.copy()
        real_r_img = I_img.copy()
        real_I_img, real_r_img = self._crop_pair(real_I_img, real_r_img)

        real_I_img = to_pseudo_rgb(real_I_img)
        real_r_img = to_pseudo_rgb(real_r_img)

        if getattr(self.opt, 'save_processed_debug', False):
            real_r_img.save(os.path.join(self.save_dir, 'infrared-radiation', f'{base_name}_r.png'))

        return {
            'infrared': self.transform_real_I(real_I_img),
            'infrared-radiation': self.transform_real_r(real_r_img),
            'paths': I_path,
        }

    def __len__(self):
        return self.I_size
