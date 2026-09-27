import os.path
import random
import hashlib

import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data.preprocess_config import (
    PREPROCESS_CACHE_VERSION,
    build_radiation_component,
    build_visible_texture,
    to_pseudo_rgb,
)


class pairedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_I = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_J = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_GT = os.path.join(opt.dataroot, opt.phase + 'GT')
        self.save_dir = os.path.join(opt.dataroot, 'processed')
        self.cache_enabled = bool(int(getattr(opt, 'cache_preprocess', 1)))
        self.cache_dir = os.path.join(opt.dataroot, getattr(opt, 'preprocess_cache_dir', 'preprocess_cache'), opt.phase)
        crop_size = int(getattr(opt, 'crop_size', 300))
        self.crop_size = (crop_size, crop_size)
        self.do_crop = True

        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

        if getattr(self.opt, 'save_processed_debug', False):
            os.makedirs(os.path.join(self.save_dir, 'infrared-radiation'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'gt-radiation'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'visible'), exist_ok=True)

        infrared_paths = sorted(make_dataset(self.dir_I))
        raw_J_paths = sorted(make_dataset(self.dir_J))
        raw_GT_paths = sorted(make_dataset(self.dir_GT))
        self.I_paths, self.J_paths, self.GT_paths = self._align_paths_by_stem(infrared_paths, raw_J_paths, raw_GT_paths)
        self._apply_dataset_limit(getattr(opt, 'max_dataset_size', float('inf')))
        self.I_size = len(self.I_paths)
        self.J_size = len(self.J_paths)
        self.GT_size = len(self.GT_paths)

        grayscale = self.opt.input_nc == 1
        self.transform_real_I = get_transform(self.opt, params={}, grayscale=grayscale)
        self.transform_real_r = get_transform(self.opt, params={}, grayscale=grayscale)
        self.transform_real_r_GT = get_transform(self.opt, params={}, grayscale=grayscale)
        self.transform_J = get_transform(self.opt, params={}, grayscale=grayscale)
        self.transform_GT = get_transform(self.opt, params={}, grayscale=grayscale)

    @staticmethod
    def _align_paths_by_stem(I_paths, J_paths, GT_paths):
        I_map = {os.path.splitext(os.path.basename(path))[0]: path for path in I_paths}
        J_map = {os.path.splitext(os.path.basename(path))[0]: path for path in J_paths}
        GT_map = {os.path.splitext(os.path.basename(path))[0]: path for path in GT_paths}
        common = sorted(set(I_map) & set(J_map) & set(GT_map))
        if not common:
            raise RuntimeError('No paired samples with matching filename stems were found.')
        return [I_map[key] for key in common], [J_map[key] for key in common], [GT_map[key] for key in common]

    def _apply_dataset_limit(self, max_dataset_size):
        try:
            max_size = int(max_dataset_size)
        except (OverflowError, TypeError, ValueError):
            return
        if max_size < len(self.I_paths):
            self.I_paths = self.I_paths[:max_size]
            self.J_paths = self.J_paths[:max_size]
            self.GT_paths = self.GT_paths[:max_size]

    def _cache_path(self, image_path, tag):
        stat = os.stat(image_path)
        key = f'{PREPROCESS_CACHE_VERSION}|{os.path.abspath(image_path)}|{tag}|{stat.st_mtime_ns}|{stat.st_size}'
        name = hashlib.md5(key.encode('utf-8')).hexdigest() + '.png'
        return os.path.join(self.cache_dir, tag, name)

    def _load_or_build_cached(self, image_path, tag, build_fn):
        if not self.cache_enabled:
            return build_fn()
        cache_path = self._cache_path(image_path, tag)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.exists(cache_path):
            return Image.open(cache_path).convert('L')
        image = build_fn()
        image.save(cache_path)
        return image
    def _choose_paths(self, index):
        I_path = self.I_paths[index % self.I_size]
        GT_path = self.GT_paths[index % self.GT_size]
        if self.opt.serial_batches:
            J_path = self.J_paths[index % self.J_size]
        else:
            J_path = self.J_paths[index % self.J_size] if index < self.J_size else self.J_paths[random.randint(0, self.J_size - 1)]
        return I_path, J_path, GT_path

    def _crop_aligned(self, images):
        if not self.do_crop or not self.crop_size:
            return images
        crop_width, crop_height = self.crop_size
        min_width = min(img.width for img in images)
        min_height = min(img.height for img in images)
        if min_width < crop_width or min_height < crop_height:
            return images
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
        return [img.crop(crop_box) for img in images]

    def __getitem__(self, index):
        I_path, J_path, GT_path = self._choose_paths(index)
        base_name = os.path.splitext(os.path.basename(J_path))[0]

        I_img = Image.open(I_path).convert('L')
        J_img = Image.open(J_path).convert('L')
        GT_img = Image.open(GT_path).convert('L')
        real_I_img = I_img.copy()
        real_r_img = I_img.copy()

        real_r_GT_img = self._load_or_build_cached(
            GT_path,
            'gt_radiation',
            lambda: build_radiation_component(GT_img)
        )
        J_img = self._load_or_build_cached(
            J_path,
            'visible_texture',
            lambda: build_visible_texture(J_img)
        )

        real_I_img, real_r_img, real_r_GT_img, J_img, GT_img = self._crop_aligned([
            real_I_img, real_r_img, real_r_GT_img, J_img, GT_img
        ])

        real_I_img = to_pseudo_rgb(real_I_img)
        real_r_img = to_pseudo_rgb(real_r_img)
        real_r_GT_img = to_pseudo_rgb(real_r_GT_img)
        J_img = to_pseudo_rgb(J_img)
        GT_img = to_pseudo_rgb(GT_img)

        if getattr(self.opt, 'save_processed_debug', False):
            real_r_img.save(os.path.join(self.save_dir, 'infrared-radiation', f'{base_name}_r.png'))
            real_r_GT_img.save(os.path.join(self.save_dir, 'gt-radiation', f'{base_name}_r_GT.png'))
            J_img.save(os.path.join(self.save_dir, 'visible', f'{base_name}_J.png'))

        return {
            'infrared': self.transform_real_I(real_I_img),
            'infrared-radiation': self.transform_real_r(real_r_img),
            'gt-radiation': self.transform_real_r_GT(real_r_GT_img),
            'visible': self.transform_J(J_img),
            'gt': self.transform_GT(GT_img),
            'paths': I_path,
            'J_paths': J_path,
        }

    def __len__(self):
        return max(self.I_size, self.J_size, self.GT_size)



