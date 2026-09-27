import argparse
import json
import os
import random
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.base_dataset import get_transform
from data.image_folder import make_dataset
from data.preprocess_config import to_pseudo_rgb
from models import create_model
from util import util


class TestAOnlyTrainLikeDataset(Dataset):
    """A-side inference dataset matching paired_dataset preprocessing without B/GT."""

    def __init__(self, opt):
        self.opt = opt
        self.dir_i = os.path.join(opt.dataroot, opt.phase + 'A')
        self.i_paths = sorted(make_dataset(self.dir_i, opt.max_dataset_size))
        if not self.i_paths:
            raise RuntimeError(f'No images found in {self.dir_i}')
        crop_size = int(getattr(opt, 'crop_size', 300))
        self.crop_size = (crop_size, crop_size)
        self.preserve_original_size = bool(int(getattr(opt, 'preserve_original_size', 1)))
        self.do_crop = not self.preserve_original_size
        grayscale = self.opt.input_nc == 1
        self.transform_i = get_transform(self.opt, params={}, grayscale=grayscale)
        self.transform_r = get_transform(self.opt, params={}, grayscale=grayscale)

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
        phase = str(getattr(self.opt, 'phase', 'test')).lower()
        if phase == 'train':
            left = 0 if max_left == 0 else np.random.randint(0, max_left + 1)
            top = 0 if max_top == 0 else np.random.randint(0, max_top + 1)
        else:
            left = max_left // 2
            top = max_top // 2
        crop_box = (left, top, left + crop_width, top + crop_height)
        return [img.crop(crop_box) for img in images]

    def __getitem__(self, index):
        i_path = self.i_paths[index]
        i_img = Image.open(i_path).convert('L')
        original_width, original_height = i_img.size
        real_i_img = i_img.copy()
        real_r_img = i_img.copy()
        real_i_img, real_r_img = self._crop_aligned([real_i_img, real_r_img])
        return {
            'infrared': self.transform_i(to_pseudo_rgb(real_i_img)),
            'infrared-radiation': self.transform_r(to_pseudo_rgb(real_r_img)),
            'paths': i_path,
            'original_size': torch.tensor([original_width, original_height], dtype=torch.int64),
        }

    def __len__(self):
        return len(self.i_paths)


def _set_reproducible(seed, deterministic):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def _tensor_to_image(tensor):
    return util.tensor2im(torch.clamp(tensor, 0.0, 1.0))


def _crop_visual_to_size(tensor, width, height):
    if not isinstance(tensor, torch.Tensor) or tensor.dim() < 4:
        return tensor
    return tensor[..., :height, :width]


def _save_visuals(visuals, out_dir, base_name, original_size, save_all_visuals=False):
    width, height = int(original_size[0]), int(original_size[1])
    saved = []
    if 'fused_I' in visuals:
        fused = _crop_visual_to_size(visuals['fused_I'], width, height)
        util.save_image(_tensor_to_image(fused), os.path.join(out_dir, base_name + '.png'))
        saved.append('fused_I')
    if save_all_visuals:
        for name, tensor in visuals.items():
            if name == 'fused_I':
                continue
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            tensor = _crop_visual_to_size(tensor, width, height)
            subdir = os.path.join(out_dir, name)
            out_path = os.path.join(subdir, base_name + '.png')
            util.save_image(_tensor_to_image(tensor), out_path)
            saved.append(name)
    return saved


def _pad_data_to_multiple(data, multiple):
    tensors = ['infrared', 'infrared-radiation']
    sample = data['infrared']
    height, width = sample.shape[-2], sample.shape[-1]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return data, (0, 0)
    for key in tensors:
        data[key] = F.pad(data[key], (0, pad_w, 0, pad_h), mode='reflect')
    return data, (pad_w, pad_h)


def main():
    parser = argparse.ArgumentParser(description='Train-like testA inference without visible/GT references.')
    parser.add_argument('--checkpoints_dir', default='/home/jpc/jpc/VIFPM/checkpoints/checkpoint-datasets2')
    parser.add_argument('--name', default='datasets2_psnr30_ssim085_V3_continue051_200_joint_val_20260731_021200')
    parser.add_argument('--dataroot', default='/home/jpc/jpc/VIFPM/datasets2')
    parser.add_argument('--phase', default='test')
    parser.add_argument('--epoch', default='best_val')
    parser.add_argument('--results_dir', default='/home/jpc/jpc/VIFPM/results/data2-test')
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--preserve_original_size', type=int, default=1)
    parser.add_argument('--pad_multiple', type=int, default=16)
    parser.add_argument('--save_all_visuals', type=int, default=0, help='Save all intermediate visuals when set to 1')
    args = parser.parse_args()

    opt_path = os.path.join(args.checkpoints_dir, args.name, 'train_opt.json')
    with open(opt_path, 'r', encoding='utf-8') as handle:
        options = json.load(handle)

    gpu_ids = [int(item) for item in str(args.gpu_ids).split(',') if item.strip() and int(item) >= 0]
    options.update({
        'checkpoints_dir': args.checkpoints_dir,
        'name': args.name,
        'dataroot': args.dataroot,
        'phase': args.phase,
        'dataset_mode': 'testA_only_trainlike',
        'isTrain': False,
        'continue_train': False,
        'epoch': args.epoch,
        'load_iter': 0,
        'gpu_ids': gpu_ids,
        'serial_batches': True,
        'no_flip': True,
        'batch_size': 1,
        'num_threads': args.num_threads,
        'display_id': -1,
        'max_dataset_size': float('inf'),
        'preserve_original_size': int(args.preserve_original_size),
    })
    opt = SimpleNamespace(**options)
    _set_reproducible(getattr(opt, 'seed', 20260730), bool(int(getattr(opt, 'deterministic', 0))))
    if torch.cuda.is_available() and opt.gpu_ids:
        torch.cuda.set_device(opt.gpu_ids[0])

    os.makedirs(args.results_dir, exist_ok=True)
    dataset = TestAOnlyTrainLikeDataset(opt)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=max(0, args.num_threads))

    model = create_model(opt)
    model.setup(opt)
    for name in getattr(model, 'model_names', []):
        if isinstance(name, str):
            getattr(model, name).train()

    visual_names_seen = None
    count = 0
    for data in loader:
        original_size = data['original_size'][0].tolist()
        data, pad_applied = _pad_data_to_multiple(data, max(1, int(args.pad_multiple)))
        model.isTrain = False
        model.set_input(data)
        # Forward in train-mode networks, but without any visible/GT tensors.
        model.test()
        visuals = model.get_current_visuals()
        src_path = data['paths'][0] if isinstance(data['paths'], (list, tuple)) else data['paths']
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        visual_names_seen = _save_visuals(
            visuals,
            args.results_dir,
            base_name,
            original_size,
            save_all_visuals=bool(int(args.save_all_visuals)),
        )
        count += 1

    summary = {
        'count': count,
        'dataroot': args.dataroot,
        'phase': args.phase,
        'checkpoint': args.epoch,
        'checkpoints_dir': args.checkpoints_dir,
        'name': args.name,
        'uses_visible_reference': False,
        'uses_gt': False,
        'preserve_original_size': bool(int(args.preserve_original_size)),
        'pad_multiple': int(args.pad_multiple),
        'pad_applied_last': pad_applied,
        'output_dir': args.results_dir,
        'visuals': visual_names_seen or [],
    }
    with open(os.path.join(args.results_dir, 'inference_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print('INFER_DONE ' + json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
