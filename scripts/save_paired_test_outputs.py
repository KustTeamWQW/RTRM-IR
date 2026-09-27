import argparse
import copy
import json
import os
import random
import sys
from types import SimpleNamespace

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import torch.nn.functional as F

from data import create_dataset
from models import create_model
from util import util


def _set_reproducible(seed, deterministic=True):
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


def _parse_gpu_ids(value):
    ids = []
    for item in str(value).split(','):
        item = item.strip()
        if item:
            gpu_id = int(item)
            if gpu_id >= 0:
                ids.append(gpu_id)
    return ids


def _pad_tensor(tensor, pad_w, pad_h):
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
        return tensor
    if pad_w == 0 and pad_h == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')


def _pad_data_to_multiple(data, multiple):
    sample = data['infrared']
    height, width = sample.shape[-2], sample.shape[-1]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    for key in ['infrared', 'infrared-radiation', 'gt-radiation', 'visible', 'gt']:
        if key in data:
            data[key] = _pad_tensor(data[key], pad_w, pad_h)
    return data, width, height, pad_w, pad_h


def _tensor_to_uint8_image(tensor, width, height):
    tensor = tensor[..., :height, :width]
    image = util.tensor2im(tensor, np.uint8)
    return image[:height, :width]


def _load_train_options(checkpoints_dir, name):
    opt_path = os.path.join(checkpoints_dir, name, 'train_opt.json')
    with open(opt_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _build_options(base_options, args):
    options = copy.deepcopy(base_options)
    options.update({
        'checkpoints_dir': args.checkpoints_dir,
        'name': args.name,
        'dataroot': args.dataroot,
        'dataset_mode': 'paired',
        'phase': args.phase,
        'gpu_ids': _parse_gpu_ids(args.gpu_ids),
        'isTrain': True,
        'continue_train': True,
        'epoch': args.epoch,
        'load_iter': 0,
        'serial_batches': True,
        'no_flip': True,
        'batch_size': 1,
        'num_threads': args.num_threads,
        'display_id': -1,
        'max_dataset_size': float('inf'),
        # pairedDataset crops to crop_size directly. A very large crop disables
        # cropping because images smaller than crop_size are returned unchanged.
        'crop_size': 1000000,
        'load_size': 1000000,
    })
    if torch.cuda.is_available() and options['gpu_ids']:
        torch.cuda.set_device(options['gpu_ids'][0])
    return SimpleNamespace(**options)


def main():
    parser = argparse.ArgumentParser(description='Save paired test outputs with original input size.')
    parser.add_argument('--checkpoints_dir', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--phase', default='test')
    parser.add_argument('--epoch', default='best_val')
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--pad_multiple', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    base_options = _load_train_options(args.checkpoints_dir, args.name)
    opt = _build_options(base_options, args)
    _set_reproducible(getattr(opt, 'seed', 20260730), bool(int(getattr(opt, 'deterministic', 0))))

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    for net_name in getattr(model, 'model_names', []):
        if isinstance(net_name, str):
            getattr(model, net_name).train()

    records = []
    count = 0
    for data in dataset:
        data, width, height, pad_w, pad_h = _pad_data_to_multiple(data, max(1, int(args.pad_multiple)))
        model.isTrain = True
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        fused = visuals['fused_I']
        src_path = data['paths'][0] if isinstance(data['paths'], (list, tuple)) else data['paths']
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        out_path = os.path.join(args.results_dir, base_name + '.png')
        image = _tensor_to_uint8_image(fused, width, height)
        util.save_image(image, out_path)
        records.append({
            'input': src_path,
            'output': out_path,
            'width': int(width),
            'height': int(height),
            'pad_w': int(pad_w),
            'pad_h': int(pad_h),
        })
        count += 1

    summary = {
        'count': count,
        'checkpoint': args.epoch,
        'checkpoints_dir': args.checkpoints_dir,
        'name': args.name,
        'dataroot': args.dataroot,
        'phase': args.phase,
        'output_dir': args.results_dir,
        'pad_multiple': int(args.pad_multiple),
        'preserve_input_size': True,
        'records': records,
    }
    with open(os.path.join(args.results_dir, 'inference_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print('SAVE_OUTPUTS_DONE ' + json.dumps({k: v for k, v in summary.items() if k != 'records'}, sort_keys=True))


if __name__ == '__main__':
    main()
