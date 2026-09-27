import argparse
import json
import os
import sys
import types

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import create_dataset
from models import create_model
from util import util

ABLATION_COMMON = os.path.join(REPO_ROOT, 'Ablation', 'common')
if ABLATION_COMMON not in sys.path:
    sys.path.insert(0, ABLATION_COMMON)

import strict_ablation_train


def _load_strict_opt(checkpoints_dir, name, results_dir, phase, gpu_ids):
    opt_path = os.path.join(checkpoints_dir, name, 'train_opt.json')
    with open(opt_path, 'r', encoding='utf-8') as handle:
        opt = json.load(handle)

    opt.update({
        'checkpoints_dir': checkpoints_dir,
        'name': name,
        'results_dir': results_dir,
        'dataroot': os.path.join(REPO_ROOT, 'datasets2'),
        'phase': phase,
        'dataset_mode': 'paired',
        'isTrain': True,
        'continue_train': True,
        'epoch': 'best',
        'load_iter': 0,
        'gpu_ids': gpu_ids,
        'serial_batches': True,
        'no_flip': True,
        'batch_size': 1,
        'num_threads': 0,
        'display_id': -1,
        'max_dataset_size': float('inf'),
        'save_processed_debug': False,
        'seed': 20260730,
        'deterministic': 0,
    })

    return argparse.Namespace(**opt)


def _save_tensor(tensor, save_path):
    if tensor is None:
        return False
    if not isinstance(tensor, torch.Tensor):
        return False
    util.save_image(util.tensor2im(torch.clamp(tensor, 0.0, 1.0)), save_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', default=os.path.join(REPO_ROOT, 'Ablation', 'M1', 'checkpoints'))
    parser.add_argument('--name', default='datasets2_ablation_M1_corrected_ep200_gpu1_bs10')
    parser.add_argument('--phase', default='test')
    parser.add_argument('--results_dir', default=os.path.join(REPO_ROOT, 'Ablation', 'M1', 'final_texture_reconstruction'))
    parser.add_argument('--gpu_ids', default='-1')
    args = parser.parse_args()

    gpu_ids = [int(x) for x in str(args.gpu_ids).split(',') if x.strip() and int(x) >= 0]
    opt = _load_strict_opt(args.checkpoints_dir, args.name, args.results_dir, args.phase, gpu_ids)

    if torch.cuda.is_available() and opt.gpu_ids:
        torch.cuda.set_device(opt.gpu_ids[0])

    os.makedirs(args.results_dir, exist_ok=True)
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.forward = types.MethodType(strict_ablation_train._forward_M1, model)
    for name in getattr(model, 'model_names', []):
        if isinstance(name, str):
            getattr(model, name).train()

    saved = 0
    for index, data in enumerate(dataset):
        model.set_input(data)
        with torch.no_grad():
            model.forward()

        out = getattr(model, 'paper_15_texture_reconstruction_Sir_plus_Tenhanced', None)
        if out is None:
            out = getattr(model, 'fused_I', None)
        if out is None:
            continue

        if out.dim() == 4:
            out = out[0:1]
        image_name = os.path.splitext(os.path.basename(data['paths'][0]))[0]
        save_path = os.path.join(args.results_dir, f'{image_name}.png')
        _save_tensor(out, save_path)
        saved += 1

    summary = {
        'count': saved,
        'checkpoint': 'best',
        'name': args.name,
        'output_dir': args.results_dir,
        'visual': 'paper_15_texture_reconstruction_Sir_plus_Tenhanced',
    }
    with open(os.path.join(args.results_dir, 'inference_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print('INFER_DONE ' + json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
