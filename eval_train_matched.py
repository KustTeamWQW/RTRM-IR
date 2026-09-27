import argparse
import copy
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch

from models import create_model
from util.train_matched_eval import evaluate_train_matched


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
    if isinstance(value, list):
        return [int(item) for item in value if int(item) >= 0]
    ids = []
    for item in str(value).split(','):
        item = item.strip()
        if item:
            gpu_id = int(item)
            if gpu_id >= 0:
                ids.append(gpu_id)
    return ids


def _load_train_options(checkpoints_dir, name):
    opt_path = os.path.join(checkpoints_dir, name, 'train_opt.json')
    with open(opt_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _build_model_options(base_options, args, label):
    options = copy.deepcopy(base_options)
    options.update({
        'checkpoints_dir': args.checkpoints_dir,
        'name': args.name,
        'dataroot': args.dataroot or base_options.get('dataroot'),
        'dataset_mode': 'paired',
        'phase': args.phase,
        'gpu_ids': _parse_gpu_ids(args.gpu_ids),
        'isTrain': True,
        'continue_train': True,
        'epoch': label,
        'load_iter': 0,
        'serial_batches': False,
        'no_flip': False,
        'num_threads': args.num_threads if args.num_threads is not None else int(base_options.get('num_threads', 0)),
        'display_id': -1,
    })
    if args.batch_size is not None:
        options['batch_size'] = args.batch_size
    if torch.cuda.is_available() and options['gpu_ids']:
        torch.cuda.set_device(options['gpu_ids'][0])
    return SimpleNamespace(**options)


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoints with the same paired/train-state logic used by training.')
    parser.add_argument('--checkpoints_dir', default='/home/jpc/jpc/VIFPM/checkpoints')
    parser.add_argument('--name', required=True)
    parser.add_argument('--dataroot', default='')
    parser.add_argument('--labels', nargs='+', default=['best_eval', 'latest', 'best'])
    parser.add_argument('--phase', default='test', help='Dataset split to evaluate: test, val, train, etc.')
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--max_images', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_threads', type=int, default=0)
    args = parser.parse_args()

    base_options = _load_train_options(args.checkpoints_dir, args.name)
    results = {}
    for label in args.labels:
        opt = _build_model_options(base_options, args, label)
        _set_reproducible(getattr(opt, 'seed', 20260727), bool(int(getattr(opt, 'deterministic', 0))))
        print(f'\n===== TRAIN_MATCHED_EVAL {label} =====', flush=True)
        model = create_model(opt)
        model.setup(opt)
        stats = evaluate_train_matched(
            model,
            opt,
            max_images=args.max_images,
            batch_size=getattr(opt, 'batch_size', 1),
            phase=args.phase,
        )
        results[label] = stats
        print('RESULT_JSON', json.dumps({'label': label, **stats}, sort_keys=True), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('\nSUMMARY_JSON', json.dumps(results, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
