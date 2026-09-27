#!/usr/bin/env python
"""Launch Dataset1 V77 training with the saved checkpoint options.

The checkpoint's train_opt.json supplies the training arguments. Runtime paths
and the visible GPU are overridden so the project can run from another machine
without editing source files.
"""

import argparse
import json
import math
import os
import runpy
import shlex
import sys
from pathlib import Path


ACTION_FLAGS = {
    'continue_train',
    'no_dropout',
    'no_flip',
    'no_html',
    'save_by_iter',
    'save_processed_debug',
    'serial_batches',
    'verbose',
}


def _value_to_args(key, value):
    if key == 'isTrain' or value is None:
        return []
    if isinstance(value, float) and math.isinf(value):
        return []
    flag = '--' + key
    if key in ACTION_FLAGS:
        return [flag] if bool(value) else []
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (dict, list)):
        raise TypeError('Unsupported option value for %s' % key)
    return [flag, str(value)]


def _shell_join(args):
    join = getattr(shlex, 'join', None)
    if join is not None:
        return join(args)
    return ' '.join(shlex.quote(str(item)) for item in args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='datasets1')
    parser.add_argument('--checkpoints-dir', default='reproduced_training/checkpoints')
    parser.add_argument('--name', default='datasets1_ODinMJ')
    parser.add_argument('--gpu-id', default='0', help='logical GPU id inside CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    bundle_dir = Path(__file__).resolve().parent
    source_dir = bundle_dir
    checkpoint_dir = bundle_dir / 'checkpoints' / args.name
    option_path = checkpoint_dir / 'train_opt.json'
    if not option_path.is_file():
        raise FileNotFoundError(str(option_path))

    with option_path.open('r', encoding='utf-8') as handle:
        options = json.load(handle)

    overrides = {
        'dataroot': args.dataset_root,
        'checkpoints_dir': args.checkpoints_dir,
        'name': args.name,
        'gpu_ids': '0',
        'phase': 'train',
        'epoch': 'latest',
        'continue_train': False,
    }
    options.update(overrides)
    command_args = []
    for key in sorted(options):
        command_args.extend(_value_to_args(key, options[key]))

    os.chdir(bundle_dir)
    sys.path.insert(0, str(source_dir))
    sys.argv = [str(source_dir / 'train.py')] + command_args
    if args.dry_run:
        from options.train_options import TrainOptions

        parsed = TrainOptions().gather_options()
        print('V77_OPTIONS_VALID ' + str(len(vars(parsed))))
        print(_shell_join(sys.argv))
        return
    runpy.run_path(str(source_dir / 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
