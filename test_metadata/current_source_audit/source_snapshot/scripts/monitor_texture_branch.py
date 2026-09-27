#!/usr/bin/env python
import argparse
import copy
import json
import os
import sys
import time
from argparse import Namespace

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data import create_dataset
from models import create_model


def _stats(name, x):
    if x is None:
        return f'{name}: None'
    if not isinstance(x, torch.Tensor):
        return f'{name}: {type(x).__name__}'
    y = x.detach().float()
    flat = y.flatten()
    if flat.numel() == 0:
        return f'{name}: empty'
    return (
        f'{name}: shape={tuple(y.shape)} '
        f'min={flat.min().item():.6f} max={flat.max().item():.6f} '
        f'mean={flat.mean().item():.6f} std={flat.std(unbiased=False).item():.6f} '
        f'p01={flat.quantile(0.01).item():.6f} p50={flat.quantile(0.50).item():.6f} '
        f'p99={flat.quantile(0.99).item():.6f} '
        f'pos={(flat > 0).float().mean().item():.6f} '
        f'neg={(flat < 0).float().mean().item():.6f} '
        f'zero={(flat == 0).float().mean().item():.6f}'
    )


def _load_opt(exp_dir, label, phase, max_images, batch_size, gpu_ids):
    opt_path = os.path.join(exp_dir, 'train_opt.json')
    with open(opt_path, 'r', encoding='utf-8') as handle:
        cfg = json.load(handle)
    cfg['isTrain'] = True
    cfg['continue_train'] = True
    cfg['epoch'] = label
    cfg['phase'] = phase
    cfg['serial_batches'] = True
    cfg['no_flip'] = True
    cfg['max_dataset_size'] = max_images
    cfg['batch_size'] = batch_size
    cfg['num_threads'] = 0
    cfg['gpu_ids'] = [int(gpu_ids)] if str(gpu_ids) != '-1' else []
    cfg['display_id'] = -1
    cfg['verbose'] = False
    cfg['load_iter'] = 0
    return Namespace(**cfg)


def _safe_load(model, label):
    model.load_networks(label)
    for name in model.model_names:
        if isinstance(name, str):
            getattr(model, name).train()


def _raw_texture_delta(model, data):
    if not bool(int(getattr(model.opt, 'texture_residual_mode', 0))):
        return None
    current_texture = model.T_ir.detach()
    ir_focus_gain = max(0.0, min(1.0, float(getattr(model.opt, 'texture_residual_ir_focus_gain', 0.0))))
    if ir_focus_gain > 0.0:
        ir_focus_map = model._ir_texture_selector(model.T_ir, model.S_ir)
        current_texture = current_texture * (1.0 + ir_focus_gain * ir_focus_map)
    if bool(int(getattr(model.opt, 'texture_frontend_enabled', 0))):
        current_texture = model.texture_frontend(current_texture)
    generator = model.G_J
    return generator(current_texture)


def inspect_once(args):
    exp_dir = os.path.join(args.checkpoints_dir, args.name)
    opt = _load_opt(exp_dir, args.label, args.phase, args.max_images, args.batch_size, args.gpu_id)
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_ids[0])

    dataset = create_dataset(opt)
    model = create_model(opt)
    _safe_load(model, args.label)
    model.isTrain = True

    totals = []
    with torch.no_grad():
        for index, data in enumerate(dataset):
            if index >= args.max_images:
                break
            model.set_input(data)
            model.forward()
            t_ir = getattr(model, 'T_ir', None)
            t_vis = getattr(model, 'T_vis', None)
            gt = getattr(model, 'gt', None)
            t_gt_ref = None
            if gt is not None and hasattr(model, 'decom_net'):
                _, t_gt_ref = model.decom_net(gt.detach())
            raw_delta = _raw_texture_delta(model, data)
            texture_delta = model._texture_delta(raw_delta) if raw_delta is not None else None
            r_gt = t_gt_ref.detach() - t_ir.detach() if t_gt_ref is not None and t_ir is not None else None
            r_vis = t_vis.detach() - t_ir.detach() if t_vis is not None and t_ir is not None else None
            rows = [
                f'sample={index} path={data.get("paths", [""])[0]}',
                _stats('real_I', getattr(model, 'real_I', None)),
                _stats('S_ir', getattr(model, 'S_ir', None)),
                _stats('T_ir', t_ir),
                _stats('T_vis', t_vis),
                _stats('T_gt', t_gt_ref),
                _stats('R_gt(T_gt-T_ir)', r_gt),
                _stats('R_vis(T_vis-T_ir)', r_vis),
                _stats('G_J_raw_delta', raw_delta),
                _stats('texture_delta_step0', texture_delta),
                _stats('R_pred_after_gain_steps', getattr(model, 'R_pred', None)),
                _stats('T_enhanced', getattr(model, 'T_enhanced', None)),
                _stats('refine_t', getattr(model, 'refine_t', None)),
                _stats('refine_r', getattr(model, 'refine_r', None)),
                _stats('fused_I', getattr(model, 'fused_I', None)),
            ]
            report = '\n'.join(rows)
            print(report, flush=True)
            print('-' * 100, flush=True)
            totals.append(report)
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--checkpoints_dir', default='/home/jpc/jpc/VIFPM/checkpoints')
    parser.add_argument('--label', default='latest')
    parser.add_argument('--phase', default='val')
    parser.add_argument('--max_images', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu_id', default='1')
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--rounds', type=int, default=1)
    args = parser.parse_args()

    for round_index in range(max(1, args.rounds)):
        print(f'### texture monitor round {round_index + 1}/{max(1, args.rounds)} label={args.label}', flush=True)
        inspect_once(args)
        if args.interval > 0 and round_index + 1 < args.rounds:
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
