import copy
import random

import numpy as np
import torch

from IQA.psnr_ssim import calculate_psnr, calculate_ssim
from data import create_dataset
from util import util


def _tensor_item_to_image(tensor, index):
    if isinstance(tensor, torch.Tensor) and tensor.dim() == 4:
        tensor = tensor[index:index + 1]
    return util.tensor2im(tensor, np.float64)


def _network_modes(model):
    modes = {}
    for name in model.model_names:
        if isinstance(name, str):
            modes[name] = getattr(model, name).training
    return modes


def _set_network_modes(model, modes):
    for name, training in modes.items():
        getattr(model, name).train(training)


def _set_all_networks_train(model):
    for name in model.model_names:
        if isinstance(name, str):
            getattr(model, name).train()


def _buffer_snapshot(model):
    snapshots = []
    for name in model.model_names:
        if not isinstance(name, str):
            continue
        net = getattr(model, name)
        net = net.module if hasattr(net, 'module') else net
        for module in net.modules():
            for buffer_name, buffer in module._buffers.items():
                if buffer is not None:
                    snapshots.append((module, buffer_name, buffer.detach().clone()))
    return snapshots


def _restore_buffers(snapshots):
    for module, buffer_name, value in snapshots:
        buffer = module._buffers.get(buffer_name)
        if buffer is not None:
            buffer.copy_(value)


def _rng_snapshot():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def evaluate_train_matched(model, opt, max_images=None, batch_size=None, phase='test'):
    """Evaluate with the same data/forward branch used during training.

    This intentionally keeps ``model.isTrain`` true and the networks in train mode.
    Gradients are still disabled by ``model.test()``. Metrics are computed per
    image so larger eval batches do not silently drop images through tensor2im().
    """
    rng_state = _rng_snapshot()
    old_is_train = model.isTrain
    old_modes = _network_modes(model)
    old_buffers = _buffer_snapshot(model)
    try:
        eval_opt = copy.deepcopy(opt)
        eval_opt.phase = str(phase)
        eval_opt.dataset_mode = 'paired'
        eval_opt.serial_batches = True
        eval_opt.no_flip = True
        eval_opt.display_id = -1

        if max_images is None:
            max_images = getattr(opt, 'eval_max_images', 160)
        max_images = max(1, int(max_images))
        eval_opt.num_test = max_images
        # Keep the paired dataset complete until it aligns A/B/GT stems. Truncating
        # before alignment can drop all matches when folders are enumerated differently.
        if not hasattr(eval_opt, 'max_dataset_size'):
            eval_opt.max_dataset_size = float('inf')
        eval_opt.batch_size = max(1, int(batch_size if batch_size is not None else getattr(opt, 'batch_size', 1)))
        eval_opt.num_threads = max(0, int(getattr(opt, 'num_threads', 0)))

        eval_dataset = create_dataset(eval_opt)
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0

        model.isTrain = True
        _set_all_networks_train(model)
        for data in eval_dataset:
            if count >= max_images:
                break
            model.set_input(data)
            model.test()
            pred_tensor = model.get_current_visuals()['fused_I']
            gt_tensor = data.get('gt', getattr(model, 'gt', None))
            if gt_tensor is None:
                continue

            batch_count = pred_tensor.shape[0] if isinstance(pred_tensor, torch.Tensor) and pred_tensor.dim() == 4 else 1
            for batch_index in range(batch_count):
                if count >= max_images:
                    break
                pred = _tensor_item_to_image(pred_tensor, batch_index)
                gt = _tensor_item_to_image(gt_tensor, batch_index)
                if pred.shape != gt.shape:
                    height = min(pred.shape[0], gt.shape[0])
                    width = min(pred.shape[1], gt.shape[1])
                    pred = pred[:height, :width, ...]
                    gt = gt[:height, :width, ...]
                total_psnr += calculate_psnr(
                    gt,
                    pred,
                    crop_border=getattr(opt, 'eval_crop_border', 0),
                    input_order='HWC',
                    test_y_channel=False,
                )
                total_ssim += calculate_ssim(
                    gt,
                    pred,
                    crop_border=getattr(opt, 'eval_crop_border', 0),
                    input_order='HWC',
                    test_y_channel=False,
                )
                count += 1
    finally:
        _restore_buffers(old_buffers)
        model.isTrain = old_is_train
        _set_network_modes(model, old_modes)
        _restore_rng_state(rng_state)

    if count == 0:
        return {'psnr': 0.0, 'ssim': 0.0, 'score': 0.0, 'images': 0, 'requested_images': max_images}

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    score = avg_psnr + float(getattr(opt, 'eval_ssim_weight', 35.0)) * avg_ssim
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'score': score,
        'images': count,
        'requested_images': max_images,
        'batch_size': eval_opt.batch_size,
    }
