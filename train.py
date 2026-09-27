import json
import os
import time
import numpy as np
import random
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.train_matched_eval import evaluate_train_matched
from util.visualizer import Visualizer


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


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    _set_reproducible(getattr(opt, 'seed', 20260713), bool(int(getattr(opt, 'deterministic', 1))))
    print('Reproducibility: seed=%d deterministic=%d' % (
        int(getattr(opt, 'seed', 20260713)),
        int(getattr(opt, 'deterministic', 1)),
    ))
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    best_metric = float('inf')
    best_val_score = -float('inf')
    val_enabled = bool(int(getattr(opt, 'val_during_train', 0)))
    val_freq = max(1, int(getattr(opt, 'val_freq', 5)))
    val_phase = str(getattr(opt, 'val_phase', 'val'))
    val_save_label = str(getattr(opt, 'val_save_label', 'best_val'))
    val_metrics_path = os.path.join(model.save_dir, 'val_metrics.jsonl')
    best_val_meta_path = os.path.join(model.save_dir, 'best_val_meta.json')
    best_val_epoch = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_loss = 0.0
        num_batches = 0
        model.current_epoch = epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            # Keep this best checkpoint based only on training losses.
            current_precision_loss = (
                0.25 * losses.get('Gg', 0.0) +
                0.30 * losses.get('recon', 0.0) +
                0.20 * losses.get('detail_gt', 0.0) +
                0.15 * losses.get('radiation', 0.0) +
                0.10 * losses.get('fusion', 0.0)
            )
            epoch_loss += current_precision_loss

            num_batches += 1
            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                model.save_networks('latest')

            iter_data_time = time.time()

        if num_batches == 0:
            print(f"Warning: Epoch {epoch} has no batches, skipping loss calculation.")
            avg_loss = float('inf')
        else:
            avg_loss = epoch_loss / num_batches

        if avg_loss < best_metric:
            best_metric = avg_loss
            print(f'Saving the best model at epoch {epoch} with precision loss {avg_loss:.4f}')
            model.save_networks('best')  # 保存为best权重

        if val_enabled and (epoch % val_freq == 0):
            old_eval_ssim_weight = getattr(opt, 'eval_ssim_weight', None)
            opt.eval_ssim_weight = float(getattr(opt, 'val_ssim_weight', getattr(opt, 'eval_ssim_weight', 30.0)))
            try:
                val_stats = evaluate_train_matched(
                    model,
                    opt,
                    max_images=getattr(opt, 'val_max_images', 64),
                    batch_size=getattr(opt, 'batch_size', 1),
                    phase=val_phase,
                )
            finally:
                if old_eval_ssim_weight is not None:
                    opt.eval_ssim_weight = old_eval_ssim_weight
            val_record = {
                'epoch': epoch,
                'phase': val_phase,
                **val_stats,
            }
            print('VAL_JSON ' + json.dumps(val_record, sort_keys=True), flush=True)
            with open(val_metrics_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(val_record, sort_keys=True) + '\n')
            if val_stats['images'] > 0 and val_stats['score'] > best_val_score:
                best_val_score = val_stats['score']
                best_val_epoch = epoch
                print(
                    'Saving the validation-best model at epoch %d with PSNR %.4f SSIM %.4f score %.4f'
                    % (epoch, val_stats['psnr'], val_stats['ssim'], val_stats['score'])
                )
                model.save_networks(val_save_label)
                best_val_record = dict(val_record)
                best_val_record['checkpoint_label'] = val_save_label
                best_val_record['best_val_epoch'] = best_val_epoch
                with open(best_val_meta_path, 'w', encoding='utf-8') as handle:
                    json.dump(best_val_record, handle, ensure_ascii=False, indent=2, sort_keys=True)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        
        # ======== [修复：高原学习率衰减策略(plateau)生效的前提] ========
        model.metric = avg_loss 
        model.update_learning_rate()

