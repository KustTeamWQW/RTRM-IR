# Corresponding Best-Val Test Summary

Date: 2026-08-10

Evaluation script:

`eval_train_matched.py`

Common settings:

- checkpoint label: `best_val`
- phase: `test`
- GPU: physical GPU 6 via `CUDA_VISIBLE_DEVICES=6`, script `--gpu_ids 0`
- batch size: 4
- max images: 100000, effectively all paired test images found
- score formula from training config: `PSNR + eval_ssim_weight * SSIM`

## datasets3

- checkpoint directory: `/home/jpc/jpc/VIFPM-data3/checkpoints/datasets3_V77_guidedK19_TirBlur31_cosPrompt_fusionRes2_scratch_bs4_ep520_gpu6_20260808_081212`
- dataroot: `/home/jpc/jpc/VIFPM-data3/datasets3`
- images: 102
- PSNR: 30.200176294038087
- SSIM: 0.843905590051329
- Score: 55.51734399557796

Command:

```bash
CUDA_VISIBLE_DEVICES=6 /home/jpc/miniconda3/envs/vifpm/bin/python eval_train_matched.py --checkpoints_dir /home/jpc/jpc/VIFPM-data3/checkpoints --name datasets3_V77_guidedK19_TirBlur31_cosPrompt_fusionRes2_scratch_bs4_ep520_gpu6_20260808_081212 --dataroot /home/jpc/jpc/VIFPM-data3/datasets3 --labels best_val --phase test --gpu_ids 0 --max_images 100000 --batch_size 4 --num_threads 0
```

## datasets1

- checkpoint directory: `/home/jpc/jpc/VIFPM-data3/checkpoints/datasets1_V77_guidedK19_TirBlur31_cosPrompt_fusionRes1_scratch_bs4_ep520_gpu6_20260809_054006`
- dataroot: `/home/jpc/jpc/VIFPM-data3/datasets1`
- images: 160
- PSNR: 31.52806576227396
- SSIM: 0.8291946760768647
- Score: 56.4039060445799

Command:

```bash
CUDA_VISIBLE_DEVICES=6 /home/jpc/miniconda3/envs/vifpm/bin/python eval_train_matched.py --checkpoints_dir /home/jpc/jpc/VIFPM-data3/checkpoints --name datasets1_V77_guidedK19_TirBlur31_cosPrompt_fusionRes1_scratch_bs4_ep520_gpu6_20260809_054006 --dataroot /home/jpc/jpc/VIFPM-data3/datasets1 --labels best_val --phase test --gpu_ids 0 --max_images 100000 --batch_size 4 --num_threads 0
```
