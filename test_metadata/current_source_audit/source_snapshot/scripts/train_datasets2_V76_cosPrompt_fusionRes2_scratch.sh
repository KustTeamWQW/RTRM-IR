#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jpc/jpc/VIFPM}"
DATAROOT="${DATAROOT:-/home/jpc/jpc/VIFPM/datasets2}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/home/jpc/jpc/VIFPM/checkpoints}"
LOG_DIR="${LOG_DIR:-/home/jpc/jpc/VIFPM/logs}"
EXP_NAME="${EXP_NAME:-datasets2_V76_cosPrompt_fusionRes2_scratch_bs3_ep520_gpu1_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_THREADS="${NUM_THREADS:-1}"
SEED="${SEED:-20260730}"

source /home/jpc/miniconda3/etc/profile.d/conda.sh
conda activate vifpm

mkdir -p "$LOG_DIR" "$CHECKPOINTS_DIR/$EXP_NAME"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

LOG_PATH="$LOG_DIR/${EXP_NAME}_train.log"
echo "Experiment: $EXP_NAME"
echo "Source weights: none; training from scratch."
echo "GPU: physical $GPU via CUDA_VISIBLE_DEVICES=$GPU"
echo "Clean launch: V76, V74 base with fusion_residual_steps=2 and cosine visible-texture prompt gating."
echo "log: $LOG_PATH"

python -u train.py \
  --dataroot "$DATAROOT" \
  --dataset_mode paired \
  --name "$EXP_NAME" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --gpu_ids 0 \
  --epoch_count 1 \
  --seed "$SEED" \
  --deterministic 0 \
  --batch_size "$BATCH_SIZE" \
  --load_size 256 \
  --crop_size 256 \
  --model Irenhance \
  --G_J modern_unet \
  --model_type swin2_full \
  --netD basic \
  --ngf 64 \
  --ndf 64 \
  --lr 0.000005 \
  --lr_policy cosine \
  --niter 520 \
  --niter_decay 0 \
  --num_threads "$NUM_THREADS" \
  --display_id -1 \
  --display_freq 200 \
  --update_html_freq 400 \
  --print_freq 25 \
  --save_latest_freq 1000 \
  --save_epoch_freq 5 \
  --cache_preprocess 1 \
  --decom_enabled 1 \
  --decom_mode ir_radiation_texture_ref \
  --decom_base_ch 32 \
  --decom_radiation_ksize 31 \
  --decom_radiation_iters 2 \
  --decom_guided_ksize 15 \
  --decom_guided_eps 0.01 \
  --gt_radiation_ksize 31 \
  --gt_radiation_raw_blend 0.20 \
  --ir_radiation_raw_blend 0.00 \
  --texture_frontend_enabled 0 \
  --texture_domain_norm_enabled 0 \
  --texture_residual_mode 1 \
  --texture_residual_gain 0.16 \
  --texture_residual_max_delta 0.35 \
  --texture_residual_gain_per_step 0 \
  --texture_residual_steps 2 \
  --texture_residual_step_scale 1.00 \
  --texture_residual_step_scales 1.00,1.00 \
  --texture_residual_step_loss_weights 1.00,1.00 \
  --texture_residual_ir_focus_gain 0.18 \
  --texture_ir_skip 0.08 \
  --texture_prompt_align_enabled 1 \
  --texture_prompt_align_window 7 \
  --texture_prompt_align_conf_threshold 0.15 \
  --texture_prompt_align_temperature 0.10 \
  --texture_prompt_align_highpass 1 \
  --texture_vis_supervision_mode gradtex \
  --texture_vis_conf_adaptive 1 \
  --texture_vis_conf_min 0.25 \
  --texture_vis_conf_power 1.5 \
  --training_stage joint \
  --stage1_epochs 0 \
  --texture_bootstrap_epochs 0 \
  --fusion_stage1_scale 1.0 \
  --fusion_warmup_epochs 1 \
  --fusion_only_finetune 0 \
  --fusion_only_zero_aux 0 \
  --fusion_low_flops 1 \
  --fusion_shared_weight 1 \
  --fusion_residual_steps 2 \
  --gan_fixed_scale 0.0 \
  --gan_stage1_scale 0.0 \
  --gan_stage2_scale 0.0 \
  --lambda_Gg 0.0 \
  --lambda_D 0.0 \
  --lambda_texture 1.00 \
  --lambda_texture_gt 0.75 \
  --lambda_texture_vis 0.04 \
  --lambda_texture_step 0.005 \
  --lambda_decom 0.10 \
  --lambda_structure_lowfreq 0.00 \
  --lambda_structure_underfill 0.00 \
  --lambda_radiation 0.25 \
  --radiation_stage1_scale 1.0 \
  --radiation_stage2_scale 1.0 \
  --radiation_grad_weight 0.00 \
  --lambda_fusion_I 1.80 \
  --lambda_fusion_l1 2.00 \
  --lambda_fusion_ssim 8.00 \
  --lambda_fusion_psnr 18.00 \
  --val_during_train 1 \
  --val_freq 1 \
  --val_max_images 64 \
  --val_save_label best_val \
  "$@" \
  2>&1 | tee "$LOG_PATH"

python eval_train_matched.py \
  --name "$EXP_NAME" \
  --dataroot "$DATAROOT" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --gpu_ids 0 \
  --max_images 64 \
  --batch_size "$BATCH_SIZE" \
  --labels latest best best_val
