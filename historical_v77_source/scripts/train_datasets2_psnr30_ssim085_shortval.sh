#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jpc/jpc/VIFPM}"
DATAROOT="${DATAROOT:-/home/jpc/jpc/VIFPM/datasets2}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/home/jpc/jpc/VIFPM/checkpoints}"
LOG_DIR="${LOG_DIR:-/home/jpc/jpc/VIFPM/logs}"
GPU="${GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_THREADS="${NUM_THREADS:-1}"
SEED="${SEED:-20260730}"
EXP_NAME="${EXP_NAME:-datasets2_psnr30_ssim085_shortval_$(date +%Y%m%d_%H%M%S)}"
EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-64}"

source /home/jpc/miniconda3/etc/profile.d/conda.sh
conda activate vifpm

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

COMMON_ARGS=(
  --dataroot "$DATAROOT"
  --dataset_mode paired
  --name "$EXP_NAME"
  --checkpoints_dir "$CHECKPOINTS_DIR"
  --gpu_ids 0
  --seed "$SEED"
  --deterministic 0
  --batch_size "$BATCH_SIZE"
  --load_size 256
  --crop_size 256
  --model Irenhance
  --G_J modern_unet
  --model_type swin2_full
  --netD basic
  --ngf 64
  --ndf 64
  --lr_policy cosine
  --num_threads "$NUM_THREADS"
  --display_id -1
  --display_freq 200
  --update_html_freq 400
  --print_freq 25
  --save_latest_freq 1000
  --save_epoch_freq 5
  --cache_preprocess 1
  --decom_enabled 1
  --decom_base_ch 32
  --texture_frontend_enabled 0
  --fusion_local_psnr_calibrator 1
  --fusion_low_flops 1
  --fusion_efficient_v16 0
  --fusion_shared_weight 1
  --fusion_lowres_refiners 0
  --fusion_separable_convs 1
  --fusion_compact_post_refiner 0
  --texture_residual_mode 1
  --texture_residual_gain 0.16
  --texture_residual_max_delta 0.35
  --texture_residual_gain_per_step 0
  --texture_residual_steps 2
  --texture_residual_step_scale 1.00
  --texture_residual_step_scales 1.00,1.00
  --texture_residual_step_loss_weights 1.00,1.00
  --texture_residual_supervision 1
  --texture_residual_ir_focus_gain 0.18
  --texture_ir_skip 0.08
  --texture_lcn_blend 0.42
  --ir_texture_prompt_strength 0.24
  --ir_texture_prompt_bootstrap_strength 0.34
  --ir_texture_prompt_min_weight 0.00
  --ir_texture_prompt_threshold 0.44
  --ir_texture_prompt_temperature 0.12
  --ir_texture_prompt_power 0.90
  --texture_prompt_align_enabled 1
  --texture_prompt_align_radius 3
  --texture_prompt_align_window 7
  --texture_prompt_align_conf_threshold 0.15
  --texture_prompt_align_temperature 0.10
  --texture_prompt_align_highpass 1
  --texture_vis_supervision_mode gradtex
  --texture_vis_gt_consistency 0
  --texture_vis_conf_adaptive 1
  --texture_vis_conf_min 0.25
  --texture_vis_conf_power 1.5
  --gan_fixed_scale 0.0
  --gan_stage1_scale 0.0
  --gan_stage2_scale 0.0
  --lambda_Gg 0.0
  --lambda_D 0.0
  --target_stop 0
  --adaptive_tuning 0
)

run_stage() {
  local stage_name="$1"
  shift
  local log_path="$LOG_DIR/${EXP_NAME}_${stage_name}.log"
  echo "===== ${stage_name} ====="
  echo "log: ${log_path}"
  python -u train.py "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "$log_path"
}

echo "Experiment: $EXP_NAME"
echo "GPU: physical $GPU via CUDA_VISIBLE_DEVICES=$GPU"
echo "Short validation: 10 texture + 30 radiation/fusion + 10 calibration epochs."
echo "Training never uses test split; final evaluation runs only after all stages."

run_stage stage1_texture \
  --training_stage texture \
  --epoch_count 1 \
  --niter 10 \
  --niter_decay 0 \
  --lr 0.000040 \
  --stage1_epochs 10 \
  --texture_bootstrap_epochs 10 \
  --texture_bootstrap_fusion_scale 0.0 \
  --fusion_stage1_scale 0.0 \
  --fusion_warmup_epochs 1 \
  --lambda_texture 0.65 \
  --lambda_texture_gt 1.10 \
  --lambda_texture_vis 0.05 \
  --texture_align_scale 0.30 \
  --texture_preserve_scale 1.35 \
  --lambda_texture_step 0.01 \
  --lambda_texture_ag 0.00 \
  --lambda_texture_distill 0.02 \
  --lambda_texture_ms_grad 0.00 \
  --lambda_texture_contrast 0.00 \
  --lambda_texture_lowfreq 0.02 \
  --lambda_ir_hf 0.04 \
  --lambda_detail_gt 0.04 \
  --lambda_decom 0.18 \
  --lambda_structure_lowfreq 0.04 \
  --lambda_structure_underfill 0.01 \
  --lambda_radiation 0.0 \
  --lambda_fusion_I 0.0

run_stage stage2_radiation_fusion \
  --training_stage radiation_fusion \
  --continue_train \
  --epoch latest \
  --epoch_count 11 \
  --niter 40 \
  --niter_decay 0 \
  --lr 0.000035 \
  --stage1_epochs 0 \
  --texture_bootstrap_epochs 0 \
  --fusion_stage1_scale 1.0 \
  --fusion_warmup_epochs 1 \
  --lambda_texture 0.0 \
  --lambda_texture_gt 0.0 \
  --lambda_texture_vis 0.0 \
  --lambda_texture_step 0.0 \
  --lambda_texture_ag 0.0 \
  --lambda_texture_distill 0.0 \
  --lambda_texture_ms_grad 0.0 \
  --lambda_texture_lowfreq 0.0 \
  --lambda_ir_hf 0.0 \
  --lambda_detail_gt 0.0 \
  --lambda_decom 0.0 \
  --lambda_structure_lowfreq 0.0 \
  --lambda_structure_underfill 0.0 \
  --lambda_radiation 0.22 \
  --radiation_stage1_scale 1.0 \
  --radiation_stage2_scale 1.0 \
  --radiation_grad_weight 0.35 \
  --thermal_preserve_strength 0.12 \
  --lambda_fusion_I 1.75 \
  --lambda_fusion_charb 1.20 \
  --lambda_fusion_ssim 6.00 \
  --lambda_fusion_psnr 8.00 \
  --lambda_fusion_grad 0.010 \
  --lambda_fusion_structure 0.00 \
  --lambda_fusion_lap 0.00 \
  --lambda_fusion_lowfreq 0.35 \
  --lambda_fusion_stats 0.08 \
  --lambda_fusion_residual_gt 0.00 \
  --lambda_fusion_brightness_gt 0.00

run_stage stage3_fusion_calibration \
  --training_stage joint \
  --fusion_calibrator_finetune 1 \
  --fusion_only_zero_aux 1 \
  --continue_train \
  --epoch latest \
  --epoch_count 41 \
  --niter 50 \
  --niter_decay 0 \
  --lr 0.000010 \
  --stage1_epochs 0 \
  --texture_bootstrap_epochs 0 \
  --fusion_stage1_scale 1.0 \
  --fusion_warmup_epochs 1 \
  --lambda_texture 0.0 \
  --lambda_radiation 0.0 \
  --lambda_fusion_I 1.60 \
  --lambda_fusion_charb 1.10 \
  --lambda_fusion_ssim 7.00 \
  --lambda_fusion_psnr 9.00 \
  --lambda_fusion_grad 0.000 \
  --lambda_fusion_lowfreq 0.50 \
  --lambda_fusion_stats 0.10 \
  --lambda_fusion_residual_gt 0.04 \
  --lambda_fusion_brightness_gt 0.10

python eval_train_matched.py \
  --name "$EXP_NAME" \
  --dataroot "$DATAROOT" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --gpu_ids 0 \
  --max_images "$EVAL_MAX_IMAGES" \
  --batch_size "$BATCH_SIZE" \
  --labels latest best
