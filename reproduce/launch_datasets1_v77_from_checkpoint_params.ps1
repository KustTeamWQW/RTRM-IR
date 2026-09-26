$ErrorActionPreference = 'Stop'

# Reproduce the training hyper-parameters saved in:
# checkpoints/datasets1_V77_guidedK19_TirBlur31_cosPrompt_fusionRes1_scratch_bs4_ep520_gpu6_20260809_054006/train_opt.json
#
# The default output name uses a timestamp suffix to avoid overwriting the
# original checkpoint directory. Override $Experiment before running if needed.

$envPath = 'D:\Environment_2023\Anaconda3\envs\py3.7'
$repoRoot = Split-Path -Parent $PSScriptRoot
$codeRoot = $repoRoot
$logDir = Join-Path $repoRoot 'logs'
$ckptDir = Join-Path $repoRoot 'checkpoints'
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$baseExperiment = 'datasets1_V77_guidedK19_TirBlur31_cosPrompt_fusionRes1_scratch_bs4_ep520_gpu6_20260809_054006'
$experiment = if ($env:EXPERIMENT_NAME) { $env:EXPERIMENT_NAME } else { "${baseExperiment}_repro_${timestamp}" }
$logPath = Join-Path $logDir "${experiment}.log"
$errPath = Join-Path $logDir "${experiment}.err.log"

New-Item -ItemType Directory -Force $logDir | Out-Null
New-Item -ItemType Directory -Force $ckptDir | Out-Null

$env:PYTHONHOME = ''
$env:PYTHONPATH = ''
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { '0' }
$env:PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:512'
$env:PATH = @(
    $envPath,
    "$envPath\Library\mingw-w64\bin",
    "$envPath\Library\usr\bin",
    "$envPath\Library\bin",
    "$envPath\Scripts",
    "$envPath\bin",
    "$envPath\DLLs",
    $env:PATH
) -join ';'

$argsList = @(
    '-u',
    'train.py',
    '--dataroot', (Join-Path $repoRoot 'datasets1'),
    '--dataset_mode', 'paired',
    '--name', $experiment,
    '--checkpoints_dir', $ckptDir,
    '--gpu_ids', '0',
    '--seed', '20260730',
    '--deterministic', '0',
    '--batch_size', '4',
    '--load_size', '256',
    '--crop_size', '256',
    '--model', 'Irenhance',
    '--G_J', 'modern_unet',
    '--model_type', 'swin2_full',
    '--netD', 'basic',
    '--ngf', '64',
    '--ndf', '64',
    '--input_nc', '1',
    '--output_nc', '1',
    '--norm', 'instance',
    '--init_type', 'normal',
    '--init_gain', '0.02',
    '--no_dropout',
    '--niter', '520',
    '--niter_decay', '0',
    '--lr', '0.000005',
    '--lr_policy', 'cosine',
    '--lr_decay_iters', '10',
    '--beta1', '0.5',
    '--num_threads', '1',
    '--display_id', '-1',
    '--display_freq', '200',
    '--update_html_freq', '400',
    '--print_freq', '25',
    '--save_latest_freq', '1000',
    '--save_epoch_freq', '5',
    '--cache_preprocess', '1',
    '--preprocess_cache_dir', 'preprocess_cache',
    '--preprocess', 'scale_min_and_crop',
    '--pool_size', '50',
    '--decom_enabled', '1',
    '--decom_mode', 'ir_radiation_texture_ref',
    '--decom_base_ch', '32',
    '--decom_guided_ksize', '15',
    '--decom_guided_eps', '0.01',
    '--decom_radiation_ksize', '31',
    '--decom_radiation_iters', '2',
    '--ir_radiation_raw_blend', '0.0',
    '--ir_radiation_scale_init', '-2.0',
    '--gt_radiation_ksize', '31',
    '--gt_radiation_raw_blend', '0.2',
    '--legacy_radiation_from_decom', '0',
    '--texture_domain_norm_enabled', '0',
    '--texture_domain_target_std', '0.08',
    '--texture_frontend_enabled', '0',
    '--texture_frontend_hidden', '24',
    '--texture_frontend_orientations', '8',
    '--texture_frontend_wavelet_levels', '2',
    '--texture_frontend_gate_init', '0.1',
    '--fusion_low_flops', '1',
    '--fusion_shared_weight', '1',
    '--fusion_residual_steps', '1',
    '--fusion_lowres_refiners', '0',
    '--fusion_separable_convs', '0',
    '--fusion_compact_post_refiner', '0',
    '--fusion_local_psnr_calibrator', '0',
    '--fusion_efficient_v16', '0',
    '--fusion_only_finetune', '0',
    '--fusion_only_zero_aux', '0',
    '--post_refiner_only_finetune', '0',
    '--fusion_calibrator_finetune', '0',
    '--fusion_teacher_distill', '0',
    '--fusion_teacher_path', './checkpoints/_unified_weights/current_best_full/best_eval_net_fusion_model.pth',
    '--lambda_fusion_teacher', '0.0',
    '--lambda_fusion_teacher_grad', '0.0',
    '--gan_mode', 'lsgan',
    '--gan_fixed_scale', '0.0',
    '--gan_stage1_scale', '0.0',
    '--gan_stage2_scale', '0.0',
    '--lambda_Gg', '0.0',
    '--lambda_D', '0.0',
    '--training_stage', 'joint',
    '--stage1_epochs', '0',
    '--texture_bootstrap_epochs', '0',
    '--texture_bootstrap_fusion_scale', '0.2',
    '--fusion_stage1_scale', '1.0',
    '--fusion_warmup_epochs', '1',
    '--texture_residual_mode', '1',
    '--texture_residual_gain', '0.16',
    '--texture_residual_max_delta', '0.35',
    '--texture_residual_gain_per_step', '0',
    '--texture_residual_steps', '2',
    '--texture_residual_step_scale', '1.0',
    '--texture_residual_step_scales', '1.00,1.00',
    '--texture_residual_step_loss_weights', '1.00,1.00',
    '--texture_residual_ir_focus_gain', '0.18',
    '--texture_ir_skip', '0.08',
    '--texture_mask_floor', '0.15',
    '--texture_blur_ksize', '9',
    '--ir_texture_prompt_strength', '0.5',
    '--ir_texture_prompt_bootstrap_strength', '0.65',
    '--ir_texture_prompt_min_weight', '0.03',
    '--ir_texture_prompt_threshold', '0.58',
    '--ir_texture_prompt_temperature', '0.08',
    '--ir_texture_prompt_power', '1.5',
    '--texture_prompt_align_enabled', '1',
    '--texture_prompt_align_radius', '3',
    '--texture_prompt_align_window', '7',
    '--texture_prompt_align_conf_threshold', '0.15',
    '--texture_prompt_align_temperature', '0.10',
    '--texture_prompt_align_highpass', '1',
    '--texture_vis_supervision_mode', 'gradtex',
    '--texture_vis_conf_adaptive', '1',
    '--texture_vis_conf_min', '0.25',
    '--texture_vis_conf_power', '1.5',
    '--texture_vis_decay_start', '-1',
    '--texture_vis_decay_end', '-1',
    '--texture_vis_min_scale', '1.0',
    '--texture_vis_gt_consistency', '0',
    '--texture_vis_gt_consistency_temp', '0.08',
    '--lambda_L1', '2.5',
    '--lambda_charb', '0.8',
    '--lambda_ssim', '1.5',
    '--lambda_recon_full', '0.35',
    '--recon_ssim_ratio', '0.35',
    '--lambda_texture', '1.0',
    '--lambda_texture_gt', '0.75',
    '--lambda_texture_vis', '0.04',
    '--lambda_texture_step', '0.005',
    '--lambda_decom', '0.10',
    '--lambda_structure_lowfreq', '0.0',
    '--lambda_structure_underfill', '0.0',
    '--structure_underfill_ratio', '0.2',
    '--structure_warmup_epochs', '20',
    '--struct_cos_thresh', '0.85',
    '--struct_ir_grad', '0.02',
    '--struct_vis_grad', '0.05',
    '--lambda_radiation', '0.25',
    '--radiation_stage1_scale', '1.0',
    '--radiation_stage2_scale', '1.0',
    '--radiation_region_block', '16',
    '--radiation_grad_weight', '0.0',
    '--thermal_hot_threshold', '0.65',
    '--thermal_preserve_strength', '0.1',
    '--lambda_identity', '0.0',
    '--lambda_fusion_I', '1.8',
    '--lambda_fusion_l1', '2.0',
    '--lambda_fusion_ssim', '8.0',
    '--lambda_fusion_psnr', '18.0',
    '--lambda_fusion_edge', '0.0',
    '--lambda_fusion_ms', '0.0',
    '--lambda_struct_consistency', '0.0',
    '--fusion_hot_weight', '0.0',
    '--fusion_grad_weight', '0.0',
    '--fusion_hard_weight', '0.0',
    '--fusion_detach_texture_epochs', '0',
    '--grad_clip_norm_g', '1.0',
    '--grad_clip_norm_d', '0.0',
    '--gf_radius_ir', '20',
    '--gf_radius_vis', '12',
    '--gf_eps', '0.001',
    '--val_during_train', '0',
    '--val_freq', '1',
    '--val_max_images', '64',
    '--val_save_label', 'best_val',
    '--val_ssim_weight', '30.0',
    '--eval_ssim_weight', '30.0'
)

if ($env:DRY_RUN -eq '1') {
    Write-Output (($argsList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' ')
    exit 0
}

$process = Start-Process -FilePath "$envPath\python.exe" `
    -ArgumentList $argsList `
    -WorkingDirectory $codeRoot `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError $errPath `
    -WindowStyle Hidden `
    -PassThru

Write-Output "PID: $($process.Id)"
Write-Output "NAME: $experiment"
Write-Output "CODE: $codeRoot"
Write-Output "CHECKPOINTS: $ckptDir"
Write-Output "LOG: $logPath"
Write-Output "ERR: $errPath"
