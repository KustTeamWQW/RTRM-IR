import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import os
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import HACM
from . import HCCM
from models.fusion import Fusion
from models.texture_frontend import LearnableGaborWaveletFrontend

# ======== [寮曞叆鏍囧噯鐨勭涓夋柟 SSIM 搴揮 ========
try:
    import pytorch_msssim
except ImportError:
    pytorch_msssim = None


def _build_gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    gauss = gauss / gauss.sum()
    window_1d = gauss.view(1, 1, 1, window_size)
    window_2d = window_1d.transpose(-1, -2) * window_1d
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _fallback_ssim(pred, target, data_range=1.0, window_size=11, sigma=1.5):
    channels = pred.shape[1]
    window = _build_gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
    return ssim_map.mean()


class IrenhanceModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Modify dataset-specific options and rewrite default values"""
        parser.set_defaults(no_dropout=True, input_nc=1, output_nc=1, init_gain=0.02)
        if is_train:
            parser.add_argument('--stage1_epochs', type=int, default=6, help='Short texture warmup; fusion/radiation remain active with warmup scales')
            parser.add_argument('--lambda_Gg', type=float, default=0.03, help='Small adversarial loss weight for texture generator')
            parser.add_argument('--lambda_D', type=float, default=0.01, help='Small discriminator loss weight')
            parser.add_argument('--gan_fixed_scale', type=float, default=0.2, help='Small global GAN scaling factor')
            parser.add_argument('--gan_stage1_scale', type=float, default=0.00, help='GAN scale during stage-1 warmup')
            parser.add_argument('--gan_stage2_scale', type=float, default=0.00, help='GAN scale after stage-1 for texture refinement')
            parser.add_argument('--lambda_radiation', type=float, default=0.35, help='Radiation loss weight')
            parser.add_argument('--lambda_identity', type=float, default=0.0, help='Disabled by default to keep the objective compact')
            parser.add_argument('--lambda_fusion_I', type=float, default=1.8, help='Fusion loss weight')
            parser.add_argument('--model_type', type=str, default='swin', help='Generator backbone: convnext | swin | swin2 | swin2_full')
            # High-level supervision weights.
            parser.add_argument('--lambda_L1', type=float, default=2.5, help='L1 reconstruction loss weight')
            parser.add_argument('--lambda_recon_full', type=float, default=0.35, help='Full-image L1 anchor ratio inside texture reconstruction to avoid black collapse')
            parser.add_argument('--lambda_charb', type=float, default=0.8, help='Charbonnier robust loss weight')
            parser.add_argument('--lambda_ssim', type=float, default=1.5, help='SSIM structural loss weight')
            parser.add_argument('--lambda_fusion_edge', type=float, default=0.0, help='Edge-consistency weight inside structural consistency branch')
            parser.add_argument('--fusion_hot_weight', type=float, default=0.0, help='Hot-region consistency weight inside structural consistency branch')
            parser.add_argument('--fusion_grad_weight', type=float, default=0.0, help='Gradient-vector consistency weight inside structural consistency branch')
            parser.add_argument('--lambda_fusion_ms', type=float, default=0.0, help='Weight for multi-scale deep supervision on fusion intermediates')
            parser.add_argument('--lambda_struct_consistency', type=float, default=0.0, help='Weight for structural consistency branch loss')
            parser.add_argument('--structure_warmup_epochs', type=int, default=20, help='Epoch span after stage-1 to ramp structural consistency branch')
            parser.add_argument('--recon_ssim_ratio', type=float, default=0.35, help='SSIM ratio inside merged recon loss [0,1]')
            parser.add_argument('--texture_mask_floor', type=float, default=0.15, help='Base weight for low-texture regions in weighted L1')
            parser.add_argument('--texture_blur_ksize', type=int, default=9, help='Kernel size used to extract high-frequency texture from visible image')
            parser.add_argument('--thermal_preserve_strength', type=float, default=0.1, help='How strongly hot radiation regions preserve infrared brightness in recon target [0,1]')
            parser.add_argument('--thermal_hot_threshold', type=float, default=0.65, help='Normalized radiation threshold for hot-region emphasis [0,1]')
            parser.add_argument('--radiation_stage1_scale', type=float, default=0.25, help='Stage-1 radiation scale')
            parser.add_argument('--radiation_stage2_scale', type=float, default=0.8, help='Stage-2 radiation scale')
            parser.add_argument('--radiation_region_block', type=int, default=16, help='Block size for regional radiation statistics')
            parser.add_argument('--radiation_grad_weight', type=float, default=0.65, help='Gradient alpha in [0,1]; region weight is (1-alpha)')
            parser.add_argument('--texture_ir_skip', type=float, default=0.0, help='Fixed skip ratio that preserves original IR high-frequency texture in T_enhanced')
            parser.add_argument('--texture_residual_mode', type=int, default=1, help='Use decomposed infrared texture as input and let G_J learn a texture residual')
            parser.add_argument('--texture_residual_gain', type=float, default=0.35, help='Residual gain used when texture_residual_mode=1')
            parser.add_argument('--texture_residual_max_delta', type=float, default=0.0, help='If >0, bound each predicted texture enhancement delta; 0 keeps the direct network output')
            parser.add_argument('--texture_domain_norm_enabled', type=int, default=1, help='Normalize signed texture-domain tensors to a shared scale')
            parser.add_argument('--texture_domain_target_std', type=float, default=0.08, help='Target per-image std for normalized texture maps')
            parser.add_argument('--texture_residual_gain_per_step', type=int, default=0, help='If 1, apply the residual gain at every recurrent step instead of splitting it across steps')
            parser.add_argument('--texture_residual_steps', type=int, default=1, help='Shared-weight iterative residual prediction steps for texture_residual_mode')
            parser.add_argument('--texture_residual_step_scale', type=float, default=1.0, help='Spatial scale for each shared residual prediction step; <1 lowers texture FLOPs')
            parser.add_argument('--texture_residual_step_scales', type=str, default='', help='Optional comma-separated per-step scales, e.g. 1.0,0.5,0.25')
            parser.add_argument('--texture_residual_step_loss_weights', type=str, default='', help='Optional comma-separated per-step supervision weights, e.g. 0.20,0.35,1.00')
            parser.add_argument('--lambda_texture_step', type=float, default=0.0, help='Auxiliary supervision weight for intermediate residual prediction steps')
            parser.add_argument('--texture_residual_ir_focus_gain', type=float, default=0.0, help='IR-only texture-focus gain applied to predicted residuals; adds no learnable parameters')
            parser.add_argument('--grad_clip_norm_g', type=float, default=1.0, help='Global grad-norm clipping for generator-side modules; <=0 disables clipping')
            parser.add_argument('--grad_clip_norm_d', type=float, default=0.0, help='Global grad-norm clipping for discriminator; <=0 disables clipping')
            parser.add_argument(
                '--training_stage',
                type=str,
                default='joint',
                choices=['joint', 'texture', 'radiation_fusion'],
                help='Training strategy stage: joint trains all branches; texture trains texture/decomposition only; radiation_fusion freezes texture and trains radiation+fusion.',
            )
            parser.add_argument('--fusion_only_finetune', type=int, default=0, help='Train only the fusion module while keeping texture/radiation branches fixed')
            parser.add_argument('--fusion_only_zero_aux', type=int, default=1, help='When fusion_only_finetune=1, backpropagate only the final fusion loss')
            parser.add_argument('--post_refiner_only_finetune', type=int, default=0, help='Freeze all pretrained modules and train only the additive post fusion refiner')
            parser.add_argument('--fusion_calibrator_finetune', type=int, default=0, help='Freeze pretrained branches and train only lightweight fusion calibration scalars/refiners')
            parser.add_argument('--fusion_local_psnr_calibrator', type=int, default=0, help='Enable lightweight local gain/bias/residual calibrator in fusion_model')
            parser.add_argument('--fusion_low_flops', type=int, default=0, help='Enable low-FLOPs fusion path with half-resolution heavy branches and lightweight full-resolution compensation')
            parser.add_argument('--fusion_efficient_v16', type=int, default=0, help='Enable efficient residual fusion v16 path for low-FLOPs full-resolution detail recovery')
            parser.add_argument('--fusion_shared_weight', type=int, default=0, help='Share AG/LC fusion-weight prediction by using one learned logit and its complement')
            parser.add_argument('--fusion_residual_steps', type=int, default=1, help='Number of independent fusion residual prediction passes')
            parser.add_argument('--fusion_lowres_refiners', type=int, default=0, help='Run only the heavy restoration/post refiners at half resolution with lightweight HR correction')
            parser.add_argument('--fusion_separable_convs', type=int, default=0, help='Use depthwise-separable blocks in selected fusion decision/calibration modules')
            parser.add_argument('--fusion_compact_post_refiner', type=int, default=0, help='Use a smaller additive post-restoration refiner while keeping the main restoration refiner unchanged')
            parser.add_argument('--fusion_teacher_distill', type=int, default=0, help='Use frozen full-resolution fusion checkpoint as teacher for low-FLOPs fusion recovery')
            parser.add_argument('--fusion_teacher_path', type=str, default='./checkpoints/_unified_weights/current_best_full/best_eval_net_fusion_model.pth', help='Path to frozen fusion teacher checkpoint')
            parser.add_argument('--lambda_fusion_teacher', type=float, default=0.0, help='Teacher output distillation weight for fusion fine-tuning')
            parser.add_argument('--lambda_fusion_teacher_grad', type=float, default=0.0, help='Teacher gradient-map distillation weight for fusion fine-tuning')
            # Decom-Net parameters
            parser.add_argument('--decom_enabled', type=int, default=1, help='Enable USTDFuse-style DecomNet')
            parser.add_argument(
                '--decom_mode',
                type=str,
                default='legacy',
                choices=['legacy', 'ir_wavelet_ref', 'ir_guided_ref', 'ir_radiation_texture_ref'],
                help='legacy uses the old DecomNet split; ir_radiation_texture_ref learns raw-IR radiation/texture and uses fixed smooth-radiation GT references.',
            )
            parser.add_argument('--decom_base_ch', type=int, default=32, help='Base channels in DecomNet')
            parser.add_argument('--decom_guided_ksize', type=int, default=15, help='Kernel size for fixed guided-filter decomposition references')
            parser.add_argument('--decom_guided_eps', type=float, default=0.01, help='Regularization eps for fixed guided-filter decomposition references')
            parser.add_argument('--decom_radiation_ksize', type=int, default=31, help='Kernel size for fixed smooth radiation decomposition references')
            parser.add_argument('--decom_radiation_iters', type=int, default=2, help='Number of smoothing passes for fixed radiation decomposition references')
            parser.add_argument('--ir_radiation_raw_blend', type=float, default=0.0, help='Raw-IR blend ratio in learned IR radiation anchor')
            parser.add_argument('--ir_radiation_scale_init', type=float, default=-2.0, help='Initial logit for learned IR radiation correction scale')
            parser.add_argument('--gt_radiation_ksize', type=int, default=7, help='Kernel size for light GT radiation/texture reference split')
            parser.add_argument('--gt_radiation_raw_blend', type=float, default=0.65, help='Original GT blend ratio in light GT radiation reference')
            parser.add_argument('--legacy_radiation_from_decom', type=int, default=0, help='In legacy DecomNet mode, feed learned S_ir to the radiation branch instead of dataset infrared-radiation')
            parser.add_argument('--lambda_decom', type=float, default=0.2, help='Lightweight decomposition regularization weight')
            # Texture frontend (optional, for raw infrared texture enhancement)
            parser.add_argument('--texture_frontend_enabled', type=int, default=1, help='Enable learnable Gabor+Wavelet texture frontend before G_J')
            parser.add_argument('--texture_frontend_hidden', type=int, default=24, help='Hidden channels in learnable texture frontend')
            parser.add_argument('--texture_frontend_orientations', type=int, default=8, help='Gabor orientations in learnable texture frontend')
            parser.add_argument('--texture_frontend_wavelet_levels', type=int, default=2, help='Haar high-frequency levels in learnable texture frontend')
            parser.add_argument('--texture_frontend_gate_init', type=float, default=0.1, help='Initial residual gate for texture frontend')
            parser.add_argument('--lambda_texture_gt', type=float, default=1.0, help='Primary GT-derived texture supervision weight')
            parser.add_argument('--lambda_texture_vis', type=float, default=0.35, help='Selected visible-texture co-supervision weight')
            parser.add_argument('--texture_vis_decay_start', type=int, default=-1, help='Epoch to start decaying visible residual texture supervision; <0 disables decay')
            parser.add_argument('--texture_vis_decay_end', type=int, default=-1, help='Epoch to end visible residual texture supervision decay')
            parser.add_argument('--texture_vis_min_scale', type=float, default=1.0, help='Minimum visible residual supervision scale after decay')
            parser.add_argument('--texture_vis_gt_consistency', type=int, default=0, help='Gate visible residual supervision by consistency with GT residual during training')
            parser.add_argument('--texture_vis_gt_consistency_temp', type=float, default=0.08, help='Temperature for GT-visible residual consistency gate')
            parser.add_argument('--texture_vis_conf_adaptive', type=int, default=0, help='Scale visible residual supervision by GT-visible residual NCC confidence during training')
            parser.add_argument('--texture_vis_conf_min', type=float, default=0.25, help='Minimum NCC confidence used by adaptive visible residual supervision')
            parser.add_argument('--texture_vis_conf_power', type=float, default=1.5, help='Power applied to NCC confidence for adaptive visible residual supervision')
            parser.add_argument('--texture_vis_supervision_mode', type=str, default='pixel', choices=['pixel', 'stat', 'gradtex', 'signed'], help='Visible residual supervision: pixel L1, texture statistics, gradient/laplacian texture alignment, or signed residual alignment')
            parser.add_argument('--texture_bootstrap_epochs', type=int, default=0, help='Extra early epochs that prioritize texture learning before full-strength fusion')
            parser.add_argument('--texture_bootstrap_fusion_scale', type=float, default=0.20, help='Fusion supervision scale during texture bootstrap epochs')
            parser.add_argument('--ir_texture_prompt_strength', type=float, default=0.50, help='How strongly train-time visible reference can modify IR texture targets')
            parser.add_argument('--ir_texture_prompt_bootstrap_strength', type=float, default=0.65, help='Prompt strength during texture bootstrap')
            parser.add_argument('--ir_texture_prompt_min_weight', type=float, default=0.03, help='Small lower prompt weight in IR-selected textured regions')
            parser.add_argument('--ir_texture_prompt_threshold', type=float, default=0.58, help='IR-derived score threshold for accepting visible texture prompts')
            parser.add_argument('--ir_texture_prompt_temperature', type=float, default=0.08, help='Softness of the IR-derived prompt selector')
            parser.add_argument('--ir_texture_prompt_power', type=float, default=1.5, help='Sharpens prompt weights after thresholding')
            parser.add_argument('--texture_prompt_align_enabled', type=int, default=0, help='Use train-time local texture cosine gating before visible texture prompting')
            parser.add_argument('--texture_prompt_align_radius', type=int, default=3, help='Local shift radius for NCC confidence used by visible texture weighting')
            parser.add_argument('--texture_prompt_align_window', type=int, default=7, help='Local window size for IR/visible texture cosine gating and NCC confidence')
            parser.add_argument('--texture_prompt_align_conf_threshold', type=float, default=0.15, help='Cosine/NCC confidence threshold for trusting visible texture prompts')
            parser.add_argument('--texture_prompt_align_temperature', type=float, default=0.10, help='Softness of the texture prompt confidence gate')
            parser.add_argument('--texture_prompt_align_highpass', type=int, default=1, help='Apply an extra high-pass filter before texture prompt confidence calculation')
            parser.add_argument('--lambda_structure_lowfreq', type=float, default=0.12, help='Low-frequency anchor for S_ir/S_vis to prevent structure branch collapse')
            parser.add_argument('--lambda_structure_underfill', type=float, default=0.0, help='Small one-sided penalty when S_ir/S_vis falls far below the low-frequency base')
            parser.add_argument('--structure_underfill_ratio', type=float, default=0.20, help='Fraction of low-frequency base that structure maps should softly keep')
            parser.add_argument('--fusion_detach_texture_epochs', type=int, default=0, help='Warmup epochs where fusion loss sees refine_t but does not backpropagate into the texture branch')
            # Main texture loss weights.
            parser.add_argument('--lambda_texture', type=float, default=1.0, help='Legacy texture multiplier used only by guided-filter fallback mode')
            # Structure consistency mask thresholds
            parser.add_argument('--struct_cos_thresh', type=float, default=0.85, help='Gradient direction cosine threshold')
            parser.add_argument('--struct_vis_grad', type=float, default=0.05, help='Visible base gradient threshold')
            parser.add_argument('--struct_ir_grad', type=float, default=0.02, help='Infrared base gradient threshold (lower)')
            # Legacy guided-filter parameters kept for backward compatibility.
            parser.add_argument('--gf_radius_ir', type=int, default=20, help='Guided filter radius for infrared base layer')
            parser.add_argument('--gf_radius_vis', type=int, default=12, help='Guided filter radius for visible base layer')
            parser.add_argument('--gf_eps', type=float, default=1e-3, help='Guided filter regularization')
            # Fusion loss weights.
            parser.add_argument('--lambda_fusion_l1', type=float, default=1.2, help='Weight for L1 fidelity in fusion loss')
            parser.add_argument('--lambda_fusion_ssim', type=float, default=6.0, help='Weight for SSIM in fusion loss')
            parser.add_argument('--lambda_fusion_psnr', type=float, default=8.5, help='Weight for PSNR (MSE) in fusion loss')
            parser.add_argument('--fusion_hard_weight', type=float, default=0.0, help='[LEGACY/ignored] Extra fusion loss multiplier')
            parser.add_argument('--fusion_stage1_scale', type=float, default=0.35, help='Fusion supervision scale during short texture warmup')
            parser.add_argument('--fusion_warmup_epochs', type=int, default=5, help='Epochs used to ramp fusion supervision after stage-1')
        else:
            pass

        return parser

    def __init__(self, opt):
        """Initialize the model"""
        BaseModel.__init__(self, opt)

        self.decom_enabled = bool(int(getattr(opt, 'decom_enabled', 1)))
        self.decom_mode = str(getattr(opt, 'decom_mode', 'legacy')).lower()
        self.texture_frontend_enabled = bool(int(getattr(opt, 'texture_frontend_enabled', 0)))
        self.training_stage = str(getattr(opt, 'training_stage', 'joint')).lower()
        self.texture_stage = self.isTrain and self.training_stage == 'texture'
        self.radiation_fusion_stage = self.isTrain and self.training_stage == 'radiation_fusion'
        self.fusion_only_finetune = self.isTrain and bool(int(getattr(opt, 'fusion_only_finetune', 0)))
        self.post_refiner_only_finetune = self.isTrain and bool(int(getattr(opt, 'post_refiner_only_finetune', 0)))
        self.fusion_calibrator_finetune = self.isTrain and bool(int(getattr(opt, 'fusion_calibrator_finetune', 0)))
        self.loss_names = [
            'recon', 'texture_gt', 'texture_vis', 'texture_step', 'ir_hf', 'detail_gt',
            'radiation',
            'fusion', 'fusion_charb', 'fusion_l1', 'fusion_psnr', 'fusion_grad'
        ]
        if self.decom_enabled:
            self.loss_names.append('decom')
        self.loss_names.extend([
            'tex_Tir_mean', 'tex_Tir_std',
            'tex_Rtex_mean', 'tex_Rtex_std',
            'tex_Tenh_mean', 'tex_Tenh_std',
            'tex_Tgt_mean', 'tex_Tgt_std',
            'tex_Tvis_mean', 'tex_Tvis_std',
        ])

        self.visual_names = [
            'paper_01_input_ir_image',
            'paper_02_ir_input_radiation_Sir',
            'paper_03_train_visible_reference',
            'paper_04_train_gt_image',
            'paper_05_gt_radiation_Sgt',
            'paper_06_radiation_stage1_ATF2D',
            'paper_07_radiation_output_TCM',
            'paper_08_ir_input_texture_Tir_visualized',
            'paper_09_ir_texture_internal_structure',
            'paper_10_structure_visible_S_vis',
            'paper_11_texture_visible_T_vis',
            'paper_12_ir_texture_focus_M',
            'paper_13_visible_supervision_select_M',
            'paper_14_texture_enhanced_T_enhanced',
            'paper_15_texture_reconstruction_Sir_plus_Tenhanced',
            'paper_16_final_fused_output',
            'paper_17_texture_gt_Tgt',
            'fused_I',
        ] if self.isTrain else [
            'paper_01_input_ir_image',
            'paper_02_ir_input_radiation_Sir',
            'paper_06_radiation_stage1_ATF2D',
            'paper_07_radiation_output_TCM',
            'paper_08_ir_input_texture_Tir_visualized',
            'paper_09_ir_texture_internal_structure',
            'paper_12_ir_texture_focus_M',
            'paper_14_texture_enhanced_T_enhanced',
            'paper_15_texture_reconstruction_Sir_plus_Tenhanced',
            'paper_16_final_fused_output',
            'fused_I',
        ]

        self.model_names = ['ATF2D', 'TCMNet', 'G_J', 'D', 'fusion_model'] if self.isTrain \
                 else ['ATF2D', 'TCMNet', 'G_J', 'fusion_model']
        if self.decom_enabled:
            self.model_names.append('decom_net')
            if self._use_learned_ir_decom():
                self.model_names.append('visible_decom_net')
        if self.texture_frontend_enabled:
            self.model_names.append('texture_frontend')


        # Initialize texture generator.
        if opt.G_J == 'modern_unet':
            self.G_J = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.G_J, opt.norm,
                                   not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                   model_type=opt.model_type,
                                   texture_vq_enabled=False)
        else:
            self.G_J = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.G_J, opt.norm,
                                   not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if bool(int(getattr(opt, 'texture_residual_mode', 0))):
            self._configure_texture_residual_generator()

        # Decom-Net for structure-texture decomposition
        if self.decom_enabled:
            from models.decom_net import (
                DecomNet,
                IRGuidedSeparationNet,
                IRRadiationTextureSeparationNet,
                IRWaveletSeparationNet,
            )
            if self.decom_mode == 'ir_wavelet_ref':
                self.decom_net = IRWaveletSeparationNet(
                    in_channels=opt.input_nc,
                    base_ch=int(getattr(opt, 'decom_base_ch', 32)),
                )
                self.visible_decom_net = DecomNet(
                    in_channels=opt.input_nc,
                    base_ch=int(getattr(opt, 'decom_base_ch', 32)),
                )
                self.visible_decom_net.to(self.device)
            elif self.decom_mode == 'ir_guided_ref':
                self.decom_net = IRGuidedSeparationNet(
                    in_channels=opt.input_nc,
                    base_ch=int(getattr(opt, 'decom_base_ch', 32)),
                    kernel_size=int(getattr(opt, 'decom_guided_ksize', 15)),
                    eps=float(getattr(opt, 'decom_guided_eps', 0.01)),
                )
                self.visible_decom_net = DecomNet(
                    in_channels=opt.input_nc,
                    base_ch=int(getattr(opt, 'decom_base_ch', 32)),
                )
                self.visible_decom_net.to(self.device)
            elif self.decom_mode == 'ir_radiation_texture_ref':
                self.decom_net = IRRadiationTextureSeparationNet(
                    in_channels=opt.input_nc,
                    base_ch=int(getattr(opt, 'decom_base_ch', 32)),
                    kernel_size=int(getattr(opt, 'decom_radiation_ksize', 31)),
                    smooth_iters=int(getattr(opt, 'decom_radiation_iters', 2)),
                    raw_blend=float(getattr(opt, 'ir_radiation_raw_blend', 0.0)),
                    scale_init=float(getattr(opt, 'ir_radiation_scale_init', -2.0)),
                )
                self.visible_decom_net = DecomNet(
                    in_channels=opt.input_nc,
                    base_ch=int(getattr(opt, 'decom_base_ch', 32)),
                )
                self.visible_decom_net.to(self.device)
            else:
                self.decom_net = DecomNet(in_channels=opt.input_nc, base_ch=int(getattr(opt, 'decom_base_ch', 32)))
                self.visible_decom_net = self.decom_net
            self.decom_net.to(self.device)

        # Optional: lightweight texture frontend for T_ir enhancement
        if self.texture_frontend_enabled:
            self.texture_frontend = LearnableGaborWaveletFrontend(
                in_channels=opt.input_nc,
                hidden_channels=int(getattr(opt, 'texture_frontend_hidden', 16)),
                orientations=int(getattr(opt, 'texture_frontend_orientations', 6)),
                wavelet_levels=int(getattr(opt, 'texture_frontend_wavelet_levels', 2)),
                gate_init=float(getattr(opt, 'texture_frontend_gate_init', 0.1)),
            ).to(self.device)

        self.ATF2D = HACM.ATF2D(input_channels=opt.input_nc).to(self.device)
        self.TCMNet = HCCM.TCMNetwork(in_channels=opt.input_nc).to(self.device)
        self.fusion_model = Fusion(
            use_local_psnr_calibrator=bool(int(getattr(opt, 'fusion_local_psnr_calibrator', 0))),
            use_low_flops=bool(int(getattr(opt, 'fusion_low_flops', 0))),
            use_efficient_v16=bool(int(getattr(opt, 'fusion_efficient_v16', 0))),
            use_shared_weight=bool(int(getattr(opt, 'fusion_shared_weight', 0))),
            residual_steps=int(getattr(opt, 'fusion_residual_steps', 1)),
            use_lowres_refiners=bool(int(getattr(opt, 'fusion_lowres_refiners', 0))),
            use_separable_convs=bool(int(getattr(opt, 'fusion_separable_convs', 0))),
            use_compact_post_refiner=bool(int(getattr(opt, 'fusion_compact_post_refiner', 0)))
        ).to(self.device)
        self.fusion_teacher = None
        if self.isTrain and bool(int(getattr(opt, 'fusion_teacher_distill', 0))):
            self.fusion_teacher = Fusion(
                use_local_psnr_calibrator=bool(int(getattr(opt, 'fusion_local_psnr_calibrator', 0))),
                use_low_flops=False,
                use_efficient_v16=False,
                use_shared_weight=False,
                use_lowres_refiners=False,
                use_separable_convs=False,
                use_compact_post_refiner=False
            ).to(self.device)
            self._load_fusion_teacher(getattr(opt, 'fusion_teacher_path', ''))
            self.fusion_teacher.eval()
            self.set_requires_grad(self.fusion_teacher, False)

        if self.isTrain:
            self.D = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                     opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        self.laplace_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert (opt.input_nc == opt.output_nc)
            
            self.fake_I_pool = ImagePool(opt.pool_size)
            self.fake_J_pool = ImagePool(opt.pool_size)
            
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            
            if self.texture_stage:
                generator_params = self._freeze_for_texture_stage()
                print('[TrainingStage:texture] Training texture/decomposition modules only; radiation and fusion modules are frozen.')
            elif self.radiation_fusion_stage:
                generator_params = self._freeze_for_radiation_fusion_stage()
                print('[TrainingStage:radiation_fusion] Training radiation and fusion modules only; texture/decomposition modules are frozen.')
            elif self.fusion_calibrator_finetune:
                generator_params = self._freeze_for_fusion_calibrator()
                print('[FusionCalibrator] Training lightweight fusion calibration scalars, calibrators, and post refiner; pretrained branches are frozen.')
            elif self.post_refiner_only_finetune:
                generator_params = self._freeze_for_post_refiner_only()
                print('[PostRefinerOnly] Training only post_refine_gain and post_restoration_refiner; all pretrained modules are frozen.')
            elif self.fusion_only_finetune:
                self._freeze_non_fusion_modules()
                generator_params = self.fusion_model.parameters()
                print('[FusionOnly] Fine-tuning fusion_model only; texture/radiation/decomposition modules are frozen.')
            else:
                generator_params = itertools.chain(
                    self.ATF2D.parameters(),
                    self.TCMNet.parameters(),
                    self.G_J.parameters(),
                    self.fusion_model.parameters()
                )
                if self.decom_enabled:
                    generator_params = itertools.chain(generator_params, self.decom_net.parameters())
                    if self._use_learned_ir_decom():
                        generator_params = itertools.chain(generator_params, self.visible_decom_net.parameters())
                if self.texture_frontend_enabled:
                    generator_params = itertools.chain(generator_params, self.texture_frontend.parameters())

            self.optimizer_G = torch.optim.Adam(
                generator_params,
                lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizer_D = torch.optim.Adam(
                self.D.parameters(),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.current_epoch = 0
        self.total_train_epochs = int(getattr(self.opt, 'niter', 0)) + int(getattr(self.opt, 'niter_decay', 0))
        self.stage1_epochs_effective = int(getattr(self.opt, 'stage1_epochs', 20))
        self.stage2_texture_finetune_scale = 1.0
        self.texture_prior_condition = None
        self.fusion_aux = {}
        self.loss_fusion_ms = torch.tensor(0.0, device=self.device)
        self.loss_struct = torch.tensor(0.0, device=self.device)
        self.loss_texture_ag = torch.tensor(0.0, device=self.device)
        self.loss_texture_gt = torch.tensor(0.0, device=self.device)
        self.loss_texture_vis = torch.tensor(0.0, device=self.device)
        self.loss_texture_distill = torch.tensor(0.0, device=self.device)
        self.loss_texture_ms_grad = torch.tensor(0.0, device=self.device)
        self.loss_texture_contrast = torch.tensor(0.0, device=self.device)
        self.loss_ir_hf = torch.tensor(0.0, device=self.device)
        self.loss_detail_gt = torch.tensor(0.0, device=self.device)
        self.loss_texture_step = torch.tensor(0.0, device=self.device)
        self.loss_tex_Tir_mean = torch.tensor(0.0, device=self.device)
        self.loss_tex_Tir_std = torch.tensor(0.0, device=self.device)
        self.loss_tex_Rtex_mean = torch.tensor(0.0, device=self.device)
        self.loss_tex_Rtex_std = torch.tensor(0.0, device=self.device)
        self.loss_tex_Tenh_mean = torch.tensor(0.0, device=self.device)
        self.loss_tex_Tenh_std = torch.tensor(0.0, device=self.device)
        self.loss_tex_Tgt_mean = torch.tensor(0.0, device=self.device)
        self.loss_tex_Tgt_std = torch.tensor(0.0, device=self.device)
        self.loss_tex_Tvis_mean = torch.tensor(0.0, device=self.device)
        self.loss_tex_Tvis_std = torch.tensor(0.0, device=self.device)
        self.loss_fusion_charb = torch.tensor(0.0, device=self.device)
        self.loss_fusion_l1 = torch.tensor(0.0, device=self.device)
        self.loss_fusion_structure = torch.tensor(0.0, device=self.device)
        self.loss_fusion_lap = torch.tensor(0.0, device=self.device)
        self.loss_fusion_lowfreq = torch.tensor(0.0, device=self.device)
        self.loss_fusion_stats = torch.tensor(0.0, device=self.device)
        self.loss_fusion_residual = torch.tensor(0.0, device=self.device)
        self.loss_fusion_brightness = torch.tensor(0.0, device=self.device)
        self.loss_fusion_teacher = torch.tensor(0.0, device=self.device)
        self.texture_mask_M = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.texture_weight_W = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.texture_prompt_W = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.S_ir_vis = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.S_vis_struct_vis = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.T_ir_vis = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.T_vis_vis = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.T_enhanced_vis = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_ir_input = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_ir_radiation_input = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_visible_reference = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_gt_target = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_gt_radiation = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_radiation_stage1 = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_radiation_enhanced = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_ir_structure = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_ir_texture = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_visible_structure = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_visible_texture = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_texture_residual = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_enhanced_texture = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_ir_texture_focus_M = torch.ones(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_visible_select_M = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_texture_reconstruction = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_fused_output = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_01_input_ir_image = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_02_ir_input_radiation_Sir = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_03_train_visible_reference = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_04_train_gt_image = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_05_gt_radiation_Sgt = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_06_radiation_stage1_ATF2D = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_07_radiation_output_TCM = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_08_ir_input_texture_Tir_visualized = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_09_ir_texture_internal_structure = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_10_structure_visible_S_vis = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_11_texture_visible_T_vis = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_12_ir_texture_focus_M = torch.ones(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_13_visible_supervision_select_M = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_14_texture_residual_Rtex = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_15_enhanced_texture_Tenhanced = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_14_texture_enhanced_T_enhanced = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_15_texture_reconstruction_Sir_plus_Tenhanced = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_16_final_fused_output = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)
        self.paper_17_texture_gt_Tgt = torch.zeros(1, opt.input_nc, 1, 1, device=self.device)

        # Runtime-configurable hyper-parameters from CLI/options.
        self.lambda_D = self.opt.lambda_D
        self.gan_scale = self.opt.gan_fixed_scale
        self.gan_stage1_scale = max(0.0, getattr(self.opt, 'gan_stage1_scale', 0.0))
        self.gan_stage2_scale = max(0.0, getattr(self.opt, 'gan_stage2_scale', 0.0))
        self.radiation_stage1_scale = self.opt.radiation_stage1_scale
        self.radiation_stage2_scale = self.opt.radiation_stage2_scale
        self.radiation_region_block = self.opt.radiation_region_block
        # Use a single knob to reduce coupling: grad alpha in [0,1], region weight is (1-alpha).
        self.radiation_grad_alpha = max(0.0, min(1.0, self.opt.radiation_grad_weight))
        self.lambda_fusion_edge = max(0.0, float(getattr(self.opt, 'lambda_fusion_edge', 0.0)))
        self.fusion_hot_weight = max(0.0, float(getattr(self.opt, 'fusion_hot_weight', 0.0)))
        self.fusion_grad_weight = max(0.0, float(getattr(self.opt, 'fusion_grad_weight', 0.0)))
        self.lambda_fusion_ms = max(0.0, float(getattr(self.opt, 'lambda_fusion_ms', 0.0)))
        self.lambda_struct_consistency = max(0.0, float(getattr(self.opt, 'lambda_struct_consistency', 0.0)))
        self.structure_warmup_epochs = max(1, int(getattr(self.opt, 'structure_warmup_epochs', 20)))
        self.fusion_stage1_scale = max(0.0, min(1.0, float(getattr(self.opt, 'fusion_stage1_scale', 0.35))))
        self.fusion_warmup_epochs = max(1, int(getattr(self.opt, 'fusion_warmup_epochs', 5)))
        self.grad_clip_norm_g = float(getattr(self.opt, 'grad_clip_norm_g', 1.0))
        self.grad_clip_norm_d = float(getattr(self.opt, 'grad_clip_norm_d', 0.0))
        model_type = str(getattr(self.opt, 'model_type', '')).lower()
        self.swin_texture_clip_norm = self.grad_clip_norm_g

    def _load_fusion_teacher(self, checkpoint_path):
        if self.fusion_teacher is None:
            return
        if not checkpoint_path:
            raise RuntimeError('fusion_teacher_distill=1 but fusion_teacher_path is empty')
        checkpoint_path = os.path.expanduser(checkpoint_path)
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError('Fusion teacher checkpoint not found: %s' % checkpoint_path)

        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if isinstance(state, dict):
            cleaned = {}
            for key, value in state.items():
                if key.startswith('module.'):
                    key = key[len('module.'):]
                cleaned[key] = value
            state = cleaned
        missing, unexpected = self.fusion_teacher.load_state_dict(state, strict=False)
        print('[FusionTeacher] Loaded frozen teacher from %s (missing=%d, unexpected=%d)' % (
            checkpoint_path, len(missing), len(unexpected)
        ))

    def _fusion_teacher_loss(self):
        lambda_teacher = float(getattr(self.opt, 'lambda_fusion_teacher', 0.0))
        lambda_teacher_grad = float(getattr(self.opt, 'lambda_fusion_teacher_grad', 0.0))
        if self.fusion_teacher is None or (lambda_teacher <= 0.0 and lambda_teacher_grad <= 0.0):
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            teacher_raw = self.fusion_teacher(
                self.refine_r.detach(),
                self.refine_t.detach(),
                texture_prior_cond=None if self.texture_prior_condition is None else self.texture_prior_condition.detach(),
                return_aux=False,
            )
            teacher_fused = torch.clamp(teacher_raw, 0.0, 1.0)

        student = torch.clamp(self.fused_I, 0.0, 1.0)
        loss = lambda_teacher * self._charbonnier(student, teacher_fused)
        if lambda_teacher_grad > 0.0:
            student_grad = self._normalized_gradient_map(student)
            teacher_grad = self._normalized_gradient_map(teacher_fused)
            loss = loss + lambda_teacher_grad * self._charbonnier(student_grad, teacher_grad)
        return loss

    @staticmethod
    def _set_module_trainable(module, trainable):
        for param in module.parameters():
            param.requires_grad = trainable

    @staticmethod
    def _collect_trainable_params(modules, stage_name):
        params = []
        for module in modules:
            params.extend([param for param in module.parameters() if param.requires_grad])
        if not params:
            raise RuntimeError('%s found no trainable parameters' % stage_name)
        return params

    def _texture_modules(self):
        modules = [self.G_J]
        if self.decom_enabled:
            modules.append(self.decom_net)
            if self._use_learned_ir_decom():
                modules.append(self.visible_decom_net)
        if self.texture_frontend_enabled:
            modules.append(self.texture_frontend)
        return modules

    def _radiation_fusion_modules(self):
        return [self.ATF2D, self.TCMNet, self.fusion_model]

    def _freeze_for_texture_stage(self):
        for module in self._radiation_fusion_modules():
            self._set_module_trainable(module, False)
        texture_modules = self._texture_modules()
        for module in texture_modules:
            self._set_module_trainable(module, True)
        return self._collect_trainable_params(texture_modules, 'training_stage=texture')

    def _freeze_for_radiation_fusion_stage(self):
        for module in self._texture_modules():
            self._set_module_trainable(module, False)
        train_modules = self._radiation_fusion_modules()
        for module in train_modules:
            self._set_module_trainable(module, True)
        return self._collect_trainable_params(train_modules, 'training_stage=radiation_fusion')

    def _freeze_non_fusion_modules(self):
        frozen_modules = [self.ATF2D, self.TCMNet, self.G_J]
        if self.decom_enabled:
            frozen_modules.append(self.decom_net)
        if self.texture_frontend_enabled:
            frozen_modules.append(self.texture_frontend)

        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False
        for param in self.fusion_model.parameters():
            param.requires_grad = True

    def _freeze_for_post_refiner_only(self):
        self._freeze_non_fusion_modules()
        trainable_params = []
        trainable_names = ('post_refine_gain', 'post_restoration_refiner.')
        for name, param in self.fusion_model.named_parameters():
            is_trainable = name == trainable_names[0] or name.startswith(trainable_names[1])
            param.requires_grad = is_trainable
            if is_trainable:
                trainable_params.append(param)
        if not trainable_params:
            raise RuntimeError('post_refiner_only_finetune found no trainable post refiner parameters')
        return trainable_params

    def _freeze_for_fusion_calibrator(self):
        self._freeze_non_fusion_modules()
        trainable_params = []
        scalar_names = {
            'detail_gain',
            'calibration_gain',
            'final_refine_gain',
            'supervised_residual_gain',
            'restoration_gain',
            'post_refine_gain',
            'local_psnr_gain',
        }
        module_prefixes = (
            'efficient_v16.',
            'weight_edge_corrector.',
            'detail_edge_corrector.',
            'output_calibrator.',
            'final_refiner.',
            'global_calibrator.',
            'supervised_residual_corrector.',
            'restoration_refiner.',
            'post_restoration_refiner.',
            'local_psnr_calibrator.',
        )
        if bool(int(getattr(self.opt, 'fusion_shared_weight', 0))):
            module_prefixes = module_prefixes + (
                'ag_net.',
                'joint_ag_head.',
                'joint_lc_head.',
            )
        for name, param in self.fusion_model.named_parameters():
            is_trainable = name in scalar_names or name.startswith(module_prefixes)
            param.requires_grad = is_trainable
            if is_trainable:
                trainable_params.append(param)
        if not trainable_params:
            raise RuntimeError('fusion_calibrator_finetune found no trainable calibration parameters')
        return trainable_params

    def _fusion_hard_sample_multiplier(self, target):
        max_extra = max(0.0, float(getattr(self.opt, 'fusion_hard_weight', 0.0)))
        if max_extra <= 0.0:
            return torch.tensor(1.0, device=target.device)
        return torch.tensor(1.0, device=target.device)

    def _radiation_structure_loss(self, pred, target):
        channels = pred.shape[1]
        sobel_x = self.sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(channels, 1, 1, 1)

        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=channels)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=channels)
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=channels)
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=channels)

        pred_grad_mag = torch.sqrt(pred_grad_x * pred_grad_x + pred_grad_y * pred_grad_y + 1e-6)
        target_grad_mag = torch.sqrt(target_grad_x * target_grad_x + target_grad_y * target_grad_y + 1e-6)

        # Radiation intensity constraint from regional statistics.
        # Divide image into small regions via block average pooling (e.g., 128x128 -> 8x8 blocks when block=16).
        block = max(1, int(self.radiation_region_block))
        pred_region_mean = F.avg_pool2d(pred, kernel_size=block, stride=block)
        target_region_mean = F.avg_pool2d(target, kernel_size=block, stride=block)

        # Add regional variance statistics to preserve thermal distribution in each local region.
        pred_region_sq_mean = F.avg_pool2d(pred * pred, kernel_size=block, stride=block)
        target_region_sq_mean = F.avg_pool2d(target * target, kernel_size=block, stride=block)
        pred_region_std = torch.sqrt(torch.clamp(pred_region_sq_mean - pred_region_mean * pred_region_mean, min=1e-6))
        target_region_std = torch.sqrt(torch.clamp(target_region_sq_mean - target_region_mean * target_region_mean, min=1e-6))

        loss_region_mean = self.criterionL1(pred_region_mean, target_region_mean)
        loss_region_std = self.criterionL1(pred_region_std, target_region_std)
        loss_region = 0.7 * loss_region_mean + 0.3 * loss_region_std

        # Gradient constraint to preserve radiation structures/edges.
        loss_grad = self.criterionL1(pred_grad_mag, target_grad_mag)

        # Hot-region brightness consistency is part of radiation supervision.
        thermal_mask = self._build_thermal_mask(target)
        loss_hot_brightness = (torch.abs(pred - target) * thermal_mask).mean()

        # Expose component magnitudes for monitoring and tuning.
        self.loss_rad_region = loss_region
        self.loss_rad_grad = loss_grad

        stage_scale = self._radiation_brightness_scale()
        hot_mix = max(0.0, min(1.0, self.opt.thermal_preserve_strength))
        region_term = (1.0 - hot_mix) * loss_region + hot_mix * loss_hot_brightness
        region_weight = 1.0 - self.radiation_grad_alpha
        grad_weight = self.radiation_grad_alpha
        merged = region_weight * region_term + grad_weight * loss_grad
        return merged * stage_scale

    def _ssim_score(self, pred, target):
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        if pytorch_msssim is not None:
            return pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=True)
        return _fallback_ssim(pred, target, data_range=1.0)

    def _is_gan_enabled(self):
        if not self.isTrain:
            return False
        if self.radiation_fusion_stage:
            return False
        return self._gan_weight_scale() > 0.0

    def _gradient_magnitude(self, x):
        channels = x.shape[1]
        sobel_x = self.sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)

    def _gradient_components(self, x):
        channels = x.shape[1]
        sobel_x = self.sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)
        return grad_x, grad_y

    def _laplacian(self, x):
        channels = x.shape[1]
        kernel = self.laplace_kernel.repeat(channels, 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=channels)

    def _gradient_direction_loss(self, pred, target):
        pred_x, pred_y = self._gradient_components(pred)
        target_x, target_y = self._gradient_components(target)
        pred_mag = torch.sqrt(pred_x * pred_x + pred_y * pred_y + 1e-6)
        target_mag = torch.sqrt(target_x * target_x + target_y * target_y + 1e-6)
        cosine = (pred_x * target_x + pred_y * target_y) / (pred_mag * target_mag + 1e-6)
        edge_weight = target_mag / (target_mag.mean(dim=(2, 3), keepdim=True) + 1e-6)
        edge_weight = torch.clamp(edge_weight.detach(), min=0.25, max=3.0)
        return ((1.0 - cosine) * edge_weight).mean()

    def _average_gradient(self, x):
        return self._gradient_magnitude(x).mean(dim=(1, 2, 3))

    def _normalized_gradient_map(self, x):
        grad = self._gradient_magnitude(x)
        b = grad.shape[0]
        flat = grad.view(b, -1)
        scale = flat.mean(dim=1, keepdim=True).view(b, 1, 1, 1)
        return grad / (scale + 1e-6)

    def _masked_texture_ag_loss(self, pred_texture, ref_texture, focus_source=None, mask=None):
        pred_grad = self._normalized_gradient_map(pred_texture)
        ref_grad = self._normalized_gradient_map(ref_texture.detach())

        if focus_source is None:
            focus_source = ref_texture
        focus_grad = self._gradient_magnitude(focus_source.detach())
        b = focus_grad.shape[0]
        flat = focus_grad.view(b, -1)
        mean = flat.mean(dim=1, keepdim=True).view(b, 1, 1, 1)
        std = flat.std(dim=1, keepdim=True, unbiased=False).view(b, 1, 1, 1)
        texture_mask = torch.sigmoid((focus_grad - mean) / (std + 1e-6))

        if mask is not None:
            texture_mask = texture_mask * mask.detach()

        texture_mask = texture_mask.clamp(min=0.05, max=1.0)
        diff = torch.sqrt((pred_grad - ref_grad) ** 2 + 1e-6)
        return (diff * texture_mask).sum() / (texture_mask.sum() + 1e-6)

    def _texture_lowfreq_loss(self, texture, kernel_size=15):
        if kernel_size % 2 == 0:
            kernel_size += 1
        low = F.avg_pool2d(texture, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return low.abs().mean()

    def _texture_distill_loss(self, student_texture, teacher_texture):
        teacher_texture = teacher_texture.detach()
        loss = self._charbonnier(student_texture, teacher_texture)
        student_grad = self._normalized_gradient_map(student_texture)
        teacher_grad = self._normalized_gradient_map(teacher_texture)
        loss = loss + 0.5 * self._charbonnier(student_grad, teacher_grad)
        for scale in (2, 4):
            if min(student_texture.shape[-2:]) >= scale:
                student_down = F.avg_pool2d(student_texture, kernel_size=scale, stride=scale)
                teacher_down = F.avg_pool2d(teacher_texture, kernel_size=scale, stride=scale)
                loss = loss + 0.25 * self._charbonnier(student_down, teacher_down)
        return loss

    def _texture_multiscale_grad_loss(self, student_texture, teacher_texture):
        teacher_texture = teacher_texture.detach()
        loss = self._charbonnier(
            self._normalized_gradient_map(student_texture),
            self._normalized_gradient_map(teacher_texture),
        )
        for scale in (2, 4, 8):
            if min(student_texture.shape[-2:]) < scale:
                continue
            student_down = F.avg_pool2d(student_texture, kernel_size=scale, stride=scale)
            teacher_down = F.avg_pool2d(teacher_texture, kernel_size=scale, stride=scale)
            loss = loss + (1.0 / float(scale)) * self._charbonnier(
                self._normalized_gradient_map(student_down),
                self._normalized_gradient_map(teacher_down),
            )
        return loss

    def _texture_contrast_loss(self, student_texture, teacher_texture, kernel_size=7):
        teacher_texture = teacher_texture.detach()
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        student_mean = F.avg_pool2d(student_texture, kernel_size=kernel_size, stride=1, padding=pad)
        teacher_mean = F.avg_pool2d(teacher_texture, kernel_size=kernel_size, stride=1, padding=pad)
        student_var = F.avg_pool2d(student_texture * student_texture, kernel_size=kernel_size, stride=1, padding=pad) - student_mean * student_mean
        teacher_var = F.avg_pool2d(teacher_texture * teacher_texture, kernel_size=kernel_size, stride=1, padding=pad) - teacher_mean * teacher_mean
        student_std = torch.sqrt(torch.clamp(student_var, min=1e-6))
        teacher_std = torch.sqrt(torch.clamp(teacher_var, min=1e-6))
        return self._charbonnier(student_std, teacher_std)

    def _weighted_texture_l1(self, pred_texture, ref_texture, weight):
        weight = weight.detach()
        ref_texture = ref_texture.detach()
        diff = torch.sqrt((pred_texture - ref_texture) ** 2 + 1e-6)
        return (diff * weight).sum() / (weight.sum() + 1e-6)

    def _visible_texture_stat_loss(self, pred_residual, ref_residual, weight):
        weight = weight.detach()
        ref_residual = ref_residual.detach()
        pred_grad = self._gradient_magnitude(pred_residual)
        ref_grad = self._gradient_magnitude(ref_residual)
        pred_std = self._local_std_map(pred_residual)
        ref_std = self._local_std_map(ref_residual)
        grad_diff = torch.sqrt((pred_grad - ref_grad) ** 2 + 1e-6)
        std_diff = torch.sqrt((pred_std - ref_std) ** 2 + 1e-6)
        return ((0.65 * grad_diff + 0.35 * std_diff) * weight).sum() / (weight.sum() + 1e-6)

    def _visible_texture_gradtex_loss(self, pred_residual, ref_residual, weight):
        weight = weight.detach()
        ref_residual = ref_residual.detach()
        pred_fine = torch.abs(self._laplacian(pred_residual))
        ref_fine = torch.abs(self._laplacian(ref_residual))
        diff = torch.sqrt((pred_fine - ref_fine) ** 2 + 1e-6)
        return (diff * weight).sum() / (weight.sum() + 1e-6)

    def _visible_texture_abs_residual_loss(self, pred_residual, ref_residual, weight, scale_weights=None):
        weight = weight.detach()
        ref_residual = ref_residual.detach()
        loss = torch.tensor(0.0, device=pred_residual.device)
        total = 0.0
        default_scale_weights = (1.0, 0.65, 0.45)
        if scale_weights is None:
            scale_weights = default_scale_weights
        for idx, scale in enumerate((1, 2, 4)):
            scale_weight = float(scale_weights[min(idx, len(scale_weights) - 1)])
            if scale_weight <= 0.0:
                continue
            if min(pred_residual.shape[-2:]) < scale:
                continue
            if scale == 1:
                pred_s = pred_residual
                ref_s = ref_residual
                weight_s = weight
            else:
                pred_s = F.avg_pool2d(pred_residual, kernel_size=scale, stride=scale)
                ref_s = F.avg_pool2d(ref_residual, kernel_size=scale, stride=scale)
                weight_s = F.avg_pool2d(weight, kernel_size=scale, stride=scale)

            diff = torch.sqrt((torch.abs(pred_s) - torch.abs(ref_s)) ** 2 + 1e-6)
            loss = loss + scale_weight * (diff * weight_s).sum() / (weight_s.sum() + 1e-6)
            total += scale_weight
        return loss / max(total, 1e-6)

    def _visible_texture_prompt_grad_loss(self, pred_texture, visible_texture, weight):
        weight = weight.detach()
        visible_texture = visible_texture.detach()
        loss = torch.tensor(0.0, device=pred_texture.device)
        total = 0.0
        for scale, scale_weight in ((1, 1.0), (2, 0.65), (4, 0.45)):
            if min(pred_texture.shape[-2:]) < scale:
                continue
            if scale == 1:
                pred_s = pred_texture
                vis_s = visible_texture
                weight_s = weight
            else:
                pred_s = F.avg_pool2d(pred_texture, kernel_size=scale, stride=scale)
                vis_s = F.avg_pool2d(visible_texture, kernel_size=scale, stride=scale)
                weight_s = F.avg_pool2d(weight, kernel_size=scale, stride=scale)

            pred_gx, pred_gy = self._gradient_components(pred_s)
            vis_gx, vis_gy = self._gradient_components(vis_s)
            pred_mag = torch.sqrt(pred_gx * pred_gx + pred_gy * pred_gy + 1e-6)
            vis_mag = torch.sqrt(vis_gx * vis_gx + vis_gy * vis_gy + 1e-6)
            pred_norm = pred_mag / (F.avg_pool2d(pred_mag, kernel_size=7, stride=1, padding=3) + 1e-6)
            vis_norm = vis_mag / (F.avg_pool2d(vis_mag, kernel_size=7, stride=1, padding=3) + 1e-6)
            cosine = torch.abs((pred_gx * vis_gx + pred_gy * vis_gy) / (pred_mag * vis_mag + 1e-6))
            direction_loss = torch.sqrt(1.0 - torch.clamp(cosine, 0.0, 1.0) + 1e-6)
            magnitude_loss = torch.sqrt((pred_norm - vis_norm) ** 2 + 1e-6)
            scale_loss = ((0.55 * magnitude_loss + 0.45 * direction_loss) * weight_s).sum() / (weight_s.sum() + 1e-6)
            loss = loss + scale_weight * scale_loss
            total += scale_weight
        return loss / max(total, 1e-6)

    def _visible_texture_signed_residual_loss(self, pred_residual, ref_residual, weight, scale_weights=None):
        weight = weight.detach()
        ref_residual = ref_residual.detach()
        loss = torch.tensor(0.0, device=pred_residual.device)
        total = 0.0
        default_scale_weights = (1.0, 0.65, 0.45)
        if scale_weights is None:
            scale_weights = default_scale_weights
        for idx, scale in enumerate((1, 2, 4)):
            scale_weight = float(scale_weights[min(idx, len(scale_weights) - 1)])
            if scale_weight <= 0.0:
                continue
            if min(pred_residual.shape[-2:]) < scale:
                continue
            if scale == 1:
                pred_s = pred_residual
                ref_s = ref_residual
                weight_s = weight
            else:
                pred_s = F.avg_pool2d(pred_residual, kernel_size=scale, stride=scale)
                ref_s = F.avg_pool2d(ref_residual, kernel_size=scale, stride=scale)
                weight_s = F.avg_pool2d(weight, kernel_size=scale, stride=scale)

            diff = torch.sqrt((pred_s - ref_s) ** 2 + 1e-6)
            loss = loss + scale_weight * (diff * weight_s).sum() / (weight_s.sum() + 1e-6)
            total += scale_weight
        return loss / max(total, 1e-6)

    def _ncc_scale_loss_weights(self, ir_detail, visible_detail, return_confidence=False):
        base_weights = (1.0, 0.65, 0.45)
        corr_means = []
        for scale in (1, 2, 4):
            if min(ir_detail.shape[-2:]) < scale:
                corr_means.append(torch.tensor(0.0, device=ir_detail.device))
                continue
            if scale == 1:
                ir_s = ir_detail
                vis_s = visible_detail
            else:
                ir_s = F.avg_pool2d(ir_detail, kernel_size=scale, stride=scale)
                vis_s = F.avg_pool2d(visible_detail, kernel_size=scale, stride=scale)
            _, corr_conf = self._local_ncc_align(vis_s, ir_s)
            corr_means.append(torch.clamp(corr_conf.mean(), 0.0, 1.0))

        raw_weights = [
            torch.clamp(corr_means[i], min=0.05) * base_weights[i]
            for i in range(len(base_weights))
        ]
        raw_sum = raw_weights[0] + raw_weights[1] + raw_weights[2] + 1e-6
        base_sum = sum(base_weights)
        scale_weights = [float((w / raw_sum * base_sum).detach().cpu().item()) for w in raw_weights]
        if return_confidence:
            confidence = torch.clamp(sum(corr_means) / float(len(corr_means)), 0.0, 1.0)
            return scale_weights, float(confidence.detach().cpu().item())
        return scale_weights

    def _texture_residual_stage_visible_signed_loss(self, base_texture, stage_textures, stage_scales, visible_target, prompt_weight, mode='signed'):
        if not stage_textures:
            return torch.tensor(0.0, device=base_texture.device)
        default_weights = [0.20, 0.35, 1.00]
        step_weights = self._parse_float_list(
            getattr(self.opt, 'texture_residual_step_loss_weights', ''),
            default_weights
        )
        vis_loss_total = torch.tensor(0.0, device=base_texture.device)
        total = 0.0
        for idx, stage_texture in enumerate(stage_textures):
            step_weight = float(step_weights[min(idx, len(step_weights) - 1)])
            if step_weight <= 0.0:
                continue
            scale = float(stage_scales[min(idx, len(stage_scales) - 1)]) if stage_scales else 1.0
            stage_residual = stage_texture - base_texture.detach()
            if scale < 1.0:
                size = stage_texture.shape[-2:]
                scaled_h = max(16, int(round(float(size[0]) * scale)))
                scaled_w = max(16, int(round(float(size[1]) * scale)))
                scaled_h = max(16, (scaled_h // 16) * 16)
                scaled_w = max(16, (scaled_w // 16) * 16)
                stage_residual = F.interpolate(stage_residual, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
                vis_s = F.interpolate(visible_target.detach(), size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
                weight_s = F.interpolate(prompt_weight.detach(), size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
            else:
                vis_s = visible_target.detach()
                weight_s = prompt_weight.detach()

            if mode == 'gradtex':
                vis_loss = self._visible_texture_abs_residual_loss(stage_residual, vis_s, weight_s)
            else:
                vis_loss = self._visible_texture_signed_residual_loss(stage_residual, vis_s, weight_s)
            vis_loss_total = vis_loss_total + step_weight * vis_loss
            total += step_weight
        return vis_loss_total / max(total, 1e-6)

    def _ir_highfreq_consistency_loss(self, pred_texture, ir_texture, mask=None):
        pred_grad = self._normalized_gradient_map(pred_texture)
        ir_grad = self._normalized_gradient_map(ir_texture.detach())
        diff = torch.sqrt((pred_grad - ir_grad) ** 2 + 1e-6)
        if mask is not None:
            weight = torch.clamp(mask.detach(), min=0.05, max=1.0)
            return (diff * weight).sum() / (weight.sum() + 1e-6)
        return diff.mean()

    def _fusion_loss(self, pred, target):
        pred_clamp = torch.clamp(pred, 0.0, 1.0)
        target_clamp = torch.clamp(target, 0.0, 1.0)

        loss_l1 = F.l1_loss(pred_clamp, target_clamp)

        ssim_val = self._ssim_score(pred_clamp, target_clamp)
        loss_ssim = 1.0 - ssim_val

        loss_psnr = F.mse_loss(pred, target)

        lambda_l1 = float(getattr(self.opt, 'lambda_fusion_l1', 0.0))
        lambda_ssim = float(getattr(self.opt, 'lambda_fusion_ssim', 1.0))
        lambda_psnr = float(getattr(self.opt, 'lambda_fusion_psnr', 1.0))

        total_loss = (
            lambda_l1 * loss_l1
            + lambda_ssim * loss_ssim
            + lambda_psnr * loss_psnr
        )
        zero_loss = torch.tensor(0.0, device=pred.device)

        return total_loss, {
            'fusion_charb': zero_loss.detach(),
            'fusion_l1': loss_l1.detach(),
            'fusion_ssim': loss_ssim.detach(),
            'fusion_psnr': loss_psnr.detach(),
            'fusion_grad': zero_loss.detach(),
            'fusion_structure': zero_loss.detach(),
            'fusion_lap': zero_loss.detach(),
            'fusion_lowfreq': zero_loss.detach(),
            'fusion_stats': zero_loss.detach(),
        }

    def _build_texture_mask(self, infrared, visible=None):
        grad_mag_ir = self._gradient_magnitude(infrared)
        if visible is not None:
            grad_mag_vis = self._gradient_magnitude(visible)
            # Visible texture should dominate the supervision focus; infrared provides auxiliary structure.
            grad_mag = 0.7 * grad_mag_vis + 0.3 * grad_mag_ir
        else:
            grad_mag = grad_mag_ir

        # Per-sample normalization keeps mask stable across dynamic ranges.
        b = grad_mag.shape[0]
        flat = grad_mag.view(b, -1)
        max_val = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        norm = grad_mag / (max_val + 1e-6)

        floor = max(0.0, min(1.0, self.opt.texture_mask_floor))
        return floor + (1.0 - floor) * norm

    def _build_thermal_mask(self, radiation):
        b = radiation.shape[0]
        flat = radiation.view(b, -1)
        min_val = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        max_val = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        norm = (radiation - min_val) / (max_val - min_val + 1e-6)

        threshold = max(0.0, min(1.0, self.opt.thermal_hot_threshold))
        mask = (norm - threshold) / (1.0 - threshold + 1e-6)
        return torch.clamp(mask, 0.0, 1.0)

    def _extract_texture(self, x):
        k = int(self.opt.texture_blur_ksize)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1

        base = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        texture = x - base
        return texture

    def _guided_filter_decompose(self, x, r=16, eps=1e-3):
        """Structure-texture decomposition via guided filter (I=p=x).
        Returns:
            base: low-frequency structure layer.
            detail: residual texture layer, x - base.
        """
        mean_x = F.avg_pool2d(x, kernel_size=r, stride=1, padding=r // 2)
        mean_xx = F.avg_pool2d(x * x, kernel_size=r, stride=1, padding=r // 2)
        var_x = mean_xx - mean_x * mean_x

        # Guided-filter coefficient: preserve edges, smooth flat regions.
        a = var_x / (var_x + eps)
        b = mean_x * (1.0 - a)

        mean_a = F.avg_pool2d(a, kernel_size=r, stride=1, padding=r // 2)
        mean_b = F.avg_pool2d(b, kernel_size=r, stride=1, padding=r // 2)

        base = mean_a * x + mean_b
        detail = x - base
        return base, detail

    def _build_structure_consistency_mask(self, base_ir, base_vis,
                                          cos_thresh=0.85,
                                          vis_grad_thresh=0.05,
                                          ir_grad_thresh=0.02):
        """Build a soft mask for structure regions shared by IR and visible images."""
        channels = base_ir.shape[1]
        sobel_x = self.sobel_x.repeat(channels, 1, 1, 1).to(base_ir.device)
        sobel_y = self.sobel_y.repeat(channels, 1, 1, 1).to(base_ir.device)

        gx_ir = F.conv2d(base_ir, sobel_x, padding=1, groups=channels)
        gy_ir = F.conv2d(base_ir, sobel_y, padding=1, groups=channels)
        gx_vis = F.conv2d(base_vis, sobel_x, padding=1, groups=channels)
        gy_vis = F.conv2d(base_vis, sobel_y, padding=1, groups=channels)

        mag_ir = torch.sqrt(gx_ir ** 2 + gy_ir ** 2 + 1e-8)
        mag_vis = torch.sqrt(gx_vis ** 2 + gy_vis ** 2 + 1e-8)
        dot = gx_vis * gx_ir + gy_vis * gy_ir
        cos_theta = torch.clamp(dot / (mag_vis * mag_ir + 1e-8), -1.0, 1.0)

        mask = (cos_theta > cos_thresh) & (mag_vis > vis_grad_thresh) & (mag_ir > ir_grad_thresh)
        mask = mask.float()
        mask = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
        mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
        return mask

    def _sample_norm_map(self, x):
        b = x.shape[0]
        flat = x.flatten(1)
        mean = flat.mean(dim=1, keepdim=True).view(b, 1, 1, 1)
        std = flat.std(dim=1, keepdim=True, unbiased=False).view(b, 1, 1, 1)
        return torch.clamp((x - mean) / (std + 1e-6) * 0.25 + 0.5, 0.0, 1.0)

    def _local_std_map(self, x, kernel_size=7):
        pad = kernel_size // 2
        mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        mean_sq = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=pad)
        return torch.sqrt(torch.clamp(mean_sq - mean * mean, min=1e-8))

    def _highpass_map(self, x, kernel_size=9):
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        low = F.avg_pool2d(x_pad, kernel_size=kernel_size, stride=1)
        return x - low

    def _texture_branch_internal_split(self, texture, kernel_size=7):
        if texture is None or texture.dim() != 4:
            return texture, texture
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        texture_pad = F.pad(texture, (pad, pad, pad, pad), mode='reflect')
        texture_structure = F.avg_pool2d(texture_pad, kernel_size=kernel_size, stride=1)
        texture_detail = texture - texture_structure
        return texture_structure, texture_detail

    def _shift_like_ir(self, x, dy, dx):
        h, w = x.shape[-2:]
        pad_l = max(dx, 0)
        pad_r = max(-dx, 0)
        pad_t = max(dy, 0)
        pad_b = max(-dy, 0)
        x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode='reflect')
        y0 = pad_b
        x0 = pad_r
        return x_pad[:, :, y0:y0 + h, x0:x0 + w]

    def _local_ncc_align(self, visible_detail, ir_detail):
        radius = max(0, int(getattr(self.opt, 'texture_prompt_align_radius', 3)))
        window = max(3, int(getattr(self.opt, 'texture_prompt_align_window', 7)))
        if window % 2 == 0:
            window += 1
        pad = window // 2

        ir_mean = F.avg_pool2d(F.pad(ir_detail, (pad, pad, pad, pad), mode='reflect'), window, stride=1)
        ir_centered = ir_detail - ir_mean
        ir_var = F.avg_pool2d(F.pad(ir_centered * ir_centered, (pad, pad, pad, pad), mode='reflect'), window, stride=1)

        best_corr = torch.full_like(ir_detail, -1.0)
        best_visible = visible_detail
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted = self._shift_like_ir(visible_detail, dy, dx)
                vis_mean = F.avg_pool2d(F.pad(shifted, (pad, pad, pad, pad), mode='reflect'), window, stride=1)
                vis_centered = shifted - vis_mean
                vis_var = F.avg_pool2d(F.pad(vis_centered * vis_centered, (pad, pad, pad, pad), mode='reflect'), window, stride=1)
                cov = F.avg_pool2d(F.pad(ir_centered * vis_centered, (pad, pad, pad, pad), mode='reflect'), window, stride=1)
                corr = cov / torch.sqrt(torch.clamp(ir_var * vis_var, min=1e-8))
                take = corr > best_corr
                best_corr = torch.where(take, corr, best_corr)
                best_visible = torch.where(take, shifted, best_visible)

        conf_threshold = float(getattr(self.opt, 'texture_prompt_align_conf_threshold', 0.15))
        conf_temp = max(1e-3, float(getattr(self.opt, 'texture_prompt_align_temperature', 0.10)))
        conf = torch.sigmoid((best_corr - conf_threshold) / conf_temp)
        return best_visible.detach(), torch.clamp(conf, 0.0, 1.0).detach()

    def _ir_texture_selector(self, ir_texture, ir_structure=None):
        ir_texture = ir_texture.detach()
        ir_texture_reliability = self._build_texture_mask(ir_texture, None)
        ir_texture_grad = self._sample_norm_map(self._gradient_magnitude(ir_texture))
        ir_texture_var = self._sample_norm_map(self._local_std_map(ir_texture))

        prompt_score = 0.45 * ir_texture_reliability + 0.35 * ir_texture_grad + 0.20 * ir_texture_var
        prompt_score = torch.clamp(prompt_score, 0.0, 1.0)

        threshold = max(0.0, min(1.0, float(getattr(self.opt, 'ir_texture_prompt_threshold', 0.58))))
        temperature = max(1e-3, float(getattr(self.opt, 'ir_texture_prompt_temperature', 0.08)))
        prompt_power = max(0.25, float(getattr(self.opt, 'ir_texture_prompt_power', 1.5)))
        selector = torch.sigmoid((prompt_score - threshold) / temperature)
        selector = torch.pow(selector, prompt_power)

        min_weight = max(0.0, min(0.25, float(getattr(self.opt, 'ir_texture_prompt_min_weight', 0.03))))
        if min_weight > 0.0:
            selected = (prompt_score > threshold).to(dtype=selector.dtype)
            selector = torch.maximum(selector, selected * min_weight)
        return torch.clamp(selector, 0.0, 1.0)

    def _ir_texture_prompt(self, visible_texture):
        visible_texture = visible_texture.detach()
        ir_texture = self.T_ir.detach()
        ir_structure = self.S_ir.detach()

        prompt_weight = self._ir_texture_selector(ir_texture, ir_structure)

        if self._is_texture_bootstrap():
            strength = float(getattr(self.opt, 'ir_texture_prompt_bootstrap_strength', 0.65))
        else:
            strength = float(getattr(self.opt, 'ir_texture_prompt_strength', 0.50))
        strength = max(0.0, min(1.0, strength))

        if bool(int(getattr(self.opt, 'texture_prompt_align_enabled', 0))):
            if bool(int(getattr(self.opt, 'texture_prompt_align_highpass', 1))):
                ir_detail = self._highpass_map(ir_texture)
                visible_detail = self._highpass_map(visible_texture)
            else:
                ir_detail = ir_texture
                visible_detail = visible_texture
            prompt_residual = visible_texture - ir_texture

            window = max(3, int(getattr(self.opt, 'texture_prompt_align_window', 7)))
            if window % 2 == 0:
                window += 1
            pad = window // 2
            ir_mean = F.avg_pool2d(F.pad(ir_detail, (pad, pad, pad, pad), mode='reflect'), window, stride=1)
            vis_mean = F.avg_pool2d(F.pad(visible_detail, (pad, pad, pad, pad), mode='reflect'), window, stride=1)
            ir_centered = ir_detail - ir_mean
            vis_centered = visible_detail - vis_mean
            local_dot = F.avg_pool2d(
                F.pad(ir_centered * vis_centered, (pad, pad, pad, pad), mode='reflect'),
                window,
                stride=1
            )
            ir_energy = F.avg_pool2d(
                F.pad(ir_centered * ir_centered, (pad, pad, pad, pad), mode='reflect'),
                window,
                stride=1
            )
            vis_energy = F.avg_pool2d(
                F.pad(vis_centered * vis_centered, (pad, pad, pad, pad), mode='reflect'),
                window,
                stride=1
            )
            local_cos = torch.clamp(local_dot / torch.sqrt(torch.clamp(ir_energy * vis_energy, min=1e-8)), -1.0, 1.0)
            cos_threshold = max(-1.0, min(1.0, float(getattr(self.opt, 'texture_prompt_align_conf_threshold', 0.15))))
            cos_temperature = max(1e-3, float(getattr(self.opt, 'texture_prompt_align_temperature', 0.10)))
            cos_gate = torch.sigmoid((local_cos - cos_threshold) / cos_temperature)
            prompt_weight = prompt_weight * torch.clamp(cos_gate, 0.0, 1.0)
        else:
            self.texture_ncc_scale_weights = (1.0, 0.65, 0.45)
            prompt_residual = visible_texture - ir_texture

        prompt_weight = strength * prompt_weight
        self.texture_prompt_residual = prompt_residual.detach()
        self.texture_prompt_visible = visible_texture.detach()
        self.texture_prompt_W = prompt_weight.detach()
        return visible_texture.detach(), prompt_weight.detach()

    def _update_visible_residual_ncc_weights(self, residual_gt, residual_vis):
        gt_residual_mag = torch.abs(residual_gt.detach())
        vis_residual_mag = torch.abs(residual_vis.detach())
        if bool(int(getattr(self.opt, 'texture_prompt_align_highpass', 1))):
            gt_residual_mag = self._highpass_map(gt_residual_mag)
            vis_residual_mag = self._highpass_map(vis_residual_mag)
        scale_weights, confidence = self._ncc_scale_loss_weights(
            gt_residual_mag,
            vis_residual_mag,
            return_confidence=True
        )
        self.texture_ncc_scale_weights = scale_weights
        self.texture_vis_ncc_confidence = confidence

    def _texture_vis_confidence_scale(self):
        if not self.isTrain or not bool(int(getattr(self.opt, 'texture_vis_conf_adaptive', 0))):
            return 1.0
        confidence = float(getattr(self, 'texture_vis_ncc_confidence', 1.0))
        min_conf = max(0.0, min(1.0, float(getattr(self.opt, 'texture_vis_conf_min', 0.25))))
        power = max(0.1, float(getattr(self.opt, 'texture_vis_conf_power', 1.5)))
        confidence = max(min_conf, min(1.0, confidence))
        return confidence ** power

    def _structure_lowfreq_loss(self, image, structure, kernel_size=15):
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        image_pad = F.pad(image, (pad, pad, pad, pad), mode='reflect')
        target = F.avg_pool2d(image_pad, kernel_size=kernel_size, stride=1)
        return self._charbonnier(structure, target.detach())

    def _structure_underfill_loss(self, image, structure, kernel_size=15):
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        image_pad = F.pad(image, (pad, pad, pad, pad), mode='reflect')
        base = F.avg_pool2d(image_pad, kernel_size=kernel_size, stride=1).detach()
        ratio = max(0.0, min(0.6, float(getattr(self.opt, 'structure_underfill_ratio', 0.20))))
        return F.relu(base * ratio - structure).mean()

    def _visualize_residual(self, x):
        x = x.detach()
        if x.dim() != 4:
            return x
        flat = x.flatten(1)
        lo = flat.quantile(0.01, dim=1).view(-1, 1, 1, 1)
        hi = flat.quantile(0.99, dim=1).view(-1, 1, 1, 1)
        return torch.clamp((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    def _visualize_signed_residual(self, x):
        x = x.detach()
        if x.dim() != 4:
            return x
        flat = torch.abs(x).flatten(1)
        scale = flat.quantile(0.99, dim=1).view(-1, 1, 1, 1)
        scale = torch.clamp(scale, min=1e-4)
        return torch.clamp(0.5 + 0.5 * x / scale, 0.0, 1.0)

    def _texture_stats_pair(self, x):
        if x is None:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero
        y = x.detach().float()
        return y.mean(), y.std(unbiased=False)

    def _cache_texture_stats(self, t_gt=None, t_vis=None):
        self.loss_tex_Tir_mean, self.loss_tex_Tir_std = self._texture_stats_pair(getattr(self, 'T_ir', None))
        self.loss_tex_Rtex_mean, self.loss_tex_Rtex_std = self._texture_stats_pair(getattr(self, 'R_pred', None))
        self.loss_tex_Tenh_mean, self.loss_tex_Tenh_std = self._texture_stats_pair(getattr(self, 'T_enhanced', None))
        self.loss_tex_Tgt_mean, self.loss_tex_Tgt_std = self._texture_stats_pair(t_gt)
        self.loss_tex_Tvis_mean, self.loss_tex_Tvis_std = self._texture_stats_pair(t_vis)

    def _texture_delta(self, raw_delta):
        max_delta = max(0.0, float(getattr(self.opt, 'texture_residual_max_delta', 0.0)))
        if max_delta > 0.0:
            return torch.clamp(raw_delta, -max_delta, max_delta)
        return raw_delta

    def _match_texture_delta_scale(self, texture_delta, reference_texture):
        if texture_delta is None or reference_texture is None:
            return texture_delta
        if texture_delta.dim() != 4 or reference_texture.dim() != 4:
            return texture_delta

        eps = 1e-6
        centered_delta = texture_delta - texture_delta.mean(dim=(2, 3), keepdim=True)
        delta_std = centered_delta.flatten(2).std(dim=2, unbiased=False).view(
            centered_delta.shape[0], centered_delta.shape[1], 1, 1
        )

        reference = reference_texture.detach()
        centered_reference = reference - reference.mean(dim=(2, 3), keepdim=True)
        reference_std = centered_reference.flatten(2).std(dim=2, unbiased=False).view(
            centered_reference.shape[0], centered_reference.shape[1], 1, 1
        )
        return centered_delta / (delta_std + eps) * reference_std

    def _texture_domain_normalize(self, texture, force_target_std=True):
        if not bool(int(getattr(self.opt, 'texture_domain_norm_enabled', 0))):
            return texture
        if texture is None or texture.dim() != 4:
            return texture

        eps = 1e-6
        centered = texture - texture.mean(dim=(2, 3), keepdim=True)
        std = centered.flatten(2).std(dim=2, unbiased=False).view(
            centered.shape[0], centered.shape[1], 1, 1
        )
        target_std = max(0.0, float(getattr(self.opt, 'texture_domain_target_std', 0.08)))

        if target_std > 0.0:
            if force_target_std:
                normalized = centered / (std + eps) * target_std
            else:
                normalized = centered
        else:
            normalized = centered
        return normalized

    def _visualize_structure_map(self, x):
        x = x.detach()
        if x.dim() != 4:
            return x
        base_vis = self._visualize_residual(x)
        grad = self._gradient_magnitude(x)
        grad_vis = self._visualize_residual(grad)
        vis = torch.maximum(base_vis, 0.45 * grad_vis)

        flat = vis.flatten(1)
        hi = flat.quantile(0.995, dim=1).view(-1, 1, 1, 1)
        active = (vis > 0.01).float()
        floor = 0.06 * active
        return torch.clamp(torch.maximum(vis / (hi + 1e-6), floor), 0.0, 1.0)

    def _local_contrast_normalize(self, x, r=7):
        """Local contrast normalization for texture-only comparison."""
        mu = F.avg_pool2d(x, kernel_size=r, stride=1, padding=r // 2)
        sigma_sq = F.avg_pool2d(x * x, kernel_size=r, stride=1, padding=r // 2) - mu * mu
        sigma_sq = torch.clamp(sigma_sq, min=1e-6)
        sigma = torch.sqrt(sigma_sq + 1e-6)
        return (x - mu) / (sigma + 1e-4)

    def _charbonnier(self, pred, target, eps=1e-3, weight=None):
        diff = pred - target
        loss = torch.sqrt(diff * diff + eps * eps)
        if weight is not None:
            loss = loss * weight
        return loss.mean()

    def _tv_loss(self, x):
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return tv_h + tv_w

    def _radiation_brightness_scale(self):
        if not self.isTrain:
            return 1.0

        if self.current_epoch <= self.stage1_epochs_effective:
            return self.radiation_stage1_scale

        return self.radiation_stage2_scale

    def _fusion_supervision_scale(self):
        if not self.isTrain:
            return 1.0

        texture_bootstrap_epochs = int(getattr(self.opt, 'texture_bootstrap_epochs', 0))
        if texture_bootstrap_epochs > 0 and self.current_epoch <= texture_bootstrap_epochs:
            return max(0.0, min(1.0, float(getattr(self.opt, 'texture_bootstrap_fusion_scale', 0.20))))

        if self.current_epoch <= self.stage1_epochs_effective:
            return self.fusion_stage1_scale

        progress = (float(self.current_epoch) - float(self.stage1_epochs_effective)) / float(self.fusion_warmup_epochs)
        return self.fusion_stage1_scale + (1.0 - self.fusion_stage1_scale) * min(1.0, max(0.0, progress))

    def _gan_weight_scale(self):
        if not self.isTrain:
            return 0.0
        return self.gan_scale

    def _is_stage1(self):
        return self.isTrain and (self.current_epoch <= self.stage1_epochs_effective)

    def _is_texture_bootstrap(self):
        texture_bootstrap_epochs = int(getattr(self.opt, 'texture_bootstrap_epochs', 0))
        return self.isTrain and texture_bootstrap_epochs > 0 and self.current_epoch <= texture_bootstrap_epochs

    def _texture_supervision_scale(self):
        return 1.0

    def _texture_vis_supervision_scale(self):
        if not self.isTrain:
            return 1.0
        decay_start = int(getattr(self.opt, 'texture_vis_decay_start', -1))
        decay_end = int(getattr(self.opt, 'texture_vis_decay_end', -1))
        min_scale = max(0.0, min(1.0, float(getattr(self.opt, 'texture_vis_min_scale', 1.0))))
        if decay_start >= 0 and decay_end > decay_start and self.current_epoch >= decay_start:
            span = float(decay_end - decay_start)
            progress = min(1.0, max(0.0, (float(self.current_epoch) - float(decay_start)) / span))
            return 1.0 - (1.0 - min_scale) * progress
        return 1.0

    def _structure_curriculum_scale(self):
        if not self.isTrain or self._is_stage1():
            return 0.0
        span = float(max(1, self.structure_warmup_epochs))
        progress = (float(self.current_epoch) - float(self.stage1_epochs_effective)) / span
        return min(1.0, max(0.0, progress))

    def set_input(self, input):
        self.real_I = input['infrared'].to(self.device)
        self.real_r = input['infrared-radiation'].to(self.device)
        self.image_paths = input['paths']
        self.paper_ir_input = self.real_I
        self.paper_ir_radiation_input = self.real_r
        self.paper_01_input_ir_image = self.real_I
        self.paper_02_ir_input_radiation_Sir = self.real_r

        if self.isTrain:
            self.real_r_GT = input['gt-radiation'].to(self.device)
            self.ref_t = input['visible'].to(self.device)
            self.gt = input['gt'].to(self.device)
            self.paper_visible_reference = self.ref_t
            self.paper_gt_target = self.gt
            self.paper_gt_radiation = self.real_r_GT
            self.paper_03_train_visible_reference = self.ref_t
            self.paper_04_train_gt_image = self.gt
            self.paper_05_gt_radiation_Sgt = self.real_r_GT
        else:
            self.real_r_GT = None
            self.ref_t = None
            self.gt = None

    def _texture_input(self, x):
        if self.texture_frontend_enabled:
            return self.texture_frontend(x)
        return x

    @staticmethod
    def _parse_float_list(value, default_values):
        if value is None or str(value).strip() == '':
            return list(default_values)
        parsed = []
        for item in str(value).split(','):
            item = item.strip()
            if item:
                parsed.append(float(item))
        return parsed if parsed else list(default_values)

    def _visible_texture_reference(self, t_vis):
        if self.texture_frontend_enabled:
            with torch.no_grad():
                return self.texture_frontend(t_vis.detach())
        return t_vis.detach()

    def _configure_texture_residual_generator(self):
        generator = self.G_J.module if hasattr(self.G_J, 'module') else self.G_J
        if hasattr(generator, 'final_act'):
            generator.final_act = nn.Identity()
        if hasattr(generator, 'up4'):
            for layer in reversed(generator.up4):
                if isinstance(layer, nn.Conv2d):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                    break

    def _gt_texture_reference(self, gt):
        if self._use_wavelet_ref_decom():
            _, t_gt = self._gt_wavelet_reference(gt)
            return self._texture_domain_normalize(t_gt, force_target_std=True).detach()
        if self._use_guided_ref_decom():
            _, t_gt = self._gt_guided_reference(gt)
            return self._texture_domain_normalize(t_gt, force_target_std=True).detach()
        if self._use_radiation_texture_ref_decom():
            _, t_gt = self._gt_radiation_texture_reference(gt)
            return self._texture_domain_normalize(t_gt, force_target_std=True).detach()
        with torch.no_grad():
            _, t_gt = self.decom_net(gt.detach())
            if self.texture_frontend_enabled:
                t_gt = self.texture_frontend(t_gt)
        return self._texture_domain_normalize(t_gt, force_target_std=True).detach()

    def _use_learned_ir_decom(self):
        return self.decom_enabled and self.decom_mode in (
            'ir_wavelet_ref',
            'ir_guided_ref',
            'ir_radiation_texture_ref',
        )

    def _use_wavelet_ref_decom(self):
        return self.decom_enabled and self.decom_mode == 'ir_wavelet_ref'

    def _use_guided_ref_decom(self):
        return self.decom_enabled and self.decom_mode == 'ir_guided_ref'

    def _use_radiation_texture_ref_decom(self):
        return self.decom_enabled and self.decom_mode == 'ir_radiation_texture_ref'

    def _use_legacy_radiation_from_decom(self):
        return (
            self.decom_enabled
            and self.decom_mode == 'legacy'
            and bool(int(getattr(self.opt, 'legacy_radiation_from_decom', 0)))
        )

    def _gt_wavelet_reference(self, gt):
        from models.decom_net import haar_wavelet_decompose
        with torch.no_grad():
            s_gt, t_gt = haar_wavelet_decompose(gt.detach())
        return s_gt.detach(), t_gt.detach()

    def _gt_guided_reference(self, gt):
        from models.decom_net import guided_filter_decompose
        kernel_size = int(getattr(self.opt, 'decom_guided_ksize', 15))
        eps = float(getattr(self.opt, 'decom_guided_eps', 0.01))
        with torch.no_grad():
            s_gt, t_gt = guided_filter_decompose(gt.detach(), kernel_size=kernel_size, eps=eps)
        return s_gt.detach(), t_gt.detach()

    def _gt_radiation_texture_reference(self, gt):
        from models.decom_net import gt_light_radiation_texture_decompose
        kernel_size = int(getattr(self.opt, 'gt_radiation_ksize', 7))
        raw_blend = float(getattr(self.opt, 'gt_radiation_raw_blend', 0.65))
        with torch.no_grad():
            s_gt, t_gt = gt_light_radiation_texture_decompose(
                gt.detach(),
                kernel_size=kernel_size,
                raw_blend=raw_blend,
            )
        return s_gt.detach(), t_gt.detach()

    def forward(self):
        learned_ir_decom = self._use_learned_ir_decom()
        legacy_radiation_from_decom = self._use_legacy_radiation_from_decom()
        if learned_ir_decom or legacy_radiation_from_decom:
            if self._use_radiation_texture_ref_decom():
                self.S_ir, self.T_ir = self._gt_radiation_texture_reference(self.real_I)
            else:
                self.S_ir, self.T_ir = self.decom_net(self.real_I)
            self.T_ir_decom = self.T_ir
            self.T_ir = self._texture_domain_normalize(self.T_ir, force_target_std=True)
            radiation_input = self.S_ir
            self.paper_ir_radiation_input = radiation_input
            self.paper_02_ir_input_radiation_Sir = radiation_input
        else:
            radiation_input = self.real_r
        if self.texture_stage:
            self.refine1_r = radiation_input.detach()
            self.refine_r = radiation_input.detach()
        else:
            self.refine1_r = self.ATF2D(radiation_input)
            self.refine_r = self.TCMNet(self.refine1_r)
        self.paper_radiation_stage1 = self.refine1_r
        self.paper_radiation_enhanced = self.refine_r
        self.paper_06_radiation_stage1_ATF2D = self.refine1_r
        self.paper_07_radiation_output_TCM = self.refine_r

        if self.decom_enabled:
            if not (learned_ir_decom or legacy_radiation_from_decom):
                self.S_ir, self.T_ir = self.decom_net(self.real_I)
                self.T_ir_decom = self.T_ir
                self.T_ir = self._texture_domain_normalize(self.T_ir, force_target_std=True)
            ir_texture_base = self.T_ir.detach()
            ir_structure_base = self.S_ir.detach()
            self.T_ir_internal_structure = ir_structure_base
            self.T_ir_internal_detail = ir_texture_base
            self.paper_09_ir_texture_internal_structure = self._visualize_residual(ir_structure_base)
            self.S_ir_vis = self._visualize_residual(self.S_ir)
            self.paper_ir_structure = self.S_ir_vis
            # Visible texture is train-only reference supervision; inference uses IR only.
            if self.isTrain and self.ref_t is not None:
                self.S_vis, self.T_vis = self.visible_decom_net(self.ref_t)
                self.T_vis_decom = self.T_vis
                self.T_vis = self._texture_domain_normalize(self.T_vis, force_target_std=True)
                self.S_vis_struct_vis = self._visualize_residual(self.S_vis)
                self.paper_visible_structure = self.S_vis_struct_vis
                self.paper_10_structure_visible_S_vis = self.S_vis_struct_vis
            else:
                self.S_vis = None
                self.T_vis = None

            ir_focus_map = self._ir_texture_selector(
                ir_texture_base,
                ir_structure_base,
            )
            self.paper_ir_texture_focus_M = ir_focus_map.detach()
            self.paper_12_ir_texture_focus_M = self.paper_ir_texture_focus_M
            ir_focus_gain = max(0.0, min(1.0, float(getattr(self.opt, 'texture_residual_ir_focus_gain', 0.0))))
            ir_residual_gain_map = 1.0 + ir_focus_gain * ir_focus_map

            if bool(int(getattr(self.opt, 'texture_residual_mode', 0))):
                residual_gain = max(0.0, min(1.0, float(getattr(self.opt, 'texture_residual_gain', 0.35))))
                residual_steps = max(1, min(4, int(getattr(self.opt, 'texture_residual_steps', 1))))
                residual_step_scale = max(0.25, min(1.0, float(getattr(self.opt, 'texture_residual_step_scale', 1.0))))
                residual_step_scales = self._parse_float_list(
                    getattr(self.opt, 'texture_residual_step_scales', ''),
                    [residual_step_scale] * residual_steps
                )
                if bool(int(getattr(self.opt, 'texture_residual_gain_per_step', 0))):
                    step_gain = residual_gain
                else:
                    step_gain = residual_gain / float(residual_steps)
                current_texture = ir_texture_base
                self.R_pred = torch.zeros_like(ir_texture_base)
                self.texture_residual_stage_preds = []
                self.texture_residual_stage_scales = []
                for step_idx in range(residual_steps):
                    current_scale = residual_step_scales[min(step_idx, len(residual_step_scales) - 1)]
                    current_scale = max(0.25, min(1.0, float(current_scale)))
                    g_input = current_texture
                    if ir_focus_gain > 0.0:
                        g_input = g_input * ir_residual_gain_map
                    full_size = current_texture.shape[2:]
                    if current_scale < 1.0:
                        scaled_h = max(16, int(round(float(full_size[0]) * current_scale)))
                        scaled_w = max(16, int(round(float(full_size[1]) * current_scale)))
                        scaled_h = max(16, (scaled_h // 16) * 16)
                        scaled_w = max(16, (scaled_w // 16) * 16)
                        g_input = F.interpolate(g_input, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
                    if self.texture_frontend_enabled:
                        g_input = self.texture_frontend(g_input)
                    texture_delta = self._texture_delta(self.G_J(g_input))
                    if current_scale < 1.0:
                        texture_delta = F.interpolate(texture_delta, size=full_size, mode='bilinear', align_corners=False)
                    self.R_pred = self.R_pred + step_gain * texture_delta
                    current_texture = ir_texture_base + self.R_pred
                    self.texture_residual_stage_preds.append(current_texture)
                    self.texture_residual_stage_scales.append(current_scale)
                self.T_enhanced = current_texture
            else:
                g_input = ir_texture_base
                self.texture_residual_stage_preds = []
                self.texture_residual_stage_scales = []
                if self.texture_frontend_enabled:
                    g_input = self.texture_frontend(g_input)
                texture_delta = self._texture_delta(self.G_J(g_input))
                self.R_pred = texture_delta
                self.T_enhanced = ir_texture_base + self.R_pred
                ir_skip = max(0.0, min(0.8, float(getattr(self.opt, 'texture_ir_skip', 0.0))))
                if ir_skip > 0.0:
                    self.T_enhanced = self.T_enhanced + ir_skip * ir_texture_base.detach()
            self.refine_t = torch.clamp(self.S_ir + self.T_enhanced, 0.0, 1.0)
            self.T_ir_vis = self._visualize_residual(self.T_ir)
            self.R_pred_vis = self._visualize_signed_residual(self.R_pred)
            self.T_enhanced_vis = self._visualize_residual(self.T_enhanced)
            self.paper_ir_texture = self.T_ir_vis
            self.paper_texture_residual = self.R_pred_vis
            self.paper_enhanced_texture = self.T_enhanced_vis
            self.paper_texture_reconstruction = self.refine_t
            self.paper_08_ir_input_texture_Tir_visualized = self.T_ir_vis
            self.paper_14_texture_residual_Rtex = self.R_pred_vis
            self.paper_15_enhanced_texture_Tenhanced = self.T_enhanced_vis
            self.paper_14_texture_enhanced_T_enhanced = self.T_enhanced_vis
            self.paper_15_texture_reconstruction_Sir_plus_Tenhanced = self.refine_t
            self._cache_texture_stats(t_vis=self.T_vis)
            if self.isTrain and self.T_vis is not None:
                self.T_vis_vis = self._visualize_residual(self.T_vis)
                self.paper_visible_texture = self.T_vis_vis
                self.paper_11_texture_visible_T_vis = self.T_vis_vis
        else:
            if self.texture_frontend_enabled:
                g_input = self.texture_frontend(self.real_I)
            else:
                g_input = self.real_I
            self.refine_t = torch.clamp(0.5 * (self.G_J(g_input) + 1.0), 0.0, 1.0)
            self.paper_texture_reconstruction = self.refine_t
            self.paper_15_enhanced_texture_Tenhanced = self.refine_t
            self.paper_14_texture_enhanced_T_enhanced = self.refine_t
            self.paper_15_texture_reconstruction_Sir_plus_Tenhanced = self.refine_t
            self._cache_texture_stats()

        generator = self.G_J.module if hasattr(self.G_J, 'module') else self.G_J
        if hasattr(generator, 'get_fusion_condition'):
            self.texture_prior_condition = generator.get_fusion_condition(size=self.refine_t.shape[2:])
        else:
            self.texture_prior_condition = None

        if self.texture_stage:
            self.fused_I = torch.clamp(self.refine_t, 0.0, 1.0)
            self.paper_fused_output = self.fused_I
            self.paper_16_final_fused_output = self.fused_I
            return

        fusion_refine_t = self.refine_t
        detach_texture_epochs = max(0, int(getattr(self.opt, 'fusion_detach_texture_epochs', 0)))
        if self.isTrain and detach_texture_epochs > 0:
            current_epoch = int(getattr(self, 'current_epoch', 0))
            if current_epoch <= detach_texture_epochs:
                fusion_refine_t = self.refine_t.detach()

        fusion_out = self.fusion_model(
            self.refine_r,
            fusion_refine_t,
            texture_prior_cond=self.texture_prior_condition,
            return_aux=self.isTrain,
        )
        if isinstance(fusion_out, tuple):
            fusion_raw, self.fusion_aux = fusion_out
        else:
            fusion_raw = fusion_out
            self.fusion_aux = {}
        self.fused_I = torch.clamp(fusion_raw, 0.0, 1.0)
        self.paper_fused_output = self.fused_I
        self.paper_16_final_fused_output = self.fused_I
    def backward_D_basic(self, netD, real, fake):
        global_pred_real, pixel_pred_real, _ = self._forward_discriminator(real)
        loss_D_real = self.criterionGAN(global_pred_real, True)
        if pixel_pred_real is not None:
            loss_D_real += self.criterionGAN(pixel_pred_real, True)

        global_pred_fake, pixel_pred_fake, _ = self._forward_discriminator(fake.detach())
        loss_D_fake = self.criterionGAN(global_pred_fake, False)
        if pixel_pred_fake is not None:
            loss_D_fake += self.criterionGAN(pixel_pred_fake, False)

        gan_scale = self._gan_weight_scale()
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * gan_scale * self.lambda_D
        loss_D.backward()
        return loss_D

    def backward_D(self):
        if not self._is_gan_enabled():
            self.loss_D = torch.tensor(0.0, device=self.device)
            return

        if self.decom_enabled:
            # Discriminate texture layers only: real = T_vis, fake = T_enhanced.
            # Detach T_vis so discriminator gradients do not update decom_net.
            fake_J = self.fake_I_pool.query(self.T_enhanced)
            self.loss_D = self.backward_D_basic(self.D, self.T_vis.detach(), fake_J)
        else:
            # Fallback: discriminate the full refined image.
            fake_J = self.fake_I_pool.query(self.refine_t)
            self.loss_D = self.backward_D_basic(self.D, self.ref_t, fake_J)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        # Use CLI/runtime value so experiments can tune adversarial strength.
        lambda_Gg = self.opt.lambda_Gg
        lambda_fusion_I = self.opt.lambda_fusion_I
        lambda_radiation = self.opt.lambda_radiation
        lambda_L1 = self.opt.lambda_L1
        lambda_charb = getattr(self.opt, 'lambda_charb', 0.0)
        lambda_ssim = self.opt.lambda_ssim
        lambda_recon_full = max(0.0, min(1.0, float(getattr(self.opt, 'lambda_recon_full', 0.0))))
        recon_ssim_ratio = max(0.0, min(1.0, self.opt.recon_ssim_ratio))

        # 1. Adversarial Loss (only on texture layer T_enhanced)
        if self._is_gan_enabled():
            gan_scale = self._gan_weight_scale()
            if self.decom_enabled:
                # GAN loss constrains T_enhanced, not the structure layer S_ir.
                global_pred_fake, pixel_pred_fake, _ = self._forward_discriminator(self.T_enhanced)
            else:
                # Fallback: constrain the full refined image.
                global_pred_fake, pixel_pred_fake, _ = self._forward_discriminator(self.refine_t)
            self.loss_Gg = self.criterionGAN(global_pred_fake, True) * lambda_Gg * gan_scale
            if pixel_pred_fake is not None:
                self.loss_Gg += self.criterionGAN(pixel_pred_fake, True) * lambda_Gg * gan_scale
        else:
            self.loss_Gg = torch.tensor(0.0, device=self.device)

        # ========================================================
        # 2. Decomposition + conditional residual texture branch.
        # ========================================================

        if self.decom_enabled and self.isTrain:
            from models.decom_net import decom_loss_fn

            # --- 2.1 Decom-Net loss: IR/visible are both decomposed for the three-branch pipeline.
            lambda_decom = float(getattr(self.opt, 'lambda_decom', 1.0))
            if self._use_radiation_texture_ref_decom():
                loss_decom_ir = torch.tensor(0.0, device=self.device)
                decom_ir_dict = {}
            else:
                loss_decom_ir, decom_ir_dict = decom_loss_fn(
                    self.real_I, self.S_ir, getattr(self, 'T_ir_decom', self.T_ir),
                    lambda_recon=1.0, lambda_corr=0.1, lambda_reg=0.01
                )
            loss_decom_vis, decom_vis_dict = decom_loss_fn(
                self.ref_t, self.S_vis, getattr(self, 'T_vis_decom', self.T_vis),
                lambda_recon=1.0, lambda_corr=0.1, lambda_reg=0.01
            )
            if self._use_wavelet_ref_decom():
                self.S_gt, self.T_gt = self._gt_wavelet_reference(self.gt)
                self.paper_05_gt_radiation_Sgt = torch.clamp(self.S_gt, 0.0, 1.0)
            elif self._use_guided_ref_decom():
                self.S_gt, self.T_gt = self._gt_guided_reference(self.gt)
                self.paper_05_gt_radiation_Sgt = torch.clamp(self.S_gt, 0.0, 1.0)
            elif self._use_radiation_texture_ref_decom():
                self.S_gt, self.T_gt = self._gt_radiation_texture_reference(self.gt)
                self.paper_05_gt_radiation_Sgt = torch.clamp(self.S_gt, 0.0, 1.0)
            else:
                with torch.no_grad():
                    self.S_gt, self.T_gt = self.decom_net(self.gt)
            self.T_gt = self._texture_domain_normalize(self.T_gt, force_target_std=True)
            self.loss_decom = 0.5 * (loss_decom_ir + loss_decom_vis) * lambda_decom
            lambda_structure_lowfreq = float(getattr(self.opt, 'lambda_structure_lowfreq', 0.08))
            if lambda_structure_lowfreq > 0.0:
                self.loss_decom = self.loss_decom + lambda_structure_lowfreq * self._structure_lowfreq_loss(self.ref_t, self.S_vis)
            lambda_structure_underfill = float(getattr(self.opt, 'lambda_structure_underfill', 0.0))
            if lambda_structure_underfill > 0.0:
                self.loss_decom = self.loss_decom + lambda_structure_underfill * self._structure_underfill_loss(self.ref_t, self.S_vis)

            # --- 2.2 Texture references keep the decomposition scale.
            T_vis_direct_ref = self._visible_texture_reference(self.T_vis)
            T_vis_ref = T_vis_direct_ref

            # --- 2.3 Texture branch: supervise enhanced texture with three losses.
            lambda_texture_gt = float(getattr(self.opt, 'lambda_texture_gt', 1.0))
            lambda_texture_vis = float(getattr(self.opt, 'lambda_texture_vis', 0.35))
            texture_vis_supervision_mode = str(getattr(self.opt, 'texture_vis_supervision_mode', 'pixel')).lower()
            lambda_texture_step = float(getattr(self.opt, 'lambda_texture_step', 0.0))
            texture_supervision_scale = self._texture_supervision_scale()
            texture_vis_supervision_scale = self._texture_vis_supervision_scale()
            lambda_texture_vis_eff = lambda_texture_vis * texture_vis_supervision_scale

            visible_texture_hint = T_vis_ref
            visible_texture_target, prompt_weight = self._ir_texture_prompt(visible_texture_hint)
            T_gt_direct_ref = self.T_gt.detach()
            if self.texture_frontend_enabled:
                with torch.no_grad():
                    T_gt_direct_ref = self.texture_frontend(T_gt_direct_ref)
            T_gt_ref = T_gt_direct_ref
            ir_texture_base = self.T_ir.detach()
            residual_gt = T_gt_ref - ir_texture_base
            residual_vis = getattr(self, 'texture_prompt_residual', visible_texture_target.detach() - ir_texture_base)
            residual_pred = self.R_pred if getattr(self, 'R_pred', None) is not None else self.T_enhanced - ir_texture_base
            self.paper_17_texture_gt_Tgt = self._visualize_residual(T_gt_ref)
            self._cache_texture_stats(t_gt=T_gt_ref, t_vis=T_vis_ref)
            self.texture_weight_W = prompt_weight.detach()
            self.paper_visible_select_M = self.texture_weight_W
            self.paper_13_visible_supervision_select_M = self.texture_weight_W
            self._update_visible_residual_ncc_weights(residual_gt, residual_vis)
            lambda_texture_vis_eff = lambda_texture_vis_eff * self._texture_vis_confidence_scale()
            visible_texture_weight = prompt_weight
            self.texture_mask_M = visible_texture_weight.detach()
            if bool(int(getattr(self.opt, 'texture_vis_gt_consistency', 0))):
                gt_consistency_temp = max(1e-3, float(getattr(self.opt, 'texture_vis_gt_consistency_temp', 0.08)))
                gt_consistency = torch.exp(-torch.abs(residual_vis.detach() - residual_gt.detach()) / gt_consistency_temp)
                visible_texture_weight = prompt_weight * torch.clamp(gt_consistency, 0.0, 1.0)
                self.texture_weight_W = visible_texture_weight.detach()
                self.paper_visible_select_M = self.texture_weight_W
                self.paper_13_visible_supervision_select_M = self.texture_weight_W

            self.loss_texture_gt = self._charbonnier(self.T_enhanced, T_gt_ref)
            if texture_vis_supervision_mode == 'stat':
                self.loss_texture_vis = self._visible_texture_stat_loss(residual_pred, residual_vis, visible_texture_weight)
            elif texture_vis_supervision_mode == 'gradtex':
                self.loss_texture_vis = self._visible_texture_gradtex_loss(
                    residual_pred,
                    residual_vis,
                    visible_texture_weight
                )
            elif texture_vis_supervision_mode == 'signed':
                self.loss_texture_vis = self._visible_texture_signed_residual_loss(
                    residual_pred,
                    residual_vis,
                    visible_texture_weight,
                    getattr(self, 'texture_ncc_scale_weights', None)
                )
            else:
                self.loss_texture_vis = self._weighted_texture_l1(residual_pred, residual_vis, visible_texture_weight)

            if lambda_texture_step > 0.0:
                self.loss_texture_step = self._texture_residual_stage_visible_signed_loss(
                    self.T_ir.detach(),
                    getattr(self, 'texture_residual_stage_preds', []),
                    getattr(self, 'texture_residual_stage_scales', []),
                    residual_vis,
                    visible_texture_weight,
                    texture_vis_supervision_mode
                )
            else:
                self.loss_texture_step = torch.tensor(0.0, device=self.device)

            zero_loss = torch.tensor(0.0, device=self.device)
            self.loss_texture_ag = zero_loss
            self.loss_texture_distill = zero_loss
            self.loss_texture_ms_grad = zero_loss
            self.loss_texture_contrast = zero_loss
            self.loss_ir_hf = zero_loss
            self.loss_detail_gt = zero_loss

            self.loss_recon = (
                self.loss_texture_gt * lambda_texture_gt * texture_supervision_scale
                + self.loss_texture_vis * lambda_texture_vis_eff * texture_supervision_scale
                + self.loss_texture_step * lambda_texture_step * texture_supervision_scale
                + self.loss_decom
            )

        else:
            # --- Fallback: guided filter decomposition (legacy) ---
            lambda_texture = float(getattr(self.opt, 'lambda_texture', 1.0))
            if lambda_texture > 0.0:
                r_ir = int(getattr(self.opt, 'gf_radius_ir', 20))
                r_vis = int(getattr(self.opt, 'gf_radius_vis', 12))
                eps = float(getattr(self.opt, 'gf_eps', 1e-3))

                base_ir,   detail_ir   = self._guided_filter_decompose(self.real_I,   r=r_ir,  eps=eps)
                base_vis,  detail_vis  = self._guided_filter_decompose(self.ref_t,    r=r_vis, eps=eps)
                base_pred, detail_pred = self._guided_filter_decompose(self.refine_t, r=r_ir,  eps=eps)

                M = self._build_structure_consistency_mask(
                    base_ir, base_vis,
                    cos_thresh=float(getattr(self.opt, 'struct_cos_thresh', 0.85)),
                    vis_grad_thresh=float(getattr(self.opt, 'struct_vis_grad', 0.05)),
                    ir_grad_thresh=float(getattr(self.opt, 'struct_ir_grad', 0.02))
                )

                detail_vis_ref = self._local_contrast_normalize(detail_vis, r=7)
                detail_vis_ref = torch.clamp(detail_vis_ref, -3.0, 3.0)
                loss_align = (M * torch.abs(detail_pred - detail_vis_ref)).mean()
                loss_preserve = ((1.0 - M) * torch.abs(detail_pred - detail_ir)).mean()
                self.loss_recon = (loss_align + loss_preserve) * lambda_texture
            else:
                self.loss_recon = torch.tensor(0.0, device=self.device)
            self.loss_decom = torch.tensor(0.0, device=self.device)
            self.loss_texture_gt = torch.tensor(0.0, device=self.device)
            self.loss_texture_vis = torch.tensor(0.0, device=self.device)
            self.loss_texture_ag = torch.tensor(0.0, device=self.device)
            self.loss_texture_distill = torch.tensor(0.0, device=self.device)
            self.loss_texture_ms_grad = torch.tensor(0.0, device=self.device)
            self.loss_texture_contrast = torch.tensor(0.0, device=self.device)
            self.loss_texture_step = torch.tensor(0.0, device=self.device)
            self.loss_ir_hf = torch.tensor(0.0, device=self.device)
            self.loss_detail_gt = torch.tensor(0.0, device=self.device)
        # ========================================================

        if self.texture_stage:
            zero_loss = torch.tensor(0.0, device=self.device)
            self.loss_idt_t = zero_loss
            self.loss_radiation = zero_loss
            self.loss_fusion = zero_loss
            self.loss_fusion_ssim = zero_loss
            self.loss_fusion_charb = zero_loss
            self.loss_fusion_l1 = zero_loss
            self.loss_fusion_psnr = zero_loss
            self.loss_fusion_grad = zero_loss
            self.loss_fusion_structure = zero_loss
            self.loss_fusion_lap = zero_loss
            self.loss_fusion_lowfreq = zero_loss
            self.loss_fusion_stats = zero_loss
            self.loss_fusion_teacher = zero_loss
            self.loss_G = self.loss_Gg + self.loss_recon
            self.loss_G.backward()
            return

        # 3. Optional identity loss. Disabled by default to keep the objective compact.
        lambda_identity = float(getattr(self.opt, 'lambda_identity', 0.0))
        if lambda_identity > 0.0:
            if self.decom_enabled:
                T_ref = self.ref_t
                if self.texture_frontend_enabled:
                    T_ref = self.texture_frontend(T_ref)
                ref_residual = self.G_J(T_ref)
                self.ref_real_t = torch.clamp(self.ref_t + ref_residual, 0.0, 1.0)
                self.loss_idt_t = self.criterionL1(self.ref_real_t, self.ref_t) * lambda_identity
            else:
                if self.texture_frontend_enabled:
                    ref_input = self.texture_frontend(self.ref_t)
                else:
                    ref_input = self.ref_t
                ref_residual = self.G_J(ref_input)
                self.ref_real_t = torch.clamp(self.ref_t + ref_residual, 0.0, 1.0)
                self.loss_idt_t = self.criterionL1(self.ref_real_t, self.ref_t) * lambda_identity
        else:
            self.loss_idt_t = torch.tensor(0.0, device=self.device)
        radiation_target = self.S_gt.detach() if (
            (self._use_learned_ir_decom() or self._use_legacy_radiation_from_decom())
            and hasattr(self, 'S_gt')
        ) else self.real_r_GT
        self.loss_radiation = self._radiation_structure_loss(self.refine_r, radiation_target) * lambda_radiation

        fusion_loss, fusion_aux = self._fusion_loss(self.fused_I, self.gt)
        self.loss_fusion_teacher = self._fusion_teacher_loss()
        fusion_loss = fusion_loss + self.loss_fusion_teacher
        fusion_loss = fusion_loss * self._fusion_hard_sample_multiplier(self.gt)
        self.loss_fusion = fusion_loss * lambda_fusion_I * self._fusion_supervision_scale()

        self.loss_fusion_ssim = fusion_aux['fusion_ssim']
        self.loss_fusion_charb = fusion_aux['fusion_charb']
        self.loss_fusion_l1 = fusion_aux['fusion_l1']
        self.loss_fusion_psnr = fusion_aux['fusion_psnr']
        self.loss_fusion_grad = fusion_aux['fusion_grad']
        self.loss_fusion_structure = fusion_aux['fusion_structure']
        self.loss_fusion_lap = fusion_aux['fusion_lap']
        self.loss_fusion_lowfreq = fusion_aux['fusion_lowfreq']
        self.loss_fusion_stats = fusion_aux['fusion_stats']

        self.loss_struct = torch.tensor(0.0, device=self.device)
        self.loss_fusion_ms = torch.tensor(0.0, device=self.device)

        if self.radiation_fusion_stage:
            self.loss_G = self.loss_radiation + self.loss_fusion
            self.loss_G.backward()
            return

        if (self.fusion_only_finetune or self.post_refiner_only_finetune or self.fusion_calibrator_finetune) and bool(int(getattr(self.opt, 'fusion_only_zero_aux', 1))):
            self.loss_G = self.loss_fusion
            self.loss_G.backward()
            return

        self.loss_G = self.loss_Gg + self.loss_recon + self.loss_radiation + self.loss_fusion
        self.loss_G.backward()

    def _forward_discriminator(self, x, return_features=False):
        if self.opt.netD == 'unet':
            if return_features:
                global_pred, pixel_pred, features = self.D(x, return_features=True)
                return global_pred, pixel_pred, features
            global_pred, pixel_pred = self.D(x)
            return global_pred, pixel_pred, []

        pred = self.D(x)
        return pred, None, []

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        if self.grad_clip_norm_g > 0.0:
            if self.post_refiner_only_finetune or self.fusion_calibrator_finetune:
                clip_params = (p for p in self.fusion_model.parameters() if p.requires_grad)
            elif self.texture_stage:
                clip_params = itertools.chain(*[module.parameters() for module in self._texture_modules()])
            elif self.radiation_fusion_stage:
                clip_params = itertools.chain(*[module.parameters() for module in self._radiation_fusion_modules()])
            else:
                clip_params = itertools.chain(
                    self.ATF2D.parameters(),
                    self.TCMNet.parameters(),
                    self.G_J.parameters(),
                    self.fusion_model.parameters(),
                )
                if self.decom_enabled:
                    clip_params = itertools.chain(clip_params, self.decom_net.parameters())
                    if self._use_learned_ir_decom():
                        clip_params = itertools.chain(clip_params, self.visible_decom_net.parameters())
                if self.texture_frontend_enabled:
                    clip_params = itertools.chain(clip_params, self.texture_frontend.parameters())
            torch.nn.utils.clip_grad_norm_(
                clip_params,
                self.grad_clip_norm_g,
            )
        self.optimizer_G.step()

        if self._is_gan_enabled():
            self.set_requires_grad(self.D, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            if self.grad_clip_norm_d > 0.0:
                torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip_norm_d)
            self.optimizer_D.step()
        else:
            self.set_requires_grad(self.D, False)
            self.loss_D = torch.tensor(0.0, device=self.device)

    def test(self):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
