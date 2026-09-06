from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        
        # Dropout and Batchnorm has different behavioir during training and test.
        # 原来是 store_true (默认False), 这里保留参数但后续通过 set_defaults 强制开启
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        
        # ======== [修改点：解除 201 张图片的限制，默认测试所有图片] ========
        parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')
        # ====================================================================
        
        parser.add_argument('--save_image', action='store_true', help='save result images.')
        parser.add_argument('--tta', action='store_true', help='use flip test-time augmentation and average predictions')

        # ======== [修改点：修正祖传代码残留，改为红外增强任务相关] ========
        parser.add_argument('--method_name', type=str, default='irenhance', help='short name for your infrared enhancement method')
        # ====================================================================

        # ======== [添加训练参数以兼容模型加载] ========
        parser.add_argument('--model_type', type=str, default='convnext', help='model type for generator')
        parser.add_argument('--decom_enabled', type=int, default=1, help='enable decomposition network')
        parser.add_argument('--decom_base_ch', type=int, default=32, help='base channels for decom net')
        parser.add_argument('--lambda_D', type=float, default=1.0, help='weight for discriminator loss')
        parser.add_argument('--lambda_Gg', type=float, default=1.0, help='weight for GAN loss')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--lambda_ssim', type=float, default=1.0, help='weight for SSIM loss')
        parser.add_argument('--lambda_texture', type=float, default=1.0, help='weight for texture loss')
        parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for identity loss')
        parser.add_argument('--lambda_radiation', type=float, default=0.5, help='weight for radiation loss')
        parser.add_argument('--lambda_decom', type=float, default=1.0, help='weight for decomposition loss')
        parser.add_argument('--lambda_fusion_I', type=float, default=1.5, help='weight for fusion loss')
        parser.add_argument('--lambda_fusion_ssim', type=float, default=2.0, help='weight for fusion SSIM')
        parser.add_argument('--lambda_fusion_psnr', type=float, default=1.0, help='weight for fusion PSNR')
        parser.add_argument('--stage1_epochs', type=int, default=100, help='stage 1 epochs')
        parser.add_argument('--texture_frontend_enabled', type=int, default=0, help='enable texture frontend')
        parser.add_argument('--gan_fixed_scale', type=float, default=1.0, help='fixed scale for GAN loss')
        parser.add_argument('--grad_clip_norm_g', type=float, default=0.5, help='gradient clipping norm')
        parser.add_argument('--grad_clip_norm_d', type=float, default=1.0, help='gradient clipping norm for D')
        parser.add_argument('--radiation_stage1_scale', type=float, default=0.0, help='radiation scale for stage 1')
        parser.add_argument('--radiation_stage2_scale', type=float, default=1.0, help='radiation scale for stage 2')
        parser.add_argument('--radiation_region_block', type=int, default=8, help='region block size')
        parser.add_argument('--radiation_grad_weight', type=float, default=0.5, help='radiation gradient weight')
        parser.add_argument('--recon_ssim_ratio', type=float, default=0.35, help='SSIM ratio in recon loss')
        parser.add_argument('--texture_blur_ksize', type=int, default=5, help='texture blur kernel size')
        parser.add_argument('--texture_mask_floor', type=float, default=0.1, help='texture mask floor')
        parser.add_argument('--thermal_hot_threshold', type=float, default=0.85, help='thermal hot threshold')
        parser.add_argument('--thermal_preserve_strength', type=float, default=0.3, help='thermal preserve strength')
        # ====================================================================

        # rewrite default values
        parser.set_defaults(model='Irenhance')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        
        # ======== [极其重要：强制默认开启 eval 模式，保证测试精度不下降] ========
        parser.set_defaults(eval=True)
        # ====================================================================
        
        self.isTrain = False
        return parser
