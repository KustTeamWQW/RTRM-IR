from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=20, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display; set < 0 to disable visdom')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=200, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        
        # ======== [核心修改：针对高精度大模型的参数优化] ========
        # 1. 延长训练轮数：复杂纹理生成需要更多时间收敛
        parser.add_argument('--niter', type=int, default=300, help='# of epochs to keep the initial learning rate')
        parser.add_argument('--niter_decay', type=int, default=150, help='# of epochs to linearly decay learning rate to zero')
        
        # 2. 降低学习率：现代骨干(ConvNeXt/Swin)遇到 0.001 容易梯度爆炸，0.0001 是标配
        parser.add_argument('--lr', type=float, default=0.000005 , help='initial learning rate for adam')
        # ========================================================
        
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='cosine', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
        # Validation uses an explicit validation split (for example valA/valB/valGT).
        # It is read-only: train.py never backpropagates validation metrics or changes
        # training loss weights from validation/test data.
        parser.add_argument('--val_during_train', type=int, default=0, help='run read-only validation during training on --val_phase')
        parser.add_argument('--val_phase', type=str, default='val', help='dataset phase used for validation, e.g. val -> valA/valB/valGT')
        parser.add_argument('--val_freq', type=int, default=5, help='validate every N epochs when --val_during_train=1')
        parser.add_argument('--val_max_images', type=int, default=64, help='maximum validation images per validation pass')
        parser.add_argument('--val_save_label', type=str, default='best_val', help='checkpoint label saved when validation score improves')
        parser.add_argument('--val_ssim_weight', type=float, default=30.0, help='validation score = PSNR + val_ssim_weight * SSIM')

        # Kept for validation scoring compatibility inside train.py.
        parser.add_argument('--eval_ssim_weight', type=float, default=30.0, help='used only by standalone evaluation scripts')

        parser.set_defaults(dataset_mode='paired', load_size=256, crop_size=256)

        self.isTrain = True
        return parser
