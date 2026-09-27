import torch
import torch.nn as nn
from torch.nn import init
import functools
import math
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn import Conv2d
import numpy as np
from torch.nn.functional import l1_loss
import torch


class Identity(nn.Module):
    def forward(self, x):
        return x




def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptionsï¼Žã€€
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        total_epochs = max(1, int(opt.niter) + int(opt.niter_decay))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[], model_type='convnext', texture_vq_enabled=True, texture_vq_codebook_size=512,
             texture_vq_beta=0.25, texture_ar_layers=4, texture_cond_drop=0.1, texture_cfg_scale=1.5,
             texture_prior_strength=0.5):
    
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'modern_unet':
        if model_type == 'swin':
            net = OfficialSwinTextureGenerator(
                input_nc,
                output_nc,
                ngf=ngf,
                texture_vq_enabled=texture_vq_enabled,
                texture_vq_codebook_size=texture_vq_codebook_size,
                texture_vq_beta=texture_vq_beta,
                texture_ar_layers=texture_ar_layers,
                texture_cond_drop=texture_cond_drop,
                texture_cfg_scale=texture_cfg_scale,
                texture_prior_strength=texture_prior_strength,
            )
        else:
            net = ModernUNetGenerator(
                input_nc,
                output_nc,
                ngf=ngf,
                model_type=model_type,
                texture_vq_enabled=texture_vq_enabled,
                texture_vq_codebook_size=texture_vq_codebook_size,
                texture_vq_beta=texture_vq_beta,
                texture_ar_layers=texture_ar_layers,
                texture_cond_drop=texture_cond_drop,
                texture_cfg_scale=texture_cfg_scale,
                texture_prior_strength=texture_prior_strength,
            )
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'unet':
        net = UNetDiscriminator(input_nc, ndf)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 9, stride=4, padding=4, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=4, padding=3, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=64, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        self.convf = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=64, num_groups=8, affine=True),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        R = x[:, 0:1, :, :]
        G = x[:, 1:2, :, :]
        B = x[:, 2:3, :, :]
        xR = torch.unsqueeze(self.conv(R), 1)
        xG = torch.unsqueeze(self.conv(G), 1)
        xB = torch.unsqueeze(self.conv(B), 1)
        x = torch.cat([xR, xG, xB], 1)
        x, _ = torch.min(x, dim=1)
        return self.convf(x)


class SKConv(nn.Module):
    def __init__(self, outfeatures=64, infeatures=1, M=4, L=32):

        super(SKConv, self).__init__()
        self.M = M
        self.convs = nn.ModuleList([])
        in_conv = InConv(in_ch=infeatures, out_ch=outfeatures)
        for i in range(M):
            if i == 0:
                self.convs.append(in_conv)
            else:
                self.convs.append(nn.Sequential(
                    nn.Upsample(scale_factor=1 / (2 ** i), mode='bilinear', align_corners=True),
                    in_conv,
                    nn.Upsample(scale_factor=2 ** i, mode='bilinear', align_corners=True)
                ))
        self.fc = nn.Linear(outfeatures, L)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(L, outfeatures)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def _to_2tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value, value)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class OfficialPatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=2, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = _to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            b, c, h, w = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(b, c, h, w)
        return x


class OfficialPatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        b, c, h, w = x.shape
        pad_b = h % 2
        pad_r = w % 2
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
            h = h + pad_b
            w = w + pad_r

        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, 4 * c)
        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(b, h // 2, w // 2, 2 * c).permute(0, 3, 1, 2).contiguous()
        return x


class OfficialSwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, drop_path_rate=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, hp, wp, 1), device=x.device)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    img_mask[:, h_slice, w_slice, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.permute(0, 3, 1, 2).contiguous()


class OfficialSwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, drop_path_rates=None, downsample=True):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        self.blocks = nn.ModuleList([
            OfficialSwinBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                drop_path_rate=drop_path_rates[i] if isinstance(drop_path_rates, (list, tuple)) else float(drop_path_rates),
            )
            for i in range(depth)
        ])
        self.downsample = OfficialPatchMerging(dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return skip, x


class OfficialSwinTextureGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, texture_vq_enabled=True,
                 texture_vq_codebook_size=512, texture_vq_beta=0.25, texture_ar_layers=4,
                 texture_cond_drop=0.1, texture_cfg_scale=1.5, texture_prior_strength=0.5):
        super().__init__()
        self.texture_prior = None
        self._last_fusion_condition = None

        self.patch_embed = OfficialPatchEmbed(in_chans=input_nc, embed_dim=ngf, patch_size=2, norm_layer=nn.LayerNorm)
        depths = [2, 2, 6, 2]
        self.swin_drop_path = [0.0] * sum(depths)
        self.stage1 = OfficialSwinStage(ngf, depth=depths[0], num_heads=max(1, ngf // 32), window_size=7,
                                        drop_path_rates=self.swin_drop_path[0:depths[0]], downsample=True)
        self.stage2 = OfficialSwinStage(ngf * 2, depth=depths[1], num_heads=max(1, (ngf * 2) // 32), window_size=7,
                                        drop_path_rates=self.swin_drop_path[depths[0]:sum(depths[:2])], downsample=True)
        self.stage3 = OfficialSwinStage(ngf * 4, depth=depths[2], num_heads=max(1, (ngf * 4) // 32), window_size=7,
                                        drop_path_rates=self.swin_drop_path[sum(depths[:2]):sum(depths[:3])], downsample=True)
        self.stage4 = OfficialSwinStage(ngf * 8, depth=depths[3], num_heads=max(1, (ngf * 8) // 32), window_size=7,
                                        drop_path_rates=self.swin_drop_path[sum(depths[:3]):], downsample=False)

        if texture_vq_enabled:
            self.texture_prior = TextureVQARPrior(
                ngf * 8,
                cond_channels=ngf * 4,
                codebook_size=texture_vq_codebook_size,
                beta=texture_vq_beta,
                ar_layers=texture_ar_layers,
                cond_drop_prob=texture_cond_drop,
                cfg_scale=texture_cfg_scale,
                prior_strength=texture_prior_strength,
            )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.dblock1 = OfficialSwinStage(ngf * 4, depth=2, num_heads=max(1, (ngf * 4) // 32), window_size=7,
                                         drop_path_rates=[0.0, 0.0], downsample=False)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 4 * 2, ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.dblock2 = OfficialSwinStage(ngf * 2, depth=2, num_heads=max(1, (ngf * 2) // 32), window_size=7,
                                         drop_path_rates=[0.0, 0.0], downsample=False)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 2 * 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.dblock3 = OfficialSwinStage(ngf, depth=2, num_heads=max(1, ngf // 32), window_size=7,
                                         drop_path_rates=[0.0, 0.0], downsample=False)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 2, output_nc, kernel_size=3, stride=1, padding=1)
        )
        self.final_act = nn.Tanh()

    def _pad_to_multiple(self, x, multiple=16):
        h, w = x.shape[2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (h, w)

    def forward(self, x):
        x, orig_size = self._pad_to_multiple(x, multiple=16)
        x = self.patch_embed(x)

        d1, x = self.stage1(x)
        d2, x = self.stage2(x)
        d3, x = self.stage3(x)
        d4, _ = self.stage4(x)

        if self.texture_prior is not None:
            d4 = self.texture_prior(d4, d3)
            self._last_fusion_condition = self.texture_prior.get_fusion_condition()
        else:
            self._last_fusion_condition = None

        u1, _ = self.dblock1(self.up1(d4))
        if d3.shape[2:] != u1.shape[2:]:
            d3 = F.interpolate(d3, size=u1.shape[2:], mode='bilinear', align_corners=False)

        u2, _ = self.dblock2(self.up2(torch.cat([u1, d3], dim=1)))
        if d2.shape[2:] != u2.shape[2:]:
            d2 = F.interpolate(d2, size=u2.shape[2:], mode='bilinear', align_corners=False)

        u3, _ = self.dblock3(self.up3(torch.cat([u2, d2], dim=1)))
        if d1.shape[2:] != u3.shape[2:]:
            d1 = F.interpolate(d1, size=u3.shape[2:], mode='bilinear', align_corners=False)

        out = self.final_act(self.up4(torch.cat([u3, d1], dim=1)))
        if out.shape[2:] != orig_size:
            out = out[:, :, :orig_size[0], :orig_size[1]].contiguous()
        return out

    def get_aux_losses(self):
        if self.texture_prior is None:
            zero = next(self.parameters()).new_tensor(0.0)
            return {'vq': zero, 'prior': zero, 'perplexity': zero}
        return self.texture_prior.get_aux_losses()

    def get_fusion_condition(self, size=None):
        if self._last_fusion_condition is None:
            zero = next(self.parameters()).new_zeros(1, 1, 1, 1)
            return zero if size is None else F.interpolate(zero, size=size, mode='bilinear', align_corners=False)
        condition = self._last_fusion_condition
        if size is not None and condition.shape[2:] != size:
            condition = F.interpolate(condition, size=size, mode='bilinear', align_corners=False)
        return condition


class ResnetGWithIntermediate(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=6, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', filtering='guided', r=10, eps=1e-3):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGWithIntermediate, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.filtering = filtering

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        model_up_part = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_up_part += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                 kernel_size=3, stride=2,
                                                 padding=1, output_padding=1,
                                                 bias=use_bias),
                              norm_layer(int(ngf * mult / 2)),
                              nn.ReLU(True)]
        model_up_part += [nn.ReflectionPad2d(3)]
        model_up_part += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_up_part += [nn.Tanh()]

        self.downsampling = nn.Sequential(*model)
        self.upsampling = nn.Sequential(*model_up_part)

        if self.filtering is not None:
            if self.filtering == 'max':
                self.last_layer = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
            elif self.filtering == 'guided':
                self.last_layer = GuidedFilter(r=r, eps=eps)

    def forward(self, x):
        """Standard forward"""
        down_out = self.downsampling(x)
        up_out = self.upsampling(down_out)
        # rescale to [0,1]
        up_out = (up_out + 1) / 2

        # rgb2gray
        guidance = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]
        # rescale to [0,1]
        guidance = (guidance + 1) / 2
        guidance = torch.unsqueeze(guidance, dim=1)

        if up_out.shape[2:4] != guidance.shape[2:4]:
            up_out = F.interpolate(up_out, size=guidance.shape[2:4], mode='nearest')

        # up_out = self.last_layer(guidance, up_out)
        return self.last_layer(guidance, up_out), up_out


# Guided image filtering for grayscale images
class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3, gpu_ids=None):  # only work for gpu case at this moment
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        self.boxfilter = nn.AvgPool2d(kernel_size=2 * self.r + 1, stride=1, padding=self.r)

    def forward(self, I, p):
        """
        I -- guidance image, should be [0, 1]
        p -- filtering input image, should be [0, 1]
        """

        # N = self.boxfilter(self.tensor(p.size()).fill_(1))
        N = self.boxfilter(torch.ones(p.size()))

        if I.is_cuda:
            N = N.cuda()

        '''print(N.shape)
        print(I.shape)

        print('-----------')'''

        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I * p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(I * I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        return mean_a * I + mean_b


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        # model += [nn.Linear()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return (torch.clamp(self.model(input), -1, 1) + 1) / 2
        # return self.model(input)





class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetTransGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=6, norm_layer=nn.BatchNorm2d, use_dropout=False, r=10,
                 eps=1e-3):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetTransGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetAlignedSkipBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                          innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetAlignedSkipBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                              norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetAlignedSkipBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetAlignedSkipBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetAlignedSkipBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetAlignedSkipBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                          norm_layer=norm_layer)  # add the outermost layer

        self.guided_filter = GuidedFilter(r=r, eps=eps)

    def forward(self, x):
        if x.shape[1] > 1:
            # rgb2gray
            guidance = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]
        else:
            guidance = x
        # rescale to [0,1]
        # guidance = (guidance + 1) / 2
        guidance = torch.unsqueeze(guidance, dim=1)

        trans_raw = self.model(x) # transmission ranges [0,1]

        if trans_raw.shape[2:4] != guidance.shape[2:4]:
            trans_raw = F.interpolate(trans_raw,size=guidance.shape[2:4], mode='nearest')
        a = self.guided_filter(guidance, trans_raw)

        return a


# for trans refination
class UnetAlignedSkipBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetAlignedSkipBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetAlignedSkipBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            # up = [uprelu, upconv, nn.Tanh()]
            up = [uprelu, upconv, downrelu]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            y = self.model(x)
            # print(x.shape, y.shape)
            if x.shape != y.shape:
                y = F.interpolate(y, size=x.shape[2:4], mode='nearest')
            return torch.cat([x, y], 1)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """

        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            # up = [uprelu, upconv, nn.Tanh()]
            up = [uprelu, upconv, downrelu]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    '''def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            print(len(x))
            print('x:', x.shape)
            print('self.model(x):', self.model(x).shape)
            return torch.cat([x, self.model(x)], 1)'''

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            y = self.model(x)
            # print(x.shape, y.shape)
            if x.shape != y.shape:
                y = F.interpolate(y, size=x.shape[2:4], mode='nearest')
            return torch.cat([x, y], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


# Defines the Multiscale-PatchGAN discriminator with the specified arguments.
class MultiDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(MultiDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # cannot deal with use_sigmoid=True case at thie moment
        assert (use_sigmoid == False)

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        scale1 = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            scale1 += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        self.scale1 = nn.Sequential(*scale1)
        scale1_output = []
        scale1_output += [
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        scale1_output += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # compress to 1 channel
        self.scale1_output = nn.Sequential(*scale1_output)

        scale2 = []
        nf_mult = nf_mult
        for n in range(3, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            scale2 += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        scale2 += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        scale2 += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            scale2 += [nn.Sigmoid()]

        self.scale2 = nn.Sequential(*scale2)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            scale1 = nn.parallel.data_parallel(self.scale1, input, self.gpu_ids)
            output1 = nn.parallel.data_parallel(self.scale1_output, scale1, self.gpu_ids)
            output2 = nn.parallel.data_parallel(self.scale2, scale1, self.gpu_ids)
        else:
            scale1 = self.scale1(input)
            output1 = self.scale1_output(scale1)
            output2 = self.scale2(scale1)

        return output1, output2

# =========================================================================
# 1. 官方验证: Facebook Research ConvNeXt Block
# Source: https://github.com/facebookresearch/ConvNeXt
# =========================================================================
class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block from official Meta AI repository"""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity() # 简化了 drop_path，保证不依赖 timm

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


# =========================================================================
# 2. 官方验证: SwinIR (ICCV 2021) 图像恢复专用 Swin Block
# Source: https://github.com/JingyunLiang/SwinIR
# 核心优势：完美处理任意分辨率的 Padding 问题，不产生伪影
# =========================================================================
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block (from SwinIR)."""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        shortcut = x
        x = self.norm1(x)

        # 官方精髓：动态 padding，防止非 window_size 倍数的图像报错
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_x = x
            attn_mask = None

        # W-MSA/SW-MSA
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 移除 padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 3, 1, 2)


class WindowAttentionV2(nn.Module):
    """Window based multi-head self-attention (Swin Transformer V2 style)."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, pretrained_window_size=(0, 0)):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.pretrained_window_size = pretrained_window_size

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # Relative coords table for continuous relative position bias.
        relative_coords_h = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij'))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)

        if pretrained_window_size[0] > 0 and pretrained_window_size[1] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (window_size[1] - 1)

        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        self.register_buffer('relative_coords_table', relative_coords_table)

        # Pair-wise relative position index.
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
            qkv = F.linear(x, self.qkv.weight, qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Cosine attention with trainable logit scale.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlockV2(nn.Module):
    """Swin Transformer V2 block for image restoration branches."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0, pretrained_window_size=(0, 0)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionV2(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=True,
            pretrained_window_size=pretrained_window_size,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 3, 1, 2)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mask_type not in {'A', 'B'}:
            raise ValueError('mask_type must be A or B')
        self.mask_type = mask_type
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, kernel_h, kernel_w = self.weight.shape
        center_h = kernel_h // 2
        center_w = kernel_w // 2
        self.mask[:, :, center_h + 1:, :] = 0
        self.mask[:, :, center_h, center_w + (1 if mask_type == 'B' else 0):] = 0

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class VectorQuantizer2D(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        batch_size, channels, height, width = z.shape
        flat_z = z.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(batch_size, height, width, channels)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        quantized = z + (quantized - z).detach()

        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_z.dtype)
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())

        return quantized, {
            'vq': codebook_loss,
            'commit': self.beta * commitment_loss,
            'perplexity': perplexity,
        }


class AutoRegressiveTexturePrior(nn.Module):
    def __init__(self, channels, cond_channels=None, num_layers=4):
        super().__init__()
        hidden_channels = channels
        if cond_channels is None:
            cond_channels = channels
        self.entry = MaskedConv2d('A', channels, hidden_channels, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([
            MaskedConv2d('B', hidden_channels, hidden_channels, kernel_size=3, padding=1)
            for _ in range(max(1, num_layers))
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(num_groups=max(1, min(8, hidden_channels)), num_channels=hidden_channels)
            for _ in range(max(1, num_layers))
        ])
        self.input_proj = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.cond_proj = nn.Conv2d(cond_channels, hidden_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def _forward_single(self, latent, cond_feat):
        hidden = self.input_proj(latent)
        hidden = F.gelu(self.entry(hidden) + cond_feat)
        for conv, norm in zip(self.layers, self.norms):
            hidden = F.gelu(norm(conv(hidden) + cond_feat))
        return self.out_proj(hidden)

    def forward(self, latent, cond, cfg_scale=1.0, cond_drop_prob=0.0):
        if cond is None:
            cond = torch.zeros_like(latent)
        if cond.shape[2:] != latent.shape[2:]:
            cond = F.interpolate(cond, size=latent.shape[2:], mode='bilinear', align_corners=False)

        if self.training and cond_drop_prob > 0:
            keep_mask = (torch.rand(latent.shape[0], 1, 1, 1, device=latent.device) >= cond_drop_prob).to(latent.dtype)
            cond = cond * keep_mask

        cond_feat = self.cond_proj(cond)
        guided = self._forward_single(latent, cond_feat)
        if self.training or abs(float(cfg_scale) - 1.0) < 1e-6:
            return guided

        unguided = self._forward_single(latent, torch.zeros_like(cond_feat))
        return unguided + cfg_scale * (guided - unguided)


class TextureVQARPrior(nn.Module):
    def __init__(self, channels, cond_channels=None, codebook_size=512, beta=0.25, ar_layers=4, cond_drop_prob=0.1, cfg_scale=1.5,
                 prior_strength=0.5):
        super().__init__()
        self.quantizer = VectorQuantizer2D(codebook_size, channels, beta=beta)
        self.prior = AutoRegressiveTexturePrior(channels, cond_channels=cond_channels, num_layers=ar_layers)
        self.cond_drop_prob = cond_drop_prob
        self.cfg_scale = cfg_scale
        self.prior_strength = prior_strength
        self._last_losses = {
            'vq': torch.tensor(0.0),
            'prior': torch.tensor(0.0),
            'perplexity': torch.tensor(0.0),
        }
        self._last_condition = torch.tensor(0.0)

    def forward(self, latent, cond):
        quantized, q_stats = self.quantizer(latent)
        prior_delta = self.prior(
            quantized,
            cond,
            cfg_scale=self.cfg_scale,
            cond_drop_prob=self.cond_drop_prob,
        )
        prior_recon = quantized + prior_delta
        prior_loss = F.smooth_l1_loss(prior_recon, latent.detach())
        enhanced = latent + self.prior_strength * (prior_recon - latent)
        condition = prior_delta.abs().mean(dim=1, keepdim=True)
        condition = condition / (condition.amax(dim=(2, 3), keepdim=True) + 1e-6)

        self._last_losses = {
            'vq': q_stats['vq'] + q_stats['commit'],
            'prior': prior_loss,
            'perplexity': q_stats['perplexity'].detach(),
        }
        self._last_condition = condition
        return enhanced

    def get_aux_losses(self):
        return self._last_losses

    def get_fusion_condition(self):
        return self._last_condition


# =========================================================================
# 3. 高精度双主干 U-Net 生成器 (整合官方 Block)
# =========================================================================
class ModernUNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, model_type='convnext', texture_vq_enabled=True,
                 texture_vq_codebook_size=512, texture_vq_beta=0.25, texture_ar_layers=4,
                 texture_cond_drop=0.1, texture_cfg_scale=1.5, texture_prior_strength=0.5):
        super().__init__()
        self.model_type = model_type.lower()
        self.texture_prior = None
        self._last_fusion_condition = None
        
        # Encoder
        self.down1 = nn.Sequential(nn.Conv2d(input_nc, ngf, 4, 2, 1), nn.GELU())
        if self.model_type == 'swin2_full':
            self.block1 = self._make_swin2_v2_stack(ngf, depth=2)
        else:
            self.block1 = self._build_block(ngf)
        
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, 4, 2, 1), nn.GELU())
        if self.model_type == 'swin2_full':
            self.block2 = self._make_swin2_v2_stack(ngf * 2, depth=2)
        else:
            self.block2 = self._build_block(ngf*2)
        
        self.down3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, 4, 2, 1), nn.GELU())
        if self.model_type == 'swin2_full':
            self.block3 = self._make_swin2_v2_stack(ngf * 4, depth=6)
        else:
            self.block3 = nn.Sequential(self._build_block(ngf*4), self._build_block(ngf*4))

        self.down4 = nn.Sequential(nn.Conv2d(ngf*4, ngf*8, 4, 2, 1), nn.GELU())
        if self.model_type == 'swin2_full':
            self.block4 = self._make_swin2_v2_stack(ngf * 8, depth=2)
        else:
            self.block4 = nn.Sequential(self._build_block(ngf*8), self._build_block(ngf*8))
        if texture_vq_enabled:
            self.texture_prior = TextureVQARPrior(
                ngf * 8,
                cond_channels=ngf * 4,
                codebook_size=texture_vq_codebook_size,
                beta=texture_vq_beta,
                ar_layers=texture_ar_layers,
                cond_drop_prob=texture_cond_drop,
                cfg_scale=texture_cfg_scale,
                prior_strength=texture_prior_strength,
            )

        # Decoder: use resize-convolution to avoid checkerboard artifacts from transposed convolutions.
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        if self.model_type == 'swin2_full':
            self.dblock1 = self._make_swin2_v2_stack(ngf * 4, depth=2)
        else:
            self.dblock1 = self._build_block(ngf*4)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 4 * 2, ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        if self.model_type == 'swin2_full':
            self.dblock2 = self._make_swin2_v2_stack(ngf * 2, depth=2)
        else:
            self.dblock2 = self._build_block(ngf*2)
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 2 * 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        if self.model_type == 'swin2_full':
            self.dblock3 = self._make_swin2_v2_stack(ngf, depth=2)
        else:
            self.dblock3 = self._build_block(ngf)
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 2, output_nc, kernel_size=3, stride=1, padding=1)
        )
        self.final_act = nn.Tanh()

    def _make_swin2_v2_stack(self, dim, depth=2, window_size=8):
        num_heads = max(1, dim // 32)
        blocks = []
        for i in range(max(1, int(depth))):
            shift = 0 if (i % 2 == 0) else (window_size // 2)
            blocks.append(
                SwinTransformerBlockV2(
                    dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    pretrained_window_size=(window_size, window_size),
                )
            )
        return nn.Sequential(*blocks)

    def _build_block(self, dim):
        if self.model_type == 'convnext':
            return ConvNeXtBlock(dim)
        elif self.model_type == 'swin':
            num_heads = max(1, dim // 32)
            # 成对出现的 Swin Block (Shift 0 和 Shift WindowSize//2)
            return nn.Sequential(
                SwinTransformerBlock(dim, num_heads=num_heads, window_size=8, shift_size=0),
                SwinTransformerBlock(dim, num_heads=num_heads, window_size=8, shift_size=4)
            )
        elif self.model_type == 'swin2':
            num_heads = max(1, dim // 32)
            # Swin V2 保持与 Swin 相同的层级和宽度，不压缩结构，只替换注意力形式。
            return nn.Sequential(
                SwinTransformerBlockV2(dim, num_heads=num_heads, window_size=8, shift_size=0, pretrained_window_size=(0, 0)),
                SwinTransformerBlockV2(dim, num_heads=num_heads, window_size=8, shift_size=4, pretrained_window_size=(0, 0))
            )
        elif self.model_type == 'swin2_full':
            return self._make_swin2_v2_stack(dim, depth=2, window_size=8)
        raise NotImplementedError('Unsupported model_type: %s' % self.model_type)

    def forward(self, x):
        d1 = self.block1(self.down1(x))
        d2 = self.block2(self.down2(d1))
        d3 = self.block3(self.down3(d2))
        d4 = self.block4(self.down4(d3))
        if self.texture_prior is not None:
            d4 = self.texture_prior(d4, d3)
            self._last_fusion_condition = self.texture_prior.get_fusion_condition()
        else:
            self._last_fusion_condition = None
        
        u1 = self.dblock1(self.up1(d4))
        if d3.shape[2:] != u1.shape[2:]:
            d3 = F.interpolate(d3, size=u1.shape[2:], mode='bilinear', align_corners=False)

        u2 = self.dblock2(self.up2(torch.cat([u1, d3], dim=1)))
        if d2.shape[2:] != u2.shape[2:]:
            d2 = F.interpolate(d2, size=u2.shape[2:], mode='bilinear', align_corners=False)

        u3 = self.dblock3(self.up3(torch.cat([u2, d2], dim=1)))
        if d1.shape[2:] != u3.shape[2:]:
            d1 = F.interpolate(d1, size=u3.shape[2:], mode='bilinear', align_corners=False)

        out = self.final_act(self.up4(torch.cat([u3, d1], dim=1)))
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

    def get_aux_losses(self):
        if self.texture_prior is None:
            zero = next(self.parameters()).new_tensor(0.0)
            return {'vq': zero, 'prior': zero, 'perplexity': zero}
        return self.texture_prior.get_aux_losses()

    def get_fusion_condition(self, size=None):
        if self._last_fusion_condition is None:
            zero = next(self.parameters()).new_zeros(1, 1, 1, 1)
            return zero if size is None else F.interpolate(zero, size=size, mode='bilinear', align_corners=False)
        condition = self._last_fusion_condition
        if size is not None and condition.shape[2:] != size:
            condition = F.interpolate(condition, size=size, mode='bilinear', align_corners=False)
        return condition


# =========================================================================
# 4. 官方验证: U-Net GAN Discriminator (CVPR 2020)
# Source: https://github.com/boschresearch/unetgan
# 极大地增强逐像素的特征对齐和高频约束
# =========================================================================
class UNetDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super().__init__()
        # Encoder
        self.down1 = nn.Sequential(nn.Conv2d(input_nc, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.down2 = nn.Sequential(nn.Conv2d(ndf, ndf*2, 4, 2, 1), nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2, True))
        self.down3 = nn.Sequential(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1), nn.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2, True))
        self.down4 = nn.Sequential(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1), nn.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True))

        # Global Head (PatchGAN style)
        self.global_head = nn.Conv2d(ndf*8, 1, 4, 1, 1)

        # Decoder (Pixel-wise Discriminator)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ndf*8, ndf*4, 4, 2, 1), nn.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2, True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ndf*8, ndf*2, 4, 2, 1), nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2, True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(ndf*4, ndf, 4, 2, 1), nn.InstanceNorm2d(ndf), nn.LeakyReLU(0.2, True))

        self.pixel_head = nn.Conv2d(ndf*2, 1, 3, 1, 1)

    def forward(self, x, return_features=False):
        features = []
        if return_features: features.append(x)

        d1 = self.down1(x)
        if return_features: features.append(d1)

        d2 = self.down2(d1)
        if return_features: features.append(d2)

        d3 = self.down3(d2)
        if return_features: features.append(d3)

        d4 = self.down4(d3)
        if return_features: features.append(d4)

        global_out = self.global_head(d4)

        u1 = self.up1(d4)
        if d3.shape[2:] != u1.shape[2:]:
            d3 = F.interpolate(d3, size=u1.shape[2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d3], dim=1)

        u2 = self.up2(u1)
        if d2.shape[2:] != u2.shape[2:]:
            d2 = F.interpolate(d2, size=u2.shape[2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d2], dim=1)

        u3 = self.up3(u2)
        if d1.shape[2:] != u3.shape[2:]:
            d1 = F.interpolate(d1, size=u3.shape[2:], mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, d1], dim=1)

        pixel_out = self.pixel_head(u3)

        if pixel_out.shape[2:] != x.shape[2:]:
            pixel_out = F.interpolate(pixel_out, size=x.shape[2:], mode='bilinear', align_corners=False)

        if return_features:
            return global_out, pixel_out, features
        return global_out, pixel_out
