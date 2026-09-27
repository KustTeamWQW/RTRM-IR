import torch
import torch.nn as nn
import torch.nn.functional as F


def _haar_filters(device, dtype):
    return torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5]],      # LL
            [[-0.5, -0.5], [0.5, 0.5]],    # LH
            [[-0.5, 0.5], [-0.5, 0.5]],    # HL
            [[0.5, -0.5], [-0.5, 0.5]],    # HH
        ],
        device=device,
        dtype=dtype,
    ).view(4, 1, 2, 2)


def haar_wavelet_decompose(x):
    """Return spatial-domain Haar structure/detail maps.

    The coefficient definition follows the paper notation:
        S = LL
        T = LH + HL + HH

    LL and high-frequency coefficients are projected back to the input
    resolution with the inverse Haar basis, so S + T reconstructs x.
    """
    b, c, h, w = x.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x_in = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    else:
        x_in = x

    filters = _haar_filters(x.device, x.dtype)
    weight = filters.repeat(c, 1, 1, 1)
    coeff = F.conv2d(x_in, weight, stride=2, groups=c)
    coeff = coeff.view(b, c, 4, coeff.shape[-2], coeff.shape[-1])

    structure_coeff = torch.zeros_like(coeff)
    texture_coeff = torch.zeros_like(coeff)
    structure_coeff[:, :, 0] = coeff[:, :, 0]
    texture_coeff[:, :, 1:] = coeff[:, :, 1:]

    structure = F.conv_transpose2d(
        structure_coeff.view(b, 4 * c, coeff.shape[-2], coeff.shape[-1]),
        weight,
        stride=2,
        groups=c,
    )
    texture = F.conv_transpose2d(
        texture_coeff.view(b, 4 * c, coeff.shape[-2], coeff.shape[-1]),
        weight,
        stride=2,
        groups=c,
    )
    return structure[:, :, :h, :w], texture[:, :, :h, :w]


def _box_filter(x, kernel_size):
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.avg_pool2d(x_pad, kernel_size=kernel_size, stride=1)


def guided_filter_decompose(x, kernel_size=15, eps=1e-2):
    """Fixed edge-preserving structure/texture decomposition.

    S is the self-guided structure layer, and T is the residual texture:
        S = guided_filter(x, x)
        T = x - S
    """
    mean_x = _box_filter(x, kernel_size)
    mean_xx = _box_filter(x * x, kernel_size)
    var_x = mean_xx - mean_x * mean_x

    a = var_x / (var_x + eps)
    b = mean_x * (1.0 - a)
    mean_a = _box_filter(a, kernel_size)
    mean_b = _box_filter(b, kernel_size)

    structure = mean_a * x + mean_b
    texture = x - structure
    return structure, texture


def radiation_texture_decompose(x, kernel_size=31, smooth_iters=2):
    """Fixed radiation/texture decomposition.

    The radiation layer is deliberately smooth and should not preserve edges.
    Edges, object contours, and structural details remain in the texture layer.
    """
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    smooth_iters = max(1, int(smooth_iters))

    radiation = x
    for _ in range(smooth_iters):
        radiation = _box_filter(radiation, kernel_size)
    radiation = torch.clamp(radiation, 0.0, 1.0)
    texture = x - radiation
    return radiation, texture


def gt_light_radiation_texture_decompose(x, kernel_size=7, raw_blend=0.65):
    """Guided-radiation / residual-texture split.

    S uses self-guided filtering to keep brightness and edge-aware structure.
    T stays tied to the pure large-kernel smoothing residual so the texture
    branch input remains unchanged.
    """
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    smooth = _box_filter(x, kernel_size)
    radiation_kernel = min(kernel_size, 19)
    if radiation_kernel % 2 == 0:
        radiation_kernel += 1
    radiation, _ = guided_filter_decompose(x, kernel_size=radiation_kernel, eps=1e-3)
    radiation = torch.clamp(radiation, 0.0, 1.0)
    texture = x - smooth
    return radiation, texture


class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class IRWaveletSeparationNet(nn.Module):
    """Learned raw-IR separator anchored by Haar low/high-frequency components."""

    def __init__(self, in_channels=1, base_ch=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            _ResidualBlock(base_ch),
            _ResidualBlock(base_ch),
        )
        self.structure_head = nn.Conv2d(base_ch, in_channels, kernel_size=3, stride=1, padding=1)
        self.structure_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        wavelet_structure, _ = haar_wavelet_decompose(x)
        feat = self.encoder(x)
        structure_delta = torch.tanh(self.structure_head(feat))
        S = wavelet_structure + torch.sigmoid(self.structure_scale) * structure_delta
        S = torch.clamp(S, 0.0, 1.0)
        T = x - S
        return S, T


class IRGuidedSeparationNet(nn.Module):
    """Learned raw-IR separator anchored by fixed edge-preserving filtering."""

    def __init__(self, in_channels=1, base_ch=32, kernel_size=15, eps=1e-2):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            _ResidualBlock(base_ch),
            _ResidualBlock(base_ch),
        )
        self.structure_head = nn.Conv2d(base_ch, in_channels, kernel_size=3, stride=1, padding=1)
        self.structure_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        guided_structure, _ = guided_filter_decompose(
            x,
            kernel_size=self.kernel_size,
            eps=self.eps,
        )
        feat = self.encoder(x)
        structure_delta = torch.tanh(self.structure_head(feat))
        S = guided_structure + torch.sigmoid(self.structure_scale) * structure_delta
        S = torch.clamp(S, 0.0, 1.0)
        T = x - S
        return S, T


class IRRadiationTextureSeparationNet(nn.Module):
    """Learned raw-IR separator for radiation and texture layers.

    Each output path is a simple 5-layer CNN: two shared layers followed by
    three branch-specific layers. The radiation branch learns a bounded
    adjustment around a low-frequency raw-IR anchor instead of generating the
    radiation layer from scratch. The texture layer is the residual T = I - S,
    so the split keeps the input information.
    """

    def __init__(self, in_channels=1, base_ch=32, kernel_size=31, smooth_iters=2, raw_blend=0.0, scale_init=-2.0):
        super().__init__()
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.smooth_iters = max(1, int(smooth_iters))
        self.raw_blend = max(0.0, min(0.95, float(raw_blend)))
        mid_ch = max(16, base_ch // 2)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.radiation_branch = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_channels, kernel_size=3, stride=1, padding=1),
        )
        self.texture_branch = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_channels, kernel_size=3, stride=1, padding=1),
        )
        self.texture_suppression = nn.Parameter(torch.tensor(-1.5))
        self.radiation_delta_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self._init_anchor_split()

    def _init_anchor_split(self):
        for branch in (self.radiation_branch, self.texture_branch):
            last = branch[-1]
            if isinstance(last, nn.Conv2d):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)

    def forward(self, x):
        radiation_anchor = x
        for _ in range(self.smooth_iters):
            radiation_anchor = _box_filter(radiation_anchor, self.kernel_size)
        radiation_anchor = self.raw_blend * x + (1.0 - self.raw_blend) * radiation_anchor
        feat = self.shared(x)
        radiation_delta = torch.sigmoid(self.radiation_delta_scale) * torch.tanh(self.radiation_branch(feat))
        texture_hint = torch.tanh(self.texture_branch(feat))
        texture_scale = 0.10 * torch.sigmoid(self.texture_suppression)
        S = radiation_anchor + radiation_delta - texture_scale * texture_hint
        S = torch.clamp(S, 0.0, 1.0)
        T = x - S
        return S, T


class DecomNet(nn.Module):
    """Lightweight structure-texture decomposition network.

    The structure map is anchored by a low-pass version of the input, while
    the texture map is a signed high-frequency residual. This avoids the
    unconstrained S/T split drifting during training.
    """
    def __init__(self, in_channels=1, base_ch=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.structure_head = nn.Conv2d(base_ch, in_channels, kernel_size=3, stride=1, padding=1)
        self.texture_head = nn.Conv2d(base_ch, in_channels, kernel_size=3, stride=1, padding=1)
        self.structure_scale = nn.Parameter(torch.tensor(-2.0))
        self.texture_scale = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _avg_blur(x, kernel_size):
        pad = kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        return F.avg_pool2d(x_pad, kernel_size=kernel_size, stride=1)

    def _multi_scale_detail(self, x):
        fine_base = self._avg_blur(x, 3)
        mid_base = self._avg_blur(x, 7)
        coarse_base = self._avg_blur(x, 15)

        fine_detail = x - fine_base
        mid_detail = fine_base - mid_base
        coarse_detail = mid_base - coarse_base

        base = coarse_base
        detail = 0.90 * fine_detail + 0.10 * mid_detail
        return base, detail

    def forward(self, x):
        base, detail = self._multi_scale_detail(x)
        feat = self.encoder(x)

        structure_residual = torch.tanh(self.structure_head(feat))
        texture_residual = torch.tanh(self.texture_head(feat))

        S = base + torch.sigmoid(self.structure_scale) * structure_residual
        S = torch.clamp(S, 0.0, 1.0)
        T = detail + torch.sigmoid(self.texture_scale) * texture_residual
        return S, T

def decom_loss_fn(I, S, T, lambda_recon=1.0, lambda_corr=0.5, lambda_reg=0.1):
    """USTDFuse decomposition losses.

    Args:
        I: original image
        S: predicted structure
        T: predicted texture
    """
    # 1. Reconstruction: I should be recoverable from S + T
    recon = F.l1_loss(S + T, I)

    # 2. Correlation loss: S and T should be statistically independent (orthogonal)
    #    Compute channel-wise correlation and minimize its absolute value
    b, c, h, w = S.shape
    S_flat = S.view(b, c, -1)
    T_flat = T.view(b, c, -1)

    S_mean = S_flat.mean(dim=-1, keepdim=True)
    T_mean = T_flat.mean(dim=-1, keepdim=True)

    S_centered = S_flat - S_mean
    T_centered = T_flat - T_mean

    corr = (S_centered * T_centered).sum(dim=-1)
    S_var = (S_centered ** 2).sum(dim=-1)
    T_var = (T_centered ** 2).sum(dim=-1)

    # Pearson correlation coefficient
    corr_coef = corr / (torch.sqrt(S_var * T_var) + 1e-8)
    corr_loss = corr_coef.abs().mean()

    # 3. Regularity: 鏀圭敤姊害鎯╃綒锛堥紦鍔卞钩婊戯級鑰屼笉鏄?L1锛堜細鍘嬪埗骞呭害锛?    # 璁＄畻 T 鐨勬搴︼紝榧撳姳骞虫粦浣嗕笉鎯╃綒骞呭害
    T_dy = torch.abs(T[:, :, 1:, :] - T[:, :, :-1, :])
    T_dx = torch.abs(T[:, :, :, 1:] - T[:, :, :, :-1])
    reg_loss = T_dy.mean() + T_dx.mean()

    loss = lambda_recon * recon + lambda_corr * corr_loss + lambda_reg * reg_loss
    return loss, {
        'decom_recon': recon,
        'decom_corr': corr_loss,
        'decom_reg': reg_loss,
    }
