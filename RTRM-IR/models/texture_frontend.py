import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableGaborWaveletFrontend(nn.Module):
    """Dual-branch learnable texture frontend with background suppression.

    The local branch extracts directional/high-frequency details using Gabor
    and Haar-wavelet filters. The global branch uses large-kernel gated context
    to keep long-range contours consistent. A confidence gate suppresses flat
    areas so the output remains a signed texture residual, not a brightness map.
    """

    def __init__(
        self,
        in_channels=1,
        hidden_channels=16,
        gabor_scales=(4.0, 8.0),
        orientations=6,
        wavelet_levels=2,
        gate_init=0.1,
        background_kernel=15,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.gabor_scales = tuple(float(s) for s in gabor_scales)
        self.orientations = int(orientations)
        self.wavelet_levels = int(wavelet_levels)
        self.background_kernel = int(background_kernel)
        if self.background_kernel % 2 == 0:
            self.background_kernel += 1

        gabor_kernels = []
        max_ksize = 0
        for scale in self.gabor_scales:
            max_ksize = max(max_ksize, 2 * int(1.5 * float(scale)) + 1)
        for scale in self.gabor_scales:
            for orient_idx in range(self.orientations):
                theta = orient_idx * math.pi / self.orientations
                kernel = self._create_gabor_kernel(scale, theta)
                pad_total = max_ksize - kernel.shape[-1]
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                kernel = F.pad(kernel, (pad_left, pad_right, pad_left, pad_right))
                gabor_kernels.append(kernel)
        gabor = torch.stack(gabor_kernels, dim=0).unsqueeze(1)
        self.gabor_kernels = nn.Parameter(gabor)

        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 4.0
        lh = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) / 4.0
        hl = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) / 4.0
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 4.0
        wavelets = torch.stack([lh, hl, hh], dim=0).unsqueeze(1)
        self.wavelet_kernels = nn.Parameter(wavelets)
        self.register_buffer('wavelet_ll', ll.view(1, 1, 2, 2))

        gabor_out = len(gabor_kernels) * self.in_channels
        wavelet_out = 3 * self.wavelet_levels * self.in_channels
        feature_channels = self.in_channels + gabor_out + wavelet_out

        self.local_fuse = nn.Sequential(
            nn.Conv2d(feature_channels, self.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.in_channels, kernel_size=1),
        )

        self.global_embed = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=1)
        self.global_context = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=7, padding=3, groups=self.hidden_channels),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=11, padding=5, groups=self.hidden_channels),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels * 2, kernel_size=1),
        )
        self.global_out = nn.Conv2d(self.hidden_channels, self.in_channels, kernel_size=1)

        self.confidence = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels * 3, self.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    @staticmethod
    def _create_gabor_kernel(lambd, theta, gamma=0.5, psi=0.0):
        sigma = 0.56 * float(lambd)
        ksize = 2 * int(1.5 * float(lambd)) + 1
        half = ksize // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32),
            torch.arange(-half, half + 1, dtype=torch.float32),
            indexing='ij',
        )
        x_rot = x * math.cos(theta) + y * math.sin(theta)
        y_rot = -x * math.sin(theta) + y * math.cos(theta)
        kernel = torch.exp(-(x_rot ** 2 + (gamma ** 2) * (y_rot ** 2)) / (2.0 * sigma ** 2))
        kernel = kernel * torch.cos(2.0 * math.pi * x_rot / float(lambd) + psi)
        kernel = kernel - kernel.mean()
        kernel = kernel / (kernel.abs().sum() + 1e-8)
        return kernel

    def _background_suppressed_detail(self, x):
        pad = self.background_kernel // 2
        low = F.avg_pool2d(x, kernel_size=self.background_kernel, stride=1, padding=pad)
        return x - low

    def _gabor_features(self, detail):
        channels = detail.shape[1]
        kernels = self.gabor_kernels.to(dtype=detail.dtype)
        k = kernels.shape[-1]
        pad = k // 2
        weight = kernels.repeat(channels, 1, 1, 1)
        detail_pad = F.pad(detail, (pad, pad, pad, pad), mode='reflect')
        responses = F.conv2d(detail_pad, weight, groups=channels)
        return responses.abs()

    def _wavelet_features(self, detail):
        b, channels, height, width = detail.shape
        current = detail
        features = []
        hf_kernels = self.wavelet_kernels.to(dtype=detail.dtype)
        ll_kernel = self.wavelet_ll.to(dtype=detail.dtype)
        for _ in range(self.wavelet_levels):
            current_pad = F.pad(current, (0, 1, 0, 1), mode='reflect')
            hf_weight = hf_kernels.repeat(channels, 1, 1, 1)
            high = F.conv2d(current_pad, hf_weight, stride=2, groups=channels).abs()
            high = F.interpolate(high, size=(height, width), mode='bilinear', align_corners=False)
            features.append(high)
            ll_weight = ll_kernel.repeat(channels, 1, 1, 1)
            current = F.conv2d(current_pad, ll_weight, stride=2, groups=channels)
            if min(current.shape[-2:]) < 2:
                break
        return torch.cat(features, dim=1)

    def _local_variance(self, x, kernel_size=7):
        pad = kernel_size // 2
        mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        mean_sq = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=pad)
        return torch.sqrt(torch.clamp(mean_sq - mean * mean, min=1e-6))

    def _global_features(self, detail):
        feat = self.global_embed(detail)
        value, gate = self.global_context(feat).chunk(2, dim=1)
        feat = value * torch.sigmoid(gate)
        return self.global_out(feat)

    def forward(self, x):
        detail = self._background_suppressed_detail(x)
        gabor = self._gabor_features(detail)
        wavelet = self._wavelet_features(detail)
        texture = torch.cat([detail, gabor, wavelet], dim=1)
        local_texture = self.local_fuse(texture)
        global_texture = self._global_features(detail)

        variance = self._local_variance(detail)
        confidence = self.confidence(torch.cat([detail.abs(), variance, local_texture.abs()], dim=1))
        branch_gate = self.branch_gate(torch.cat([detail.abs(), local_texture.abs(), global_texture.abs()], dim=1))
        enhanced = branch_gate * local_texture + (1.0 - branch_gate) * global_texture
        return detail + confidence * self.gate * enhanced
