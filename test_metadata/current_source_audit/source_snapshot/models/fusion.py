import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import ModernUNetGenerator


def _group_norm(channels):
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class FixedSobelGradient(nn.Module):
    """Parameter-free gradient magnitude used as a lightweight full-resolution prior."""

    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        kernel_y = kernel_x.transpose(-1, -2).contiguous()
        self.register_buffer('kernel_x', kernel_x, persistent=False)
        self.register_buffer('kernel_y', kernel_y, persistent=False)

    def forward(self, x):
        gx = F.conv2d(x, self.kernel_x.to(dtype=x.dtype), padding=1)
        gy = F.conv2d(x, self.kernel_y.to(dtype=x.dtype), padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)


class RestormerFeedForward(nn.Module):
    def __init__(self, dim, expansion=2.66):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class RestormerAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('dim must be divisible by num_heads')
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).view(b, c, h, w)
        return self.project_out(out)


class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, expansion=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = RestormerAttention(dim, num_heads=num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = RestormerFeedForward(dim, expansion=expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class JointRestormerEncoder(nn.Module):
    def __init__(self, in_channels=2, dim=64, depth=4, num_heads=4):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[RestormerBlock(dim, num_heads=num_heads) for _ in range(depth)])
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        feat = self.embed(x)
        feat = self.blocks(feat)
        return feat + self.refine(feat)


class SpatialChannelCrossAttention(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.proj_r = nn.Conv2d(1, dim, 1)
        self.proj_t = nn.Conv2d(1, dim, 1)
        self.downsample = nn.AvgPool2d(kernel_size=4, stride=4)
        self.spatial_q_r = nn.Conv2d(dim, dim // 8, 1)
        self.spatial_k_t = nn.Conv2d(dim, dim // 8, 1)
        self.spatial_v_t = nn.Conv2d(dim, dim, 1)
        self.spatial_q_t = nn.Conv2d(dim, dim // 8, 1)
        self.spatial_k_r = nn.Conv2d(dim, dim // 8, 1)
        self.spatial_v_r = nn.Conv2d(dim, dim, 1)
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim * 2 // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2 // 4, dim * 2, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(dim * 2, dim, 1)
        self.out_r = nn.Conv2d(dim, 1, 1)
        self.out_t = nn.Conv2d(dim, 1, 1)

    def forward(self, r, t):
        fr = self.proj_r(r)
        ft = self.proj_t(t)
        fr_down = self.downsample(fr)
        ft_down = self.downsample(ft)

        qr = self.spatial_q_r(fr_down)
        kt = self.spatial_k_t(ft_down)
        vt = self.spatial_v_t(ft_down)
        b, c, h_down, w_down = qr.shape
        qr_flat = qr.view(b, c, -1)
        kt_flat = kt.view(b, c, -1)
        vt_flat = vt.view(b, -1, h_down * w_down)
        attn_r2t = F.softmax(torch.bmm(qr_flat.permute(0, 2, 1), kt_flat) / (c ** 0.5), dim=-1)
        fr_attended_down = torch.bmm(vt_flat, attn_r2t.permute(0, 2, 1)).view(b, self.dim, h_down, w_down)
        fr_attended = F.interpolate(fr_attended_down, size=fr.shape[2:], mode='bilinear', align_corners=False)

        qt = self.spatial_q_t(ft_down)
        kr = self.spatial_k_r(fr_down)
        vr = self.spatial_v_r(fr_down)
        qt_flat = qt.view(b, c, -1)
        kr_flat = kr.view(b, c, -1)
        vr_flat = vr.view(b, -1, h_down * w_down)
        attn_t2r = F.softmax(torch.bmm(qt_flat.permute(0, 2, 1), kr_flat) / (c ** 0.5), dim=-1)
        ft_attended_down = torch.bmm(vr_flat, attn_t2r.permute(0, 2, 1)).view(b, self.dim, h_down, w_down)
        ft_attended = F.interpolate(ft_attended_down, size=ft.shape[2:], mode='bilinear', align_corners=False)

        channel_weight = self.channel_fc(torch.cat([fr, ft], dim=1))
        cw_r, cw_t = torch.chunk(channel_weight, 2, dim=1)
        fr = fr * cw_r + fr_attended
        ft = ft * cw_t + ft_attended
        fused = self.fuse(torch.cat([fr, ft], dim=1))
        return r + self.out_r(fused), t + self.out_t(fused)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _group_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _group_norm(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, norm=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
            ),
            _group_norm(in_channels) if norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class SeparableResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            SeparableConvBlock(channels, channels),
            _group_norm(channels),
            nn.ReLU(inplace=True),
            SeparableConvBlock(channels, channels),
            _group_norm(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=4, dilation=4)
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        out = torch.cat([
            F.relu(self.conv1(x)),
            F.relu(self.conv2(x)),
            F.relu(self.conv3(x)),
            F.relu(self.conv4(x)),
        ], dim=1)
        return self.fusion(out)


class SeparableMultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        branch = in_channels // 4
        self.conv1 = SeparableConvBlock(in_channels, branch, kernel_size=3, padding=1, dilation=1, norm=False)
        self.conv2 = SeparableConvBlock(in_channels, branch, kernel_size=3, padding=2, dilation=2, norm=False)
        self.conv3 = SeparableConvBlock(in_channels, branch, kernel_size=3, padding=3, dilation=3, norm=False)
        self.conv4 = SeparableConvBlock(in_channels, branch, kernel_size=3, padding=4, dilation=4, norm=False)
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        out = torch.cat([
            F.relu(self.conv1(x)),
            F.relu(self.conv2(x)),
            F.relu(self.conv3(x)),
            F.relu(self.conv4(x)),
        ], dim=1)
        return self.fusion(out)


class WeightPredictionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            _group_norm(64),
            nn.ReLU(inplace=True),
            MultiScaleBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class SeparableWeightPredictionNet(nn.Module):
    def __init__(self, channels=64, depth=4):
        super().__init__()
        blocks = [
            SeparableConvBlock(2, channels, kernel_size=5, padding=2),
            _group_norm(channels),
            nn.ReLU(inplace=True),
            SeparableMultiScaleBlock(channels),
        ]
        blocks.extend(SeparableResidualBlock(channels) for _ in range(depth))
        blocks.extend([
            SeparableConvBlock(channels, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        ])
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class PriorGuidedWeightPredictionNet(nn.Module):
    """Predict fusion logits from branch features and explicit reliability priors."""

    def __init__(self, in_channels=8, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            SeparableConvBlock(in_channels, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            SeparableResidualBlock(hidden),
            SeparableResidualBlock(hidden),
            SeparableConvBlock(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class EfficientChannelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        reduction = max(1, dim // 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, reduction, 1),
            nn.GELU(),
            nn.Conv2d(reduction, dim, 1),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        identity = x
        x = x * self.fc(self.gap(x))
        return self.spatial_conv(x) + identity


class SmoothFusionBlock(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.attn1 = EfficientChannelAttention(dim)
        self.attn2 = EfficientChannelAttention(dim)
        self.cross = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1),
        )

    def forward(self, x1, x2):
        fused = self.cross(torch.cat([self.attn1(x1), self.attn2(x2)], dim=1))
        return fused + self.ffn(fused)


class TexturePromptModulation(nn.Module):
    def __init__(self, channels=1, hidden=16):
        super().__init__()
        self.prompt_encoder = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_scale_shift = nn.Conv2d(hidden, channels * 2, kernel_size=1)
        self.local_prompt = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, feat, texture_prompt):
        if texture_prompt.shape[2:] != feat.shape[2:]:
            texture_prompt = F.interpolate(texture_prompt, size=feat.shape[2:], mode='bilinear', align_corners=False)
        scale_shift = self.to_scale_shift(self.prompt_encoder(texture_prompt))
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        scale = 0.25 * torch.tanh(scale)
        shift = 0.10 * torch.tanh(shift)
        local = self.local_prompt(texture_prompt)
        gate = torch.sigmoid(self.gate)
        return feat * (1.0 + gate * scale * local) + gate * shift * local


class SupervisedResidualCorrector(nn.Module):
    def __init__(self, in_channels=10, hidden=48, separable=False, depth=3):
        super().__init__()
        if separable:
            blocks = [
                SeparableConvBlock(in_channels, hidden),
                nn.GELU(),
            ]
            blocks.extend(SeparableResidualBlock(hidden) for _ in range(depth))
            blocks.extend([
                SeparableConvBlock(hidden, hidden, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
                nn.Conv2d(hidden, hidden // 2, kernel_size=1),
                nn.GELU(),
                SeparableConvBlock(hidden // 2, 1),
                nn.Tanh(),
            ])
            self.head = nn.Sequential(*blocks)
        else:
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                ResidualBlock(hidden),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
                ResidualBlock(hidden),
                nn.Conv2d(hidden, hidden // 2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden // 2, 1, kernel_size=3, padding=1),
                nn.Tanh(),
            )
        if separable:
            nn.init.zeros_(self.head[-2].net[-1].weight)
            nn.init.zeros_(self.head[-2].net[-1].bias)
        else:
            nn.init.zeros_(self.head[-2].weight)
            nn.init.zeros_(self.head[-2].bias)

    def forward(self, x):
        return self.head(x)


class LocalPSNRCalibrator(nn.Module):
    """Lightweight full-resolution local gain/bias/residual calibrator."""

    def __init__(self, in_channels=8, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        gain, bias, residual = torch.chunk(self.net(x), 3, dim=1)
        gain = 0.08 * torch.tanh(gain)
        bias = 0.04 * torch.tanh(bias)
        residual = 0.05 * torch.tanh(residual)
        return gain, bias, residual


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFRestorationBlock(nn.Module):
    def __init__(self, channels=64, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2)
        self.gate = SimpleGate()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.ffn_gate = SimpleGate()
        self.ffn2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.gate(y)
        y = y * self.channel_attn(y)
        x = x + self.beta * self.conv2(y)

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.ffn_gate(y)
        return x + self.gamma * self.ffn2(y)


class SameResolutionRestorationRefiner(nn.Module):
    def __init__(self, in_channels=12, channels=64, num_blocks=8, low_resolution=False):
        super().__init__()
        self.low_resolution = bool(low_resolution)
        self.intro = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[NAFRestorationBlock(channels=channels) for _ in range(num_blocks)])
        self.out = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        if self.low_resolution:
            self.hr_correction = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
                nn.GELU(),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
            )
            self.hr_gain = nn.Parameter(torch.tensor(-2.0))
            nn.init.zeros_(self.hr_correction[-1].weight)
            nn.init.zeros_(self.hr_correction[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        original_size = x.shape[2:]
        trunk_input = x
        if self.low_resolution:
            trunk_input = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        feat = self.intro(trunk_input)
        residual = self.out(self.blocks(feat))
        if self.low_resolution:
            residual = F.interpolate(residual, size=original_size, mode='bilinear', align_corners=False)
            residual = residual + torch.sigmoid(self.hr_gain) * self.hr_correction(x)
        return torch.tanh(residual)


class MultiScalePostRefiner(nn.Module):
    """Additional post-refinement residual. It is additive and starts near zero."""

    def __init__(self, in_channels=12, channels=48, num_blocks=3, low_resolution=False, separable=False):
        super().__init__()
        self.low_resolution = bool(low_resolution)
        self.separable = bool(separable)
        mid = channels
        high = channels * 2
        bottleneck = channels * 4
        conv3 = SeparableConvBlock if self.separable else nn.Conv2d
        self.intro = conv3(in_channels, mid, kernel_size=3, padding=1)
        self.enc0 = nn.Sequential(*[NAFRestorationBlock(channels=mid) for _ in range(num_blocks)])
        self.down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            conv3(mid, high, kernel_size=3, padding=1),
        ) if self.separable else nn.Conv2d(mid, high, kernel_size=3, stride=2, padding=1)
        self.enc1 = nn.Sequential(*[NAFRestorationBlock(channels=high) for _ in range(num_blocks)])
        self.down2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            conv3(high, bottleneck, kernel_size=3, padding=1),
        ) if self.separable else nn.Conv2d(high, bottleneck, kernel_size=3, stride=2, padding=1)
        self.body = nn.Sequential(*[NAFRestorationBlock(channels=bottleneck) for _ in range(num_blocks + 1)])
        self.up1_reduce = nn.Conv2d(bottleneck, high, kernel_size=1)
        self.dec1 = nn.Sequential(*[NAFRestorationBlock(channels=high) for _ in range(num_blocks)])
        self.up0_reduce = nn.Conv2d(high, mid, kernel_size=1)
        self.dec0 = nn.Sequential(*[NAFRestorationBlock(channels=mid) for _ in range(num_blocks)])
        self.out = conv3(mid, 1, kernel_size=3, padding=1)
        if self.low_resolution:
            self.hr_correction = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
                nn.GELU(),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
            )
            self.hr_gain = nn.Parameter(torch.tensor(-2.0))
            nn.init.zeros_(self.hr_correction[-1].weight)
            nn.init.zeros_(self.hr_correction[-1].bias)
        if self.separable:
            nn.init.zeros_(self.out.net[-1].weight)
            nn.init.zeros_(self.out.net[-1].bias)
        else:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        original_size = x.shape[2:]
        trunk_input = x
        if self.low_resolution:
            trunk_input = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        feat0 = self.enc0(self.intro(trunk_input))
        feat1 = self.enc1(self.down1(feat0))
        body = self.body(self.down2(feat1))
        up1 = F.interpolate(self.up1_reduce(body), size=feat1.shape[2:], mode='bilinear', align_corners=False)
        up1 = self.dec1(up1 + feat1)
        up0 = F.interpolate(self.up0_reduce(up1), size=feat0.shape[2:], mode='bilinear', align_corners=False)
        up0 = self.dec0(up0 + feat0)
        residual = self.out(up0)
        if self.low_resolution:
            residual = F.interpolate(residual, size=original_size, mode='bilinear', align_corners=False)
            residual = residual + torch.sigmoid(self.hr_gain) * self.hr_correction(x)
        return torch.tanh(residual)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.gain = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        return x + torch.sigmoid(self.gain) * self.net(x)


class EfficientResidualFusionV16(nn.Module):
    """Low-FLOPs residual fusion: low-res decisions, full-res detail bypass."""

    def __init__(self, hidden=32, detail_hidden=24):
        super().__init__()
        self.gradient = FixedSobelGradient()
        self.context_stem = nn.Sequential(
            nn.Conv2d(8, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseSeparableBlock(hidden),
            DepthwiseSeparableBlock(hidden),
            DepthwiseSeparableBlock(hidden),
        )
        self.context_head = nn.Conv2d(hidden, 3, kernel_size=1)
        self.edge_weight_head = nn.Sequential(
            nn.Conv2d(8, detail_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(detail_hidden, detail_hidden, kernel_size=3, padding=1, groups=detail_hidden),
            nn.GELU(),
            nn.Conv2d(detail_hidden, 2, kernel_size=1),
        )
        self.detail_stem = nn.Sequential(
            nn.Conv2d(12, detail_hidden, kernel_size=1),
            nn.GELU(),
            DepthwiseSeparableBlock(detail_hidden),
            DepthwiseSeparableBlock(detail_hidden),
            DepthwiseSeparableBlock(detail_hidden),
            nn.Conv2d(detail_hidden, 1, kernel_size=1),
            nn.Tanh(),
        )
        self.local_calibrator = nn.Sequential(
            nn.Conv2d(10, detail_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(detail_hidden, detail_hidden, kernel_size=3, padding=1, groups=detail_hidden),
            nn.GELU(),
            nn.Conv2d(detail_hidden, 1, kernel_size=1),
            nn.Tanh(),
        )
        self.global_calibrator = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(24, 2, kernel_size=1),
        )
        self.context_residual_gain = nn.Parameter(torch.tensor(-2.5))
        self.detail_gain = nn.Parameter(torch.tensor(-2.0))
        self.local_gain = nn.Parameter(torch.tensor(-3.0))

        nn.init.zeros_(self.context_head.weight)
        nn.init.zeros_(self.context_head.bias)
        nn.init.zeros_(self.edge_weight_head[-1].weight)
        nn.init.zeros_(self.edge_weight_head[-1].bias)
        nn.init.zeros_(self.detail_stem[-2].weight)
        nn.init.zeros_(self.detail_stem[-2].bias)
        nn.init.zeros_(self.local_calibrator[-2].weight)
        nn.init.zeros_(self.local_calibrator[-2].bias)
        nn.init.zeros_(self.global_calibrator[-1].weight)
        nn.init.zeros_(self.global_calibrator[-1].bias)

    @staticmethod
    def _mean_std_pair(x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True, unbiased=False)
        return mean, std

    def _stats(self, *sources):
        feats = []
        for source in sources:
            mean, std = self._mean_std_pair(source)
            feats.extend([mean, std])
        return torch.cat(feats, dim=1)

    def forward(self, img1, img2, texture_prompt, anchor_gray=None, return_aux=False):
        img1_gray = img1.mean(dim=1, keepdim=True)
        img2_gray = img2.mean(dim=1, keepdim=True)
        if anchor_gray is None:
            anchor_gray = img1_gray

        diff = torch.abs(img1_gray - img2_gray)
        grad1 = self.gradient(img1_gray)
        grad2 = self.gradient(img2_gray)
        grad_t = self.gradient(texture_prompt)
        grad_gap = torch.abs(grad1 - grad2)

        context_input = torch.cat([
            img1_gray,
            img2_gray,
            anchor_gray,
            texture_prompt,
            diff,
            grad1,
            grad2,
            grad_t,
        ], dim=1)
        context_low = F.avg_pool2d(context_input, kernel_size=2, stride=2, ceil_mode=True)
        context_logits = self.context_head(self.context_stem(context_low))
        context_logits = F.interpolate(context_logits, size=img1_gray.shape[2:], mode='bilinear', align_corners=False)

        edge_logits = self.edge_weight_head(context_input)
        weight_logits = context_logits[:, :2] + edge_logits
        weights = torch.softmax(weight_logits, dim=1)
        lc = weights[:, 0:1]
        ag = weights[:, 1:2]
        context_residual = torch.tanh(context_logits[:, 2:3])

        lc_expand = lc.expand_as(img1)
        ag_expand = ag.expand_as(img2)
        base = img1 * lc_expand + img2 * ag_expand

        base_gray = base.mean(dim=1, keepdim=True)
        detail_input = torch.cat([
            base_gray,
            img1_gray,
            img2_gray,
            anchor_gray,
            texture_prompt,
            diff,
            grad1,
            grad2,
            grad_t,
            grad_gap,
            ag,
            lc,
        ], dim=1)
        detail_residual = self.detail_stem(detail_input)
        fused = base + torch.sigmoid(self.context_residual_gain) * context_residual.expand_as(base)
        fused = fused + torch.sigmoid(self.detail_gain) * detail_residual.expand_as(base)
        fused = torch.clamp(fused, 0.0, 1.0)

        fused_gray = fused.mean(dim=1, keepdim=True)
        local_input = torch.cat([
            fused_gray,
            base_gray,
            img1_gray,
            img2_gray,
            anchor_gray,
            texture_prompt,
            diff,
            torch.abs(fused_gray - img1_gray),
            torch.abs(fused_gray - img2_gray),
            grad_t,
        ], dim=1)
        local_residual = self.local_calibrator(local_input)
        fused = torch.clamp(fused + torch.sigmoid(self.local_gain) * local_residual.expand_as(fused), 0.0, 1.0)

        fused_gray = fused.mean(dim=1, keepdim=True)
        stat_features = self._stats(fused_gray, base_gray, img1_gray, img2_gray, anchor_gray, texture_prompt)
        scale_bias = self.global_calibrator(stat_features)
        scale, bias = torch.chunk(scale_bias, 2, dim=1)
        fused = torch.clamp(fused * (1.0 + 0.20 * torch.tanh(scale)) + 0.08 * torch.tanh(bias), 0.0, 1.0)

        post_delta = fused - base
        if return_aux:
            return fused, {
                'fusion_ag': ag.detach(),
                'fusion_lc': lc.detach(),
                'joint_residual': context_residual.detach(),
                'texture_prompt': texture_prompt.detach(),
                'calibration_residual': local_residual.detach(),
                'final_residual': detail_residual.detach(),
                'supervised_residual': local_residual.detach(),
                'restoration_residual': detail_residual.detach(),
                'post_restoration_residual': post_delta.detach(),
                'post_base': base,
                'post_delta': post_delta,
                'local_psnr_delta': local_residual.detach(),
            }
        return fused


class FusionResidualDriftPredictor(nn.Module):
    """Texture-branch-style signed residual predictor for final fusion reconstruction."""

    def __init__(self, in_channels=3, channels=64, num_blocks=8):
        super().__init__()
        del num_blocks
        self.residual_net = ModernUNetGenerator(
            input_nc=in_channels,
            output_nc=1,
            ngf=channels,
            model_type='swin2_full',
            texture_vq_enabled=False,
        )
        self.delta_gain = nn.Parameter(torch.tensor(0.10))

    def forward(self, x):
        raw_delta = self.residual_net(x)
        gain = torch.abs(self.delta_gain)
        return gain * raw_delta


class Fusion(nn.Module):
    def __init__(
        self,
        use_local_psnr_calibrator=False,
        use_low_flops=False,
        use_efficient_v16=False,
        use_shared_weight=False,
        residual_steps=1,
        use_lowres_refiners=False,
        use_separable_convs=False,
        use_compact_post_refiner=False,
    ):
        super(Fusion, self).__init__()
        self.use_local_psnr_calibrator = bool(use_local_psnr_calibrator)
        self.use_low_flops = bool(use_low_flops)
        self.use_efficient_v16 = bool(use_efficient_v16)
        self.use_shared_weight = bool(use_shared_weight)
        self.residual_steps = max(1, int(residual_steps))
        self.use_lowres_refiners = bool(use_lowres_refiners) or self.use_low_flops
        self.use_separable_convs = bool(use_separable_convs)
        self.use_compact_post_refiner = bool(use_compact_post_refiner)
        self.efficient_v16 = EfficientResidualFusionV16(hidden=32, detail_hidden=24) if self.use_efficient_v16 else None
        self.cross_modal = SpatialChannelCrossAttention(dim=32)
        weight_net = SeparableWeightPredictionNet if self.use_separable_convs else WeightPredictionNet
        self.ag_net = None if self.use_shared_weight else weight_net()
        self.lc_net = None if self.use_shared_weight else weight_net()
        self.prior_weight_net = PriorGuidedWeightPredictionNet(in_channels=8, hidden=32)
        self.prior_residual_gain = nn.Parameter(torch.tensor(-2.0))
        self.joint_encoder = JointRestormerEncoder(in_channels=2, dim=64, depth=4, num_heads=4)
        self.joint_ag_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.joint_lc_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.joint_residual_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.texture_prior_gain = nn.Parameter(torch.tensor(-1.5))
        self.gradient = FixedSobelGradient()
        if self.use_low_flops:
            self.weight_edge_corrector = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
                nn.GELU(),
                nn.Conv2d(16, 2, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.weight_edge_corrector[-1].weight)
            nn.init.zeros_(self.weight_edge_corrector[-1].bias)
        self.drift_residual_predictor = FusionResidualDriftPredictor(
            in_channels=3,
            channels=64,
            num_blocks=8,
        )
        self.extra_drift_residual_predictors = nn.ModuleList([
            FusionResidualDriftPredictor(
                in_channels=3,
                channels=64,
                num_blocks=8,
            )
            for _ in range(self.residual_steps - 1)
        ])

    def _prepare_texture_prompt(self, img2_gray, texture_prior_cond):
        if texture_prior_cond is None:
            return img2_gray
        cond = texture_prior_cond.float()
        if cond.dim() == 3:
            cond = cond.unsqueeze(0)
        if cond.shape[2:] != img2_gray.shape[2:]:
            cond = F.interpolate(cond, size=img2_gray.shape[2:], mode='bilinear', align_corners=False)
        if cond.shape[1] != 1:
            cond = cond.mean(dim=1, keepdim=True)

        b = cond.shape[0]
        flat = cond.view(b, -1)
        cond_min = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        cond_max = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        cond = (cond - cond_min) / (cond_max - cond_min + 1e-6)
        gain = torch.sigmoid(self.texture_prior_gain)
        return torch.clamp(img2_gray + gain * (cond - 0.5), 0.0, 1.0)

    @staticmethod
    def _mean_std_pair(x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True, unbiased=False)
        return mean, std

    def _global_stat_features(self, fused_gray, img1_gray, img2_gray, texture_prompt):
        diff_12 = torch.abs(img1_gray - img2_gray)
        diff_f1 = torch.abs(fused_gray - img1_gray)
        sources = [fused_gray, img1_gray, img2_gray, texture_prompt, diff_12]
        features = []
        for source in sources:
            mean, std = self._mean_std_pair(source)
            features.extend([mean, std])
        return torch.cat(features, dim=1)

    @staticmethod
    def _sample_minmax_norm(x):
        b = x.shape[0]
        flat = x.flatten(1)
        x_min = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        x_max = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        return torch.clamp((x - x_min) / (x_max - x_min + 1e-6), 0.0, 1.0)

    def _local_variance_prior(self, x, kernel_size=7):
        pad = kernel_size // 2
        mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        mean_sq = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=pad)
        return torch.sqrt(torch.clamp(mean_sq - mean * mean, min=1e-8))

    def _fusion_weight_priors(self, initial_fused_gray):
        grad_init = self.gradient(initial_fused_gray)
        low_init = F.avg_pool2d(initial_fused_gray, kernel_size=15, stride=1, padding=7)
        high_init = torch.abs(initial_fused_gray - low_init)

        thermal_saliency = self._sample_minmax_norm(initial_fused_gray)
        low_frequency_body = self._sample_minmax_norm(low_init)
        p_rad = torch.clamp(0.60 * thermal_saliency + 0.40 * low_frequency_body, 0.0, 1.0)

        texture_gradient = self._sample_minmax_norm(grad_init)
        high_frequency_detail = self._sample_minmax_norm(high_init)
        p_tex = torch.clamp(0.60 * texture_gradient + 0.40 * high_frequency_detail, 0.0, 1.0)
        return p_rad, p_tex

    def _fusion_weight_features(self, img1_gray, texture_prompt):
        grad_r = self.gradient(img1_gray)
        grad_t = self.gradient(texture_prompt)
        return grad_r, grad_t

    def forward(self, img1, img2, texture_prior_cond=None, anchor=None, return_aux=False):
        orig_shape = img1.shape
        dtype = img1.dtype
        img1 = img1.float()
        img2 = img2.float()
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        img1_gray = torch.mean(img1, dim=1, keepdim=True)
        img2_gray = torch.mean(img2, dim=1, keepdim=True)
        anchor_gray = img1_gray
        texture_prompt = self._prepare_texture_prompt(img2_gray, texture_prior_cond)
        initial_fused_gray = 0.5 * (img1_gray + img2_gray)
        p_rad, p_tex = self._fusion_weight_priors(initial_fused_gray)
        grad_r, grad_t = self._fusion_weight_features(img1_gray, texture_prompt)
        if self.use_efficient_v16:
            fused_out = self.efficient_v16(
                img1,
                img2,
                texture_prompt=texture_prompt,
                anchor_gray=anchor_gray,
                return_aux=return_aux,
            )
            if return_aux:
                fused, aux = fused_out
            else:
                fused = fused_out
                aux = None
            if len(orig_shape) == 3:
                fused = fused.squeeze(0)
            fused = fused.to(dtype)
            if return_aux:
                return fused, aux
            return fused
        joint_input = torch.cat([img1_gray, img2_gray], dim=1)
        if self.use_low_flops:
            joint_input_low = F.avg_pool2d(joint_input, kernel_size=2, stride=2, ceil_mode=True)
            joint_feat = self.joint_encoder(joint_input_low)
        else:
            joint_feat = self.joint_encoder(joint_input)
        # Keep the joint encoder as a global structure path; no TGP-style prompt modulation here.

        prior_weight_input = torch.cat([
            img1_gray,
            img2_gray,
            p_rad,
            p_tex,
            torch.abs(img1_gray - img2_gray),
            grad_r,
            grad_t,
            texture_prompt,
        ], dim=1)
        prior_residual_scale = torch.sigmoid(self.prior_residual_gain)
        if self.use_low_flops:
            if self.use_shared_weight:
                prior_weight_low = F.avg_pool2d(prior_weight_input, kernel_size=2, stride=2, ceil_mode=True)
                prior_logits_low = torch.log(torch.clamp(prior_weight_low[:, 2:4], min=1e-4))
                learned_logits_low = self.prior_weight_net(prior_weight_low) + torch.cat([
                    self.joint_lc_head(joint_feat),
                    self.joint_ag_head(joint_feat),
                ], dim=1)
                raw_weights_low = prior_logits_low + prior_residual_scale * learned_logits_low
                raw_weights = F.interpolate(
                    raw_weights_low,
                    size=img1_gray.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
                raw_lc, raw_ag = torch.chunk(raw_weights, 2, dim=1)
            else:
                feat_r, feat_t = self.cross_modal(img1_gray, img2_gray)
                attended_feat = torch.cat([feat_r, feat_t], dim=1)
                attended_feat_low = F.avg_pool2d(attended_feat, kernel_size=2, stride=2, ceil_mode=True)
                raw_ag_low = self.ag_net(attended_feat_low) + self.joint_ag_head(joint_feat)
                raw_lc_low = self.lc_net(attended_feat_low) + self.joint_lc_head(joint_feat)
                raw_weights = F.interpolate(
                    torch.cat([raw_ag_low, raw_lc_low], dim=1),
                    size=img1_gray.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
                raw_ag, raw_lc = torch.chunk(raw_weights, 2, dim=1)
            weight_edge_input = torch.cat([
                img1_gray,
                img2_gray,
                torch.abs(img1_gray - img2_gray),
                grad_t,
            ], dim=1)
            edge_delta = self.weight_edge_corrector(weight_edge_input)
            if self.use_shared_weight:
                raw_lc, raw_ag = torch.chunk(
                    torch.cat([raw_lc, raw_ag], dim=1) + prior_residual_scale * edge_delta,
                    2,
                    dim=1,
                )
            else:
                raw_ag, raw_lc = torch.chunk(torch.cat([raw_ag, raw_lc], dim=1) + edge_delta, 2, dim=1)
        else:
            if self.use_shared_weight:
                prior_logits = torch.log(torch.clamp(torch.cat([p_rad, p_tex], dim=1), min=1e-4))
                learned_logits = self.prior_weight_net(prior_weight_input) + torch.cat([
                    self.joint_lc_head(joint_feat),
                    self.joint_ag_head(joint_feat),
                ], dim=1)
                raw_weights = prior_logits + prior_residual_scale * learned_logits
                raw_lc, raw_ag = torch.chunk(raw_weights, 2, dim=1)
            else:
                feat_r, feat_t = self.cross_modal(img1_gray, img2_gray)
                attended_feat = torch.cat([feat_r, feat_t], dim=1)
                raw_ag = self.ag_net(attended_feat) + self.joint_ag_head(joint_feat)
                raw_lc = self.lc_net(attended_feat) + self.joint_lc_head(joint_feat)
        weights = torch.softmax(torch.cat([raw_lc, raw_ag], dim=1), dim=1)
        lc, ag = torch.chunk(weights, 2, dim=1)
        if img1.size(1) == 3:
            ag = ag.expand(-1, 3, -1, -1)
            lc = lc.expand(-1, 3, -1, -1)

        joint_residual = self.joint_residual_head(joint_feat)
        if self.use_low_flops:
            joint_residual = F.interpolate(
                joint_residual, size=img1_gray.shape[2:], mode='bilinear', align_corners=False
            )
        base = img1 * lc + img2 * ag
        base_gray = torch.mean(base, dim=1, keepdim=True)
        drift_input = torch.cat([img1_gray, img2_gray, base_gray], dim=1)
        drift_delta = self.drift_residual_predictor(drift_input)
        corrected_base = base + drift_delta
        for residual_predictor in self.extra_drift_residual_predictors:
            corrected_gray = torch.mean(corrected_base, dim=1, keepdim=True)
            drift_input = torch.cat([img1_gray, img2_gray, corrected_gray], dim=1)
            step_delta = residual_predictor(drift_input)
            drift_delta = drift_delta + step_delta
            corrected_base = corrected_base + step_delta
        if base.size(1) == 3:
            drift_delta = drift_delta.expand(-1, 3, -1, -1)
            corrected_base = base + drift_delta
        fused = torch.clamp(corrected_base, 0.0, 1.0)

        calibration_residual = drift_delta
        final_residual = drift_delta
        supervised_residual = drift_delta
        restoration_residual = drift_delta
        post_restoration_residual = drift_delta
        post_base = base
        post_delta = drift_delta
        local_psnr_delta = None

        if len(orig_shape) == 3:
            fused = fused.squeeze(0)
        fused = fused.to(dtype)
        if return_aux:
            return fused, {
                'fusion_ag': ag.detach(),
                'fusion_lc': lc.detach(),
                'fusion_prior_rad': p_rad.detach(),
                'fusion_prior_tex': p_tex.detach(),
                'joint_residual': joint_residual.detach(),
                'texture_prompt': texture_prompt.detach(),
                'calibration_residual': calibration_residual.detach(),
                'final_residual': final_residual.detach(),
                'supervised_residual': supervised_residual.detach(),
                'restoration_residual': restoration_residual.detach(),
                'post_restoration_residual': post_restoration_residual.detach(),
                'post_base': post_base,
                'post_delta': post_delta,
                'local_psnr_delta': local_psnr_delta.detach() if local_psnr_delta is not None else None,
            }
        return fused
