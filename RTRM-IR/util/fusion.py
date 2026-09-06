import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块增强特征提取能力"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return self.relu(out)


class MultiScaleBlock(nn.Module):
    """多尺度特征提取模块"""

    def __init__(self, in_channels):
        super().__init__()
        # 不同空洞率的卷积
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=4, dilation=4)
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.fusion(out)


class WeightPredictionNet(nn.Module):
    """增强的权重预测网络"""

    def __init__(self):
        super().__init__()
        # 特征提取主干
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            MultiScaleBlock(64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            ResidualBlock(128),
            ResidualBlock(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            ResidualBlock(256),
            ResidualBlock(256),
        )

        # 上采样解码器
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            ResidualBlock(128),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            ResidualBlock(64),
        )

        # 最终预测层
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.final_conv(x)


class Fusion(nn.Module):
    def __init__(self):
        """
        图像融合模块
        初始化时不接受任何图像参数
        """
        super(Fusion, self).__init__()

        # 空间注意力模块 (保持原设计)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 增强的权重预测网络
        self.ag_net = WeightPredictionNet()
        self.lc_net = WeightPredictionNet()

        # Sobel滤波器 (保持不变)
        self.register_buffer('sobel_x',
                             torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y',
                             torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, img1, img2):
        """
        融合两张输入图像
        参数:
            img1: 第一张输入图像 (B,C,H,W) 或 (C,H,W)
            img2: 第二张输入图像 (与img1形状相同)
        返回:
            融合后的图像 (与输入形状相同)
        """
        # 保存原始输入类型和维度 (保持不变)
        orig_shape = img1.shape
        input_dtype = img1.dtype

        # 统一转换为4D float32张量 (保持不变)
        img1 = img1.float()
        img2 = img2.float()
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        # 转换为单通道灰度图 (保持不变)
        if img1.size(1) == 3:
            img1_gray = torch.mean(img1, dim=1, keepdim=True)
            img2_gray = torch.mean(img2, dim=1, keepdim=True)
        else:
            img1_gray, img2_gray = img1, img2

        # 计算基础权重 (保持不变)
        with torch.no_grad():
            ag1 = self.ag(img1_gray)
            ag2 = self.ag(img2_gray)
            base_ag = ag2 / (ag1 + ag2 + 1e-6)

            lc1 = self.local_contrast(img1_gray)
            lc2 = self.local_contrast(img2_gray)
            base_lc = lc1 / (lc1 + lc2 + 1e-6)

        # 拼接输入特征 (保持不变)
        feat_input = torch.cat([img1_gray, img2_gray], dim=1)

        # 应用空间注意力 (保持不变)
        spatial_att = self.spatial_attention(feat_input)
        attended_feat = feat_input * spatial_att

        # 使用增强的网络预测权重偏差
        delta_ag = self.ag_net(attended_feat)
        delta_lc = self.lc_net(attended_feat)

        # 调整后的权重 (保持不变)
        adjusted_ag = base_ag + delta_ag
        adjusted_lc = base_lc + delta_lc
        # 归一化权重
        total_weight = adjusted_ag + adjusted_lc
        adjusted_ag = adjusted_ag / (total_weight + 1e-6)
        adjusted_lc = adjusted_lc / (total_weight + 1e-6)
        # 融合图像（保持原始通道数）(保持不变)
        if img1.size(1) == 3:
            adjusted_ag = adjusted_ag.expand(-1, 3, -1, -1)
            adjusted_lc = adjusted_lc.expand(-1, 3, -1, -1)

        fused = img2 * adjusted_ag + img1 * adjusted_lc

        # 恢复原始维度和类型 (保持不变)
        if len(orig_shape) == 3:
            fused = fused.squeeze(0)
        return fused.to(input_dtype)

    def ag(self, image):
        """平均梯度计算 (保持不变)"""
        grad_x = F.conv2d(image, self.sobel_x, padding=1)
        grad_y = F.conv2d(image, self.sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def local_contrast(self, img, window_size=5):
        """局部对比度计算 (保持不变)"""
        kernel = torch.ones((1, 1, window_size, window_size), dtype=img.dtype, device=img.device) / (window_size ** 2)
        mean = F.conv2d(img, kernel, padding=window_size // 2)
        var = F.conv2d(img ** 2, kernel, padding=window_size // 2) - mean ** 2
        return torch.sqrt(torch.abs(var) + 1e-6)