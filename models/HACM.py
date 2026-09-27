import torch
import torch.nn as nn
import torch.nn.functional as F

'''
大气传输校正模型
'''
class ATF2D(nn.Module):
    def __init__(self, D=8, W=256, input_channels=3, param_scale=0.5):
        """处理图像输入 [B, C, H, W]，输出按公式计算后的结果 [B, C, H, W]"""
        super().__init__()
        self.D = D
        self.W = W
        self.skips = [D // 2]  # 跳跃连接层索引
        self.input_channels = input_channels
        self.param_scale = max(0.25, min(1.0, float(param_scale)))

        # 首先用CNN提取特征
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 生成归一化坐标网格
        self.coord_conv = CoordConv2d()

        # 网络主干
        self.linear = nn.ModuleList()
        input_ch = 128 + 2  # CNN特征 + 坐标特征
        for i in range(D):
            if i in self.skips:
                self.linear.append(nn.Linear(input_ch + 2, W))
                input_ch = W
            else:
                self.linear.append(nn.Linear(input_ch, W))
                input_ch = W

        # 输出头
        self.abs = nn.Linear(W, 1)  # 吸收系数
        self.sca = nn.Linear(W, 1)  # 散射系数
        self.d = nn.Linear(W, 1)  # 等效传播距离

    def forward(self, x):
        """
        输入: x - 图像张量 [B, C, H, W]
        输出: 按公式计算后的结果 [B, C, H, W]
        """
        B, C, H, W = x.shape
        if self.param_scale < 1.0:
            param_h = max(16, int(round(H * self.param_scale)))
            param_w = max(16, int(round(W * self.param_scale)))
            x_param = F.interpolate(x, size=(param_h, param_w), mode='bilinear', align_corners=False)
        else:
            x_param = x
            param_h, param_w = H, W

        # 1. 在低分辨率参数场上提取特征，降低逐像素 MLP 的计算量
        cnn_features = self.encoder(x_param)  # [B, 128, param_h, param_w]

        # 2. 生成坐标特征
        coord_features = self.coord_conv(x_param)  # [B, 2, param_h, param_w]

        # 3. 拼接特征
        h = torch.cat([cnn_features, coord_features], dim=1)  # [B, 130, param_h, param_w]

        # 4. 转换为MLP输入格式 [B*param_h*param_w, 130]
        h = h.permute(0, 2, 3, 1).reshape(-1, 130)
        coord_features = coord_features.permute(0, 2, 3, 1).reshape(-1, 2)

        # 5. 前向传播
        for i, layer in enumerate(self.linear):
            if i in self.skips:
                h = torch.cat([h, coord_features], dim=-1)
            h = layer(h)
            h = F.relu(h)

        # 6. 计算各系数
        abs_val = torch.sigmoid(self.abs(h)).view(B, param_h, param_w, 1)  # [B, param_h, param_w, 1]
        sca_val = torch.sigmoid(self.sca(h)).view(B, param_h, param_w, 1)  # [B, param_h, param_w, 1]
        d_val = torch.exp(self.d(h)).view(B, param_h, param_w, 1)  # [B, param_h, param_w, 1]

        # 7. 按公式计算最终输出
        exponent = (abs_val + sca_val) * d_val  # [B, H, W, 1]

        # 调整维度以匹配输入x [B, C, H, W]
        exponent = exponent.permute(0, 3, 1, 2)  # [B, 1, param_h, param_w]
        if exponent.shape[2:] != (H, W):
            exponent = F.interpolate(exponent, size=(H, W), mode='bilinear', align_corners=False)
        exponent = exponent.expand(-1, C, -1, -1)  # [B, C, H, W]

        output = x * torch.exp(exponent)  # [B, C, H, W]
        return output


class CoordConv2d(nn.Module):
    """生成归一化坐标特征图"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        # 生成网格坐标
        xx = torch.linspace(-1, 1, W, device=x.device)
        yy = torch.linspace(-1, 1, H, device=x.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')

        # 扩展为batch形式 [B, 2, H, W]
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        return coords
