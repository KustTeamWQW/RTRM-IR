import torch
import torch.nn as nn
import torch.nn.functional as F


'''
热传导补偿模型 - 修复版（使用可微分梯度算子，完全等价于torch.gradient）
'''
class ReshapeConv(nn.Module):
    """双通道转单通道的1x1卷积模块"""

    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = F.normalize(x, dim=1)  # 沿通道维度归一化
        x = self.activation(x)
        return x


class NorConv(nn.Module):
    """单通道特征增强模块"""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = F.normalize(x, dim=1)
        x = self.activation(x)
        return x


def differentiable_gradient_x(x):
    """
    可微分的水平梯度，完全等价于 torch.gradient(x, dim=3)[0]
    使用中心差分 + 边界单侧差分

    输入: x [B, C, H, W]
    输出: grad_x [B, C, H, W]
    """
    B, C, H, W = x.shape
    result = torch.zeros_like(x)

    # 左边界：单侧差分 (x[1] - x[0])
    result[:, :, :, 0] = x[:, :, :, 1] - x[:, :, :, 0]

    # 中间：中心差分 (x[i+1] - x[i-1]) / 2
    if W > 2:
        result[:, :, :, 1:-1] = (x[:, :, :, 2:] - x[:, :, :, :-2]) / 2.0

    # 右边界：单侧差分 (x[-1] - x[-2])
    if W > 1:
        result[:, :, :, -1] = x[:, :, :, -1] - x[:, :, :, -2]

    return result


def differentiable_gradient_y(x):
    """
    可微分的垂直梯度，完全等价于 torch.gradient(x, dim=2)[0]
    使用中心差分 + 边界单侧差分

    输入: x [B, C, H, W]
    输出: grad_y [B, C, H, W]
    """
    B, C, H, W = x.shape
    result = torch.zeros_like(x)

    # 上边界：单侧差分 (x[1] - x[0])
    result[:, :, 0, :] = x[:, :, 1, :] - x[:, :, 0, :]

    # 中间：中心差分 (x[i+1] - x[i-1]) / 2
    if H > 2:
        result[:, :, 1:-1, :] = (x[:, :, 2:, :] - x[:, :, :-2, :]) / 2.0

    # 下边界：单侧差分 (x[-1] - x[-2])
    if H > 1:
        result[:, :, -1, :] = x[:, :, -1, :] - x[:, :, -2, :]

    return result


class TCMNetwork(nn.Module):
    """面向二维红外图像的特征增强网络 - 修复版（逻辑完全不变）"""

    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # 修改为处理多通道输入
        self.TCM = ReshapeConv(in_channels=in_channels * 2, out_channels=in_channels)
        self.TCM1 = NorConv(in_channels=in_channels, out_channels=in_channels)
        self.TCM2 = NorConv(in_channels=in_channels, out_channels=in_channels)
        self.residual_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        输入: x - 图像张量 [B, C, H, W] (如 [8, 3, 300, 300])
        输出: 增强后的特征 [B, C, H, W]

        完全等价于原始实现，只是将torch.gradient替换为可微分版本
        """
        B, C, H, W = x.shape

        # ----------------------------
        # 阶段2：梯度特征计算（逐通道计算）
        # ----------------------------
        grad_maps = []
        for c in range(C):
            channel_x = x[:, c:c + 1, :, :]  # 取单通道 [B, 1, H, W]

            # 一阶梯度（使用可微分实现，完全等价于torch.gradient）
            grad_x = differentiable_gradient_x(channel_x)  # 水平方向
            grad_y = differentiable_gradient_y(channel_x)  # 垂直方向

            # 二阶梯度
            grad_xx = differentiable_gradient_x(grad_x)
            grad_yy = differentiable_gradient_y(grad_y)

            # 二阶边缘特征
            second_order_edge = torch.abs(grad_xx) + torch.abs(grad_yy)
            grad_maps.append(second_order_edge)

        # 合并所有通道的边缘特征
        second_order_edge = torch.cat(grad_maps, dim=1)  # [B, C, H, W]

        # ----------------------------
        # 阶段3：特征融合与处理
        # ----------------------------
        # 拼接原始图像与边缘特征
        x1 = torch.cat([x, second_order_edge], dim=1)  # [B, 2*C, H, W]

        # 通过卷积模块处理
        x2 = self.TCM(x1)  # [B, C, H, W]
        x3 = self.TCM1(x2)  # [B, C, H, W]
        residual = self.TCM2(x3)  # [B, C, H, W]

        residual_scale = 2.0 * torch.sigmoid(self.residual_gain)
        output = x + residual_scale * residual  # Learnable positive TCM compensation strength.
        return output
