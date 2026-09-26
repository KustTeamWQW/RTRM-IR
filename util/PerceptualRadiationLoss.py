import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualRadiationLoss(nn.Module):
    def __init__(self, lambda_grad=0.3, lambda_hist=0.2, lambda_ms=0.1, bins=256):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_hist = lambda_hist
        self.lambda_ms = lambda_ms
        self.bins = bins

        # 注册Sobel算子（单通道）
        self.register_buffer("sobel_x",
                             torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y",
                             torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def gradient_loss(self, pred, target):
        """支持多通道输入的梯度损失"""
        # 方法1：分别计算每个通道的梯度
        sobel_x = self.sobel_x.expand(pred.size(1), 1, 3, 3)  # [C, 1, 3, 3]
        sobel_y = self.sobel_y.expand(pred.size(1), 1, 3, 3)

        grad_pred_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.size(1))
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.size(1))
        grad_target_x = F.conv2d(target, sobel_x, padding=1, groups=target.size(1))
        grad_target_y = F.conv2d(target, sobel_y, padding=1, groups=target.size(1))

        return torch.mean(torch.abs(grad_pred_x - grad_target_x)) + \
            torch.mean(torch.abs(grad_pred_y - grad_target_y))

    def histogram_loss(self, pred, target):
        """计算辐射直方图匹配损失"""
        hist_pred = torch.histc(pred, bins=self.bins, min=0, max=1)
        hist_target = torch.histc(target, bins=self.bins, min=0, max=1)
        hist_pred = hist_pred / (torch.sum(hist_pred) + 1e-6)
        hist_target = hist_target / (torch.sum(hist_target) + 1e-6)
        return F.kl_div(torch.log(hist_pred + 1e-6), hist_target + 1e-6, reduction='batchmean')

    def multi_scale_loss(self, pred, target, scales=[1.0, 0.5, 0.25]):
        """多尺度辐射一致性损失"""
        loss = 0
        for scale in scales:
            if scale != 1.0:
                p = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
                t = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                p, t = pred, target
            loss += F.l1_loss(p, t)
        return loss / len(scales)

    def forward(self, pred, target):
        """
        计算感知辐射损失
        Args:
            pred (torch.Tensor): 模型输出的辐射图 [B,C,H,W]
            target (torch.Tensor): 真实辐射图 [B,C,H,W]
        Returns:
            loss (torch.Tensor): 加权后的感知损失
            loss_dict (dict): 各分项损失（用于日志记录）
        """
        loss_grad = self.gradient_loss(pred, target)
        loss_hist = self.histogram_loss(pred, target)
        loss_ms = self.multi_scale_loss(pred, target)
        
        total_loss = (
            self.lambda_grad * loss_grad +
            self.lambda_hist * loss_hist +
            self.lambda_ms * loss_ms
        )
        

        return total_loss