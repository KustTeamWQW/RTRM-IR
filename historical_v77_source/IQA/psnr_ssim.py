import math

import cv2
import numpy as np


def _crop_border(img, crop_border):
    crop_border = int(crop_border)
    if crop_border <= 0:
        return img
    return img[crop_border:-crop_border, crop_border:-crop_border, ...]


def _to_y_channel(img):
    if img.ndim == 2 or img.shape[2] == 1:
        return img[..., 0] if img.ndim == 3 else img
    img = img.astype(np.float64)
    return 16.0 + (65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]) / 255.0


def _reorder_image(img, input_order):
    if input_order == 'HWC':
        return img
    if input_order == 'CHW':
        return np.transpose(img, (1, 2, 0))
    raise ValueError('Unsupported input_order: %s' % input_order)


def calculate_psnr(img, img2, crop_border=0, input_order='HWC', test_y_channel=False):
    img = _reorder_image(np.asarray(img), input_order).astype(np.float64)
    img2 = _reorder_image(np.asarray(img2), input_order).astype(np.float64)
    img = _crop_border(img, crop_border)
    img2 = _crop_border(img2, crop_border)
    if test_y_channel:
        img = _to_y_channel(img)
        img2 = _to_y_channel(img2)
    mse = np.mean((img - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _ssim_single_channel(img, img2):
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


def calculate_ssim(img, img2, crop_border=0, input_order='HWC', test_y_channel=False):
    img = _reorder_image(np.asarray(img), input_order).astype(np.float64)
    img2 = _reorder_image(np.asarray(img2), input_order).astype(np.float64)
    img = _crop_border(img, crop_border)
    img2 = _crop_border(img2, crop_border)
    if test_y_channel:
        img = _to_y_channel(img)
        img2 = _to_y_channel(img2)
    if img.ndim == 2:
        return _ssim_single_channel(img, img2)
    if img.shape[2] == 1:
        return _ssim_single_channel(img[..., 0], img2[..., 0])
    return float(np.mean([
        _ssim_single_channel(img[..., channel], img2[..., channel])
        for channel in range(img.shape[2])
    ]))
