import cv2
import numpy as np
import math


import math

def avgGradient(image):
    width = image.shape[1]  # 获得图像的宽度
    height = image.shape[0]  # 获得图像的高度

    # 计算梯度的总和
    tmp = 0.0
    for i in range(height - 1):  # 减 1 以避免访问最后一个元素时越界
        for j in range(width - 1):  # 同上
            dx = float(image[i, j + 1]) - float(image[i, j])  # 水平方向的梯度
            dy = float(image[i + 1, j]) - float(image[i, j])  # 垂直方向的梯度
            ds = math.sqrt((dx * dx + dy * dy) / 2)  # 梯度的欧几里得距离
            tmp += ds

    # 计算平均梯度
    imageAG = tmp / ((width - 1) * (height - 1))
    return imageAG


# if __name__ == '__main__':
#     image = cv2.imread('1.png', 0)
#     print(avgGradient(image))
