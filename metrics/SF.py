# 计算空间频率
# 根据公式推断出来的，不一定正确
import cv2
import numpy as np
import math


def spatialF(image):
    M = image.shape[0]
    N = image.shape[1]

    cf = 0
    rf = 0

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            dx = float(image[i, j - 1]) - float(image[i, j])
            rf += dx ** 2
            dy = float(image[i - 1, j]) - float(image[i, j])
            cf += dy ** 2

    RF = math.sqrt(rf / (M * N))
    CF = math.sqrt(cf / (M * N))
    SF = math.sqrt(RF ** 2 + CF ** 2)

    return SF


# if __name__ == '__main__':
#     # 这样读取后已经是灰度图
#     image = cv2.imread('1.png', 0)
#     print(spatialF(image))
