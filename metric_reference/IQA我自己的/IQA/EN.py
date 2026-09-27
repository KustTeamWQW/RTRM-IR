import cv2
import numpy as np
import math



# def en(image):
#     tmp = []
#     for i in range(len(image)):
#         tmp.append(0)
#     val = 0
#     k = 0
#     res = 0
#     img = np.array(image)
#     for i in range(len(img)):
#         for j in range(len(img[i])):
#             val = int(img[i][j])
#             tmp[val] = float(tmp[val] + 1)
#             k = float(k + 1)
#     for i in range(len(tmp)):
#         tmp[i] = float(tmp[i] / k)
#     for i in range(len(tmp)):
#         if (tmp[i] == 0):
#             res = res
#         else:
#             res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
#     return res
#     print(res)

# def en(image):
#     '''
#     :param img:narray 二维灰度图像
#     :return: float 图像约清晰越大
#     '''
#     out = 0
#     count = np.shape(image)[0] * np.shape(image)[1]
#     p = np.bincount(np.array(image).flatten())
#     for i in range(0, len(p)):
#         if p[i] != 0:
#             out -= p[i] * math.log(p[i] / count) / count
#     return out

import math
import numpy as np

def en(image):
    tmp = [0] * 256  # 假设像素值在0到255之间
    k = 0
    res = 0
    img = np.array(image)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = int(img[i][j])
            tmp[val] += 1
            k += 1
    for i in range(256):
        if tmp[i] > 0:
            res -= tmp[i] / k * math.log(tmp[i] / k) / math.log(2.0)
    return res