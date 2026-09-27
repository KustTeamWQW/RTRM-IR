import numpy as np


def eme(image, block_size=5):
    """
    计算图像增强度量 EME。

    参数:
    - image: 输入的图像数据，预期是一个二维 numpy 数组。
    - block_size: 处理图像时所使用的块的大小。

    返回:
    - E: 计算得到的 EME 值。
    """
    # 确保输入的图像是 numpy 数组
    image = np.array(image, dtype=np.double)

    # 获取图像的边长
    M = image.shape[0]

    # 计算 EME
    E = 0
    B1 = np.zeros((block_size, block_size))  # 创建 block_size x block_size 的零矩阵

    # 计算每个块的索引范围
    how_many = int(M / block_size)

    m1 = 0
    for m in range(how_many):
        n1 = 0
        for n in range(how_many):
            # 截取 block_size x block_size 大小的块
            B1 = image[m1:m1 + block_size, n1:n1 + block_size]
            b_min = np.min(B1)  # 计算块的最小值
            b_max = np.max(B1)  # 计算块的最大值

            # 如果最小值大于0，则计算比率并更新 E
            if b_min > 0:
                b_ratio = b_max / b_min
                E += 20 * np.log10(b_ratio)  # 使用 np.log10 计算以10为底的对数

            n1 += block_size
        m1 += block_size

    # 计算最终的 EME 值
    E = E / (how_many * how_many)
    return E

# 使用示例：
# 假设 I 是一个 MxM 的 numpy 数组，表示图像数据
# I = np.random.rand(256, 256)  # 这里仅为示例，生成一个随机的 256x256 图像
# block_size = 5  # 块的大小
# eme_value = eme(I, block_size)
# print(eme_value)