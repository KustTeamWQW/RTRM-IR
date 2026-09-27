import cv2
import numpy as np


def standard_deviation(image):

    # 计算图像的平均值
    mean = np.mean(image)

    # 计算标准差
    std_dev = np.sqrt(np.mean((image - mean) ** 2))

    return std_dev


def evaluate_global_contrast(image):

    std_dev = standard_deviation(image)
    contrast_evaluation = std_dev

    return contrast_evaluation


# # 示例使用
# if __name__ == "__main__":
#     image_paths = [
#         "path_to_your_first_image.jpg",
#         "path_to_your_second_image.jpg",
#         # 添加更多图像路径
#     ]
#
#     contrast_results = evaluate_global_contrast(image_paths)
#     for path, std_dev in contrast_results.items():
#         print(f"图像路径: {path}, 标准差（全局对比度）: {std_dev:.4f}")