import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from AG import avgGradient
from EME import eme
from EN import en
from psnr_ssim import calculate_psnr, calculate_ssim
from SF import spatialF
from niqe_1 import niqe_1
from piqe_1 import piqe_1
from STD import evaluate_global_contrast

import os
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate

# 假设 avgGradient, eme, en, spatialF, calculate_psnr, calculate_ssim 这些函数已经定义

def calculate_no_reference_scores(img_dir):
    # 初始化存储结果的变量
    total_ag_score = 0
    total_eme_score = 0
    total_en_score = 0
    total_sf_score = 0
    total_niqe_score = 0
    total_piqe_score = 0
    total_gc_score = 0
    count = 0

    # 遍历目录中的所有文件
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(img_dir, filename)

            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 计算分数

            niqe_score = niqe_1(image)
            piqe_score = piqe_1(image)
            ag_score = avgGradient(image)
            eme_score = eme(image)
            en_score = en(image)
            sf_score = spatialF(image)
            gc_score = evaluate_global_contrast(image)

            total_niqe_score += niqe_score
            total_piqe_score += piqe_score
            total_ag_score += ag_score
            total_eme_score += eme_score
            total_en_score += en_score
            total_sf_score += sf_score
            total_gc_score += gc_score

            count += 1

    # 计算平均分数
    average_ag_score = total_ag_score / count
    average_eme_score = total_eme_score / count
    average_en_score = total_en_score / count
    average_sf_score = total_sf_score / count
    average_niqe_score = total_niqe_score / count
    average_piqe_score = total_piqe_score / count
    average_gc_score = total_gc_score / count
    # 将结果转换为DataFrame
    df = pd.DataFrame({
        'Average AG Score': [average_ag_score],
        'Average EME Score': [average_eme_score],
        'Average EN Score': [average_en_score],
        'Average SF Score': [average_sf_score],
        'Average niqe Score': [average_niqe_score],
        'Average piqe Score': [average_piqe_score],
        'Average gc_score': [average_gc_score]
    })

    return df, average_ag_score, average_eme_score, average_en_score, average_sf_score, average_niqe_score, average_piqe_score, average_gc_score

def calculate_average_psnr_ssim(img_dir1, img_dir2):
    # 初始化存储结果的变量
    total_psnr_score = 0
    total_ssim_score = 0
    count = 0

    # 获取两个目录中的文件列表
    files1 = [f for f in os.listdir(img_dir1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    files2 = [f for f in os.listdir(img_dir2) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # 确保两个目录中的文件名可以配对
    files1.sort()
    files2.sort()

    # 遍历并配对图像
    for f1, f2 in zip(files1, files2):
        file_path1 = os.path.join(img_dir1, f1)
        file_path2 = os.path.join(img_dir2, f2)

        # 读取图像
        img1 = cv2.imread(file_path1)
        img2 = cv2.imread(file_path2)
        if img1 is None or img2 is None:
            continue

        # 计算PSNR和SSIM
        psnr_score = calculate_psnr(img1, img2, crop_border=4, input_order='HWC', test_y_channel=False)
        ssim_score = calculate_ssim(img1, img2, crop_border=4, input_order='HWC', test_y_channel=False)

        total_psnr_score += psnr_score
        total_ssim_score += ssim_score
        count += 1

    # 计算平均PSNR和SSIM分数
    average_psnr_score = total_psnr_score / count
    average_ssim_score = total_ssim_score / count

    # 将结果转换为DataFrame
    df = pd.DataFrame({
        'Average PSNR': [average_psnr_score],
        'Average SSIM': [average_ssim_score],
    })

    return df, average_psnr_score, average_ssim_score

def main():
    img_dir1 = 'datasets2/GT' # GT
    img_dir2 = 'F:\Decomposition-Reconstruction Enhancement\IQA\generalization\VIFPM\Generalization evaluation/trained on Dataset 1 and tested on Dataset 2'   # 增强图wp

    # 计算无参考图像质量评分
    df_no_reference, average_ag_score, average_eme_score, average_en_score, average_sf_score, average_niqe_score, average_piqe_score, average_gc_score = calculate_no_reference_scores(img_dir2)

    # 打印无参考图像质量评分结果
    print("\nNo-Reference Image Scores:")
    print("Average AG Score:", average_ag_score)
    print("Average EME Score:", average_eme_score)
    print('Average EN Score:', average_en_score)
    print("Average SF Score:", average_sf_score)
    print("Average gc_score:", average_gc_score)
    print(tabulate(df_no_reference, headers='keys', tablefmt='grid', showindex=False))

    # 计算有参考图像质量评分
    df_with_reference, average_psnr_score, average_ssim_score = calculate_average_psnr_ssim(img_dir1, img_dir2)

    # 打印有参考图像质量评分结果
    print("\nWith-Reference Image Scores:")
    print("Average PSNR Score:", average_psnr_score)
    print("Average SSIM Score:", average_ssim_score)
    print(tabulate(df_with_reference, headers='keys', tablefmt='grid', showindex=False))

    # 将结果保存到CSV文件
    df_no_reference.to_csv('F:\Decomposition-Reconstruction Enhancement\IQA\generalization/results\VIFPM\data1train-data2test/VIFPM_no_reference_scores.csv', index=False)
    df_with_reference.to_csv('F:\Decomposition-Reconstruction Enhancement\IQA\generalization/results\VIFPM\data1train-data2test/VIFPM_with_reference_scores.csv', index=False)

    # 打印所有DataFrames
    print("\nAll DataFrames:")
    print(tabulate(df_no_reference, headers='keys', tablefmt='grid', showindex=False))
    print(tabulate(df_with_reference, headers='keys', tablefmt='grid', showindex=False))

if __name__ == '__main__':
    main()

