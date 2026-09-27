import os
import shutil


def copy_files_with_same_basename(src_dir_b, dest_dir_c, src_dir_a):
    # 确保目标文件夹存在  
    if not os.path.exists(dest_dir_c):
        os.makedirs(dest_dir_c)

        # 获取A文件夹中所有文件（包括子文件夹中的）的基本名称（不含扩展名）
    filenames_in_a = set()
    for root, dirs, files in os.walk(src_dir_a):
        for file in files:
            # 获取文件的基本名称（不含扩展名）  
            base_name = os.path.splitext(file)[0]
            filenames_in_a.add(base_name)

            # 遍历B文件夹，复制与A中文件名相同的文件到C
    for root, dirs, files in os.walk(src_dir_b):
        for file in files:
            # 获取B中文件的基本名称（不含扩展名）  
            base_name = os.path.splitext(file)[0]

            # 如果B中的文件基本名称在A中存在  
            if base_name in filenames_in_a:
                # 构建B和C中文件的完整路径  
                src_file = os.path.join(root, file)
                rel_path = os.path.relpath(root, src_dir_b)
                dest_file = os.path.join(dest_dir_c, rel_path, file)

                # 创建目标文件夹（如果不存在）  
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)

                # 复制文件到C  
                shutil.copy2(src_file, dest_file)

            # 使用函数


src_dir_b = 'datasets/GT1/datasets1/test/LR'  # 源文件夹B
dest_dir_c = 'datasets/GT1/datasets1/test/LR_part'  # 目标文件夹C
src_dir_a = 'datasets/GT1/datasets1/test/GT_part'  # 比较文件夹A


copy_files_with_same_basename(src_dir_b, dest_dir_c, src_dir_a)