from PIL import Image
import os
from tqdm import tqdm  # 用于显示进度条


def resize_images(input_dir, output_dir, new_size=(640, 512)):
    """
    批量调整图片大小

    参数:
        input_dir (str): 输入图片目录
        output_dir (str): 输出图片目录
        new_size (tuple): 新尺寸 (width, height)
    """
    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录中的所有图片文件
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"警告: 输入目录 '{input_dir}' 中没有找到支持的图片文件")
        return

    print(f"找到 {len(image_files)} 张图片需要处理...")

    # 处理每张图片
    for filename in tqdm(image_files, desc="处理进度"):
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 打开并调整图片大小
            with Image.open(input_path) as img:
                # 检查原始尺寸是否为1280x1024
                if img.size != (1280, 1024):
                    print(f"\n警告: 图片 '{filename}' 的尺寸不是1280×1024 (实际: {img.size})")

                # 调整图片大小
                resized_img = img.resize(new_size, Image.LANCZOS)

                # 保存图片（保持原始格式）
                resized_img.save(output_path, quality=95)

        except Exception as e:
            print(f"\n处理图片 '{filename}' 时出错: {e}")

    print(f"\n处理完成！结果已保存到: {output_dir}")


if __name__ == "__main__":
    # 设置输入和输出目录
    input_directory = "F:\\Decomposition-Reconstruction Enhancement\\IQA\\datasets2\\VIFPM"  # 替换为你的输入目录
    output_directory = "F:\\Decomposition-Reconstruction Enhancement\\IQA\\datasets2\\VIFPM\\2"  # 替换为你的输出目录

    # 调用函数处理图片
    resize_images(input_directory, output_directory)