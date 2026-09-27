from PIL import Image
import os

# 设置路径和参数
input_dir = 'datasets2/ECMRNET'
original_size = (1280, 1024)  # 原尺寸
target_size = (640, 512)  # 目标尺寸

processed_count = 0
skipped_count = 0

# 处理1000-1028编号的图片
for i in range(1000, 1029):
    image_path = os.path.join(input_dir, f'{i}.jpg')

    if os.path.exists(image_path):
        try:
            # 打开图片
            img = Image.open(image_path)
            current_size = img.size
            print(f'Checking {i}.jpg, current size: {current_size[0]}x{current_size[1]}')

            # 只处理尺寸为1280×1024的图片
            if current_size == original_size:
                # 调整尺寸
                img_resized = img.resize(target_size, Image.LANCZOS)

                # 保存（覆盖原图）
                img_resized.save(image_path, quality=95)
                print(f'  [OK] Resized to {target_size[0]}x{target_size[1]}')
                processed_count += 1
            else:
                print(f'  [SKIP] Not 1280x1024')
                skipped_count += 1

        except Exception as e:
            print(f'[ERROR] Processing {i}.jpg: {e}')
    else:
        print(f'[ERROR] File not exists: {image_path}')

print(f'\nDone! Processed: {processed_count}, Skipped: {skipped_count}')
