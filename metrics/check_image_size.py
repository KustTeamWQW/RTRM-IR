from PIL import Image

# 检查几张图片的尺寸
test_images = ['1000.jpg', '1010.jpg', '1020.jpg', '1028.jpg']

for img_name in test_images:
    img_path = f'datasets2/ECMRNET/{img_name}'
    try:
        img = Image.open(img_path)
        print(f'{img_name}: {img.size[0]}×{img.size[1]}')
    except Exception as e:
        print(f'{img_name}: 无法读取 - {e}')
