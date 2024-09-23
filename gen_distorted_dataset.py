from PIL import Image
import os
import numpy as np
from pillow_heif import AvifImagePlugin
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from threading import Lock

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

image_size = 256
formats = ['jpeg', 'avif', 'webp']
qualities = [10, 20, 40, 60, 80, 90]
input_dir = '/home/ubuntu/data/datasets/imgnet1k'
output_base_dir = '/mnt/tmpfs/imgnet1k_comp'
num_threads = 64  # 

# 获取所有文件的列表
files = []
for root, dirs, filenames in os.walk(input_dir):
    for filename in filenames:
        files.append(os.path.join(root, filename))

lock = Lock()

def process_and_save_image(filepath, formats, qualities, input_dir, output_base_dir):
    try:
        pil_image = Image.open(filepath).convert('RGB')
        pil_image = center_crop_arr(pil_image, image_size)
        # 获取相对路径
        relative_path = os.path.relpath(filepath, input_dir)
        # 去掉原有的扩展名
        relative_path = os.path.splitext(relative_path)[0]
        for format in formats:
            for quality in qualities:
                # 更改文件扩展名
                extension_map = {
                    'jpeg': '.jpg',
                    'avif': '.avif',
                    'webp': '.webp',
                }
                extension = extension_map[format]
                # 构建输出文件路径
                output_dir = os.path.join(output_base_dir, format, str(quality))
                output_path = os.path.join(output_dir, relative_path + extension)
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # 保存图像
                if format == 'jpeg':
                    pil_image.save(output_path, format='JPEG', quality=quality)
                elif format == 'webp':
                    pil_image.save(output_path, format='WEBP', quality=quality)
                elif format == 'avif':
                    pil_image.save(output_path, format='AVIF', quality=quality)
    except Exception as e:
        print(f"处理文件 {filepath} 时出错: {e}")

def process_image_with_progress(filepath, formats, qualities, input_dir, output_base_dir, pbar):
    process_and_save_image(filepath, formats, qualities, input_dir, output_base_dir)
    with lock:
        pbar.update(1)

# 单一的进度条，展示整体处理进度
print("开始处理图像...")
pbar = tqdm(total=len(files), desc="处理进度")
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    for filepath in files:
        executor.submit(process_image_with_progress, filepath, formats, qualities, input_dir, output_base_dir, pbar)
pbar.close()
print("图像处理完成。")
