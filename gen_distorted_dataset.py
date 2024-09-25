import os
import numpy as np
from PIL import Image
from pillow_heif import AvifImagePlugin
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Manager

def center_crop_arr(pil_image, image_size):
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
formats = ['png']
qualities = [100]
input_dir = '/home/ubuntu/data/datasets/imgnet1k'
output_base_dir = '/mnt/tmpfs/imgnet1k_png'
num_processes = 96
batch_size = 7680  # 批次大小

files = []
for root, dirs, filenames in os.walk(input_dir):
    for filename in filenames:
        files.append(os.path.join(root, filename))

def process_and_save_image(filepath, formats, qualities, input_dir, output_base_dir):
    try:
        pil_image = Image.open(filepath).convert('RGB')
        pil_image = center_crop_arr(pil_image, image_size)
        relative_path = os.path.relpath(filepath, input_dir)
        relative_path = os.path.splitext(relative_path)[0]
        for format in formats:
            for quality in qualities:
                extension_map = {
                    'jpeg': '.jpg',
                    'avif': '.avif',
                    'png': '.png',
                    'webp': '.webp',
                }
                extension = extension_map[format]
                output_dir = os.path.join(output_base_dir, format, str(quality))
                output_path = os.path.join(output_dir, relative_path + extension)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if format == 'jpeg':
                    pil_image.save(output_path, format='JPEG', quality=quality)
                elif format == 'webp':
                    pil_image.save(output_path, format='WEBP', quality=quality)
                elif format == 'avif':
                    pil_image.save(output_path, format='AVIF', quality=quality)
                elif format == 'png':
                    pil_image.save(output_path, format='PNG', compress_level=9)
    except Exception as e:
        print(f"处理文件 {filepath} 时出错: {e}")

if __name__ == '__main__':
    with Manager() as manager:
        print("开始处理图像...")
        pbar = tqdm(total=len(files), desc="处理进度")

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]  # 获取当前批次
                futures = {executor.submit(process_and_save_image, filepath, formats, qualities, input_dir, output_base_dir): filepath for filepath in batch}
                
                for future in as_completed(futures):
                    pbar.update(1)  # 每处理一个文件更新进度条

        pbar.close()
        print("图像处理完成。")
