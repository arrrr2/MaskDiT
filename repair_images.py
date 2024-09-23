from gen_distorted_dataset import process_and_save_image


image_size = 256
formats = ['jpeg', 'avif', 'webp']
qualities = [10, 20, 40, 60, 80, 90]
input_dir = '/home/ubuntu/data/datasets/imgnet1k'
output_base_dir = '/mnt/tmpfs/imgnet1k_comp'
output_base_dir2 = '/mnt/tmpfs/repair_imgnet1k_comp'
num_threads = 8


with open('broken_file_list.txt') as f:
    