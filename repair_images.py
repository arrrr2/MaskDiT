from gen_distorted_dataset import process_and_save_image


image_size = 256
formats = ['jpeg', 'avif', 'webp']
qualities = [10, 20, 40, 60, 80, 90]
input_dir = '/home/ubuntu/data/datasets/imgnet1k'
output_base_dir = '/mnt/tmpfs/imgnet1k_comp'
output_base_dir2 = '/mnt/tmpfs/repair_imgnet1k_comp'
num_threads = 8


broken_list = 'broken_file_list.txt'
with open(broken_list, 'r') as file:
    broken_files = file.readlines()

for broken_file in broken_files:
    # remove extention
    broken_file = broken_file.strip()
    broken_file = broken_file.split('/')[-1]
    broken_file = broken_file.split('.')[0]
    print(broken_file)

    category = broken_file.split('_')[0]
    input_file = f'{input_dir}/{category}/{broken_file}.JPEG'
    process_and_save_image(input_file, formats, qualities, input_dir, output_base_dir2)
    # process_and_save_image(input_file, formats, qualities, input_dir, output_base_dir)