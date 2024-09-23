import os
from PIL import Image
from tqdm import tqdm
from pillow_heif import AvifImagePlugin
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def count_and_list_files(directories):
    file_list = []
    total_files = 0
    print("calculating: ")
    for directory in directories:
        for root, _, files in os.walk(directory):
            full_paths = [os.path.join(root, file) for file in files]
            file_list.extend(full_paths)
            total_files += len(files)
            if total_files % 100000 == 0:
                print(f"gather {total_files} files")
    print(f"total {total_files} ")
    return file_list, total_files

def process_file(file_path, pbar, lock):
    try:
        with Image.open(file_path) as img:
            img.convert('RGB')
    except Exception as e:
        print(f" {file_path} has error {e}")
    finally:
        with lock:
            pbar.update(1)

def main():
    paths = [
        "/mnt/tmpfs/imgnet1k_comp/avif/20",
        "/mnt/tmpfs/imgnet1k_comp/avif/10",
    ]

    file_list, total_files = count_and_list_files(paths)
    num_threads = 32
    lock = Lock()

    print("progressing")
    with tqdm(total=total_files, desc="total") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(process_file, file_path, pbar, lock)
                for file_path in file_list
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"error: {e}")
    print("finished. ")

if __name__ == "__main__":
    main()