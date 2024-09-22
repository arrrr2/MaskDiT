import argparse
import os
import time

import lmdb
import torch
from torch import nn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from autoencoder import get_model
from train_utils.datasets import imagenet_lmdb_dataset

codecs = ['avif', 'jpeg', 'webp']
qualities = [10, 20, 40, 60, 80, 90]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/ubuntu/data/datasets/imgnet1k_comp', type=str)
    parser.add_argument('--codec', type=str)
    parser.add_argument('--quality', type=int)
    parser.add_argument('--device_ids', nargs=1, type=str)
    parser.add_argument('--ckpt', default='assets/vae/autoencoder_kl.pth', type=str, help='checkpoint path')
    parser.add_argument('--resolution', default=256, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--xflip', action='store_true')
    parser.add_argument('--outdir', type=str, default='./latents/imagenet256', help='output directory')
    args = parser.parse_args()

    args.device_ids = [int(i) for i in args.device_ids[0].split(',')]
    assert args.split in ['train', 'val']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
 
    dataset = imagenet_lmdb_dataset(root=f'{os.path.join(args.data_dir, args.codec, str(args.quality))}', 
                                    transform=transform, resolution=args.resolution)

    print(f'data size: {len(dataset)}')

    model = get_model(args.ckpt)
    print(f'load vae weights from autoencoder_kl.pth')
    model = nn.DataParallel(model, device_ids=args.device_ids)
    device = torch.device(f"cuda:{args.device_ids[0]}") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    def extract_feature():
        target_db_dir = os.path.join(args.outdir, args.codec, str(args.quality))
        os.makedirs(target_db_dir, exist_ok=True)
        target_env = lmdb.open(target_db_dir, map_size=pow(2,40), readahead=False)

        dataset_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                    num_workers=16, pin_memory=True, persistent_workers=True)

        idx = 0
        begin = time.time()
        print('start...')
        for batch in dataset_loader:
            img, label = batch
            assert img.min() >= -1 and img.max() <= 1
        
            img = img.to(device)
            moments = model(img, fn='encode_moments')
            assert moments.shape[-1] == (args.resolution // 8)
        
            moments = moments.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
        
            with target_env.begin(write=True) as target_txn:
                for moment, lb in zip(moments, label):
                    target_txn.put(f'z-{str(idx)}'.encode('utf-8'), moment)
                    target_txn.put(f'y-{str(idx)}'.encode('utf-8'), str(lb).encode('utf-8'))
                    idx += 1
        
            if idx % 10000 == 0:
                cur_time = time.time()
                spent = cur_time - begin
                speed = 10000 / spent
                print(f'saved {idx} files with {spent:.4f}s elapsed. speed: {speed:.4f} imgs/sec.')
                begin = time.time()

        # idx = 1_281_167
        if args.xflip:
            print('starting to store the xflip latents')
            begin = time.time()
            for batch in dataset_loader:
                img, label = batch
                assert img.min() >= -1 and img.max() <= 1

                img = img.to(device)
                moments = model(img.flip(dims=[-1]), fn='encode_moments')

                moments = moments.detach().cpu().numpy()
                label = label.detach().cpu().numpy()

                with target_env.begin(write=True) as target_txn:
                    for moment, lb in zip(moments, label):
                        target_txn.put(f'z-{str(idx)}'.encode('utf-8'), moment)
                        target_txn.put(f'y-{str(idx)}'.encode('utf-8'), str(lb).encode('utf-8'))
                        idx += 1

                if idx % 10000 == 0:
                    cur_time = time.time()
                    print(f'saved {idx} files with {cur_time - begin}s elapsed')
                    begin = time.time()

        with target_env.begin(write=True) as target_txn:
            target_txn.put('length'.encode('utf-8'), str(idx).encode('utf-8'))

        print(f'[finished] saved {idx} files')

    extract_feature()


if __name__ == "__main__":
    main()
