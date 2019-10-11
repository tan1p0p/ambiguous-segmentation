import glob
import os
import random

import torch
from torchvision import transforms, datasets
from tqdm import tqdm
from PIL import Image

from utils.hard import get_device

class TrimapDataset(torch.utils.data.Dataset):
    def __init__(self, fg_root_dir, bg_dir, data_num, scale=80):
        self.fg_root_dir = fg_root_dir
        self.bg_dir = bg_dir
        self.data_num = data_num
        self.scale = scale

        self.__init_device()
        self.__init_dir_path()
        self.__init_images()

    def __init_device(self):
        self.device, self.data_type = get_device()

    def __init_dir_path(self):
        self.fg_dir = os.path.join(self.fg_root_dir, 'origin')
        self.alpha_dir = os.path.join(self.fg_root_dir, 'gt')
        self.trimap_dir = os.path.join(self.fg_root_dir, 'trimap')

    def __init_images(self):
        self.__load_portraits()
        self.__load_bgs()
        self.__pile_images()

    def __load_portraits(self):
        transform = transforms.Compose([
            transforms.Resize((self.scale, self.scale)),
            transforms.ToTensor()
        ])
        photo_list, alpha_list, trimap_list = self.__get_samples()
        self.fgs = self.__load_images(photo_list, transform=transform)
        self.alphas = self.__load_images(alpha_list, gray=True, transform=transform)
        self.trimaps = self.__load_images(trimap_list, gray=True, transform=transform)
        print('Portrait loaded.')

    def __load_bgs(self):
        transform = transforms.Compose([
            transforms.Resize(self.scale * 2),
            transforms.RandomCrop(self.scale),
            transforms.ToTensor()
        ])
        bg_list = random.choices(sorted(glob.glob(os.path.join(self.bg_dir, '*'))), k=self.data_num)
        self.bgs = self.__load_images(bg_list, transform=transform)
        print('BG loaded.')

    def __pile_images(self):
        self.piles = []
        for fg, bg, alpha in tqdm(zip(self.fgs, self.bgs, self.alphas)):
            self.piles.append(self.__pile_one_image(fg, bg, alpha))
        print('Finish piling.')

    def __pile_one_image(self, fg, bg, alpha):
        return fg * alpha + bg * (1 - alpha)

    def __get_samples(self):
        photo_list = glob.glob(os.path.join(self.fg_root_dir, 'origin', '*'))
        photo_sample = random.sample(photo_list, k=self.data_num)
        alpha_sample = [path.replace('origin', 'gt') for path in photo_sample]
        trimap_sample = [path.replace('origin', 'trimap') for path in photo_sample]
        return photo_sample, alpha_sample, trimap_sample

    def __load_images(self, path_list, gray=False, transform=None):
        img_list = []
        for filepath in tqdm(path_list):
            img = Image.open(filepath)
            if gray:
                img = img.convert('L')
            else:
                img = img.convert('RGB')

            if transform:
                img = transform(img)
            img_list.append(img)
        return img_list

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.piles[idx], self.fgs[idx], self.bgs[idx], self.alphas[idx], self.trimaps[idx]

def get_dataloader(data_dir, bg_dir, data_num, batch_size=32, scale=80):
    dataset = TrimapDataset(data_dir, bg_dir, data_num=data_num, scale=scale)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
