import glob
import os

import torch
from torchvision import transforms, datasets
from PIL import Image

from utils.hard import get_device

class TrimapDataset(torch.utils.data.Dataset):
    def __init__(self, fg_root_dir, device, data_type, scale=80):
        self.fg_root_dir = fg_root_dir
        self.scale = scale

        self.__init_device()
        self.__init_dir_path()
        self.__init_images()

    def __init_device(self):
        self.device, self.data_type = get_device()

    def __init_dir_path(self):
        self.fg_dir = os.path.join(self.fg_root_dir, 'origin')
        self.alpha_dir = os.path.join(self.fg_root_dir, 'gt')
        self.bg_dir = '/home/hassaku/dataset/VOCdevkit/VOC2007/JPEGImages/'

    def __init_images(self):
        # self.data_num = len(glob.glob(os.path.join(self.fg_dir, '*')))
        self.data_num = 10
        self.__load_portraits()
        self.__load_bgs()
        self.__pile_images()

    def __load_portraits(self):
        transform = transforms.Compose([
            transforms.Resize((self.scale, self.scale)),
            transforms.ToTensor()
        ])
        self.fgs = self.__load_images(self.fg_dir, transform=transform)
        self.data_num = len(self.fgs)
        self.alphas = self.__load_images(self.alpha_dir, gray=True, transform=transform)
        print('Portrait loaded.')

    def __load_bgs(self):
        transform = transforms.Compose([
            transforms.Resize(self.scale * 2),
            transforms.RandomCrop(self.scale),
            transforms.ToTensor()
        ])
        self.bgs = self.__load_images(self.bg_dir, transform=transform)
        print('BG loaded.')

    def __pile_images(self):
        self.piles = [self.__pile_one_image(fg, bg, alpha) for fg, bg, alpha in zip(self.fgs, self.bgs, self.alphas)]
        print('Finish piling.')

    def __pile_one_image(self, fg, bg, alpha):
        return fg * alpha + bg * (1 - alpha)

    def __load_images(self, dir_path, gray=False, transform=None):
        path_list = glob.glob(os.path.join(dir_path, '*'))[:self.data_num]

        img_list = []
        for filepath in path_list:
            img = Image.open(filepath)
            if gray:
                img = img.convert('L')
            if transform:
                img = transform(img)
            img_list.append(img.to(self.device).type(self.data_type))
        return img_list

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        # TODO: add another transforms for data augumant
        # transform = transforms.Compose([
        #     transforms.Normalize(
        #         mean = [0.485, 0.456, 0.406],
        #         std = [0.229, 0.224, 0.225])
        # ])
        return self.piles[idx], self.fgs[idx], self.bgs[idx]

def get_dataloader(data_dir, batch_size=32, scale=80):
    dataset = TrimapDataset(data_dir, scale)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    dataset = get_dataloader('./data/96x64/train/')
    for pile, fg, bg in dataset:
        print(pile.shape, fg.shape, bg.shape)
        break
