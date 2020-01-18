import glob
import os
import random

from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
from PIL import Image

from utils.hard import get_device

class TrimapDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, bg_list, shape):
        self.photo_list = data_list[0]
        self.alpha_list = data_list[1]
        self.trimap_list = data_list[2]
        self.bg_list = bg_list
        self.shape = shape

        self.__init_device()
        self.__init_images()

    def __init_device(self):
        self.device, self.data_type = get_device()

    def __init_images(self):
        self.__load_portraits()
        self.__load_bgs()
        self.__pile_images()

    def __load_portraits(self):
        transform = transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor()
        ])
        self.fgs = self.__load_images(self.photo_list, transform=transform)
        self.alphas = self.__load_images(self.alpha_list, gray=True, transform=transform)
        self.trimaps = self.__load_images(self.trimap_list, gray=True, transform=transform)
        print('Portrait loaded.')

    def __load_bgs(self):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(self.shape, scale=(1.5, 2), ratio=(1, 1)),
            transforms.ToTensor()
        ])
        self.bgs = self.__load_images(self.bg_list, transform=transform)
        print('BG loaded.')

    def __pile_images(self):
        self.piles = []
        for fg, bg, alpha in tqdm(zip(self.fgs, self.bgs, self.alphas)):
            self.piles.append(self.__pile_one_image(fg, bg, alpha))
        print('Finish piling.')

    def __pile_one_image(self, fg, bg, alpha):
        return fg * alpha + bg * (1 - alpha)

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
        return len(self.photo_list)

    def __getitem__(self, idx):
        return self.piles[idx], self.fgs[idx], self.bgs[idx], self.alphas[idx], self.trimaps[idx]


def get_dataloaders(data_dir, bg_dir, train_num, test_num, batch_size=32, shape=(512, 384), suffle=True):
    k = train_num + test_num

    photo_list = glob.glob(os.path.join(data_dir, 'photo', '*'))
    photo_sample = random.sample(photo_list, k=k)
    alpha_sample = [path.replace('photo', 'alpha') for path in photo_sample]
    trimap_sample = [path.replace('photo', 'trimap') for path in photo_sample]

    bg_sample = random.sample(glob.glob(os.path.join(bg_dir, '*')), k=k)

    ph_train, ph_test, al_train, al_test, tr_train, tr_test, bg_train, bg_test = \
        train_test_split(photo_sample, alpha_sample, trimap_sample, bg_sample, test_size=test_num/k)

    train_dataset = TrimapDataset([ph_train, al_train, tr_train], bg_train, shape=shape)
    test_dataset = TrimapDataset([ph_test, al_test, tr_test], bg_test, shape=shape)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=suffle)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader
