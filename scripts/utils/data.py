import glob
import os

import torch
from torchvision import transforms, datasets
from PIL import Image

class TrimapDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale=80):
        self.root_dir = root_dir
        self.scale = scale

        self.__init_transform()
        self.__init_images()

    def __init_transform(self):
        self.transform = transforms.Compose([
            transforms.Resize(self.scale),
            transforms.ToTensor()
        ])

    def __init_images(self):
        self.photos = self.__load_images('origin')
        self.alphas = self.__load_images('gt')
        self.data_num = len(self.photos)

    def __load_images(self, img_type):
        path_list = glob.glob(os.path.join(self.root_dir, img_type, '*'))
        img_list = []
        for filepath in path_list:
            img_list.append(self.transform(Image.open(filepath)))
        return img_list

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        # TODO: add another transforms for data augumant
        return self.photos[idx], self.alphas[idx]

def get_dataloader(data_dir, batch_size=32, scale=80):
    dataset = TrimapDataset(data_dir, scale)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    dataset = get_dataloader('./data/96x64/train/')
    for a, b in dataset:
        print(b)
        exit()
