from tqdm import tqdm
import torch
import torch.nn as nn
torch.manual_seed(0)

from modules.model import SegNet, DeepImageMatting
from utils.data import get_dataloader

class Trainer():
    def __init__(self, data_dir):
        self.train_dir = data_dir + 'train'
        self.test_dir = data_dir + 'test'
        self.matting_weight_path = '/home/hassaku/research/ambiguous-segmentation/models/stage1_sad_54.4.pth'
        self.iter_num = 20

        self.__init_device()
        self.__init_dataloader()
        self.__init_nets()
        self.__init_optim()
        self.__init_losses()

    def __init_device(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.data_type = torch.cuda.FloatTensor
        else:
            self.device = 'cpu'
            self.data_type = torch.cpu.FloatTensor

    def __init_dataloader(self):
        self.train_dataloader = get_dataloader(self.train_dir)
        print('Train images loaded.')
        self.test_dataloader = get_dataloader(self.test_dir)
        print('Test images loaded.')

    def __init_nets(self):
        self.trimap_stage = SegNet().to(self.device).type(self.data_type)
        self.matting_stage = DeepImageMatting(stage=1).to(self.device).type(self.data_type)
        self.matting_stage.load_state_dict(torch.load(self.matting_weight_path)['state_dict'], strict=True)
        self.matting_stage.eval()
        for param in self.matting_stage.parameters():
            param.requires_grad = False
        print('Model loaded.')

    def __init_optim(self):
        self.optimizer = torch.optim.Adam(self.trimap_stage.parameters())

    def __init_losses(self):
        self.loss_func = torch.nn.MSELoss()

    def optimize(self):
        for i in range(self.iter_num):
            for X, t in tqdm(self.train_dataloader, postfix='{}/{}'.format(i, self.iter_num)):
                X = X.to(self.device).type(self.data_type)
                t = t.to(self.device).type(self.data_type)

                self.optimizer.zero_grad()

                upsample = nn.UpsamplingBilinear2d(scale_factor=4)
                maxpool = nn.MaxPool2d(4)

                y1 = self.trimap_stage(X)
                concat = torch.cat((y1, X), dim=1)
                y2, _ = self.matting_stage(upsample(concat))
                loss = self.loss_func(maxpool(y2), t)
                loss.backward()
