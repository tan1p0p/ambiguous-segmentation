from tqdm import tqdm
import torch
import torch.nn as nn
torch.manual_seed(0)

from modules.loss import Loss
from modules.model import SegNet, DeepImageMatting
from utils.data import get_dataloader
from utils.hard import get_device

class Trainer():
    def __init__(self, portrait_dir, bg_dir, output_path, epochs, matting_model, is_file_saved=True):
        self.train_dir = portrait_dir + 'train'
        self.test_dir = portrait_dir + 'test'
        self.bg_dir = bg_dir
        self.matting_weight_path = matting_model
        self.output_path = output_path
        self.iter_num = epochs
        self.is_file_saved = is_file_saved

        self.__init_device()
        self.__init_dataloader()
        self.__init_nets()
        self.__init_optim()
        self.__init_losses()

    def __init_device(self):
        self.device, self.data_type = get_device()

    def __init_dataloader(self):
        self.train_dataloader = get_dataloader(self.train_dir, self.bg_dir)
        print('Train images loaded.')
        self.test_dataloader = get_dataloader(self.test_dir, self.bg_dir)
        print('Test images loaded.')

    def __init_nets(self):
        self.trimap_stage = SegNet().to(self.device).type(self.data_type)
        self.matting_stage = DeepImageMatting(stage=1).to(self.device).type(self.data_type)
        self.matting_stage.load_state_dict(torch.load(self.matting_weight_path)['state_dict'], strict=True)
        self.matting_stage.eval()
        for param in self.trimap_stage.parameters():
            param.requires_grad = False
        print('Model loaded.')

    def __init_optim(self):
        params = list(self.trimap_stage.parameters()) + list(self.matting_stage.parameters())
        self.optimizer = torch.optim.Adam(params)

    def __init_losses(self):
        self.current_loss = 1
        self.loss_func = Loss()

    def optimize(self):
        for epoch in range(self.iter_num):
            for pile, fg, bg in tqdm(
                self.train_dataloader,
                postfix='{}/{} Loss: {:05f}'.format(epoch, self.iter_num, self.current_loss)):

                pile = pile.to(self.device).type(self.data_type)
                fg = fg.to(self.device).type(self.data_type)
                bg = bg.to(self.device).type(self.data_type)

                upsample = nn.UpsamplingBilinear2d(scale_factor=4)
                maxpool = nn.MaxPool2d(4)
                pred_trimap = self.trimap_stage(pile)
                concat = torch.cat((pred_trimap, pile), dim=1)
                pred_alpha, _ = self.matting_stage(upsample(concat))

                loss = self.loss_func(maxpool(pred_alpha), fg, bg, pile)
                self.current_loss = loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.is_file_saved and epoch % 10 == 0:
                torch.save(self.trimap_stage.state_dict(), self.output_path + 'trimap_{:04d}.model'.format(epoch))
