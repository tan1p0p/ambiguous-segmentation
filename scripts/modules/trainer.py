import os
from statistics import mean

import cloudpickle
from tqdm import tqdm
import torch
import torch.nn as nn
torch.manual_seed(0)

from modules.loss import AlphaMatteLoss, AmbiguousLoss
from modules.model import SegNet, DeepImageMatting
from utils.data import get_dataloader
from utils.hard import get_device
from utils.visualize import save_line

class Trainer():
    def __init__(self, portrait_dir, bg_dir, output_path, matting_model,
        epochs, batch_size, train_data_num, test_data_num,
        is_file_saved=True):

        self.train_dir = portrait_dir + 'train'
        self.test_dir = portrait_dir + 'test'
        self.bg_dir = bg_dir
        self.matting_weight_path = matting_model
        self.output_path = output_path
        self.iter_num = epochs
        self.is_file_saved = is_file_saved
        self.batch_size = batch_size
        self.train_data_num = train_data_num
        self.test_data_num = test_data_num

        self.__init_device()
        self.__init_dataloader()
        self.__init_nets()
        self.__init_layers()
        self.__init_optim()
        self.__init_losses()

    def __init_device(self):
        self.device, self.data_type = get_device()

    def __init_dataloader(self):
        self.train_dataloader = get_dataloader(self.train_dir, self.bg_dir,
            batch_size=self.batch_size, data_num=self.train_data_num)
        print('Train images loaded.')
        self.test_dataloader = get_dataloader(self.test_dir, self.bg_dir,
            batch_size=self.batch_size, data_num=self.test_data_num)
        print('Test images loaded.')

    def __init_nets(self):
        self.trimap_stage = SegNet().to(self.device).type(self.data_type)
        self.matting_stage = DeepImageMatting(stage=1).to(self.device).type(self.data_type)
        self.matting_stage.load_state_dict(torch.load(self.matting_weight_path)['state_dict'], strict=True)
        self.matting_stage.eval()
        print('Model loaded.')

    def __init_layers(self):
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)
        self.maxpool = nn.MaxPool2d(4)

    def __init_optim(self):
        for param in list(self.matting_stage.parameters()) + \
            list(self.upsample.parameters()) + \
            list(self.maxpool.parameters()):
            param.requires_grad = False

        params = list(self.trimap_stage.parameters()) + \
            list(self.upsample.parameters()) + \
            list(self.matting_stage.parameters()) + \
            list(self.maxpool.parameters())
        self.optimizer = torch.optim.Adam(params)

    def __init_losses(self):
        self.train_loss, self.test_loss = [], []
        self.current_loss = 1
        self.alpha_matte_loss_func = AlphaMatteLoss()
        self.ambiguous_loss_func = AmbiguousLoss()
        self.mse_loss_func = nn.MSELoss()

    def __prediction(self, pile, calc_grad=True):
        for param in list(self.trimap_stage.parameters()):
            if calc_grad == True:
                param.requires_grad = True
            else:
                param.requires_grad = False

        pred_trimap = self.trimap_stage(pile)
        concat = torch.cat((pile, pred_trimap), dim=1)
        pred_alpha_big, _ = self.matting_stage(self.upsample(concat))
        return self.maxpool(pred_alpha_big), pred_trimap

    def __fit(self, pile, fg, bg, alpha, trimap, calc_grad=True):
        pile = pile.to(self.device).type(self.data_type)
        fg = fg.to(self.device).type(self.data_type)
        bg = bg.to(self.device).type(self.data_type)
        alpha = alpha.to(self.device).type(self.data_type)
        trimap = trimap.to(self.device).type(self.data_type)

        pred_alpha, pred_trimap = self.__prediction(pile, calc_grad=calc_grad)

        # calc losses
        loss = self.alpha_matte_loss_func(pred_alpha, pred_trimap, fg, bg, pile) * 0.5
        loss += self.mse_loss_func(pred_alpha, alpha) * 0.5
        # loss += self.ambiguous_loss_func(pred_trimap) * 0.1
        return loss

    def optimize(self):
        for epoch in range(self.iter_num):
            epoch_train_loss, epoch_test_loss = [], []

            for pile, fg, bg, alpha, trimap in tqdm(
                self.train_dataloader,
                postfix='{}/{} Loss: {:05f}'.format(epoch, self.iter_num, self.current_loss)):

                loss = self.__fit(pile, fg, bg, alpha, trimap, calc_grad=True)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss.append(loss.item())

            self.train_loss.append(mean(epoch_train_loss))
            self.current_loss = self.train_loss[-1]

            for pile, fg, bg, alpha, trimap in self.test_dataloader:
                loss = self.__fit(pile, fg, bg, alpha, trimap, calc_grad=False)
                epoch_test_loss.append(loss.item())

            self.test_loss.append(mean(epoch_test_loss))

            if self.is_file_saved:
                save_line([self.train_loss, self.test_loss], title='loss_{:04d}'.format(epoch), output_dir=self.output_path)
                with open(os.path.join(self.output_path, 'trimap_{:04d}.model'.format(epoch)), 'wb') as f:
                    cloudpickle.dump(self.trimap_stage, f)
                # torch.save(self.trimap_stage.state_dict(), self.output_path + 'trimap_{:04d}.model'.format(epoch))
