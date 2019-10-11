import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hard import get_device

class SegNet(nn.Module):
    def __init__(self, mode='joint'):
        super(SegNet, self).__init__()
        _, self.data_type = get_device()
        self.mode = mode

        self.dropout = nn.Dropout2d(0.3)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(3)

        self.threshold = nn.Threshold(0.5, 0)

    def forward(self, input):
        x = self.maxpool(F.relu(self.dropout(self.conv1(input))))
        x = self.maxpool(F.relu(self.dropout(self.conv2(x))))
        x = self.maxpool(F.relu(self.dropout(self.conv3(x))))
        x = self.maxpool(F.relu(self.dropout(self.conv4(x))))

        x = F.relu(self.deconv1(self.upsample(x)))
        x = F.relu(self.deconv2(self.upsample(x)))
        x = F.relu(self.deconv3(self.upsample(x)))
        x = self.batch_norm(self.deconv4(self.upsample(x)))
        x = F.softmax(x, dim=1)

        if self.mode == 'joint':
            x = self.threshold(x)
            consts = torch.Tensor(
                np.mod(np.arange(input.shape[0] * 3 * 80 * 80), 3).reshape((-1, 80, 80, 3)).transpose(0, 3, 1, 2)).type(self.data_type) / 2
            x = torch.sum(x * consts, dim=1)
            x = x.view(-1, 1, 80, 80)
        return x

class DeepImageMatting(nn.Module):
    def __init__(self, stage):
        super(DeepImageMatting, self).__init__()
        self.stage = stage

        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)

        # model released before 2019.09.09 should use kernel_size=1 & padding=0
        #self.conv6_1 = nn.Conv2d(512, 512, kernel_size=1, padding=0,bias=True)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)

        self.deconv6_1 = nn.Conv2d(512, 512, kernel_size=1, bias=True)
        self.deconv5_1 = nn.Conv2d(512, 512, kernel_size=5, padding=2, bias=True)
        self.deconv4_1 = nn.Conv2d(512, 256, kernel_size=5, padding=2, bias=True)
        self.deconv3_1 = nn.Conv2d(256, 128, kernel_size=5, padding=2, bias=True)
        self.deconv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2, bias=True)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True)

        self.deconv1 = nn.Conv2d(64, 1, kernel_size=5, padding=2, bias=True)

        if self.stage == 2:
            # for stage2 training
            for p in self.parameters():
                p.requires_grad=False
        
        if self.stage == 2 or self.stage == 3:
            self.refine_conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=True)
            self.refine_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
            self.refine_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
            self.refine_pred = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.conv1_1(x))
        x12 = F.relu(self.conv1_2(x11))
        x1p, id1 = F.max_pool2d(x12,kernel_size=(2,2), stride=(2,2), return_indices=True)

        # Stage 2
        x21 = F.relu(self.conv2_1(x1p))
        x22 = F.relu(self.conv2_2(x21))
        x2p, id2 = F.max_pool2d(x22,kernel_size=(2,2), stride=(2,2), return_indices=True)

        # Stage 3
        x31 = F.relu(self.conv3_1(x2p))
        x32 = F.relu(self.conv3_2(x31))
        x33 = F.relu(self.conv3_3(x32))
        x3p, id3 = F.max_pool2d(x33,kernel_size=(2,2), stride=(2,2), return_indices=True)

        # Stage 4
        x41 = F.relu(self.conv4_1(x3p))
        x42 = F.relu(self.conv4_2(x41))
        x43 = F.relu(self.conv4_3(x42))
        x4p, id4 = F.max_pool2d(x43,kernel_size=(2,2), stride=(2,2), return_indices=True)

        # Stage 5
        x51 = F.relu(self.conv5_1(x4p))
        x52 = F.relu(self.conv5_2(x51))
        x53 = F.relu(self.conv5_3(x52))
        x5p, id5 = F.max_pool2d(x53,kernel_size=(2,2), stride=(2,2), return_indices=True)

        # Stage 6
        x61 = F.relu(self.conv6_1(x5p))

        # Stage 6d
        x61d = F.relu(self.deconv6_1(x61))

        # Stage 5d
        x5d = F.max_unpool2d(x61d,id5, kernel_size=2, stride=2)
        x51d = F.relu(self.deconv5_1(x5d))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x41d = F.relu(self.deconv4_1(x4d))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x31d = F.relu(self.deconv3_1(x3d))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x21d = F.relu(self.deconv2_1(x2d))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.deconv1_1(x1d))

        # Should add sigmoid? github repo add so.
        raw_alpha = self.deconv1(x12d)
        pred_mattes = torch.sigmoid(raw_alpha)

        if self.stage <= 1:
            return pred_mattes, 0

        # Stage2 refine conv1
        refine0 = torch.cat((x[:, :3, :, :], pred_mattes),  1)
        refine1 = F.relu(self.refine_conv1(refine0))
        refine2 = F.relu(self.refine_conv2(refine1))
        refine3 = F.relu(self.refine_conv3(refine2))
        # Should add sigmoid?
        # sigmoid lead to refine result all converge to 0... 
        #pred_refine = torch.sigmoid(self.refine_pred(refine3))
        pred_refine = self.refine_pred(refine3)

        pred_alpha = torch.sigmoid(raw_alpha + pred_refine)

        #print(pred_mattes.mean(), pred_alpha.mean(), pred_refine.sum())

        return pred_mattes, pred_alpha
