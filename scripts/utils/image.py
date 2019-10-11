import numpy as np
import torch

def to_one_hot_trimap(input_image):
    input_image = input_image.numpy()
    labels = (np.floor(input_image / 0.5)).astype(np.int64) # [height, width]の中に0, 1, 2のintで格納
    # one_hot_label = np.zeros([input_image.shape[0], 3, input_image.shape[2], input_image.shape[3]])
    # for i in range(input_image.shape[0]):
    #     one_hot_label[i] = np.eye(3)[labels[i]].transpose(0, 3, 1, 2)
    return torch.Tensor(labels).view(labels.shape[0], labels.shape[2], labels.shape[3])
