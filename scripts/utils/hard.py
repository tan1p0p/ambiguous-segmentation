import torch

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
        data_type = torch.cuda.FloatTensor
    else:
        device = 'cpu'
        data_type = torch.cpu.FloatTensor
    return device, data_type
