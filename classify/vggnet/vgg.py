import torch
import torch.nn as nn
import os
import torch.utils.data as Data
# from classify.pre_data import pre_data
from torchvision.datasets import ImageFolder
import torchvision.transforms

def getVgg16(path='/home/hailongzhang/Models/vgg16-397923af.pth', num_classes=2):
    vgg_model = torchvision.models.vgg16(pretrained=False)
    vgg_model.load_state_dict(torch.load(path))
    for param in vgg_model.parameters():
        param.requires_grad = False

    vgg_model.classifier._modules['6'] = nn.Linear(4096, num_classes)
    # vgg_model.classifier._modules['7'] = torch.nn.LogSoftmax(dim=1)
    return vgg_model