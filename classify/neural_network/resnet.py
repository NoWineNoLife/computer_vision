#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : tsing
# @time    : 2023/7/27 下午5:34
# @function: the script is used to do something.
import torch.nn as nn
import torchvision


class ResNet(nn.Module):
    def __init__(self, a_cfg=None):
        super(ResNet, self).__init__()
        net = torchvision.models.resnet34(pretrained=False)
        net.fc = nn.Sequential()
        for param in self.parameters():
            param.requires_grad = False

        self.feafures = net
        self.classes_num = a_cfg.get('classes_num')
        self.fc = nn.Linear(512, self.classes_num)

    def forward(self, x):
        x = self.feafures(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = nn.Softmax(x, dim=1)

        return x
