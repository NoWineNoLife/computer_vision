import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        net = torchvision.models.resnet34(pretrained=False)
        net.load_state_dict(torch.load('/home/hailongzhang/Models/resnet34-b627a593.pth'))
        net.fc = nn.Sequential()
        for param in self.parameters():
            param.requires_grad = False

        self.feafures = net
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.feafures(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = nn.Softmax(x, dim=1)

        return x
