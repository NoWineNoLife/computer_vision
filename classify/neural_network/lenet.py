import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNet(nn.Module):
    def __init__(self, a_cfg=None):
        super(LinearNet, self).__init__()
        self.classes_num = a_cfg.get('classes_num')
        self.linear = nn.Linear(784, self.classes_num)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class LeNet(nn.Module):
    def __init__(self, a_cfg=None):
        super(LeNet, self).__init__()
        self.classes_num = a_cfg.get('classes_num')
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.adaptpool = nn.AdaptiveAvgPool2d(5, 5)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.classes_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(F.relu(x))
        x = self.conv2(x)
        x = self.adaptpool(F.relu(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, a_cfg=None):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, 4, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(64, 192, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.adaptpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classes_num = a_cfg.get('classes_num')
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 36, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.classes_num),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


