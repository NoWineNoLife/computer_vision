import torch
import torch.nn as nn
import torchvision
import torchvision.models.vgg
from segment.pre_data import pre_data




class DetectionNet(nn.Module):
    def __init__(self, num_classes):
        super(DetectionNet, self).__init__()
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

        self.conv = nn.Sequential(

        )


        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )


    def forward(self, x):
        x = self.feature(x)
        x = x

        return x


