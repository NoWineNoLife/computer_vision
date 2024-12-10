import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

# class Block(nn.Module):
#     '''Depthwise conv + Pointwise conv'''
#     def __init__(self, in_planes, out_planes, stride=1):
#         super(Block, self).__init__()
#         self.conv1 = nn.Conv2d\
#             (in_planes, in_planes, kernel_size=3, stride=stride,
#              padding=1, groups=in_planes, bias=False)
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv2 = nn.Conv2d\
#             (in_planes, out_planes, kernel_size=1,
#             stride=1, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         return out
#
# class MobileNet(nn.Module):
#     # (128,2) means conv planes=128, conv stride=2,
#     # by default conv stride=1
#     cfg = [64, (128,2), 128, (256,2), 256, (512,2),
#            512, 512, 512, 512, 512, (1024,2), 1024]
#
#     def __init__(self, num_classes=10):
#         super(MobileNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
#         	stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.layers = self._make_layers(in_planes=32)
#         self.linear = nn.Linear(1024, num_classes)
#
#     def _make_layers(self, in_planes):
#         layers = []
#         for x in self.cfg:
#             out_planes = x if isinstance(x, int) else x[0]
#             stride = 1 if isinstance(x, int) else x[1]
#             layers.append(Block(in_planes, out_planes, stride))
#             in_planes = out_planes
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layers(out)
#         out = F.avg_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

class MobileV2(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileV2, self).__init__()
        net = torchvision.models.mobilenet_v2(pretrained=False)
        net.load_state_dict(torch.load('/home/hailongzhang/Models/mobilenet_v2-b0353104.pth'))
        net.classifier = nn.Sequential()
        self.features = net
        for param in self.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(1280, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MobileV_2(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileV_2, self).__init__()
        net = torchvision.models.mobilenet_v2(pretrained=False)
        net.load_state_dict(torch.load('/home/hailongzhang/Models/mobilenet_v2-b0353104.pth'))
        # net.classifier = nn.Sequential()
        self.features =nn.Sequential(*list(net.children())[:-1])
        for param in self.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(1280, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# net = MobileV2()
# opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)
