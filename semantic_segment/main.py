import sys
sys.path.append('/home/hailongzhang/personal/jones')
import numpy as np
import os
import torch
import torchvision
import torch.nn as nn
import cv2
import torch.utils.data as Data
import torchvision.transforms as T

from segment.customed.fcn import *
from segment.customed.unet import UnetVgg16
from segment.pre_data.pre_data import *

os.environ["CUDA_VISIBLE_DEVI" "CES"] = "0, 1"
transformer = T.Compose([
    T.ToTensor(),
    T.Resize(size=(959, 640))
])

transformer_penn = T.Compose([
    T.ToTensor()
])

class NailData(Data.Dataset):
    def __init__(self):
        super(NailData, self).__init__()
        self.root_path = '/home/yuki/DataSets/segment/nail_segmentation/'


imgs_dir = '/home/hailongzhang/DataSets/jones/segment/carvana/train_hq'
mask_dir = '/home/hailongzhang/DataSets/jones/segment/carvana/train_masks'

imgs_dir = '/home/yuki/DataSets/segment/PennFudanPed/PNGImages'
mask_dir = '/home/yuki/DataSets/segment/PennFudanPed/PedMasks'
# data_set = CarvanaData(imgs_dir, mask_dir, transformer)
# data_set = TGS_Salt(data_dir, transformer)
data_set = PennSegmentData(imgs_dir, mask_dir, transformer_penn)

fcn_model = UnetVgg16(2)
fcn_model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
fcn_model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1, stride=1)

deeplab_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
deeplab_model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1, stride=1)
deeplab_model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1, stride=1)


# iccv_imgs_path = '/home/yuki/DataSets/PennFudanPed/PNGImages'
# iccv_labels_path = '/home/yuki/DataSets/PennFudanPed/PedMasks/'
# train_data = PennData(iccv_imgs_path, iccv_labels_path, transformer_iccv)
train_loader = Data.DataLoader(dataset=data_set, batch_size=10, shuffle=True, drop_last=True)


# vgg_model = segment.VGGNet(requires_grad=True, show_params=False)
# model = segment.FCN32s(pretrained_net=vgg_model, n_class=2)

#model = nn.DataParallel(fcn_model).cuda()
model = nn.DataParallel(fcn_model).cuda()

EPOCH = 500
loss_f = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.98)

checkpoint = {}
for turn in range(EPOCH):
    for (train_x, train_y) in train_loader:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        # torch.squeeze
        train_y = torch.squeeze(train_y, 1)
        pred = model(train_x)['out']

        loss = loss_f(pred, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('EPOCH: %d, loss: %.6f' % (turn, loss.item()))

    scheduler.step()
    if( (1 + turn) % 5 == 0):
        checkpoint['EPOCH'] = turn
        checkpoint['optimizier'] = opt.state_dict()
        checkpoint['param'] = model.module.state_dict()
        torch.save(checkpoint, '/home/hailongzhang/Models/personal/jones/deeplabv3_resnet50_' + str(turn + 1) + '_checkpoint.pth')




