# -*- coding = utf-8 -*-
#  * @file              : infer.py
#  * @author            : tsing
#  * @brief             : None
#  * @attention         : None
#  * @time              : 2023/8/25 上午11:01
# @function: the script is used to do something.


import os
import torch
import torchvision
import cv2
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import torchvision.transforms as T

from semantic_segment.dataset.mallampti import Mallampti


os.environ["CUDA_VISIBLE_DEVI" "CES"] = "0"

transformer_penn = T.Compose([
    T.ToTensor()
])

imgs_dir = '/home/hailongzhang/DataSets/jones/segment/carvana/train_hq'
mask_dir = '/home/hailongzhang/DataSets/jones/segment/carvana/train_masks'

imgs_dir = '/home/yuki/DataSets/segment/PennFudanPed/PNGImages'
mask_dir = '/home/yuki/DataSets/segment/PennFudanPed/PedMasks'
# data_set = CarvanaData(imgs_dir, mask_dir, transformer)
# data_set = TGS_Salt(data_dir, transformer)
# data_set = PennSegmentData(imgs_dir, mask_dir, transformer_penn)

# fcn_model = UnetVgg16(2)
fcn_model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
fcn_model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1, stride=1)

train_txt = "/home/hozumi/datasets/mallampti/train.txt"
val_txt = "/home/hozumi/datasets/mallampti/val.txt"
train_set = Mallampti(train_txt, transformer_penn)
val_set = Mallampti(val_txt, transformer_penn)
train_loader = Data.DataLoader(dataset=train_set, batch_size=1)
val_loader = Data.DataLoader(dataset=val_set, batch_size=1)

# vgg_model = segment.VGGNet(requires_grad=True, show_params=False)
# model = segment.FCN32s(pretrained_net=vgg_model, n_class=2)

checkpoint_path = "/home/hozumi/python_projects/computer_vision/semantic_segment/res/500_checkpoint.pth"
checkpoint = torch.load(checkpoint_path)
fcn_model.load_state_dict(checkpoint['param'])
model = nn.DataParallel(fcn_model).cuda()

EPOCH = 500
loss_f = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.98)

model.eval()
for val_x, val_y in train_loader:
    val_x, val_y = val_x.cuda(), val_y.cuda()
    pred = model(val_x)['out']
    gray = torch.argmax(pred, dim=1)
    gray = torch.squeeze(gray)
    gray_img = gray.cpu().detach().numpy()
    print(np.unique(gray_img))
    gray_img = gray_img.astype(np.uint8)
    _, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)



    cv2.imshow("11", gray_img)
    cv2.waitKey(0)








# for turn in range(EPOCH):
#     for (train_x, train_y) in train_loader:
#         train_x, train_y = train_x.cuda(), train_y.cuda()
#         # torch.squeeze
#         train_y = torch.squeeze(train_y, 1)
#         pred = model(train_x)['out']
#
#         loss = loss_f(pred, train_y)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         print('EPOCH: %d, loss: %.6f' % (turn, loss.item()))
#
#     scheduler.step()
#     if( (1 + turn) % 5 == 0):
#         checkpoint['EPOCH'] = turn
#         checkpoint['optimizier'] = opt.state_dict()
#         checkpoint['param'] = model.module.state_dict()
#         torch.save(checkpoint, './res/' + str(turn + 1) + '_checkpoint.pth')
