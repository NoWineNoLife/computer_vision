import sys
sys.path.append('/home/hailongzhang/personal/jones')
import cv2
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as T
from detection.unet.unet import UNet
# from detection.pre_data.pre_data import CarvanaData
import torchvision.models

transformer = T.Compose([
    T.ToTensor(),
    T.Resize(size=(959, 640))
])
# data_dir = '/home/qdrs/Documents/test_imgs'
# data_set = CarvanaData(data_dir, transformer)
# data_loader = Data.DataLoader(dataset=data_set, batch_size=2, shuffle=True)

# fcn_model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
# fcn_model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1, stride=1)
#
#
# deeplab_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
# deeplab_model.classifier[4] = nn.Conv2d(256, 2, 1, 1)
# deeplab_model.aux_classifier[4] = nn.Conv2d(256, 2, 1, 1)
# deeplab_model.eval()


# checkpoint = {}
# checkpoint = torch.load('/home/hailongzhang/Models/personal/jones/deeplabv3_resnet50_45_checkpoint.pth')
# deeplab_model.load_state_dict(checkpoint['param'])
# model = deeplab_model.cuda()

#
# img = Image.open('/home/hailongzhang/DataSets/jones/segment/carvana/test/3a7a8f03e713_01.jpg').convert("RGB")
# img = Image.open('/home/hailongzhang/DataSets/jones/segment/PennFudanPed/PNGImages/PennPed00017.png').convert("RGB")

# img = cv2.imread("/home/yuki/Pictures/20090517103324-1501301850.jpg", 0)
# _, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
#
# con, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# cv2.imshow("111", img)
# cv2.waitKey(0)

import glob
print(glob.glob("/home/yuki/rubbish/*"))

