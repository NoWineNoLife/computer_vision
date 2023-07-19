import os
import cv2
import torch
import torch.utils.data as Data
from PIL import Image


class HandWrittenData(Data.Dataset):

    def __init__(self, dir, transform=None):
        super(HandWrittenData, self).__init__()
        self.dir = dir
        self.all_imgs = os.listdir(self.dir)
        self.transform = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, item):
        # img = cv2.imread(os.path.join(self.dir, self.all_imgs[item]), cv2.IMREAD_GRAYSCALE)
        img = Image.open(os.path.join(self.dir,
                                      self.all_imgs[item])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        flag = int(self.all_imgs[item][4])
        return img, torch.tensor(flag)
