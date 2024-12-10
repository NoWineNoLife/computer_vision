#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : tsing
# @time    : 2023/7/27 下午3:58
# @function: the script is used to do something.

import os
import torch
import torch.utils.data as Data
from PIL import Image

from .access_util import get_properties, transform_img, preprocess


class HandWrittenData(Data.Dataset):
    def __init__(self, a_cfg, a_train_flag):
        super(HandWrittenData, self).__init__()
        self.imgs_path = None
        self.imgs_filename = None
        self.labels = None
        self.transform = None
        if a_cfg.get('split_manual'):
            # TODO: Add implementation here

            pass



        if a_train_flag is True :
            self.imgs_path, self.imgs_filename, self.labels = get_properties(a_cfg['train'])
            self.transform = transform_img(a_cfg)
        else:
            self.imgs_path, self.imgs_filename, self.labels = get_properties(a_cfg['eval'])
            self.transform = transform_img(a_cfg)

    def __len__(self):
        return len(self.imgs_filename)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.imgs_path,
                                      self.imgs_filename[item])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        flag = int(self.labels[item])
        return img, torch.tensor(flag)
