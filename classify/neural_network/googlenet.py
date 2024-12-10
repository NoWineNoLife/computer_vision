#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : tsing
# @time    : 2023/7/27 下午3:50
# @function: the script is used to do something.

import torch.nn as nn

class GoogLenet(nn.Module):
    def __init__(self, num_classes):
        super(GoogLenet, self).__init__()