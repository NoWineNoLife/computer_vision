#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : tsing
# @time    : 2023/7/27 下午3:19
# @function: the script is used to do something.

import sys
import os
import yaml

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from neural_network.lenet import LeNet
from dataset.hand_written_dataset import HandWrittenData


if __name__ == '__main__':
    yaml_file_path = 'configuration/base.yaml'
    with open(yaml_file_path, 'r', encoding='utf8') as f:
        config = yaml.safe_load(f.read())
    nn_classifier = eval(config['neural_network']['category'])(config['neural_network'])
    opt = None
    loss_fn = eval(config['loss_fn'])()
    dataset = eval(config['dataset']['train']['name'])(config['dataset'])

    # opt = eval(config['optimizer']['category'])(config['optimizer'])