#!/usr/bin/python
# -*- coding = uft-8 -*-
#  * @file           : ttt.py
#  * @author         : tsing
#  * @brief          : None
#  * @attention      : None
#  * @date           : 2023/7/27


import sys
import os
import yaml
import argparse

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from classify.dataset.hand_written_dataset import HandWrittenData
from classify.neural_network.resnet import ResNet

parser = argparse.ArgumentParser()


def arg_parse():
    parser.add_argument("--yaml_path", default='configuration/base.yaml')
    return parser


if __name__ == '__main__':
    parser = arg_parse()
    cfg_path = parser.parse_args().yaml_path
    with open(cfg_path, 'r', encoding='utf8') as f:
        config = yaml.safe_load(f.read())
    nn_classifier = eval(config['neural_network']['category'])(config['neural_network'])
    opt = eval(config['optimizer']['category'])(nn_classifier.parameters(), config['optimizer']['lr'])
    loss_fn = eval(config['loss_fn'])()
    train_dataset = eval(config['dataset']['name'])(config['dataset'], True)
    test_dataset = eval(config['dataset']['name'])(config['dataset'], False)
