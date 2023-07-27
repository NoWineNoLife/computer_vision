#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : tsing
# @time    : 2023/7/27 下午3:58
# @function: the script is used to do something.
import torchvision.transforms as T

def get_properties(a_cfg):
    imgs_path = a_cfg.get('imgs_path')
    map_file_path = a_cfg.get('map_file')
    with open(map_file_path, 'r') as f:
        lines = f.read().strip()
    lines_list = lines.splitlines()
    filename_list = []
    labels = []
    for i in lines_list:
        filename, label = i.split(',')
        filename_list.append(filename)
        labels.append(label)
    return imgs_path, filename_list, labels


def get_transforms(a_cfg):
    transf = T.Compose(
        T.ToTensor()
    )
    return transf

if __name__ == '__main__':
    cfg = {
        'imgs_path': '/',
        'map_file': '/home/hozumi/rubbish/1'
    }

    path, imgs, label = get_properties(cfg)

    print(111)
    print(22)