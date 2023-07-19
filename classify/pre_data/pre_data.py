import os
from PIL import Image
import torch.utils.data as Data
import cv2
import torch


class GetData(Data.Dataset):

    def __init__(
        self,
        path,
    ):
        super(GetData, self).__init__()
        self.root = path
        self.all_files = os.listdir(self.root)

    def __getitem__(self, item):
        file_path = os.path.join(self.root, self.all_files[item])
        flag = 0
        img = Image.open(file_path)
        return img, flag

    def __len__(self):
        return len(self.all_files)



