import torch.utils.data as Data
import os

import torchvision.transforms
from PIL import Image
import numpy as np
import torch
import cv2


import os
import numpy as np
import torch
from PIL import Image


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
            target = self.transforms(target)
            # img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class CarvanaData(Data.Dataset):
    def __init__(self, imgs_dir, masks_dir, transform):
        super(CarvanaData, self).__init__()
        self.imgs_path = imgs_dir
        self.imgs = list(sorted(os.listdir(self.imgs_path)))
        self.labels_path = masks_dir
        self.labels = list(sorted(os.listdir(self.labels_path)))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, item):
        img = Image.open(os.path.join(self.imgs_path, self.imgs[item])).convert("RGB")
        mask = Image.open(os.path.join(self.labels_path, self.labels[item])).convert("L")

        img = self.transform(img)
        mask = self.transform(mask)
        mask = torch.squeeze(mask)
        mask = mask.type(torch.LongTensor)

        return img, mask

import torchvision.transforms as T
transformer = T.Compose([
    T.ToTensor(),
])

class TGS_Salt(Data.Dataset):
    def __init__(self, root_dir, transform):
        super(TGS_Salt, self).__init__()
        self.imgs_path = os.path.join(root_dir, "images")
        self.imgs = list(sorted(os.listdir(self.imgs_path)))
        self.labels_path = os.path.join(root_dir, "masks")
        self.labels = list(sorted(os.listdir(self.labels_path)))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, item):
        img = Image.open(os.path.join(self.imgs_path, self.imgs[item])).convert("L")
        mask = Image.open(os.path.join(self.labels_path, self.labels[item])).convert("L")

        img = self.transform(img)
        mask = self.transform(mask)
        mask = torch.squeeze(mask)
        mask = mask.type(torch.LongTensor)

        return img, mask



class IccvData(Data.Dataset):
    def __init__(self, dir_img, dir_label, transform=None):
        super(IccvData, self).__init__()
        self.path_img = dir_img
        self.path_label = dir_label
        self.all_imgs = os.listdir(dir_img)
        self.transf = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.path_img, self.all_imgs[item]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        temp = cv2.imread(os.path.join(self.path_label, self.all_imgs[item][:-3] + 'png'), cv2.IMREAD_COLOR)
        tt = np.zeros([temp.shape[0], temp.shape[1]], dtype=np.long)
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if np.all(temp[i, j, :] == [203, 192, 255]):
                    tt[i, j] = 1

        img = self.transf(img)
        tt = self.transf(tt)
        return img, tt



class IccVDataTest(Data.Dataset):
    def __init__(self, dir_img, transform=None):
        super(IccVDataTest, self).__init__()
        self.path_img = dir_img
        self.all_imgs = os.listdir(dir_img)
        self.transf = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.path_img, self.all_imgs[item]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        img = self.transf(img)
        return img

np.set_printoptions(threshold=np.inf)

class PennSegmentData(Data.Dataset):
    def __init__(self, dir_img, dir_label, transform=None):
        super(PennSegmentData, self).__init__()
        self.path_img = dir_img
        self.path_label = dir_label
        self.all_imgs = os.listdir(dir_img)
        self.transf = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, item):
        # img = cv2.imread(os.path.join(self.path_img, self.all_imgs[item]), cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = Image.open(os.path.join(self.path_img, self.all_imgs[item])).convert("RGB")
        temp = cv2.imread(os.path.join(self.path_label, self.all_imgs[item][:-4] + '_mask.png'), cv2.IMREAD_GRAYSCALE)
        # print(os.path.join(self.path_label, self.all_imgs[item]))
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if temp[i][j] > 0:
                    temp[i][j] = 1
        # temp = np.array(cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY))

        # img = img.astype(np.float)

        img = self.transf(img)
        # tt = self.transf(temp)
        tt = torch.LongTensor(temp)
        return img, tt