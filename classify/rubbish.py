import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import PIL
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
import torchvision.transforms
from classify.mobilenet import mobile
from classify.resnet import resnet
from classify.vggnet import vgg
from collections import OrderedDict
transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(512, 512)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
test_path = '/home/hailongzhang/DataSets/rubbish/ttt'
# test_path = '/home/hailongzhang/DataSets/barbecue_test/'
# train_path = '/home/hailongzhang/DataSets/jones/kaggle/dogs_cats/train/'
# test_path = '/home/hailongzhang/DataSets/kaggle/dogs_cats/test/'


class DataImgs(Data.Dataset):
    def __init__(self, root_dir, transformer):
        super(DataImgs, self).__init__()
        self.root = root_dir
        self.transfomer = transformer
        self.all_imgs = os.listdir(self.root)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, item):
        img = PIL.Image.open(os.path.join(self.root, self.all_imgs[item]))
        img = self.transfomer(img)
        return img, self.all_imgs[item]



test = DataImgs(test_path, transformer)
# test = ImageFolder(test_path, transformer)
test_loader = Data.DataLoader(test, batch_size=1)

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


test_model = resnet.ResNet(2)
test_model.load_state_dict(convert_state_dict(torch.load('/home/hailongzhang/Models/barbecue_param.pth')))
test_model = test_model.cuda()
loss_f = nn.CrossEntropyLoss()
lr = 1e-4
# opt = torch.optim.Adam(test_model.module.classifier[6].parameters(), lr=lr)
# opt = torch.optim.Adam(test_model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler(opt, 'min')
test_model.eval()

for (train_x, train_y) in test_loader:
    with torch.no_grad():
        train_x = train_x.cuda()
        file_name = train_y
        out = test_model(train_x)
        # pred = torch.zeros(size=(out.size(0), 1))
        # for i in range(out.size(0)):
        #     if out[i, 1].item() > 0.99:
        #         pred[i] = 1
        out = F.softmax(out)
        print(file_name)
        print(torch.max(out, 1)[1])


model = torchvision.models.googlenet(pretrained=False)
print(model)