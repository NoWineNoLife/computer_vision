import torch
import torch.nn as nn
import os
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
import torchvision.transforms
from mobilenet import mobile
from resnet import resnet
from vggnet import vgg
from pre_data.pre_data import HandWrittenData
from lenet.lenet import *
from torch.utils.tensorboard import SummaryWriter
# from efficientnet_pytorch import EfficientNet

#tobacco
transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(512, 512)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

#tobacco_hainan
transformer_hainan = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(256, 256)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

#kaggle
transformer_kaggle = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

#hand_written_digits
transformer_hand_written_digits = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    # torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
])

os.environ["CUDA_VISIBLE_DEVI"
           "CES"] = "0"
# train_path = '/home/hailongzhang/DataSets/train_test_barbecue/'
train_path = '/home/yuki/DataSets/classify/dogs_cats/train'
# train_path = '/home/hailongzhang/DataSets/hand_written_digits/all_imgs'
# train_path = '/home/hailongzhang/DataSets/tobacco/hainan/C/'
# test_path = '/home/hailongzhang/DataSets/kaggle/dogs_cats/test/'
# train = ImageFolder(train_path, transformer_hainan)
train = ImageFolder(train_path, transformer_kaggle)
# train = HandWrittenData(train_path, transform=transformer_hand_written_digits)
train_num = int(len(train) * 0.8)
train_set, test_set = Data.random_split(
    dataset=train,
    lengths=[train_num, len(train) - train_num],
    generator=torch.Generator().manual_seed(0))
train_loader = Data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8)
test_loader = Data.DataLoader(test_set, batch_size=16, shuffle=True, num_workers=8)

#customed
# test_model = Lenet(num_classes=10)
# test_model = Alexnet(num_classes=10)
# test_model = Vggnet(num_classes=2, pretained=True)

# test_model = vgg.getVgg16(num_classes=2)
# test_model.load_state_dict(torch.load('/home/hailongzhang/Models/hainan/T_classifier.pth'))

# test_model = mobile.MobileV2(num_classes = 2)
test_model = torchvision.models.resnet18(pretrained=2)
# test_model = EfficientNet.from_name('efficientnet-b0')
fc_layer = nn.Linear(in_features=1280, out_features=2)
# test_model._fc = []
test_model._fc = fc_layer
print(test_model)

#pretained
# test_model = torchvision.models.googlenet()
# test_model = torchvision.models.vgg16_bn(pretrained=True, progress=True, num_classes=2)
# test_model = torchvision.models.AlexNet(num_classes=10)
# test_model = torchvision.models.alexnet(pretrained=True)

writer = SummaryWriter(log_dir='log/resnet34_256', comment='recorder')
test_model = nn.DataParallel(test_model)
test_model = test_model.cuda()
loss_f = nn.CrossEntropyLoss()
lr = 1e-3
EPOCH = 50

# torch.nn.init.kaiming_uniform_()
# opt = torch.optim.Adam(test_model.module.classifier[6].parameters(), lr=lr)
# opt = torch.optim.SGD(test_model.module.parameters(), lr=lr, momentum=0.9)
opt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                              test_model.module.parameters()),
                       lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
test_model.train()

loss = 0.0
acc = 0.0

for turn in range(EPOCH):
    for (train_x, train_y) in train_loader:

        train_y = train_y.cuda()
        out = test_model(train_x)
        loss = loss_f(out, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            test_model.eval()
            test_nums = 0
            test_correct = 0
            for (test_x, test_y) in test_loader:
                test_x = test_x.cuda()
                test_y = test_y.cuda()
                pred = test_model(test_x)
                test_nums += pred.size(0)
                pred = torch.max(pred, 1)[1]
                test_correct += (pred == test_y).sum()
            acc = test_correct / test_nums
            print('EPOCH: %d, loss: %.4f, acc: %.3f' % (turn, loss, acc))
            if acc > 0.5 and turn > 10:
                torch.save(test_model.module.state_dict(),
                           '/home/hailongzhang/Models/c_classifier.pth')
        test_model.train()
    writer.add_scalar('loss', loss, turn)
    writer.add_scalar('acc', acc, turn)
    scheduler.step()

