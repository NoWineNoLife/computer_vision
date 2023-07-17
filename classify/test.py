import torch
import torch.nn as nn
import os
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
import torchvision.transforms
from classify.mobilenet import mobile
from classify.resnet import resnet
from classify.vggnet import vgg

transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
# train_path = '/home/hailongzhang/DataSets/train_test_barbecue/'
# test_path = '/home/hailongzhang/DataSets/barbecue_test/'
train_path = '/home/hailongzhang/DataSets/jones/kaggle/dogs_cats/train/'
# test_path = '/home/hailongzhang/DataSets/kaggle/dogs_cats/test/'
train = ImageFolder(train_path, transformer)
# test = ImageFolder(test_path, transformer)
train_num = int(len(train) * 0.8)
train_set, test_set = Data.random_split(dataset=train, lengths=[train_num, len(train) - train_num], generator=torch.Generator().manual_seed(0))
train_loader = Data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
test_loader = Data.DataLoader(test_set, batch_size=32, shuffle=True, num_workers=4)

# test_model = vgg.getVgg16(num_classes=2)
# test_model = mobile.MobileV2(num_classes = 2)
test_model = resnet.ResNet(2)

test_model = test_model.cuda()
test_model = nn.DataParallel(test_model)
test_model = test_model.cuda()
loss_f = nn.CrossEntropyLoss()
lr = 1e-4
EPOCH = 50
# opt = torch.optim.Adam(test_model.module.classifier[6].parameters(), lr=lr)
# opt = torch.optim.Adam(test_model.parameters(), lr=lr)
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, test_model.module.parameters()), lr=1e-4)
# scheduler = torch.optim.lr_scheduler(opt, 'min')
test_model.train()

for turn in range(EPOCH):
    for (train_x, train_y) in train_loader:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        out = test_model(train_x)
        loss = loss_f(out, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()


        with torch.no_grad():
            test_nums = 0
            test_correct = 0
            for (test_x, test_y) in test_loader:
                test_x = test_x.cuda()
                test_y = test_y.cuda()
                pred = test_model(test_x)
                test_nums += pred.size(0)
                pred = torch.max(pred, 1)[1]
                test_correct += (pred==test_y).sum()
            print('EPOCH: %d, loss: %.4f, acc: %.3f' % (turn, loss, 100 * test_correct / test_nums))
    # scheduler.step()
# torch.save(test_model.state_dict(), '/home/hailongzhang/Models/barbecue_param.pth')