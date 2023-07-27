import torch
import torch.nn as nn
import os
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter
# from efficientnet_pytorch import EfficientNet


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_path = '/home/yuki/DataSets/classify/dogs_cats/train'
train = ImageFolder(train_path, transformer_kaggle)

train_num = int(len(train) * 0.8)
train_set, test_set = Data.random_split(
    dataset=train,
    lengths=[train_num, len(train) - train_num],
    generator=torch.Generator().manual_seed(0))
train_loader = Data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8)
test_loader = Data.DataLoader(test_set, batch_size=16, shuffle=True, num_workers=8)


test_model = torchvision.models.resnet18(pretrained=2)

writer = SummaryWriter(log_dir='log/resnet34_256', comment='recorder')
test_model = nn.DataParallel(test_model)
test_model = test_model.cuda()
loss_f = nn.CrossEntropyLoss()
lr = 1e-3
EPOCH = 50

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

