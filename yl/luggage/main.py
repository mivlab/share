import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
import argparse
import os
from torch.utils.data import DataLoader

from dataloader import mnist_loader as ml
from models.cnn import Net


parser = argparse.ArgumentParser(description='PyTorch MNIST Example') #创建解析器
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--batch_size', type=int, default=96, help='training batch size')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--no_cuda', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)') #增加命令行

args = parser.parse_args() #执行命令行

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed) #设置种子用于生成随机数
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True  #增加程序的运行效率
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def image_list_id(imageRoot, txt='list.txt'):
    f = open(txt, 'wt')
    for (label, filename) in enumerate(sorted(os.listdir(imageRoot), reverse=False)):
        if os.path.isdir(os.path.join(imageRoot, filename)):
            for imagename in os.listdir(os.path.join(imageRoot, filename)):
                name, ext = os.path.splitext(imagename)
                ext = ext[1:]
                if ext == 'jpg' or ext == 'png' or ext == 'bmp':
                    f.write('%s %d\n' % (os.path.join(imageRoot, filename, imagename), int(filename)))
    f.close()

def val(model, val_loader, batch_size, batch_num):
    model.eval()
    eval_acc = 0
    count = 0
    for batch_x, batch_y in val_loader:
        if args.cuda:
            batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
        else:
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
        count += batch_size
        if count >= batch_size * batch_num:
            break
    print('Val Acc: %.3f' % (eval_acc / count))
    model.train()

def train():
    os.makedirs('./output', exist_ok=True) #用于递归创建目录
    #image_list_id(r'D:\data\xb-reid\MVB_train\train_224_s', 'output/train.txt')
    #image_list_id(r'D:\data\xb-reid\MVB_train\val_224_s', 'output/val.txt')
    val_batch_size = args.batch_size // 4
    train_data = ml.MyDataset(txt='output/train.txt', transform=transforms.ToTensor())
    val_data = ml.MyDataset(txt='output/val.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=val_batch_size)

    pretrain = False

    if pretrain:
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(512, 3919)
    else:
        #model = Net()
        model = models.resnet18(num_classes=3919)  # 调用内置模型
        model.load_state_dict(torch.load('./output/params_1.pth'))

    if args.cuda:
        print('training with cuda')
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005) #优化器
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1) #学习效率
    loss_func = nn.CrossEntropyLoss() #交叉熵损失函数

    for epoch in range(args.epochs):
        # training-----------------------------------
        model.train()
        train_loss = 0
        train_acc = 0
        batch = 0
        for batch_x, batch_y in train_loader:
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)  # 256x3x28x28  out 256x10
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item() #将tensor转换为数值
            batch += 1
            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f, Acc: %.3f'
                  % (epoch + 1, args.epochs, batch, math.ceil(len(train_data) / args.batch_size),
                     loss.item(), train_correct.item() / len(batch_x)))
            if batch % 50 == 0:
                val(model, val_loader, val_batch_size, 50)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # 更新learning rate
        print('Train Loss: %.6f, Acc: %.3f' % (train_loss / (math.ceil(len(train_data)/args.batch_size)),
                                               train_acc / (len(train_data))))

        # save model --------------------------------
        torch.save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for batch_x, batch_y in val_loader:
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Val Loss: %.6f, Acc: %.3f' % (eval_loss / (math.ceil(len(val_data)/val_batch_size)),
                                             eval_acc / (len(val_data))))
if __name__ == '__main__':
    train()