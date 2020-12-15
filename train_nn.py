import argparse
import os
import time
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchsummary import summary

from dataset import TRAIN_CLF_DIR, TEST_CLF_DIR, MyDataset
from net import OneCNN, FCN


def print2logfile(s,end='\n'):
    print(s)
    with open(logfile,'a') as f:
        f.write(s.__str__())
        f.write(end)

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5],std=[0.5])
    ])
    input_shape=(228,1)
    # net=OneCNN()
    net = FCN()
    train_data = MyDataset(
        data_dir = TRAIN_CLF_DIR,
        transform=transform
    )

    test_data = MyDataset(
        data_dir=TEST_CLF_DIR,
        transform = transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum)
    #也可以选择Adam优化方法
    # optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    train_accs = []
    train_loss = []
    test_accs = []
    test_loss = []
    test_id = []
    best_acc = 0
    print2logfile(net)
    # print2logfile(summary(net, (1, input_shape[0], input_shape[1])))
    for epoch in range(opt.nepoch):
        running_loss=0
        for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
            net.train()
            # data 的格式[[inputs, labels]]
            #         inputs,labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            # 初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()

            # 前向+后向+优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # loss 的输出，每个一百个batch输出，平均的loss
            running_loss += loss.item()
            train_loss.append(loss.item())
            _, preds = torch.max(outputs.data, 1)
            total = labels.size(0)  # labels 的长度
            correct = (preds == labels).sum().item()  # 预测正确的数目
            train_acc = correct / total
            train_accs.append(train_acc)

            if (i+1) % opt.display_interval == 0:
                print2logfile('[%2d/%d,%5d/%d] train_loss:%.4f, train_acc:%.4f' %
                      (epoch+1, opt.nepoch, i+1, len(train_loader), running_loss / opt.display_interval, train_acc))
                running_loss=0

            if (i + 1) % opt.test_interval == 0:
                net.eval()
                test_correct = 0
                test_total = 0
                with torch.no_grad():  # 进行评测的时候网络不更新梯度
                    for test_data in test_loader:
                        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
                        test_outputs = net(test_images)
                        loss = criterion(test_outputs, test_labels)
                        _, test_preds = torch.max(test_outputs.data, 1)
                        test_total += labels.size(0)  # labels 的长度
                        test_correct += (test_preds == test_labels).sum().item()  # 预测正确的数目
                    test_acc=test_correct/test_total
                    test_accs.append(test_acc)
                    test_loss.append(loss.item())
                    test_id.append(epoch*len(train_loader)+i)
                    print2logfile('[%2d/%d,%5d/%d]                                  test_acc:%.4f %s' %
                          (epoch + 1, opt.nepoch, i + 1, len(train_loader), test_acc, 'best_acc update, model saved.' if test_acc>best_acc else ''))
                    if test_acc>best_acc:
                        best_acc=test_acc
                        # save model
                        path = os.path.join(opt.model_save_dir, opt.model_name)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        torch.save(net.state_dict(), os.path.join(path, 'net_%s.pth'%(timestr)))

    print2logfile('Finished Training, best accuracy:%f'%best_acc)
    #画图
    plt.figure(figsize=(20,6))
    plt.subplot(1,2,1)
    plt.plot(range(len(train_loss)), train_loss, label='train_loss')
    plt.plot(test_id, test_loss, label='test_loss',linewidth=2)
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accs)), train_accs, label='train_acc')
    plt.plot(test_id, test_accs, label='test_acc',linewidth=2)
    plt.xlabel('Iter')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(opt.expr, opt.model_name,'loss_acc_%s_%f.png'%(timestr,best_acc)),dpi=300)
    plt.savefig(os.path.join(opt.expr, opt.model_name, 'loss_acc_%s_%f.eps' % (timestr, best_acc)), dpi=300)
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default='train', help="is train or test")
    parser.add_argument("--nepoch", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="SGD: learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD: momentum")
    parser.add_argument("--img_size", type=tuple, default=(28,28), help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--predictImg", type=str, default='', help="image need to be predicted")
    parser.add_argument("--display_interval", type=int, default=10, help="")
    parser.add_argument("--test_interval", type=int, default=10, help="")
    parser.add_argument("--model_save_dir", type=str, default='weights', help="")
    parser.add_argument("--expr", type=str, default='expr', help="")
    parser.add_argument("--model_name", type=str, default='OneCNN', help="")
    parser.add_argument("--data_dir", type=str, default='data', help="")
    opt = parser.parse_args()

    timestr = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    path = os.path.join(opt.model_save_dir, opt.model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(opt.expr, opt.model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    logfile=os.path.join(path,'log_%s.txt'%timestr)
    if os.path.exists(logfile):
        os.remove(logfile)
    print2logfile(opt)

    train()