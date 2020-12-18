import pickle
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt

DATA_REG_PATH = '../data/data_joint.pkl'

class MyDataset(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        self.transform = transform
        trainX, trainY, testX, testY = pickle.load(open(data_path, 'rb'))
        if mode=='train':
            self.X=trainX
            self.Y=trainY
        elif mode=='test':
            self.X=testX
            self.Y=testY
        else:
            print('wrong mode!')
            exit(-1)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x=self.X[idx]
        y=self.Y[idx]
        # onehot=np.zeros(2)
        # onehot[int(y[1])]=1
        return torch.Tensor(x[1:]), torch.Tensor([y[0]/100,1-y[0]/100,y[1]])


class JonitLoss(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target):
        # CrossEntropyLoss+alpha*l*L1Loss
        celoss = F.cross_entropy(output[1], target[:,2].long(), reduction='sum')
        _, predicted = torch.max(output[1], 1)
        out_reg=output[0]
        sign=(1-target[:,2])
        out_reg=torch.mul(out_reg,sign.reshape((sign.shape[0],1)))
        tar_reg = torch.mul(target[:,:2], sign.reshape((sign.shape[0], 1)))
        l1loss = F.l1_loss(out_reg, tar_reg, reduction='sum')
        return celoss + self.alpha*l1loss

        # return celoss




class MyNet(nn.Module):
    def __init__(self,in_channels=3):
        super(MyNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=5),
            # nn.BatchNorm1d(1),
            nn.ReLU(inplace=True)
            # nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=1, kernel_size=5),
            # nn.BatchNorm1d(1),
            nn.ReLU(inplace=True)
            # nn.Sigmoid()
        )

        self.clf=nn.Sequential(
            nn.Linear(220,20),
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Linear(20, 2),
            # nn.ReLU(inplace=True),
            # nn.Softmax(dim=1)
        )
        self.reg = nn.Sequential(
            nn.Linear(220,20),
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Linear(20, 2),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out_clf = self.clf(out)
        out_reg = self.reg(out)
        return out_reg, out_clf

def train():
    nepoch=1000
    batch_size=32

    train_data = MyDataset(DATA_REG_PATH)
    test_data = MyDataset(DATA_REG_PATH, mode='test')
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )


    net=MyNet()
    # criterion = nn.L1Loss()
    # criterion = nn.MSELoss(reduction='mean')
    # criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion = JonitLoss()

    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.99)
    # optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    best_acc=0
    best_mae = 1e10
    train_loss_plt=[]
    test_loss_plt=[]
    train_acc_plt=[]
    test_acc_plt=[]
    train_mae_plt=[]
    test_mae_plt=[]

    for epoch in range(nepoch):
        correct = 0
        total=0
        loss_sum=0
        mae=0
        for i, data in enumerate(train_loader):  # 0是下标起始位置默认为0
            net.train()
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels.squeeze()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs[1].data, 1)
            total += labels.size(0)
            correct += (predicted == labels[:,2]).sum().item()
            loss_sum+=loss.item()

            local_mae = mean_absolute_error(torch.mul(outputs[0][:, 0].cpu().detach(),1-labels[:,2].cpu().detach()),
                                       torch.mul(labels[:, 0].cpu().detach(),1-labels[:,2].cpu().detach()))
            mae += len(labels) * local_mae

        net.eval()
        correct_test = 0
        total_test = 0
        loss_sum_test=0
        mae_test=0

        with torch.no_grad():
            for j, data_test in enumerate(test_loader):
                inputs_test, labels_test = data_test[0].to(device), data_test[1].to(device)
                labels_test = labels_test.squeeze()
                outputs_test = net(inputs_test)
                _, predicted = torch.max(outputs_test[1].data, 1)
                total_test += labels_test.size(0)
                correct_test += (predicted == labels_test[:,2]).sum().item()
                loss_sum_test+=criterion(outputs_test, labels_test).item()
                local_mae = mean_absolute_error(torch.mul(outputs_test[0][:,0].cpu().detach(),1-labels_test[:,2].cpu().detach()),
                                                torch.mul(labels_test[:,0].cpu().detach(),1-labels_test[:,2].cpu().detach()))
                mae_test += len(labels_test) * local_mae
        if correct_test/total_test>=best_acc:
            best_acc=correct_test/total_test
            if mae_test/total_test<=best_mae:
                best_mae=mae_test/total_test
                torch.save(net, 'weights/joint.pth')
                print('model saved, %f,%f'%(best_acc, best_mae))
        print("Epoch: %d/%d, train loss:%f, test loss:%f, train_acc=%d/%d=%f, test_acc=%d/%d=%f, best_acc=%f, train_mae:%f, test_mae:%f"%\
              (epoch, nepoch, loss_sum/total, loss_sum_test/total_test, correct, total, correct/total, correct_test, total_test, correct_test/total_test, best_acc, mae/total, mae_test/total_test))
        train_loss_plt.append(loss_sum/total)
        test_loss_plt.append(loss_sum_test/total_test)
        train_acc_plt.append(correct/total)
        test_acc_plt.append(correct_test/total_test)
        train_mae_plt.append(mae/len(train_loader))
        test_mae_plt.append(mae_test/len(test_loader))

    plt.plot(np.arange(len(train_loss_plt)),train_loss_plt, label='train_loss')
    plt.legend()
    plt.savefig('../expr/train_loss.png', dpi=300)
    plt.savefig('../expr/train_loss.eps', dpi=300)
    plt.show()
    plt.plot(np.arange(len(test_loss_plt)), test_loss_plt, label='test_loss')
    plt.legend()
    plt.savefig('../expr/test_loss.png',dpi=300)
    plt.savefig('../expr/test_loss.eps', dpi=300)
    plt.show()
    plt.plot(np.arange(len(train_acc_plt)), train_acc_plt, label='train_acc')
    plt.plot(np.arange(len(test_acc_plt)), test_acc_plt, label='test_acc')
    plt.legend()
    plt.savefig('../expr/acc.png', dpi=300)
    plt.savefig('../expr/acc.eps', dpi=300)
    plt.show()
    plt.plot(np.arange(len(train_mae_plt)), train_mae_plt, label='train_mae')
    plt.plot(np.arange(len(test_mae_plt)), test_mae_plt, label='test_mae')
    plt.legend()
    plt.savefig('../expr/mae.png', dpi=300)
    plt.savefig('../expr/mae.eps', dpi=300)
    plt.show()
    print('finish, %f,%f' % (best_acc, best_mae))

def test():
    trainX, trainY, testX, testY = pickle.load(open(DATA_REG_PATH, 'rb'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net=torch.load('weights/joint.pth').to(device)
    outputs = net(torch.Tensor(testX[:, 1:]).to(device))
    _, predicted = torch.max(outputs[1].data, 1)
    correct = (predicted == torch.Tensor(testY[:, 1]).long().to(device)).sum().item()
    mae = mean_absolute_error(torch.mul(outputs[0][:, 0].cpu().detach(),torch.Tensor(1-testY[:, 1])),torch.mul(torch.Tensor(testY[:, 0])/100,torch.Tensor(1-testY[:, 1])))
    print(correct/len(testY), mae)

if __name__=="__main__":
    train()
    test()