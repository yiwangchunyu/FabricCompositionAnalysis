import pickle
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch import nn, optim
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

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
        return torch.Tensor(x[1:]), torch.Tensor(y).long()


class JonitLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        # CrossEntropyLoss+alpha*l*L1Loss
        celoss = F.cross_entropy(output[1], target[:,1])
        # _, predicted = torch.max(output[1], 1)
        # out,tar=output[0][predicted==0], target[0][predicted==0]
        return celoss




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
    nepoch=300
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
    for epoch in range(nepoch):
        correct = 0
        total=0
        loss_sum=0
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
            correct += (predicted == labels[:,1]).sum().item()
            loss_sum+=loss.item()

        net.eval()
        correct_test = 0
        total_test = 0
        loss_sum_test=0
        with torch.no_grad():
            for j, data_test in enumerate(test_loader):
                inputs_test, labels_test = data_test[0].to(device), data_test[1].to(device)
                labels_test = labels_test.squeeze()
                outputs_test = net(inputs_test)
                _, predicted = torch.max(outputs_test[1].data, 1)
                total_test += labels_test.size(0)
                correct_test += (predicted == labels_test[:,1]).sum().item()
                loss_sum_test+=criterion(outputs_test, labels_test).item()
        if correct_test/total_test>best_acc:
            best_acc=correct_test/total_test

        print("Epoch: %d/%d, train loss:%f, train_acc=%d/%d=%f, test loss:%f, test_acc=%d/%d=%f, best_acc=%f"%\
              (epoch, nepoch, loss_sum/total, correct, total, correct/total, loss_sum_test/total_test, correct_test, total_test, correct_test/total_test, best_acc))

if __name__=="__main__":
    train()