import pickle

import torch
from sklearn.metrics import mean_absolute_error
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

DATA_REG_PATH = '../data/data_reg.pkl'

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
        return torch.Tensor(x[1:]), torch.Tensor([y/100,1-y/100])

class MyNet(nn.Module):
    def __init__(self,in_channels=3):
        super(MyNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=10, kernel_size=5),
            # nn.BatchNorm1d(10),
            # nn.ReLU(inplace=True)
            nn.Sigmoid(),
            # nn.MaxPool1d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=1, kernel_size=5),
            # nn.BatchNorm1d(1),
            # nn.ReLU(inplace=True)
            nn.Sigmoid(),
            # nn.MaxPool1d(2, 2)
        )

        self.fc=nn.Sequential(
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
        out = self.fc(out)
        return out

def train():
    nepoch=200
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
    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss(reduction='mean')

    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.99)
    # optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    best_mae_test=1e10
    for epoch in range(nepoch):
        for i, data in enumerate(train_loader):  # 0是下标起始位置默认为0
            net.train()
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            mae=mean_absolute_error(outputs.cpu().detach().numpy()[:,0],labels.cpu().detach().numpy()[:,0])

            net.eval()
            mae_test=0
            with torch.no_grad():
                for j, data_test in enumerate(test_loader):
                    inputs_test, labels_test = data_test[0].to(device), data_test[1].to(device)
                    outputs_test = net(inputs_test)
                    mae_test+=mean_absolute_error(outputs_test.cpu().detach().numpy()[:,0],labels_test.cpu().detach().numpy()[:,0])
            mae_test=mae_test/len(test_loader)
            if mae_test<best_mae_test:
                best_mae_test=mae_test
            print("%d/%d,%d/%d, train loss:%f, train_mae:%f, test_mae:%f, best_mae:%f"%(epoch, nepoch, i, len(train_loader), loss.item(), mae, mae_test, best_mae_test))
    print("best mae:",best_mae_test)
if __name__=="__main__":
    train()