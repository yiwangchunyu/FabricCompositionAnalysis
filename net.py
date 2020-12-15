from torch import nn
from torch.autograd.grad_mode import F


class OneCNN(nn.Module):
    def __init__(self):
        super(OneCNN, self).__init__()    # super用法:Cnn继承父类nn.Model的属性，并用父类的方法初始化这些属性
        self.conv = self.make_layers([8,'M',16,'M',32,32,'M',64,64,'M',64, 64,'M'])

        self.conv1 = nn.Sequential(
            nn.Linear(7*64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def make_layers(self,cfg):
        layers=[]
        input_channel=1
        for l in cfg:
            if l=='M':
                layers+=[nn.MaxPool1d(kernel_size=2,stride=2)]
                continue
            layers+=[nn.Conv1d(input_channel, l, kernel_size=3,padding=1)]
            layers+=[nn.ReLU(inplace=True)]
            input_channel=l
        return nn.Sequential(*layers)

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()    # super用法:Cnn继承父类nn.Model的属性，并用父类的方法初始化这些属性
        self.fc = nn.Sequential(
            nn.Linear(224, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out
