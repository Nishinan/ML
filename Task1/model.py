import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class OneStepNet1(nn.Module):
    def __init__(self):
        super(OneStepNet1, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 2048)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(2048, 4096)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(4096, 10562)

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x


class OneStepNet(nn.Module):
    def __init__(self):
        super(OneStepNet, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 2048)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(2048, 10562)

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


class OneStepData(Dataset):  # 继承Dataset
    def __init__(self, x, y):  # __init__是初始化该类的一些基础参数
        self.x = x
        self.y = y

    def __len__(self):  # 返回整个数据集的大小
        return len(self.x)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        x = self.x[index]
        y = self.y[index]
        i = np.zeros(10562)
        i[self.y[index]] = 1
        return x, i, y
