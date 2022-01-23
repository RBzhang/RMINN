from turtle import forward
from loader import loader_musk
from MIL_pooling import pooling
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.optim as optim
# dataset = loader_musk('clean1.data')

class mi_Net(torch.nn.Module):
    def __init__(self,length):
        super().__init__()
        self.linear1 = torch.nn.Linear(length,256)
        self.linear2 = torch.nn.Linear(256,128)
        self.linear3 = torch.nn.Linear(128,64)
        self.linear4 = torch.nn.Linear(64,1)
    def forward(self,x,r = 1):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
#        print(x.shape)
        x = pooling.max_pooling(x)
#        x = pooling.mean_pooling(x)
#        x = pooling.lse_pooling(x,r=0.1)
        return x

class MI_Net(torch.nn.Module):
    def __init__(self , length) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(length,256)
        self.linear2 = torch.nn.Linear(256,128)
        self.linear3 = torch.nn.Linear(128,64)
        self.linear4 = torch.nn.Linear(64,1)
    def forward(self,x,r = 1):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        x = pooling.max_pooling(x)
#        x = pooling.mean_pooling(x)
#        x = pooling.lse_pooling(x,r=0.1)
        return x