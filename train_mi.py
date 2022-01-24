from turtle import forward
from loader import loader_musk
#from MIL_pooling import pooling
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.optim as optim
# dataset = loader_musk('clean1.data')

class mi_Net(torch.nn.Module):
    def __init__(self,length):
        super().__init__()
        self.linear1 = torch.nn.Linear(length,256,bias=False)
        self.linear2 = torch.nn.Linear(256,128,bias=False)
        self.linear3 = torch.nn.Linear(128,64,bias=False)
        self.linear4 = torch.nn.Linear(64,1,bias=False)
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
        x = torch.max(x[0], 0)[0]
#        x = torch.mean(x[0] , 0)
#        x = pooling.mean_pooling(x)
#        x = pooling.lse_pooling(x,r=0.1)
        return x

class MI_Net(torch.nn.Module):
    def __init__(self , length) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(length,256,bias=False)
        self.linear2 = torch.nn.Linear(256,128,bias=False)
        self.linear3 = torch.nn.Linear(128,64,bias=False)
        self.linear4 = torch.nn.Linear(64,1,bias=False)
    def forward(self,x,r = 1):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        x = torch.max(x[0] , 0)[0]
#        x = pooling.mean_pooling(x)
#        x = pooling.lse_pooling(x,r=0.1)
        return x