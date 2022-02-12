#from pyparsing import TokenConverter
import torch
import torch.nn.functional as F
import torch.nn as nn
from MIL_pooling import pool
from MIL_pooling import attention
#import numpy
#from torch.utils.data import DataLoader
#import torch.optim as optim
#from loader import loader_musk

#dataset = loader_musk('clean2.data')
def max_pool(x):
#    x = x[0]
    x_data = torch.max(x[0] , 0)[0]
#    x = x.transpose(0,1)
    return x_data
def mean_pool(x):
    x_data = torch.mean(x[0] , 0)
    return x_data
class MI_Net(torch.nn.Module):
    def __init__(self,length) -> None:
        super(MI_Net,self).__init__()
        self.drob = 0.5
        self.layer1 = nn.Sequential(nn.Linear(length,256,bias=False), nn.ReLU(),nn.Dropout(self.drob))
        self.layer2 = nn.Sequential(nn.Linear(256,128,bias=False) , nn.ReLU(),nn.Dropout(self.drob))
        self.layer3 = nn.Sequential(nn.Linear(128,64,bias=False) ,  nn.ReLU(),nn.Dropout(self.drob))
        self.layer4 = nn.Sequential(nn.Linear(64,1,bias=False) , nn.Sigmoid())
        self.atten = attention(64 , 64)
#        self.relu = torch.nn.Relu()
    def forward(self , x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
#        print(x.shape)
#        x = max_pool(x)
#        x = torch.mean(x[0] , 0)
#        x = pool.lse(x , 1)[0]
#        print(self.atten.linear1.weight)
        x = self.atten(x,gate=False)[0]
        x = self.layer4(x)
        return x
#x = torch.rand((1,8,64))

#print(x)

#print(max_pool(x))