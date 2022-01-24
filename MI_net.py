#from pyparsing import TokenConverter
import torch
import torch.nn.functional as F
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
        super().__init__()
        self.linear1 = torch.nn.Linear(length,256,bias=False)
        self.linear2 = torch.nn.Linear(256,128,bias=False)
        self.linear3 = torch.nn.Linear(128,64,bias=False)
        self.linear4 = torch.nn.Linear(64,1,bias=False)
    def forward(self , x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
#        print(x.shape)
        x = max_pool(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        return x
#x = torch.rand((1,8,64))

#print(x)

#print(max_pool(x))