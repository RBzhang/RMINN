from turtle import forward
import torch
import numpy as np
import torch.nn.functional as F
class pool:
    def lse(x,r = 11):
        x = F.normalize(x,p=2,dim=1)
        x = torch.log(torch.mean(torch.exp(x * r) , 1)) / r
        return x
class attention(torch.nn.Module): # 论文[2]中的注意力机制
    def __init__(self, length , L) -> None:
        super(attention,self).__init__()
        self.linear1 = torch.nn.Linear(L,1 , bias = False)
        self.linearV = torch.nn.Linear(length,L, bias = False)
        self.linearU = torch.nn.Linear(length,L, bias = False)
        self.softmax = torch.nn.Softmax(dim=0)
    def forward(self, x , gate = True):
        # print(x)
        x = x[0]
        if gate:
            y = torch.sigmoid(self.linearV(x)) * torch.tanh(self.linearU(x))
#            print(y)
        else:
            y = torch.tanh(self.linearU(x))
        y = self.linear1(y)
        y = self.softmax(y)
        y = y.T
#        print(self.linear1.weight)
        x = torch.mm(y , x)
        return x
# a = attention(length=4 , L = 5)
# x = torch.rand(2,3,4)
# print(x)
# print(a(x))