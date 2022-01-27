import torch
import torch.nn.functional as F

from MI_net_DS import MI_net_DS
from MIL_pooling import pool
class MI_net_Res(torch.nn.Module):
    def __init__(self,length) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(length, 128,bias=False)
        self.linear2 = torch.nn.Linear(128 , 128,bias=False)
        self.linear3 = torch.nn.Linear(128,128,bias=False)
        self.linear4 = torch.nn.Linear(128,1,bias=False)
        self.Sigmoid = torch.nn.Sigmoid()
    def forward(self , x):
#        print(x.shape)
        x = self.linear1(x)
        x = F.relu(x)
#        x = F.normalize(x,p=2,dim=1)
        x_1 = torch.max(x , 1)[0]
#        x_1 = torch.mean(x , 1)
#        x_1 = pool.lse(x,1)
        x = self.linear2(x)
        x = F.relu(x)
#        x = F.normalize(x,p=2,dim=1)
        x_1 = torch.max(x , 1)[0] + x_1
#        x_1 = torch.mean(x , 1) + x_1
#        x_1 = pool.lse(x_1,1) + x_1
        x = self.linear3(x)
        x = F.relu(x)
#        x = F.normalize(x,p=2,dim=1)
        x = torch.max(x , 1)[0] + x_1
#        x = torch.mean(x , 1) + x_1
#        x = pool.lse(x,1) + x_1
        x = self.linear4(x)
        x = self.Sigmoid(x[0])
#        print(x.shape)
        return x
# x = torch.rand((1,4,22))

# net  = MI_net_DS(22)

# x = net(x)
# print(x.shape)