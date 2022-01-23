import torch
import torch.nn.functional as F

from MI_net_DS import MI_net_DS

class MI_net_Res(torch.nn.Module):
    def __init__(self,length) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(length, 128)
        self.linear2 = torch.nn.Linear(128 , 128)
        self.linear3 = torch.nn.Linear(128,1)
#        self.Sigmoid = torch.nn.Sigmoid(128)
    def forward(self , x):
        x = self.linear1(x)
        x = F.relu(x)
        x_1 = torch.mean(x , 1)
        x = self.linear2(x)
        x = F.relu(x)
        x_1 = torch.mean(x , 1) + x_1
        x = self.linear2(x)
        x = torch.mean(x , 1) + x_1
        x = self.linear3(x)
        x = torch.sigmoid(x[0])
        return x
# x = torch.rand((1,4,22))

# net  = MI_net_DS(22)

# x = net(x)
# print(x.shape)