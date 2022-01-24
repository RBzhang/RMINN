
import torch
import torch.nn.functional as F

class MI_net_DS(torch.nn.Module):
    def __init__(self , length) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(length , 256,bias=False)
        self.linear2 = torch.nn.Linear(256,128,bias=False)
        self.linear3 = torch.nn.Linear(128,64,bias=False)
        self.linear4 = torch.nn.Linear(64 , 1,bias=False)
        self.linear5 = torch.nn.Linear(256 , 1,bias=False)
        self.linear6 = torch.nn.Linear(128,1,bias=False)
    def forward(self , x):
        x = self.linear1(x)
        x = F.relu(x)
        x_1 = torch.max(x, 1)[0]
        x_1 = torch.sigmoid(self.linear5(x_1))
        x = self.linear2(x)
        x = F.relu(x)
        x_2 = torch.max(x, 1)[0] 
        x_2 = torch.sigmoid(self.linear6(x_2))
        x = self.linear3(x)
        x = torch.max(x, 1)[0]
        x = torch.sigmoid(self.linear4(x))
#        print(x.shape,x_1.shape,x_2.shape)
        x = torch.cat((x_1,x_2,x),1)
#        print(x.shape)
        x = torch.mean(x , 1)
        return x
        
# x = torch.rand((1,4,22))
# net = MI_net_DS(22)
# x = net(x)
# print(x)