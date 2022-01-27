import torch
import numpy as np
import torch.nn.functional as F
class pool:
    def lse(x,r = 11):
        x = F.normalize(x,p=2,dim=1)
        x = torch.log(torch.mean(torch.exp(x * r) , 1)) / r
        return x