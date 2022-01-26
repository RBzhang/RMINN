import torch
import numpy as np
class pool:
    def lse(x,r = 11):
        x = torch.log(torch.mean(torch.exp(x * r) , 1)) / r
        return x
#x = torch.ones((1,3,15))
# x = np.array([[0.7,0.4,0.8],[0.9,0.99,0.87],[0.66,0.78,0.54],[0.54,0.66,0.77]])
# x = torch.from_numpy(x)
# net = pool.lse(x)
# print(net)
# print(torch.exp(x * 1))
