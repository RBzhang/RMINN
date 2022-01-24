import torch
class pool:
    def lse(x,r = 1):
        x = torch.log(torch.mean(torch.exp(x * r) , 0)) / r
        return x
# x = torch.ones((1,3,15))
# net = pool.lse(x)
# print(net)
# print(torch.exp(x * 1))
