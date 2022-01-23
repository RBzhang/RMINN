import torch
class pooling:
    def max_pooling(x):
        return x.max()
    def mean_pooling(x):
        return x.mean()
    def lse_pooling(x, r):
        return torch.log1p(torch.exp(r*x)).mean()/ r