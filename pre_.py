import torch
import scipy.io as io

data = io.loadmat('fox.mat')
print(data.keys())
print(data['data'])