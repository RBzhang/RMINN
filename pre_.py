import torch
import scipy.io as io
from torch.utils.data import Dataset
import random
class loader_image(Dataset):
    def __init__(self, filepath) -> None:
        super().__init__()
        data = io.loadmat(filepath)
        x = data['data']
        self.x_data = []
        self.y_data = torch.zeros(x.shape[0],dtype=torch.float32)
        num = list(range(x.shape[0]))
        random.shuffle(num)
        j = 0
        for i in num:
            self.y_data[j] = (float(x[i][1][0]) + 1.) / 2
            self.x_data.append(torch.Tensor(x[i][0])[:,0:-1].clone())
            j += 1
#        self.x_data = torch.Tensor(self.x_data)
#        print(self.x_data[0].shape)
        self.len = x.shape[0]
        self.leng = self.x_data[0].shape[1]
#        print(self.y_data.shape)
    def __getitem__(self, index):
        return self.x_data[index] , self.y_data[index]
    def __len__(self):
        return self.len
    def __length__(self):
        return self.leng
#dataset = loader_image('fox.mat')