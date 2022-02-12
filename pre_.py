import torch
import torch.nn.functional as F
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
#        print(x.shape)
        random.shuffle(num)
        j = 0
        sum = 0
        for i in num:
            self.y_data[j] = (float(x[i][1][0]) + 1) / 2
            self.x_data.append(torch.Tensor(x[i][0])[:,0:-1].clone())
            sum += len(x[i][0])
            j += 1
        # print(sum)
        # print(self.x_data[0].shape[1])
        mid = torch.zeros(sum,self.x_data[0].shape[1])
        ma = max(self.y_data)
        mi = min(self.y_data)
        self.y_data = (self.y_data - mi) /(ma - mi) 
#        self.x_data = torch.Tensor(self.x_data)
        
        # print(self.y_data[0])
        self.len = x.shape[0]

        self.leng = self.x_data[0].shape[1]
        sum = 0
        for index in range(self.len):
            for j in range(self.x_data[index].shape[0]):
                mid[sum] = self.x_data[index][j].clone()
                sum += 1
#        print(mid[5000])
#        x_mean = torch.mean(mid,dim=0)
        x_max = torch.max(mid,dim=0)[0]
        x_min = torch.min(mid,dim=0)[0]
        for i in range(x_max.shape[0]):
            if x_max[i] < -x_min[i]:
                x_max[i] = - x_min[i]
            if x_max[i] < 0.01:
                x_max[i] = 1.
        # print(x_mean)
        # print(x_max)
        # print(x_min)
        for index in range(self.len):
            self.x_data[index] = self.x_data[index] / x_max
        #     self.x_data[index] = F.normalize(self.x_data[index],p=2,dim=0)
#        print(self.x_data[0])
        # std_ = torch.std(self.x_data[0],dim=0)
#        print(self.x_data[0].shape)
        # for index in self.x_data:
        #     std_ = torch.std(index, dim=0)

#        print(self.y_data)
    def __getitem__(self, index):
        return self.x_data[index] , self.y_data[index]
    def __len__(self):
        return self.len
    def __length__(self):
        return self.leng
#dataset = loader_image('fox.mat')