import torch
from torch.utils.data import Dataset
#from torch.utils.data import DataLoader
import numpy as np
class loader_musk(Dataset):
    def __init__(self , filepath):
        dic = {}
        data = np.loadtxt(filepath,dtype=object,delimiter=',')
        rows , cols= data.shape
        print(rows, cols)
        data_loader = []
        count1 = 0
        count2 = 0
        leng = 0
        self.x_data = []
        for index in data:
            if index[0] not in dic.keys():
                count1 = count1 + count2
                dic[index[0]] =  [count1,0]
                count2 = 0
            dic[index[0]][1] += 1
            count2 += 1
        y = data[:,-1]
        name = data[:,0]
        self.y_data = np.zeros(len(dic))
        self.y_data = torch.from_numpy(self.y_data)
        print(self.y_data)
        data = data[:,2:-1]
        data_in = np.zeros((rows,cols-3),dtype=np.float32)
        for row in range(rows):
            for col in range(cols-3):
                data_in[row,col] = float(data[row,col])
#        data.astype(np.float32)
        for i in dic:
            self.x_data.append(torch.from_numpy(data_in[dic[i][0]:dic[i][0]+dic[i][1],:]))
            for j in range(dic[i][0],dic[i][0]+dic[i][1]):
                if y[j][0] == '1':
                    self.y_data[leng] = 1
            leng += 1
        print(self.x_data[0])
        print(self.x_data[1][3])
        print(self.y_data)
    def __getitem__(self, index):
        return self.x_data[index] , self.y_data[index]
    def __len__(self):
        return self.len
dataset = loader_musk('clean1.data')