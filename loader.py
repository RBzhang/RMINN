import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
#from torch.utils.data import DataLoader
import numpy as np
import random
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
        name = []
        for index in data:
            if index[0] not in dic.keys():
                count1 = count1 + count2
                dic[index[0]] =  [count1,0]
                count2 = 0
                name.append(index[0])
            dic[index[0]][1] += 1
            count2 += 1
        y = data[:,-1]
#        print(dic.keys())
#        print(y)
#        name = data[:,0]
#        print(name)
#        self.y_data = np.zeros(len(dic))
        self.y_data = torch.zeros(len(dic),dtype=torch.float32)
#        print(self.y_data)
        data = data[:,2:-1]
        data_in = torch.zeros((rows,cols-3),dtype=torch.float32)
        num = list(range(len(dic)))
#        print(num)
#        name = dic.keys()
        random.shuffle(num)
#        print(num)
        for row in range(rows):
            for col in range(cols-3):
                data_in[row,col] = float(data[row,col])
        for i in range(cols-3):
            data_in[:,i] = (data_in[:,i]-data_in.min())/(data_in[:,i].max()-data_in.min())
#        print(data_in[0])
#        data.astype(np.float32)
        for i in num:
#            print(i)
#            print(dic[name[i]])
            self.x_data.append(data_in[dic[name[i]][0]:dic[name[i]][0]+dic[name[i]][1],:].clone())  #加载数据并随机化
            for j in range(dic[name[i]][0],dic[name[i]][0]+dic[name[i]][1]):
                if y[j][0] == '1':
                    self.y_data[leng] = 1
            leng += 1
        self.len = leng
#        print(leng)
        self.lengtht = cols-3
#        print(self.y_data)
 #       F.normalize(self.x_data,dim=1)
#        print(self.x_data[0].shape)
#        print(self.x_data[7])
        # print(self.y_data)
        
    def __getitem__(self, index):
        return self.x_data[index] , self.y_data[index]
    def __len__(self):
        return self.len
    def __length__(self):
        return self.lengtht
# dataset = loader_musk('clean1.data')