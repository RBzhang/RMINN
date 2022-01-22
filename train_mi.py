from turtle import forward
from loader import loader_musk
from MIL_pooling import pooling
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.optim as optim
dataset = loader_musk('clean1.data')

class mi_Net(torch.nn.Module):
    def __init__(self,length):
        super().__init__()
        self.linear1 = torch.nn.Linear(length,256)
        self.linear2 = torch.nn.Linear(256,128)
        self.linear3 = torch.nn.Linear(128,64)
        self.linear4 = torch.nn.Linear(64,1)
    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.sigmoid(x)
        x = pooling.max_pooling(x)
        return x
model = mi_Net(dataset.__length__())
criterion = torch.nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01 , momentum = 0.5)

train_data = DataLoader(dataset=dataset,batch_size=1,shuffle=True)
def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_data,0):
        print("NMB")
        inputs , y_pred = data
        print(y_pred)
        optimizer.zero_grad()

        outputs = model(inputs)
        print(outputs)
        loss  = criterion(outputs,y_pred[0])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print('[%d, %5d] loss: %0.3f' % (epoch + 1 , batch_idx + 1 , running_loss / 3))
        running_loss = 0.0            
for epoch in range(100):
    train(epoch)