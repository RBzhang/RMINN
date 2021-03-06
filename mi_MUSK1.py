import torch.optim as optim
from torch.utils.data import DataLoader
from loader import loader_musk
import torch
from train_mi import mi_Net
from MI_net import MI_Net
from MI_net_DS import MI_net_DS
from MI_Res import MI_net_Res
from pre_ import loader_image
# 修改此处代码改变数据集
#dataset = loader_musk('clean1.data')   #MUSK1 dataset
dataset = loader_musk('clean2.data')    #MUSK2 dataset
#dataset = loader_image('fox.mat')
#dataset = loader_image('elephant.mat')
#dataset = loader_image('tiger.mat')
criterion = torch.nn.BCELoss()


train_data = DataLoader(dataset=dataset,batch_size=1,shuffle=True)
def test_1(epoch,model):
    test = list(enumerate(train_data,0))
    loss = 0
    leng = len(test)
    test_size =  leng - (int(leng * 0.7) - (int(leng * 0.7)) %10)
    test_begin = (epoch*10)%(leng - leng % 10) + int(leng * 0.7) - (int(leng * 0.7))%10 
    # test_begin = 0
    # test_size = leng
    # print(test_begin,test_size)
    # print(leng)
    for i in range(test_size):
        _, data = test[(test_begin + i) % leng]
        inputs , y_pred = data
        outputs = model(inputs)
#        print(outputs, y_pred)
#        print(outputs, y_pred)
        if outputs.mean() - 0.5 > 0. and y_pred.mean() > 0.9:
            loss += 1
        elif outputs.mean() - 0.5 < 0. and y_pred.mean() < 0.1:
            loss += 1
    print("test accuracy: %0.3f %%" % ((loss / test_size) * 100))
def train_1(epoch,model):
#    model = mi_Net(dataset.__length__())
    optimizer = optim.SGD(model.parameters(), lr = 0.01 , momentum = 0.5)
    running_loss = 0
    
    train = list(enumerate(train_data,0))
    leng = len(train)
    epoch_ = (epoch*10)%(leng - leng % 10)
    train_size = int(leng * 0.7) - (int(leng * 0.7)) %10
#    epoch_ = 0
#    train_size = leng
    running_loss = 0
#    print(train_size , epoch_)
    for count in range(1000):
        for index in range(train_size):
            batch_idx , data = train[(epoch_ + index) % leng]
            inputs, y_pred = data
#            print(y_pred)
            outputs = model(inputs)
#            print(outputs , y_pred)
            loss = criterion(outputs , y_pred)
#            if index % 10 == 9:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data
        if count % 100 == 99:
            print('[第%d轮, 训练集大小:%5d] loss: %0.7f' % (count + 1 , index + 1 , running_loss.data/ (train_size * 100)))
            test_1(epoch,model)
            running_loss = 0        
        if count == 999:
            for index in range(train_size):
                _ , data = train[index]
                inputs , y_pred = data
                outputs = model(inputs)
#                print(outputs , y_pred)
for epoch in range(1):#网络在这里 选择
    # model = mi_Net(dataset.__length__())
    # model = MI_Net(dataset.__length__())
    # model = MI_net_DS(dataset.__length__())
    model = MI_net_Res(dataset.__length__())
    train_1(epoch,model)