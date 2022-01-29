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
dataset = loader_image('musk1.mat')   #MUSK1 dataset
#dataset = loader_image('musk2.mat')    #MUSK2 dataset
#dataset = loader_image('fox.mat')
#dataset = loader_image('elephant.mat')
#dataset = loader_image('tiger.mat')
gpus = [0]
# print(torch.cuda.device_count())
cuda_gpu = torch.cuda.is_available()
criterion = torch.nn.BCELoss()
train_data = DataLoader(dataset=dataset,batch_size=1,shuffle=True)

lens = len(list(train_data))

expet = 10 * int(lens / 10)
#print(lens)
num = []
for i in range(10):
    num.append(list(range(int(i)*int(lens / 10) ,int(i)*int(lens/10)  +int(lens / 10))))
for index in range(lens - int(lens / 10) * 10):
    num[9].append(index + expet)
train =list(enumerate(train_data,0))
# print(train[0])
# print(num)

# for i in range(len(train)):
#     train[i][1][0] = train[i][1][0].cuda()
#     train[i][1][1] = train[i][1][1].cuda()
def train_(model , epoch , t):
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)
#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    running_loss = 0
    ran = list(range(10))
    ran.pop(epoch)
#    print((ran))
    l_epo = len(num[epoch])
    t_c = 220             #迭代次数
    for count in range(t_c):
#        print("t")
#        index = 0
        
        for index in ran:
            for i in num[index]:
                _, data = train[i]
                inputs , y_pred = data
                if(cuda_gpu):
                    inputs = inputs.to("cuda:0")
                    y_pred = y_pred.to("cuda:0")
                
                outputs = model(inputs)
#                y_pred= y_pred
#                if(outputs[0].data > 1. or outputs[0].data < 0.):
#                print(outputs)
#                if(y_pred[0].data > 1. or y_pred[0].data < 0.):
#                print(y_pred)
                loss = criterion(outputs, y_pred)
                running_loss += loss.data
                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()
#        scheduler.step()
    print('[%d  %d loss:   %0.7f]' % (t + 1, epoch + 1, running_loss / (t_c * (lens - l_epo))))
def test(model, epoch):
    accuracy_num = 0
    l_epo = len(num[epoch])
#    print(epoch)
#    print(l_epo)
    for i in num[epoch]:
        _, data = train[i]
        inputs, y_pred = data
        if(cuda_gpu):
            inputs = inputs.to("cuda:0")
            y_pred = y_pred.to("cuda:0")
        outputs = model(inputs)
#        y_pred = y_pred
        if outputs[0].data > 0.5 and y_pred[0].data > 0.9:
            accuracy_num += 1
        elif outputs[0].data < 0.5 and y_pred[0].data < 0.1:
            accuracy_num += 1
#    print('accuracy:  %d %%' % (100 * accuracy_num /l_epo))
    return accuracy_num
if __name__ == '__main__':
    count = 0
    for i in range(5):
        accuracy = 0
        for j in range(10):
            model = mi_Net(dataset.__length__())   #mi-net
#            model = MI_Net(dataset.__length__())   #MI-net
#            model = MI_net_DS(dataset.__length__()) #MI-net-DS
#            model = MI_net_Res(dataset.__length__())  #MI-net-RS
            if(cuda_gpu):
                model = model.to("cuda:0")
            train_(model,j,i)
            accuracy += test(model, j)
            print('the accuracy number: %d' % (accuracy))
#            accracy /= 10
#            t_num += accracy
        print('accuracy:  %2.5f %%' % (100 * accuracy / lens))
        count += accuracy
    print('[the accuracy in 5 times is : ] %2.5f %%' % (100 * count / (5 * lens)))