import torch
import torch.nn as nn
from model import AutoEncoder
from data import dataloader
from data import dataprocess
from utils import loss

import time
starttime = time.time()

torch.manual_seed(1)
EPOCH = 50
BATCH_SIZE = 64
LR = 0.005
N_TEST_IMG = 5

# 读取数据
Train_Dict , Test_Dict = dataloader.Get_data()
# 数据处理，变成 40个 * 10 行 * 2列 数据形式
Train_Data,Test_Data,Train_Output,Test_Output  = dataprocess.Process_data(Train_Dict , Test_Dict)


Coder = AutoEncoder.AutoEncoder()
print(Coder)

optimizer = torch.optim.Adam(Coder.parameters(),lr=LR)
loss_func = loss.loss_func()

for epoch in range(EPOCH):
    for step in range(len(Train_Data)):
        b_x = torch.from_numpy(Train_Data[step]).to(torch.float32)
        b_y = torch.from_numpy(Train_Output[step]).to(torch.float32)
        encoded , decoded = Coder(b_x)
        loss = loss_func(decoded,b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%5 == 0:
            print('Epoch :', epoch,'|','train_loss:%.4f'%loss.data)

torch.save(Coder,'AutoEncoder.pkl')
print('________________finish training___________________')
endtime = time.time()
print('训练耗时：',(endtime - starttime))


Coder = AutoEncoder.AutoEncoder()
Coder = torch.load('AutoEncoder.pkl')

# 数据的空间形式的表示


for i in range(10):
    view_data = torch.from_numpy(Test_Data[i]).to(torch.float32)
    _ , encoded_data = Coder(view_data)
    print(encoded_data)
    print(Test_Output[i])