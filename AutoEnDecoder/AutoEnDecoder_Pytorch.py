import numpy as np
import torch
import torch.nn as nn
from model import AutoEncoder
from data import dataloader
from data import dataprocess
from utils import lossfunc
from utils import view
import time
starttime = time.time()

torch.manual_seed(1)
EPOCH = 100
BATCH_SIZE = 64
LR = 0.0001
y1 = [] # 可视化loss list
y2 = [] # 精度person_cof list
loss_min = 10 # 定位最小误差
early_stop_flag = False

# 读取数据
Train_Dict , Test_Dict = dataloader.Get_data()
# 数据处理，变成 40个 * 10 行 * 2列 数据形式
Train_Data,Test_Data,Train_Output,Test_Output  = dataprocess.Process_data(Train_Dict , Test_Dict)
# 模型示例化，自编码器模型
Coder = AutoEncoder.AutoEncoder()
# 优化器选择
optimizer = torch.optim.Adam(Coder.parameters(),lr=LR)

def train(Coder):
    # 自编码器训练过程
    for epoch in range(EPOCH):
        for step in range(len(Train_Data)):
            X_state = torch.from_numpy(Train_Data[step]).to(torch.float32) # 读取零状态
            X_state_1 = torch.from_numpy(Train_Output[step]).to(torch.float32) # 读取下一时刻数据
            encoded , decoded ,matrixloss, MATA, MATB = Coder(X_state) # 解码器结果
            loss = lossfunc.loss_compute( decoded , X_state, X_state_1 , matrixloss) # 求loss
            corr = np.corrcoef(decoded.detach().numpy() , X_state_1.view(-1).detach().numpy()) # 求相关系数
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optimizer.step() # 优化器迭代
            if loss < loss_min:
                matrix_best_A = MATA # 找到loss最小的MATA
                matrix_best_B = MATB # 找到loss最小的MATB
            # if corr[0][1] > 0.95:
            #     early_stop_flag = True
            #     break
            if step%5 == 0:
                print('Epoch :', epoch,'|','train_loss:%.4f'%loss.data)
        y1.append(loss.data) # 维系误差loss list 用于可视化训练误差
        y2.append(corr[0][1]) # 维系相关系数cof list 用于可视化训练误差
    # 可视化误差
    view.perform_loss_cof(y1,y2)

    if early_stop_flag :
        print("------early stopping ----------")

    # 保存Cober模型
    torch.save(Coder,'AutoEncoder.pkl')
    print('________________finish training___________________')
    endtime = time.time()
    print('训练耗时：',(endtime - starttime))
    # 实例化模型
    Coder = AutoEncoder.AutoEncoder()
    # 加载模型参数
    Coder = torch.load('AutoEncoder.pkl')

    # 输出最终的预测结果
    for i in range(10):
        view_data = torch.from_numpy(Test_Data[i]).to(torch.float32)
        _ , decoded_data ,_,_,_ = Coder(view_data)
        print(decoded_data)
        print(Test_Output[i])

    # 返回所要的两个矩阵
    return MATA ,MATB

if __name__ == "__main__":
    MATA , MATB =train(Coder)
    print(MATA.shape,MATB.shape) # 返回A B 矩阵 A 是40*40 B是10*10