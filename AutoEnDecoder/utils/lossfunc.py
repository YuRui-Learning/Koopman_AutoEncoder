import torch.nn as nn
import torch
import math

lose = {'mul_loss':0,'predict_lost':0,'decode_lost':0,'inf_loss':0}

def loss_compute( encoded , decoded , X_state, X_state_1 , matrixloss):
    length = len(decoded)

    X_state_1 = X_state_1.view(-1) # 降低为1维数据
    X_state_0 = X_state[1::2] # 提取前一时刻状态遍变量
    # 编码器输入和解码器输出的差距
    lose['mul_loss'] += torch.sqrt(torch.mean(torch.pow((X_state_0 - X_state_1), 2)))# Lx,x 状态变量之间的差距
    # 量化出 A , B 的作用，乘以A，B前，和乘以A,B后
    lose['predict_lost'] = matrixloss # Lo,x，encoder_loss

    lose['decode_lost'] = torch.sqrt(torch.mean(torch.pow((decoded - X_state_1), 2))) # Lo,x decoder_loss

    lose['inf_loss'] = torch.norm(decoded - X_state_1, p=1)  # L∞ inf loss

    loss_value = lose['inf_loss'] + lose['decode_lost'] + lose['predict_lost'] * 0.3 + lose['mul_loss'] * 1e-9

    return loss_value
