import torch.nn as nn
import torch
import math

lose = {'mul_loss':0,'predict_lost':0,'decode_lost':0,'inf_loss':0}

def loss_compute( encoded , decoded , X_state, X_state_1):
    length = len(decoded)
    X_state_1 = X_state_1.view(-1) # 降低为1维数据
    X_state_0 = X_state[::2] # 提取前一时刻状态遍变量

    lose['mul_loss'] = torch.norm(X_state_1 - X_state_0, p=1) # Lx,x 状态变量之间的差距

    lose['predict_lost'] = torch.sqrt(torch.mean(torch.pow((encoded - X_state), 2))) # Lo,x，encoder_loss

    lose['decode_lost'] = torch.sqrt(torch.mean(torch.pow((decoded - X_state_1), 2))) # Lo,x decoder_loss

    lose['inf_loss'] = torch.norm(decoded - X_state_1, p=1)  # L∞ inf loss

    loss_value = lose['inf_loss'] + lose['decode_lost'] + lose['predict_lost'] + lose['mul_loss']

    return loss_value
