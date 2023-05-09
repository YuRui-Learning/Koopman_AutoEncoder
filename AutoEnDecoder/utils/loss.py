import torch.nn as nn
import torch

lose = {'mul_loss':0,'predict_lost':0,'decode_lost':0,'inf_loss':0}

def loss_func(decoded,X_state_1):
    length = len(decoded)
    X_state_1 = X_state_1.view(-1) # 降低为1维数据

    lose['inf_loss'] = torch.norm(decoded - X_state_1, p=1)
    lose['decode_lost'] = nn.MSELoss(decoded,X_state_1.reshape(-1,1))

    loss_value = lose['inf_loss'] + lose['decode_lost'] + lose['predict_lost'] + lose['mul_loss']
    return loss_value
