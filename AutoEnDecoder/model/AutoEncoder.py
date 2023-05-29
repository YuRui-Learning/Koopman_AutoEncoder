import torch.nn as nn
import torch

class AutoEncoder(nn.Module):
    def __init__(self):

        super(AutoEncoder, self).__init__()
        matrix_B = torch.randn(10,10, requires_grad=True)
        matrix_A = torch.randn(40, 40, requires_grad=True)
        self.matrix_B = nn.Parameter(matrix_B)
        self.matrix_A = nn.Parameter(matrix_A)

        self.encoder = nn.Sequential(
            nn.Linear(20, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 20),

        )
        self.decoder = nn.Sequential(
            nn.Linear(50, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 10),
            nn.Sigmoid()

        )

    def forward(self, x):
        U = x[::2] # tensor 10, 并且该矩阵是一个单位向量乘以一个常数
        encoded = self.encoder(x) # 20维度向量
        flpha_e =  torch.cat((x , encoded),0) #tensor拼接 40维
        U_Convert = torch.mm(U.reshape(1,-1),self.matrix_B).reshape(10,-1) # tensor内积
        flpha_Convert = torch.mm(flpha_e.reshape(1,-1) , self.matrix_A).reshape(40,1) # tensor内积

        Encoder_Input = torch.cat((flpha_Convert,U_Convert),0)
        Base_Value = torch.cat((flpha_e , U),0) # 因为要计算一下A，B矩阵的影响，所以需要在前向传播时候，记录一下。
        Matrix_Loss = torch.sqrt(torch.mean(torch.pow((Base_Value - Encoder_Input.view(-1)), 2)))
        Encoder_Input = Encoder_Input.view(-1) # 压缩为1维向量
        decoded = self.decoder(Encoder_Input)
        # print('matrix_A',self.matrix_A)
        # print('matrix_B',self.matrix_B)
        return encoded, decoded , Matrix_Loss,self.matrix_A,self.matrix_B



