import torch.nn as nn
import torch

class AutoEncoder(nn.Module):
    def __init__(self):

        super(AutoEncoder, self).__init__()
        matrix_B = torch.ones(10,10, requires_grad=True)
        self.matrix_B = nn.Parameter(matrix_B)


        self.encoder = nn.Sequential(
            nn.Linear(20, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 10),
            nn.Sigmoid()

        )
        # super(AutoEncoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(20, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 32),
        #     nn.Tanh(),
        #     nn.Linear(32, 16),
        #     nn.Tanh(),
        #     nn.Linear(16, 3)
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(3, 16),
        #     nn.Tanh(),
        #     nn.Linear(16, 32),
        #     nn.Tanh(),
        #     nn.Linear(32, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 10),
        #     nn.Sigmoid()
        #
        # )

    def forward(self, x):
        u = x[::2]
        print(self.matrix_B)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
