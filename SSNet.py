import torch
import torch.nn as nn

class SSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(19, 32)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.25)
        self.hidden2 = nn.Linear(32, 48)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.25)
        self.hidden3 = nn.Linear(256, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.25)
        self.hidden4 = nn.Linear(512, 1024)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.25)
        self.output = nn.Linear(48, 6)

    def forward(self, x):
        x = self.drop1(self.act1(self.hidden1(x)))
        x = self.drop2(self.act2(self.hidden2(x)))
        # x = self.drop3(self.act3(self.hidden3(x)))
       #  x = self.drop4(self.act4(self.hidden4(x)))
        x = self.output(x)

        return x
