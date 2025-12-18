' based on https://nextjournal.com/gkoehler/pytorch-mnist'
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegresionDenseNet(nn.Module):
    def __init__(self):
        super(RegresionDenseNet, self).__init__()
        self.l1 = nn.Linear(1, 5)
        self.l2 = nn.Linear(5, 5)
        self.l3 = nn.Linear(5, 5)
        self.output = nn.Linear(5,1)


    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = self.output(x)
        return x

    def summary(self, size):
        print(self)
