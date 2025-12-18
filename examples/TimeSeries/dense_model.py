import torch
from torch import nn


class DenseTimeSeries(nn.Module):
    def __init__(self, window_size=4):
        super(DenseTimeSeries,self).__init__()
        self.l1 = nn.Linear(window_size, 2*window_size)
        self.l2 = nn.Linear(2*window_size, window_size+1)
        self.l3 = nn.Linear(window_size+1, window_size+1)
        self.output = nn.Linear(window_size+1, 1)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = self.output(x)
        return x

    def summary(self, size):
        print(self)