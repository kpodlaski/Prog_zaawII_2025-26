import math

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import optim, nn

from common.pytorch.ml_wrapper import ML_Wrapper
from examples.TimeSeries.dense_model import DenseTimeSeries
from examples.TimeSeries.generate_dataset import generate_dataset
from examples.TimeSeries.rnn_model import RNNNet

fun = lambda x: math.exp(x)
fun = np.vectorize(fun)

rev_fun = lambda x: math.log(x,math.e)
rev_fun = np.vectorize(rev_fun)
x_min = -1
x_max = 1
window_size = 4

base_path = "../../"
dataset_size = 1000
n_epochs = 100
batch_size_train = 64
learning_rate = 0.01
momentum = 0.5
test_size = 0.2
train_dataset, train_dataloader, test_dataset, test_dataloader = generate_dataset(dataset_size,
        fun, x_min=x_min, x_max=x_max, window_size=window_size,
        test_size=test_size, batch_size=batch_size_train)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
rnn_type = 'LSTM'
#rnn_type = 'GRU'

network = RNNNet(rnn_type, window_size=window_size)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
ml = ML_Wrapper(network, optimizer, base_path, device = device )
ml.loss_function = nn.MSELoss()

for epoch in range(1, n_epochs + 1):
    ml.train(epoch, train_dataloader, val_loader=test_dataloader)

pred = ml.network(test_dataset.tensors[0]).detach().numpy().reshape(-1)
expected = test_dataset.tensors[1].numpy().reshape(-1)
fig = plt.figure()
plt.title("TimeSeries, DenseNetwork, predictions, after {} epochs".format(n_epochs))
plt.scatter(rev_fun(expected), expected, label="test_real", s=1, c="red")
plt.scatter(rev_fun(expected),pred, label="test_pred", s=1, c="blue")
pred = ml.network(train_dataset.tensors[0]).detach().numpy().reshape(-1)
expected = train_dataset.tensors[1].numpy().reshape(-1)
plt.scatter(rev_fun(expected), expected, label="train_real", s=1, c="orange")
plt.scatter(rev_fun(expected), pred, label="train_pred", s=1, c="magenta")
plt.legend()
plt.savefig(base_path+"/out/time_series_rnn.png")




