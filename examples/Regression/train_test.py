'based on https://nextjournal.com/gkoehler/pytorch-mnist'
import math
import os
#os.sys.path.append('/vspace')
import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from common.pytorch.ml_wrapper import ML_Wrapper
from examples.Regression.dense_model import RegresionDenseNet

def f1(x):
    #return 2*x + 1
    #return math.log(x+2,math.e)
    return math.exp(x)
vf1 = np.vectorize(f1)

base_path = "../../"
log_interval = 10
batch_size_test = 1000
n_epochs = 10
batch_size_train = 64
learning_rate = 0.01
momentum = 0.5
x_min = -1
x_max = 1


network = RegresionDenseNet()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
ml = ML_Wrapper(network, optimizer, base_path, device = "cpu" )
ml.loss_function = nn.MSELoss()

train_x = np.random.rand(1000,1)*2 - 1
train_y = vf1(train_x)

test_x = np.random.rand(batch_size_test,1)*2 - 1
test_y = vf1(test_x)

dataset_p_train = TensorDataset(torch.Tensor(train_x,),
                                torch.Tensor(train_y) )
train_loader = DataLoader(dataset_p_train, batch_size = 10);

dataset_p_test = TensorDataset(torch.Tensor(test_x,),
                                torch.Tensor(test_y) )
test_loader = DataLoader(dataset_p_test, batch_size = batch_size_test);

for epoch in range(1, n_epochs + 1):
    ml.train(epoch, train_loader)

pred = ml.network(torch.Tensor(test_x)).detach().numpy()
print(test_x)
fig = plt.figure()
plt.scatter(test_x[:,0], test_y[:,0], label="real",s=1 )
plt.scatter(test_x[:,0], pred[:,0], label="pred", s=1)
plt.legend()
plt.savefig(base_path+"/out/pred_linear.png")

print(math.sqrt(np.sum((pred[:,0] - test_y[:,0])**2)/batch_size_test))
#fig.show()
