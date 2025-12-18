import math

import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def generate_dataset(size, function,
                     x_min, x_max, window_size=4,
                     batch_size=64, test_size=.2,
                     draw_dataset= False, rev_fun=None ):
    patterns = []
    values = []
    step_size = (x_max - x_min) / (window_size * size)
    for k in range(size):
        x0 = x_min + window_size*k*step_size
        xs = np.linspace(x0, x0+(window_size+1)*step_size, window_size+1, endpoint=False)
        ys =  function(xs)
        patterns.append(ys[:-1])
        values.append(ys[-1])
    patterns = np.array(patterns)
    values = np.array(values)
    train_x, test_x, train_y, test_y = train_test_split(patterns, values, test_size=test_size)
    #print(train_x.shape, train_y.shape)
    #print(test_x.shape, test_y.shape)
    tensor_patterns = torch.Tensor(train_x)
    tensor_values = torch.Tensor(train_y.reshape(-1,1))
    train_dataset = TensorDataset( tensor_patterns, tensor_values)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    tensor_patterns = torch.Tensor(test_x)
    tensor_values = torch.Tensor(test_y.reshape(-1,1))
    test_dataset = TensorDataset(tensor_patterns, tensor_values)
    test_dataloader = DataLoader(train_dataset, batch_size=len(tensor_values))
    if draw_dataset:
        draw_dataset_ys(rev_fun,train_dataset,test_dataset)
    return train_dataset, train_dataloader, test_dataset, test_dataloader

def draw_dataset_ys(rev_fun, train_dataset, test_dataset):
    fig = plt.figure()
    train = train_dataset.tensors[1].numpy().reshape(-1)
    test = test_dataset.tensors[1].numpy().reshape(-1)
    plt.scatter(rev_fun(train), train, label="train", s=1)
    plt.scatter(rev_fun(test), test, label="test", s=1)
    plt.legend()
    plt.savefig("../../out/time_series_dataset_ys.png")

if __name__ == '__main__':
    fun = lambda x: math.exp(x)
    fun = np.vectorize(fun)
    rev_fun = lambda x: math.log(x,np.e)
    rev_fun = np.vectorize(rev_fun)
    x_min = -1
    x_max = 1
    window_size = 4
    dataset_size = 1000
    test_size = 0.2
    train_dataset, train_dataloader, test_dataset, test_dataloader = generate_dataset(
        dataset_size,
        fun, x_min=x_min, x_max=x_max,
        window_size=window_size,
        test_size=test_size,
        draw_dataset=True, rev_fun=rev_fun)
    print(len(train_dataset), train_dataset.tensors[0].shape, train_dataset.tensors[1].shape)
    print(len(test_dataset), test_dataset.tensors[0].shape, test_dataset.tensors[1].shape)
