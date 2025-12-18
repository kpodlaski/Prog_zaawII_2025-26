import time

import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


from common.pytorch.ml_wrapper import ML_Wrapper
from examples.CNN.pytorch.cnn_model import ConvNet

base_path = "../../../"
n_epochs = 20
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
transfer_learning = False

#print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.KMNIST(base_path+'datasets/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=True, **data_loader_kwargs)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.KMNIST(base_path+'datasets/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_train, shuffle=True, **data_loader_kwargs)

network = ConvNet()

optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
output_filename = "kmist_network"
if transfer_learning:
    output_filename = output_filename+"_transfered"
    ml = ML_Wrapper.load_model (base_path, "conv_network_model.pth", network, optimizer, device)
    layers_to_freeze  = [ml.network.conv1, ml.network.conv2, ml.network.fc1]
    for layer in layers_to_freeze:
        for param in layer.parameters():
            param.requires_grad = False
else:
    ml = ML_Wrapper(network, optimizer, base_path, device)
print("Start training")
globalT= time.time()
for epoch in range(1, n_epochs + 1):
  ml.train(epoch, train_loader, test_loader)
globalT = time.time() - globalT
print("Total time of training(s):{:.4f}".format(globalT))
ml.save_model(output_filename)

print("Test on training set:")
ml.test(train_loader)
print("Test on test set:")
ml.test(test_loader)