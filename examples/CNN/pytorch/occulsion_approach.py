
import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


from common.pytorch.ml_wrapper import ML_Wrapper
from examples.CNN.pytorch.cnn_model import ConvNet

def add_colorbar(im, width=None, pad=None, **kwargs):

    l, b, w, h = im.axes.get_position().bounds       # get boundaries
    width = width or 0.05 * w                         # get width of the colorbar
    pad = pad or width                               # get pad between im and cbar
    fig = im.axes.figure                             # get figure of image
    cax = fig.add_axes([l + w + pad, b, width, h])   # define cbar Axes
    return fig.colorbar(im, cax=cax, **kwargs)       # draw cbar

activation = {}
def get_activation(name=None):
    def hook (model, input, output):
        activation[name]=(output.cpu().detach())
    return hook

base_path = "../../../"
n_epochs = 100
batch_size_train = 64
batch_size_test = 1
learning_rate = 0.01
momentum = 0.5
log_interval = 10
#print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(base_path+'datasets/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=True, **data_loader_kwargs)


network = ConvNet()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
ml = ML_Wrapper.load_model (base_path, "conv_network_model.pth", network, optimizer, device)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
patch_half_size = 4
patch_tensor = torch.zeros(2*patch_half_size+1,2*patch_half_size+1)
image_shape = example_data.shape
patch_shape = patch_tensor.shape
print(patch_shape, image_shape)

with torch.no_grad():
    network.to(device)
    test_image = example_data.clone()
    test_image = test_image.to(device)
    output = ml.network(test_image)
    expected_class = output.argmax(dim=1).item()
    print(example_data.shape, patch_shape)
    print(example_data.shape[2] - patch_shape[0])
    occlusion_image = np.zeros( (example_data.shape[2], example_data.shape[3]))
    for x in range(int(example_data.shape[2])):
        for y in range(int(example_data.shape[3])):
            x_s = max(x-patch_half_size,0)
            x_e = min(x+patch_half_size, example_data.shape[2])
            y_s = max(y - patch_half_size, 0)
            y_e = min(y + patch_half_size, example_data.shape[3])
            test_image = example_data.clone()
            test_image[0,0, x_s:x_e,y_s:y_e] = patch_tensor[0:x_e-x_s, 0:y_e-y_s]
            test_image = test_image.to(device)
            output = ml.network(test_image)
            value = np.exp(output[0][expected_class].item())
            occlusion_image[x,y] = value
fig = plt.figure()
f, (ax1, ax2) = plt.subplots(1,2)
im2 =ax2.imshow(occlusion_image, cmap='gnuplot', interpolation='none')
ax1.imshow(example_data[0][0], cmap='grey', interpolation='none')
add_colorbar(im2)
plt.savefig(base_path+"/out/occlusion.png")


