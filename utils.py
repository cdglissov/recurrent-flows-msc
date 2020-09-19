import torch
import numpy as np
import torch.nn as nn

use_gpu = False
device = None


# Set the gpu if available
def set_gpu(mode):
    global use_gpu
    global device
    use_gpu = mode & torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    if torch.cuda.is_available():
        print("Note: GPU is available")
    else:
        print("Note: GPU is not available")
    return device


# Wrapper, convert to cuda tensor
def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


# Convert to numpy
def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


# Flatten layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# Unflatten layer to dimensions
class UnFlatten(nn.Module):
    def forward(self, input, C_x, H_x, W_x):
        dims = input.size(0)
        return input.view(dims, C_x, H_x, W_x)


# Print the output dimensions of layer
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.size())
        return x

def get_layer_size(dims, kernels, paddings, strides, dilations, uneven_format = False):
    out_h, out_w = dims
    if uneven_format == True:
      for kernel, padding, stride, dilation in zip(kernels, paddings, strides, dilations):
        out_h = (out_h + 2*padding[0] - dilation[0] * (kernel[0]-1) - 1) // stride[0] + 1
        out_h = (out_w + 2*padding[1] - dilation[1] * (kernel[1]-1) - 1) // stride[1] + 1
    else:
      for kernel, padding, stride, dilation in zip(kernels, paddings, strides, dilations):
        out_h = (out_h + 2*padding - dilation * (kernel-1) - 1) // stride + 1
        out_w = (out_w + 2*padding - dilation * (kernel-1) - 1) // stride + 1
    return out_h, out_w
