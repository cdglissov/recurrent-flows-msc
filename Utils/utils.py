import torch
import torch.nn as nn

use_gpu = False
device = None


# Set the gpu if available
def set_gpu(mode, verbose = False):
    global use_gpu
    global device
    use_gpu = mode & torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    if verbose:
      print("Device set to: ", device)
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
    def __init__(self, C_x, H_x, W_x):
        super(UnFlatten, self).__init__()
        self.C_x = C_x
        self.H_x = H_x
        self.W_x = W_x
        
    def forward(self, input):
        dims = input.size(0)
        return input.view(dims, self.C_x, self.H_x, self.W_x)


# Print the output dimensions of layer
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.size())
        return x

def get_layer_size(dims, kernels, paddings, strides, dilations, output_paddings = [], uneven_format = False, transpose=False):
    out_h, out_w = dims
    if transpose == False:
      if uneven_format == True:
        for kernel, padding, stride, dilation in zip(kernels, paddings, strides, dilations):
          out_h = (out_h + 2*padding[0] - dilation[0] * (kernel[0]-1) - 1) // stride[0] + 1
          out_w = (out_w + 2*padding[1] - dilation[1] * (kernel[1]-1) - 1) // stride[1] + 1
      else:
        for kernel, padding, stride, dilation in zip(kernels, paddings, strides, dilations):
          out_h = (out_h + 2*padding - dilation * (kernel-1) - 1) // stride + 1
          out_w = (out_w + 2*padding - dilation * (kernel-1) - 1) // stride + 1
    else:
      assert len(output_paddings) == len(paddings), "Please specify output_padding when using transpose"
      if uneven_format == True:
        for kernel, padding, stride, dilation, output_padding in zip(kernels, paddings, strides, dilations, output_paddings):
          out_h = (out_h - 1) * stride[0] - 2*padding[0] + dilation[0] * (kernel[0] - 1) + output_padding[0] + 1
          out_w = (out_w - 1) * stride[1] - 2*padding[1] + dilation[1] * (kernel[1] - 1) + output_padding[1] + 1
      else:
        for kernel, padding, stride, dilation, output_padding in zip(kernels, paddings, strides, dilations, output_paddings):
          out_h = (out_h - 1) * stride - 2*padding + dilation * (kernel - 1) + output_padding + 1
          out_w = (out_w - 1) * stride - 2*padding + dilation * (kernel - 1) + output_padding + 1
    return out_h, out_w

def split_feature(tensor, type="split"):
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

def free_bits_kl(kl, free_bits=0.0, batch_average = False, eps=1e-6):

    if free_bits < eps:
        return kl.mean(0)
    if batch_average:
        return kl.mean(0).clamp(min=free_bits)
    return kl.clamp(min=free_bits).mean(0)



