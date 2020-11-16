import torch
import torch.nn as nn
from utils import *
from modules import ActFun
import torch.distributions as td
import torch.nn.functional as F


## Glow Modules
class ActNorm(nn.Module):

    def __init__(self, num_channels):
        super().__init__()

        size = [1, num_channels, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size), requires_grad=True))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size), requires_grad=True))
        
        # Buffer to register if initialization has been performed
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
      if not self.training:
        return

      with torch.no_grad():
          bias = input.clone().mean(dim=[0, 2, 3], keepdim=True)
          std_input = input.clone().std(dim=[0, 2, 3], keepdim=True)
          logs = (1.0 / (std_input + 1e-6)).log()
          self.bias.data.copy_(-bias)
          self.logs.data.copy_(logs)

    def forward(self, input, logdet, reverse):
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
            
        dims = input.size(2) * input.size(3)

        if reverse == False:
            input = input + self.bias
            input = input * self.logs.exp()
            dlogdet = torch.sum(self.logs) * dims
            if logdet is not None:
              logdet = logdet + dlogdet

        if reverse == True:
            input = input * self.logs.mul(-1).exp()
            input = input - self.bias
            dlogdet = - torch.sum(self.logs) * dims
            if logdet is not None:
              logdet = logdet + dlogdet

        return input, logdet

class Conv2dZeros(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3,3], stride=[1,1]):
        super().__init__()
        
        padding = (kernel_size[0] - 1) // 2
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, input):
      output = self.conv(input)
      return output 

class Conv2dNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1], norm = "actnorm"):
        super().__init__()

        padding = [(kernel_size[0]-1)//2, (kernel_size[1]-1)//2]

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=(norm != "actnorm"))
        self.conv.weight.data.normal_(mean=0.0, std=0.05)
        
        self.norm = norm
        if self.norm == "actnorm":
          self.norm_type = ActNorm(out_channels)
        elif self.norm=="batchnorm":
          self.conv.bias.data.zero_()
          self.norm_type = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        output = self.conv(input)
        if self.norm == "actnorm":
          output,_ = self.norm_type(output, logdet=0.0, reverse=False)
        elif self.norm == "batchnorm":
          output = self.norm_type(output)
        else:
          return output
        return output


class InvConv(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u = u + torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet, reverse):
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
              logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
              logdet = logdet - dlogdet
            return z, logdet

class AffineCoupling(nn.Module):
    def __init__(self, x_size, condition_size):
        super(AffineCoupling, self).__init__()
        
        Bx, Cx, Hx, Wx = x_size
        
        B, C, H, W = condition_size
        channels = Cx // 2 + C
        non_lin = 'leakyrelu'
        hidden_channels = 128
        self.net = nn.Sequential(
            Conv2dNorm(channels, hidden_channels),
            ActFun(non_lin),
            Conv2dNorm(hidden_channels, hidden_channels, kernel_size=[1, 1]),
            ActFun(non_lin),
            Conv2dZeros(hidden_channels, Cx),
        )

        self.scale = nn.Parameter(torch.zeros(Cx//2, 1, 1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(Cx//2, 1, 1), requires_grad=True)
        
    def forward(self, x, condition, logdet, reverse): 
        z1, z2 = split_feature(x, "split")

        assert condition.shape[2:4] == x.shape[2:4], "condition and x in affine needs to match"
        h = torch.cat([z1, condition], dim=1)

        shift, log_scale = split_feature(self.net(h), "cross")

        # Here we could try to use the exponential as suggested in arXiv:1907.02392v3
        log_scale = self.scale * torch.tanh(log_scale) + self.scale_shift

        if reverse == False:
            z2 = z2 + shift
            z2 = z2 * torch.exp(log_scale)
            if logdet is not None:
              logdet = logdet + torch.sum(log_scale, dim=[1, 2, 3])
        else:
            z2 = z2 * log_scale.mul(-1).exp()
            z2 = z2 - shift
            if logdet is not None:
              logdet = logdet - torch.sum(log_scale, dim=[1, 2, 3]) 

        output = torch.cat((z1, z2), dim=1)
        return output, logdet


class Squeeze2d(nn.Module):
    def __init__(self):
        super(Squeeze2d, self).__init__()
        
    def forward(self, x, undo_squeeze):
      B, C, H, W = x.shape
      if undo_squeeze == False:
        # C x H x W -> 4C x H/2 x W/2
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * 4, H // 2, W // 2)
      else:
        # 4C x H/2 x W/2  ->  C x H x W
        x = x.reshape(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // 4, H * 2, W * 2)
      return x

class Split2d(nn.Module):
    def __init__(self, x_size, condition_size):
      super(Split2d, self).__init__()

      Bx, Cx, Hx, Wx = x_size
      B, C, H, W = condition_size
      channels = Cx // 2 + C 
        
      self.conv = nn.Sequential(Conv2dZeros(channels, Cx),)

    # TODO: We could try to use the Convnorm here, make more powerful (maybe)
    # TODO: Make option to enable conditional.
    def forward(self, x, condition, logdet, reverse):

        if reverse == False:
            z1, z2 = split_feature(x, "split")
            h = torch.cat([z1, condition], dim=1)
            out = self.conv(h)
            mean, log_scale = split_feature(out, "cross")
            if logdet is not None:
              logdet = logdet + torch.sum(td.Normal(mean, torch.exp(log_scale)).log_prob(z2), dim=(1,2,3))
            return z1, logdet
        else:
            h = torch.cat([x, condition], dim=1)
            mean, log_scale = split_feature(self.conv(h), "cross")
            z2 = td.Normal(mean, torch.exp(log_scale)).rsample()
            z = torch.cat((x, z2), dim=1)
            return z, logdet
