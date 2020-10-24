import torch
import torch.nn as nn
from .utils import *
from math import log, pi, exp
import torch.distributions as td
from .layers import (LinearNorm, LinearZeros, 
                              Conv2dZeros, Conv2dResize, 
                              Conv2dZerosy, ActNorm,
                              Conv2dNormy, ConvLSTM)
import torch.nn.functional as F

device = set_gpu(True)


class ActFun(nn.Module):
  def __init__(self, non_lin):
    super(ActFun, self).__init__()
    if non_lin=='relu':
      self.net=nn.ReLU()
    if non_lin=='leakyrelu':
      self.net=nn.LeakyReLU(negative_slope=0.20)

  def forward(self,x):
    return self.net(x)


class ConditionalActNorm(nn.Module):
    def __init__(self, x_size, condition_size):
        """
        Applied conditional ActNorm, finds the conditional shift. From the condition.
        """
        super(ConditionalActNorm, self).__init__()
        Bx, Cx, self.Hx, self.Wx = x_size
        B, C, self.H, self.W = condition_size
        
        x_hidden_channels = 128
        x_hidden_size = 64
        self.ConditionalNet = nn.Sequential(
            Conv2dResize(in_size=[C, self.H, self.W], out_size=[x_hidden_channels, self.H//2, self.W//2]),
            ActFun(non_lin),
            Conv2dResize(in_size=[x_hidden_channels, self.H//2, self.W//2], out_size=[x_hidden_channels, self.H//4, self.W//4]),
            ActFun(non_lin),
            Conv2dResize(in_size=[x_hidden_channels, self.H//4, self.W//4], out_size=[x_hidden_channels, self.H//8, self.W//8]),
            ActFun(non_lin),
        )
        
        self.ConditionalNet_Linear = nn.Sequential(
            LinearZeros(x_hidden_channels*self.H*self.W//(8*8), x_hidden_size),
            ActFun(non_lin),
            LinearZeros(x_hidden_size, x_hidden_size),
            ActFun(non_lin),
            LinearZeros(x_hidden_size, 2*Cx),
            nn.Tanh()
        )
        
    def forward(self, x, condition, logdet = 0.0, reverse=False):
        bs = x.shape[0]
        output = self.ConditionalNet(condition)
        output=output.view(bs, -1)
        output=self.ConditionalNet_Linear(output).view(bs, -1, 1, 1)
        
        log_scale, shift = utils.split_feature(output)


        dlogdet = self.Hx * self.Wx * torch.sum(log_scale, dim=(1,2,3))

        if reverse == False:
            logdet += dlogdet
            x = x + shift
            x = x * torch.exp(log_scale)
            return x, logdet
        else:
            logdet -= dlogdet
            x = x * torch.exp(-log_scale)
            x = x - shift
            return x, logdet


class ConditionalInvConv(nn.Module):
    def __init__(self, x_size, condition_size):
        super(ConditionalInvConv, self).__init__()

        Bx, Cx, self.Hx, self.Wx = x_size
        B, C, self.H, self.W = condition_size
        
        x_hidden_channels = 128
        x_hidden_size = 64
        self.ConditionalNet = nn.Sequential(
            Conv2dResize(in_size=[C, self.H, self.W], out_size=[x_hidden_channels, self.H//2, self.W//2]),
            ActFun(non_lin),
            Conv2dResize(in_size=[x_hidden_channels, self.H//2, self.W//2], out_size=[x_hidden_channels, self.H//4, self.W//4]),
            ActFun(non_lin),
            Conv2dResize(in_size=[x_hidden_channels, self.H//4, self.W//4], out_size=[x_hidden_channels, self.H//8, self.W//8]),
            ActFun(non_lin),
        )

        self.ConditionalNet_Linear = nn.Sequential(
            LinearZeros(x_hidden_channels*self.H*self.W//(8*8), x_hidden_size),
            ActFun(non_lin),
            LinearZeros(x_hidden_size, x_hidden_size),
            ActFun(non_lin),
            LinearNorm(x_hidden_size, Cx*Cx),
            nn.Tanh()
        )

    def forward(self, x, condition, logdet = None, reverse=False):
        ### WE NOTICE PROBLEMS HERE. THERE IS NOISE DUE TO W NOT BEING AN ORTHOGONAL MATRIX. 
        Bx, Cx, Hx, Wx = x.size()
        B, C, H, W = condition.size()


        bs = x.shape[0]
        W = self.ConditionalNet(condition)
        W=W.view(bs, -1)
        W=self.ConditionalNet_Linear(W).view(bs, Cx, Cx, 1)

        W = W.view(Bx, Cx, Cx)
        

        dlogdet = Hx*Wx*torch.slogdet(W)[1]

        if reverse == False:
            W = W.view(Bx, Cx, Cx, 1, 1)
        else:
            W = torch.inverse(W.double()).float().view(Bx, Cx, Cx, 1, 1)
        
        x = x.view(1, Bx*Cx, Hx, Wx)
        B_weight, C_1_weight, C_2_weight, H_weight, W_weight = W.shape
        W = W.reshape(B_weight*C_1_weight, C_2_weight, H_weight, W_weight)

        if reverse == False:
          x = F.conv2d(x, W, groups=Bx)
          x = x.view(Bx,Cx,Hx,Wx)
          logdet += dlogdet
        else:
          x = F.conv2d(x, W, groups=Bx)
          x = x.view(Bx,Cx,Hx,Wx)
          logdet -= dlogdet
        return x, logdet


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, x_size, condition_size):
        super(ConditionalAffineCoupling, self).__init__()
        
        Bx, Cx, Hx, Wx = x_size
        B, C, H, W = condition_size
        stride = [H//Hx, W//Wx]
        kernel = [H - (Hx - 1) * stride[0], W - (Wx - 1) * stride[1]]
        
        hidden_channels = 256

        self.resize = nn.Sequential(
            Conv2dZeros(C, 16),
            ActFun(non_lin),
            Conv2dResize((16, H, W), out_size=(Cx, Hx, Wx)),
            ActFun(non_lin),
            Conv2dZeros(Cx, Cx//2),
            ActFun(non_lin),
        )

        self.f = nn.Sequential(
            Conv2dNormy(Cx, hidden_channels),
            ActFun(non_lin),
            Conv2dNormy(hidden_channels, hidden_channels, kernel_size=[1, 1]),
            ActFun(non_lin),
            Conv2dZerosy(hidden_channels, Cx),
            nn.Tanh()
        )

    def forward(self, x, condition, logdet=0.0, reverse=False):
        
        z1, z2 = utils.split_feature(x, "split")
        condition = self.resize(condition)
        x = torch.cat((z1, condition), dim=1)
        
        x = self.f(x)

        scale, shift = utils.split_feature(x, "cross")
        scale = torch.sigmoid(scale + 2.)

        if reverse == False:
            z2 = z2 + shift
            z2 = z2 * scale
            logdet += torch.sum(torch.log(scale), dim=(1, 2, 3))
        else:
            z2 = z2 / scale
            z2 = z2 - shift
            logdet -= torch.sum(torch.log(scale), dim=(1, 2, 3)) 

        x = torch.cat((z1, z2), dim=1)
        return x, logdet


class GlowStep(nn.Module):
    def __init__(self, x_size, condition_size):
      super(GlowStep, self).__init__()
            
      self.actnorm = ConditionalActNorm(x_size, condition_size)
      self.invconv =  ConditionalInvConv(x_size, condition_size)
      self.affine =  ConditionalAffineCoupling(x_size, condition_size)
       

    def forward(self, x, condition, logdet=None, reverse=False):
        if reverse == False:
            x, logdet = self.actnorm(x, condition, logdet, reverse=False)
            x, logdet = self.invconv(x, condition, logdet, reverse=False)
            x, logdet = self.affine(x, condition, logdet, reverse=False)
            return x, logdet #z here
        else:
            x, logdet = self.affine(x, condition, logdet, reverse=True)
            x, logdet = self.invconv(x, condition, logdet, reverse=True)
            x, logdet = self.actnorm(x, condition, logdet, reverse=True)
            return x, logdet

class Squeeze2d(nn.Module):
    def __init__(self):
        super(Squeeze2d, self).__init__()
        
    def forward(self, x, undo_squeeze = False):
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
    def __init__(self, x_size):
      super(Split2d, self).__init__()

      Bx, Cx, Hx, Wx = x_size
      self.conv = nn.Sequential(
          Conv2dZeros(Cx // 2, Cx),
          #ActFun(non_lin),  # Some include tanh or none here, we found leakyrelu to work better
          )
      

    def forward(self, x, logdet=0.0, reverse=False):
        if reverse == False:
            z1, z2 = utils.split_feature(x, "split")
            out = self.conv(z1)
            mean, log_scale = utils.split_feature(out, "cross")
            logdet += torch.sum(td.Normal(mean, torch.exp(log_scale)).log_prob(z2), dim=(1,2,3))
            return z1, logdet
        else:
            mean, log_scale = utils.split_feature(self.conv(x), "cross")
            z2 = td.Normal(mean, torch.exp(log_scale)).rsample()
            z = torch.cat((x, z2), dim=1)
            return z, logdet

class GlowConditional(nn.Module):
    def __init__(self, x_size, condition_size, K = 4, L = 2, learn_prior = True):
        super(GlowConditional, self).__init__()

        self.L = L
        self.K = K
        Bx, Cx, Hx, Wx = x_size
        B, C, H, W = condition_size
        
        layers = []

        # Set up structure
        for l in range(0, L):
            Cx, Hx, Wx = Cx * 4, Hx // 2, Wx // 2
            x_size =  [Bx, Cx, Hx, Wx]
            layers.append(Squeeze2d())

            for i in range(0, K):
                layers.append(GlowStep(x_size, condition_size))
            
            if l < (L-1):
                layers.append(Split2d(x_size)) 
                Cx = Cx // 2 
                x_size = [Bx, Cx, Hx, Wx]

        self.glow_frame = nn.ModuleList(layers)

        self.learn_prior = learn_prior
        if learn_prior:
          self.new_mean = nn.Parameter(torch.zeros([Cx, Hx, Wx])).to(device)
          self.new_logs = nn.Parameter(torch.zeros([Cx, Hx, Wx])).to(device)
        else:
          self.new_mean = torch.zeros([Cx, Hx, Wx]).to(device)
          self.new_logs = torch.zeros([Cx, Hx, Wx]).to(device)

    def set_prior(self):
        return self.new_mean, torch.exp(self.new_logs)

    def g(self, z, condition, logdet=0.0):
        # maps z -> x
        x=z
        for step in reversed(self.glow_frame):
          if isinstance(step, Squeeze2d):
            x = step(x, undo_squeeze=True)
          elif isinstance(step, Split2d):
            x, _ = step(x, logdet = logdet, reverse=True)
          else:
            x, _ = step(x, condition=condition, logdet = logdet, reverse=True)
        return x, logdet

    def f(self, x, condition, logdet=0.0):
        # maps x -> z
        z = x
        for step in self.glow_frame:
            if isinstance(step, Squeeze2d):
                z = step(z)
            elif isinstance(step, Split2d):
                z, logdet = step(z, logdet=logdet, reverse=False)
            else:
                z, logdet = step(z, condition=condition, logdet=logdet, reverse=False)
        return z, logdet
    
    def log_prob(self, x, condition):
        x, log_det = self.uniform_binning_correction(x)
        condition, _ = self.uniform_binning_correction(condition)

        dims = torch.prod(torch.tensor(x.shape[1:]))
        z, obj = self.f(x, condition, log_det)
        mean, scale = self.set_prior()
        prior = td.Normal(mean, scale)

        obj += torch.sum(prior.log_prob(z), [1, 2, 3]) #p_z
        obj = torch.mean(obj)

        nll = (-obj) / float(np.log(2.) * dims)

        return z, nll

    def sample(self, z, condition, num_samples):
        temperature = 0.8
        with torch.no_grad():
          mean, scale = self.set_prior()
          prior = td.Normal(mean, scale*temperature)
          if z == None:
            z = prior.sample([num_samples]).to(device)
          x, logdet = self.g(z, condition)
        return x, logdet

    def uniform_binning_correction(self, x, n_bits=8):
        b, c, h, w = x.size()
        n_bins = 2 ** n_bits
        chw = c * h * w
        x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)
        objective = -np.log(n_bins) * chw * torch.ones(b, device=x.device)
        return x, objectives) / hwc ) * torch.ones(batch_size).to(device)
      return x, objective
