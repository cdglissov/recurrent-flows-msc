import torch
import torch.nn as nn
from .utils import *
from math import log, pi, exp
import torch.distributions as td
from .layers import (Conv2dZeros,Conv2dZerosy,ActNorm,Conv2dNormy, ConvLSTM)
import torch.nn.functional as F

device = set_gpu(True)


class GlowStep(nn.Module):
    def __init__(self, x_size, condition_size, LU_decompose):
      super(GlowStep, self).__init__()
      
      b, c, h, w = x_size
      bc, cc, hc, wc = condition_size
      self.actnorm = ActNorm(c)
      self.invconv =  InvConv(c, LU_decomposed = LU_decompose)
      self.affine =  AffineCoupling(x_size, condition_size)
       
    def forward(self, x, condition, logdet, reverse):
        if reverse == False:
            x, logdet = self.actnorm(x, logdet, reverse=False)
            x, logdet = self.invconv(x, logdet, reverse=False)
            x, logdet = self.affine(x, condition, logdet, reverse=False)
            return x, logdet
        else:
            x, logdet = self.affine(x, condition, logdet, reverse=True)
            x, logdet = self.invconv(x, logdet, reverse=True)
            x, logdet = self.actnorm(x, logdet, reverse=True)
            return x, logdet


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
            z1, z2 = utils.split_feature(x, "split")
            h = torch.cat([z1, condition], dim=1)
            out = self.conv(h)
            mean, log_scale = utils.split_feature(out, "cross")
            if logdet is not None:
              logdet = logdet + torch.sum(td.Normal(mean, torch.exp(log_scale)).log_prob(z2), dim=(1,2,3))
            return z1, logdet
        else:
            h = torch.cat([x, condition], dim=1)
            mean, log_scale = utils.split_feature(self.conv(h), "cross")
            z2 = td.Normal(mean, torch.exp(log_scale)).rsample()
            z = torch.cat((x, z2), dim=1)
            return z, logdet

class ListGlow(nn.Module):
    def __init__(self, x_size, condition_size, base_dist_size, K = 12, L = 2, learn_prior = True, LU_decompose = True):
        super(ListGlow, self).__init__()
        
        assert isinstance(condition_size, list), "condition_size is not a list, make sure it fits L"

        self.L = L
        self.K = K
        Bx, Cx, Hx, Wx = x_size
        Bc, Cc, Hc, Wc = base_dist_size
        layers = []
        
        for l in range(0, L):
            layers.append(Squeeze2d())
            Cx, Hx, Wx = Cx * 4, Hx // 2, Wx // 2
            x_size =  [Bx, Cx, Hx, Wx]
            
            condition_size_cur = condition_size[l]

            for i in range(0, K):
                layers.append(GlowStep(x_size, condition_size_cur, LU_decompose))
            
            if l < (L-1):
                layers.append(Split2d(x_size, condition_size_cur)) 
                Cx = Cx // 2 
                x_size = [Bx, Cx, Hx, Wx]

        self.glow_frame = nn.ModuleList(layers)

        self.learn_prior = learn_prior
        if learn_prior == True:
          # TODO: We could try to use the Convnorm here, make more powerful (maybe)
          self.prior = nn.Sequential(
            Conv2dNorm(Cc, Cc//2),
            ActFun("leakyrelu"),
            Conv2dNorm(Cc//2, Cc//4),
            ActFun("leakyrelu"),
            Conv2dZeros(in_channel=Cc//4, out_channel=2*Cx),
            )
        else:
          self.prior_in = torch.zeros([1, 2*Cx, Hx, Wx,]).to(device)
          

    def g(self, z, condition, logdet):
        # maps z -> x
        x = z
        l = len(condition)-1
        for step in reversed(self.glow_frame):
          if isinstance(step, Squeeze2d):
            x = step(x, undo_squeeze=True)
          elif isinstance(step, Split2d):
            l = l-1
            x, logdet = step(x, condition[l], logdet = logdet, reverse=True)
          else:
            x, logdet = step(x, condition[l], logdet = logdet, reverse=True)
        return x, logdet


    def f(self, x, condition, logdet):
        # maps x -> z
        z = x
        l=0
        for step in self.glow_frame:
            if isinstance(step, Squeeze2d):
                z = step(z, undo_squeeze=False)
            elif isinstance(step, Split2d):
                z, logdet = step(z, condition[l], logdet = logdet, reverse=False)
                l = l+1
            else:
                z, logdet = step(z, condition[l], logdet=logdet, reverse=False)
        return z, logdet
    
    def log_prob(self, x, condition, base_condition, logdet = None):
        
        assert isinstance(condition, list), "Condition is not a list, make sure it fits L"
        dims = torch.prod(torch.tensor(x.shape[1:]))
        z, obj = self.f(x, condition, logdet)
        
        z_in = base_condition

        if self.learn_prior:
          mean, log_scale = utils.split_feature(self.prior(z_in), type="split")
        else:
          mean, log_scale = utils.split_feature(self.prior_in.repeat(x.shape[0],1,1,1), type="split")
          
        prior = td.Normal(mean, torch.exp(log_scale))
        obj = obj + torch.sum(prior.log_prob(z), dim=(1,2,3)) #p_z
        obj = torch.mean(obj)
        nll = (-obj) / float(np.log(2.) * dims)
        return z, nll

    def sample(self, z, condition, base_condition, num_samples = 32, temperature=0.8):
    
        with torch.no_grad():
          if z == None:

            z_in = base_condition
            if self.learn_prior:
              mean, log_scale = utils.split_feature(self.prior(z_in), type="split")
            else:
              mean, log_scale = utils.split_feature(self.prior_in.repeat(num_samples,1,1,1), type="split")

            prior = td.Normal(mean, torch.exp(log_scale)*temperature)
            z = prior.sample().to(device)
          x, _ = self.g(z, condition, logdet=None)
        return x
