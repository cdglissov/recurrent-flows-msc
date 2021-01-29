import torch
import torch.nn as nn
from utils import split_feature, set_gpu
import torch.distributions as td
from glow_modules import (ActNorm,Conv2dZeros,Conv2dNorm,InvConv, AffineCoupling,Squeeze2d,Split2d)
from modules import ActFun
import numpy as np

device = set_gpu(True)

class GlowStep(nn.Module):
    def __init__(self, x_size, condition_size, args):
      super(GlowStep, self).__init__()
      LU_decomposed = args.LU_decomposed
      self.n_units_affine = args.n_units_affine
      b, c, h, w = x_size
      bc, cc, hc, wc = condition_size
      self.actnorm = ActNorm(c)
      self.invconv =  InvConv(c, LU_decomposed = LU_decomposed)
      self.affine =  AffineCoupling(x_size, condition_size, hidden_units = self.n_units_affine)
       
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

class Squeeze2dPrior(nn.Module):
    def __init__(self):
        super(Squeeze2dPrior, self).__init__()
        
    def forward(self, x):
      B, C, H, W = x.shape
      x = x.reshape(B, C // 4, 2, 2, H, W)
      x = x.permute(0, 1, 4, 2, 5, 3)
      x = x.reshape(B, C // 4, H * 2, W * 2)
      return x
  
class ListGlow(nn.Module):
    def __init__(self, x_size, condition_size, base_dist_size, args):
        super(ListGlow, self).__init__()
        
        assert isinstance(condition_size, list), "condition_size is not a list, make sure it fits L"
        self.learn_prior = args.learn_prior
        self.n_units_prior = args.n_units_prior
        self.make_conditional = args.make_conditional
        L = args.L
        K = args.K
        Bx, Cx, Hx, Wx = x_size
        Bc, Cc, Hc, Wc = base_dist_size
        layers = []
        
        for l in range(0, L):
            layers.append(Squeeze2d())
            Cx, Hx, Wx = Cx * 4, Hx // 2, Wx // 2
            x_size =  [Bx, Cx, Hx, Wx]
            
            condition_size_cur = condition_size[l]

            for i in range(0, K):
                layers.append(GlowStep(x_size, condition_size_cur, args))
            
            if l < (L-1):
                layers.append(Split2d(x_size, condition_size_cur, self.make_conditional)) 
                Cx = Cx // 2 
                x_size = [Bx, Cx, Hx, Wx]

        self.glow_frame = nn.ModuleList(layers)

        if self.learn_prior == True:
          # TODO: We could try to use the Convnorm here, make more powerful (maybe)
          self.prior = nn.Sequential(
            Conv2dNorm(Cc, self.n_units_prior),
            ActFun("leakyrelu"),
            Squeeze2dPrior(),
            Conv2dNorm(self.n_units_prior//4, self.n_units_prior//4),
            ActFun("leakyrelu"),
            Conv2dZeros(in_channel=self.n_units_prior//4, out_channel=2*Cx),
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
          mean, log_scale = split_feature(self.prior(z_in), type="split")
        else:
          mean, log_scale = split_feature(self.prior_in.repeat(x.shape[0],1,1,1), type="split")
          
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
              mean, log_scale = split_feature(self.prior(z_in), type="split")
            else:
              mean, log_scale = split_feature(self.prior_in.repeat(num_samples,1,1,1), type="split")

            prior = td.Normal(mean, torch.exp(log_scale)*temperature)
            z = prior.sample().to(device)
          x, _ = self.g(z, condition, logdet=None)
        return x
