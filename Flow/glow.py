import torch
import torch.nn as nn
from Utils import split_feature, set_gpu
import torch.distributions as td
from .glow_modules import ActNorm, Conv2dZeros, Conv2dNorm, InvConv, AffineCoupling, Squeeze2d, Split2d, BatchNormFlow
from Utils import ActFun

device = set_gpu(True)

class GlowStep(nn.Module):
    def __init__(self, x_size, condition_size, args):
      super(GlowStep, self).__init__()
      LU_decomposed = args.LU_decomposed
      self.n_units_affine = args.n_units_affine
      self.non_lin_glow = args.non_lin_glow
      self.clamp_type = args.clamp_type
      norm_type = args.flow_norm
      momentum = args.flow_batchnorm_momentum
      b, c, h, w = x_size
      bc, cc, hc, wc = condition_size
      if norm_type == 'batchnorm':
          self.norm = BatchNormFlow(x_size, momentum = momentum)
      else:
          self.norm = ActNorm(c)
      self.invconv =  InvConv(c, LU_decomposed = LU_decomposed)
      self.affine =  AffineCoupling(x_size, condition_size, 
                                    hidden_units = self.n_units_affine, 
                                    non_lin = self.non_lin_glow, 
                                    clamp_type = self.clamp_type)
       
    def forward(self, x, condition, logdet, reverse):
        if reverse == False:
            x, logdet = self.norm(x, logdet, reverse=False)
            x, logdet = self.invconv(x, logdet, reverse=False)
            x, logdet = self.affine(x, condition, logdet, reverse=False)
            return x, logdet
        else:
            x, logdet = self.affine(x, condition, logdet, reverse=True)
            x, logdet = self.invconv(x, logdet, reverse=True)
            x, logdet = self.norm(x, logdet, reverse=True)
            return x, logdet

class ListGlow(nn.Module):
    def __init__(self, x_size, condition_size, base_dist_size, args):
        super(ListGlow, self).__init__()
        
        assert isinstance(condition_size, list), "condition_size is not a list, make sure it fits L"
        self.learn_prior = args.learn_prior
        self.n_units_prior = args.n_units_prior
        self.make_conditional = args.make_conditional
        self.base_norm = args.base_norm
        self.non_lin_glow = args.non_lin_glow
        self.L = args.L
        self.K = args.K
        Bx, Cx, Hx, Wx = x_size
        Bc, Cc, Hc, Wc = base_dist_size
        layers = []
        
        
        for l in range(0, self.L):
            layers.append(Squeeze2d())
            Cx, Hx, Wx = Cx * 4, Hx // 2, Wx // 2
            x_size =  [Bx, Cx, Hx, Wx]
            
            condition_size_cur = condition_size[l]

            for i in range(0, self.K):
                layers.append(GlowStep(x_size, condition_size_cur, args))
            
            if l < (self.L-1):
                layers.append(Split2d(x_size, condition_size_cur, self.make_conditional)) 
                Cx = Cx // 2 
                x_size = [Bx, Cx, Hx, Wx]

        self.glow_frame = nn.ModuleList(layers)

        if self.learn_prior == True:
          self.prior = nn.Sequential(
            Conv2dNorm(Cc, self.n_units_prior, norm = self.base_norm),
            ActFun(self.non_lin_glow),
            Conv2dNorm(self.n_units_prior, self.n_units_prior//2, norm = self.base_norm),
            ActFun(self.non_lin_glow),
            Conv2dZeros(in_channel=self.n_units_prior//2, out_channel=2*Cx),
            )
        else:
          self.prior_in = torch.zeros([1, 2*Cx, Hx, Wx,]).to(device)
          

    def g(self, z, condition, logdet, temperature):
        # maps z -> x
        x = z
        l = len(condition)-1
        for step in reversed(self.glow_frame):
          if isinstance(step, Squeeze2d):
            x = step(x, undo_squeeze=True)
          elif isinstance(step, Split2d):
            l = l-1
            x, logdet = step(x, condition[l], logdet = logdet, reverse = True, temperature = temperature)
          else:
            x, logdet = step(x, condition[l], logdet = logdet, reverse = True)
        return x, logdet


    def f(self, x, condition, logdet):
        # maps x -> z
        z = x
        l=0
        for step in self.glow_frame:
            if isinstance(step, Squeeze2d):
                z = step(z, undo_squeeze=False)
            elif isinstance(step, Split2d):
                z, logdet = step(z, condition[l], logdet = logdet, reverse = False)
                l = l+1
            else:
                z, logdet = step(z, condition[l], logdet = logdet, reverse = False)
        return z, logdet
    
    
    def log_prob(self, x, condition, base_condition, logdet = None):
        
        assert isinstance(condition, list), "Condition is not a list, make sure it fits L"
        z, obj = self.f(x, condition, logdet)
        
        z_in = base_condition
        if self.learn_prior:
          mean, log_scale = split_feature(self.prior(z_in), type="split")
        else:
          mean, log_scale = split_feature(self.prior_in.repeat(x.shape[0],1,1,1), type="split")
          
        prior = td.Normal(mean, torch.exp(log_scale))
        obj = obj + torch.sum(prior.log_prob(z), dim=(1,2,3)) #p_z
        obj = torch.mean(obj)
        nll = -obj
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
          x, _ = self.g(z, condition, logdet = None, temperature = temperature)
        return x
