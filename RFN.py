from .glow import GlowConditional
import torch
import torch.nn as nn
from math import log, pi, exp
from .utils import *
from .layers import ConvLSTM
import torch.distributions as td

device = set_gpu(True)

class Model(nn.Module):
    def __init__(self, L = 2, K = 4):
      super(Model, self).__init__()

      self.batch_size = 64
      self.u_dim = (self.batch_size, 1, 32, 32)
      self.x_dim = (self.batch_size, 1, 32, 32)
      self.h_dim = 128
      self.z_dim = 64
      self.beta = 1
      self.L = L
      self.K = K
      # Settings #
      self.extract = True
       
      # Transition constructor
      out_dim_u = 32
      out_dim_x = out_dim_u

      # Smallest feature we can have is H, W = 16, 16, due to affine coupling. This condition_extractor is made for
      # H, W = 32, 32 only.
      h_c = [64]
      self.condition_extractor = nn.Sequential(
          nn.Conv2d(self.u_dim[1], h_c[0], kernel_size=3, stride=2, padding=1), 
          nn.ReLU(),
          nn.Conv2d(h_c[0], out_dim_u, kernel_size=3, stride=1, padding=1) 
        ).to(device)
      
      if self.extract:
        # Get dimensions from condition_extractor output
        hu, wu = get_layer_size([self.u_dim[2], self.u_dim[3]], kernels=[3, 3], 
                                    paddings=[1, 1], strides=[2, 1],
                                    dilations=[1, 1])
        self.u_dim = (self.batch_size, out_dim_u, hu, wu)
      
      # Define LSTM, eq. 11. TODO: Maybe enable ConvLSTM to take different kernel size?
      self.h_0 = nn.Parameter(torch.zeros(self.batch_size, self.h_dim, self.u_dim[2], self.u_dim[3])).to(device)
      self.c_0 = nn.Parameter(torch.zeros(self.batch_size, self.h_dim, self.u_dim[2], self.u_dim[3])).to(device)
      self.lstm = ConvLSTM(in_channels = self.u_dim[1], hidden_channels=self.h_dim, 
                         kernel_size=[3, 3], bias=True, peephole=True).to(device)


      # Define zt and net for zt (prior), eq. 12
      self.z_0 = nn.Parameter(torch.zeros(self.batch_size, self.z_dim, self.u_dim[2], self.u_dim[3]).to(device))
      
      h_c = [64, 64]
      self.prior = nn.Sequential(
        nn.Conv2d(self.h_dim + self.z_dim, h_c[0],  kernel_size=3, stride=1, padding=1),  
        nn.ReLU(),
        nn.Conv2d(h_c[0], h_c[1],  kernel_size=3, stride=1, padding=1),  
        nn.ReLU(),
        ).to(device)
      self.prior_mean =  nn.Conv2d(h_c[1], self.z_dim,  kernel_size=3, stride=1, padding=1).to(device)
      self.prior_std =  nn.Sequential(nn.Conv2d(h_c[1], self.z_dim,  kernel_size=3, stride=1, padding=1), nn.Softplus()).to(device)

      # Flow decoder
      # build prior conditioner, eq. 13. The prior needs to match b's dimensions, i.e. the output
      # of f(x,condition), so we get the dimensions of b first:
      Bx, Cx, Hx, Wx = self.x_dim
      for l in range(0, self.L):
          Cx, Hx, Wx = Cx * 4, Hx // 2, Wx // 2
          if l < (self.L-1):
              Cx = Cx // 2 
      _, _, Hu, Wu = self.u_dim
      
      # we find the appropriate downsampling.
      stride_cond_prior = [Hu//Hx, Wu//Wx]
      kernel_cond_prior = [Hu - (Hx - 1) * stride_cond_prior[0], Wu - (Wx - 1) * stride_cond_prior[1]]

      # Define conditional prior
      h_c = [64, 64]
      self.conditional_prior = nn.Sequential(
          nn.Conv2d(in_channels = self.h_dim + self.z_dim, out_channels = h_c[0], kernel_size=3, stride=1, padding = 1), 
          nn.ReLU(),
          nn.Conv2d(h_c[0], h_c[1],  kernel_size=kernel_cond_prior, stride=stride_cond_prior, padding=0),  
          nn.ReLU(),
      ).to(device)
      self.conditional_prior_mean =  nn.Conv2d(h_c[1], Cx,  kernel_size=3, stride=1, padding=1).to(device)
      self.conditional_prior_std =  nn.Sequential(nn.Conv2d(h_c[1], Cx,  kernel_size=3, stride=1, padding=1), nn.Softplus()).to(device)

      # Define flow, eq. 14
      # this is simply ht and zt concatenated dimension

      flow_condition_dim = (self.batch_size, self.z_dim + self.h_dim, Hu, Wu)
      self.flow = GlowConditional(self.x_dim, flow_condition_dim, K=self.K, L=self.L).to(device)

      # Now eq. 15. To concat ht, zt-1 and xt we need to downscale x_t first to match dimensions
      stride_x_extr = [self.x_dim[2] // self.u_dim[2], self.x_dim[3] // self.u_dim[3]]
      kernel_x_extr = [self.x_dim[2] - (self.u_dim[2] - 1) * stride_x_extr[0], self.x_dim[3] - (self.u_dim[3] - 1) * stride_x_extr[1]]

      # Define Encoder. This should spit out the same dimensions as find in your prior.
      h_c = [128, 64]
      #self.encoder = SimpleResnet(in_channels = out_dim_x + self.h_dim + self.z_dim, out_channels = h_c[1], n_blocks = 3, n_filters = 128, norm_type = "bn").to(device)
      
      self.encoder=nn.Sequential(
          nn.Conv2d(in_channels = out_dim_x + self.h_dim + self.z_dim, out_channels = h_c[0], kernel_size=3, stride=1, padding = 1), 
          nn.ReLU(),
          nn.Conv2d(h_c[0], h_c[1],  kernel_size=3, stride=1, padding=1), 
          nn.ReLU(),
      ).to(device)

      self.encoder_mean =  nn.Conv2d(h_c[1], self.z_dim,  kernel_size=3, stride=1, padding=1).to(device)
      self.encoder_std =  nn.Sequential(nn.Conv2d(h_c[1], self.z_dim,  kernel_size=3, stride=1, padding=1), nn.Softplus()).to(device)
      self.z_enc_0 = nn.Parameter(torch.zeros(self.batch_size, self.z_dim, self.u_dim[2], self.u_dim[3]).to(device))

      # For book keeping
      self.book = {"zt": 0, "b": 0, "b_loc": 0, "enc_loc": 0, "ht": 0, "kl": 0, "nll": 0}


    def loss(self, xt):
      hprev = self.h_0
      cprev = self.c_0
      zprev = self.z_0
      zprev_enc = self.z_enc_0 

      loss = 0
      kld_loss = 0
      nll_loss = 0

      t = xt.shape[1]

      features = []
      for i in range(0, t):
        features.append(self.condition_extractor(xt[:, i, :, :, :]))

      

      for i in range(1, t):
        u_feature = features[i-1]
        x_feature = features[i]
        x = xt[:, i, :, :, :]
        # Find states
        _, ht, ct = self.lstm(u_feature.unsqueeze(1), hprev, cprev) 
        
        
        # Sample zt from prior
        out = self.prior(torch.cat((zprev, ht), dim = 1))
        prior_mean = self.prior_mean(out)
        prior_std = self.prior_std(out)

        zt = td.Independent(td.Normal(prior_mean, prior_std), 0).rsample()
        
        
        # Start inference
        encoder_input = torch.cat((zprev_enc, ht, x_feature), dim = 1)
        out = self.encoder(encoder_input)
        enc_mean = self.encoder_mean(out)
        enc_std = self.encoder_std(out)
        zt_enc = td.Independent(td.Normal(enc_mean, enc_std), 0).rsample()
        

        # eq. 13
        out = self.conditional_prior(torch.cat((zt, ht), dim = 1))
        b_mean = self.conditional_prior_mean(out)
        b_std = self.conditional_prior_std(out)
        
        
        # set prior for Glow
        self.flow.set_prior(b_mean, b_std)
        flow_conditions = torch.cat((zt, ht), dim = 1)
        
        b, nll = self.flow.log_prob(x, flow_conditions)


        
        dims_x = torch.prod(torch.tensor(x_feature.shape[1:]))
        kld_loss += (self.beta) * self._kld_gauss(enc_mean, enc_std, prior_mean, prior_std).mean() #/ float(np.log(2.) * dims_x)
        nll_loss += nll 
        
        hprev = ht
        cprev = ct
        zprev = zt
        zprev_enc = zt_enc
        
      # Book keeping
      self.book["zt"] = zt.detach()
      self.book["b"] = b.detach()
      self.book["b_loc"] = b_mean.detach()
      self.book["enc_loc"] = enc_mean.detach()
      self.book["ht"] = ht.detach()
      self.book["kl"] = kld_loss.detach()
      self.book["nll"] = nll_loss.detach()

      loss = kld_loss + nll_loss
      return loss

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
      kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
        (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
        std_2.pow(2) - 1)
      return	0.5 * kld_element.sum([1,2,3])

    def sample(self, xt, sample_type = "sample", num_predictions = 6):
      samples = []
      predictions = []
      temperature = 0.8
      hprev = self.h_0
      cprev = self.c_0
      zprev = self.z_0

      t = xt.shape[1]

      features = []
      for i in range(0, t):
          features.append(self.condition_extractor(xt[:, i, :, :, :]))


      for i in range(1, t):
        u_feature = features[i-1]
        x_feature = features[i]                     
        x = xt[:, i, :, :, :]
        # Find states
        _, ht, ct = self.lstm(u_feature.unsqueeze(1), hprev, cprev) 
        
        # Sample zt from prior
        out = self.prior(torch.cat((zprev, ht), dim = 1))
        prior_mean = self.prior_mean(out)
        prior_std = self.prior_std(out)

        zt = td.Independent(td.Normal(prior_mean, prior_std),0).sample()
        
        # eq. 13
        out = self.conditional_prior(torch.cat((zt, ht), dim = 1))
        b_mean = self.conditional_prior_mean(out)
        b_std = self.conditional_prior_std(out)*temperature

        # set prior for Glow
        self.flow.set_prior(b_mean, b_std)
        flow_conditions = torch.cat((zt, ht), dim = 1)
        
        if sample_type == "sample":
          sample, _ = self.flow.sample(None, flow_conditions)
          samples.append(sample)
        else:
          b, _ = self.flow.log_prob(x, flow_conditions)
          recon, _ = self.flow.sample(b, flow_conditions)
          samples.append(recon)
        
        hprev = ht
        cprev = ct
        zprev = zt

      if num_predictions > 0:
        u = sample
        predictions = []
        for k in range(0, num_predictions):
          u_feature = self.condition_extractor(u) 
          _, ht, ct = self.lstm(u_feature.unsqueeze(1), hprev, cprev) 
          out = self.prior(torch.cat((zprev, ht), dim = 1))
          prior_mean = self.prior_mean(out)
          prior_std = self.prior_std(out)
          zt = td.Independent(td.Normal(prior_mean, prior_std), 0).sample()
          out = self.conditional_prior(torch.cat((zt, ht), dim = 1))
          b_mean = self.conditional_prior_mean(out)
          b_std = self.conditional_prior_std(out)*temperature
          self.flow.set_prior(b_mean, b_std)
          flow_conditions = torch.cat((zt, ht), dim = 1)
          prediction, _ = self.flow.sample(None, flow_conditions)
          hprev = ht
          cprev = ct
          zprev = zt
          predictions.append(prediction)
          u = prediction
      
      return samples, predictions
