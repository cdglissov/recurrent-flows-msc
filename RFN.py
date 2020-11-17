from glow import ListGlow
import torch
import torch.nn as nn
from math import log, pi, exp
from utils import *
from modules import *

import torch.distributions as td

#device = set_gpu(True)

class RFN(nn.Module):
    def __init__(self,batch_size=64):
      super(RFN, self).__init__()

      self.batch_size = batch_size
      self.u_dim = (batch_size, 1, 32, 32)
      self.x_dim = (batch_size, 1, 32, 32)
      self.h_dim = 100
      self.z_dim = 30
      self.beta = 1
      scaler = 1 # Chooses the scaling of 'conv' and 'deconv' layers, default is 2
      self.L = 4
      self.K = 8
      norm_type = "none"
      context_dim = 128 # Channel output of feature extractor
      condition_size_list = []
      
      # Tip: Use 2 convs between each pool and only 1 conv between each strided conv
      # Each 'conv' will multiply channels by 2 and each deconv will divide by 2.
      down_structure = [32, 'conv', 32, 'conv', 64, 'conv', 64, 'conv'] + [context_dim]
      up_structure = [[128], ['deconv', 64], ['deconv', 64], ['deconv',32]]
      
      # adjust channel dims to match up_structure. Reversed.
      channel_dims = [i[-1] for i in up_structure][::-1] #[32, 64, 64, 128]
      hu, wu = (self.u_dim[2], self.u_dim[3])
      for i in range(0, self.L):
        hu, wu = (hu//2, hu//2)
        condition_size_list.append([batch_size, channel_dims[i], hu, wu])
      
      self.h_0 = nn.Parameter(torch.zeros(batch_size, self.h_dim, hu, wu))
      self.c_0 = nn.Parameter(torch.zeros(batch_size, self.h_dim, hu, wu))
      
      # TODO: Maybe adjust to take the output from extractor [:, :, 1, 1] and only use this, then upscale for glow
      self.z_0 = nn.Parameter(torch.zeros(batch_size, self.z_dim, hu, wu))
      self.z_0x = nn.Parameter(torch.zeros(batch_size, self.z_dim, hu, wu))

      # Feature extractor and upscaler for flow
      self.extractor = VGG_downscaler(down_structure, in_channels = self.x_dim[1], 
                                      norm_type = norm_type, non_lin = "leakyrelu", scale = scaler)
      self.upscaler = VGG_upscaler(up_structure, L=self.L, in_channels = self.h_dim + self.z_dim, 
                                   norm_type = norm_type, non_lin = "relu", scale = scaler)

      # ConvLSTM
      self.lstm = ConvLSTM(in_channels = context_dim, hidden_channels=self.h_dim, 
                           kernel_size=[3, 3], bias=True, peephole=True)

      # Prior
      prior_struct = [128]
      self.prior = SimpleParamNet(prior_struct, in_channels = self.h_dim + self.z_dim, 
                                  out_channels = self.z_dim, norm_type = norm_type, non_lin = "leakyrelu")
      
      # Flow
      base_dim = (batch_size, self.h_dim + self.z_dim, hu, wu)
      self.flow = ListGlow(self.x_dim, condition_size_list, base_dim, K=self.K, L=self.L, 
                           learn_prior = True)

      # Variational encoder
      enc_struct = [256, 128]
      self.encoder = SimpleParamNet(enc_struct, in_channels = context_dim + self.h_dim + self.z_dim, 
                                    out_channels = self.z_dim, norm_type = norm_type, non_lin = "leakyrelu")
      
      # Bookkeeping
      self.book = {"zt": 0, "b": 0, "enc_loc": 0, "ht": 0, "kl": 0, "nll": 0}
    
    def get_inits(self):
      loss = 0
      kl_loss = 0
      nll_loss = 0
      return self.h_0, self.c_0, self.z_0, self.z_0x, loss, kl_loss, nll_loss

    def loss(self, x, logdet):
      assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
      hprev, cprev, zprev, zxprev, loss, kl_loss, nll_loss = self.get_inits()
      t = x.shape[1]

      features = []
      for i in range(0, t):
        features.append(self.extractor(x[:, i, :, :, :]))
      
      for i in range(1, t):
        condition = features[i-1]
        x_feature = features[i]

        _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev) 
        # TODO: maybe try to make another LSTM but only for the prior.

        prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
        dist_prior = td.Normal(prior_mean, prior_std)
        zt = dist_prior.rsample()

        # Try to flatten zt?
        enc_mean, enc_std = self.encoder(torch.cat((ht, zxprev, x_feature), dim = 1))
        dist_enc = td.Normal(enc_mean, enc_std)
        zxt = dist_enc.rsample()
        
        # Maybe try to split so base conditions and flow conditions have their own input seperately.
        flow_conditions = self.upscaler(torch.cat((zxt, ht), dim = 1))
        base_conditions = torch.cat((zxt, ht), dim = 1)
        
        b, nll = self.flow.log_prob(x[:, i, :, :, :], flow_conditions, base_conditions, logdet)
        
        # TODO: Probably shouldn't divide by dims_z
        dims_z = torch.prod(torch.tensor(zt.shape[1:]))
        kl_loss = kl_loss + td.kl_divergence(dist_enc, dist_prior).sum([1,2,3]).mean() / dims_z
        nll_loss = nll_loss + nll 

        hprev, cprev, zprev, zxprev = ht, ct, zt, zxt
      
      self.book["zt"] = zt.detach()
      self.book["b"] = b.detach()
      self.book["enc_loc"] = enc_mean.detach()
      self.book["ht"] = ht.detach()
      self.book["kl"] = kl_loss.detach()
      self.book["nll"] = nll_loss.detach()

      loss = (self.beta * kl_loss) + nll_loss
      return loss

    def sample(self, x, n_predictions=6, temperature = 0.8, encoder_sample = False):
      assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
      hprev, cprev, zprev, zxprev, _, _, _ = self.get_inits()
      t = x.shape[1]

      samples = torch.zeros((t-1, *x[:,0,:,:,:].shape))
      samples_recon = torch.zeros((t-1, *x[:,0,:,:,:].shape))

      features = []
      for i in range(0, t):
        features.append(self.extractor(x[:, i, :, :, :]))
      
      for i in range(1, t):
        condition = features[i-1]
        x_feature = features[i]
        
        _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)

        enc_mean, enc_std = self.encoder(torch.cat((zxprev, ht, x_feature), dim = 1))
        dist_enc = td.Normal(enc_mean, enc_std)
        zxt = dist_enc.rsample()

        prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
        dist_prior = td.Normal(prior_mean, prior_std)
        zt = dist_prior.rsample()

        if encoder_sample:
          flow_conditions = self.upscaler(torch.cat((zxt, ht), dim = 1))
          base_conditions = torch.cat((zxt, ht), dim = 1)
          zxprev = zxt
        else:
          flow_conditions = self.upscaler(torch.cat((zt, ht), dim = 1))
          base_conditions = torch.cat((zt, ht), dim = 1)
          zprev = zt
        
        sample = self.flow.sample(None, flow_conditions, base_conditions, temperature)
        z, _ = self.flow.log_prob(x[:, i, :, :, :], flow_conditions, base_conditions, 0.0)
        sample_recon = self.flow.sample(z, flow_conditions, base_conditions, temperature)
        
        hprev, cprev = ht, ct

        samples[i-1,:,:,:,:] = sample.detach()
        samples_recon[i-1,:,:,:,:] = sample_recon.detach()
      
      # Make predictions
      predictions = torch.zeros((n_predictions, *x[:,0,:,:,:].shape))
      for i in range(0, n_predictions):
        condition = self.extractor(sample)
        _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)
        prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
        dist_prior = td.Normal(prior_mean, prior_std)
        zt = dist_prior.rsample()

        flow_conditions = self.upscaler(torch.cat((zt, ht), dim = 1))
        base_conditions = torch.cat((zt, ht), dim = 1)
        prediction = self.flow.sample(None, flow_conditions, base_conditions, temperature)
        predictions[i-1,:,:,:,:] = prediction.detach()

        hprev, cprev, zprev, zxprev = ht, ct, zt, zxt
      return samples, samples_recon, predictions
