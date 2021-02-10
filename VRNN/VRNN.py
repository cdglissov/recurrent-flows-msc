import torch
import torch.nn as nn
from Utils import get_layer_size, Flatten, UnFlatten, set_gpu, batch_reduce
import torch.distributions as td
from Utils import ConvLSTM, NormLayer
import numpy as np
from Utils import DiscretizedMixtureLogits, DiscretizedMixtureLogits_1d
device = set_gpu(True)

#https://medium.com/@aminamollaysa/summary-of-the-recurrent-latent-variable-model-vrnn-4096b52e731
class VRNN(nn.Module):
    def __init__(self, args):
      super(VRNN, self).__init__()
      
      norm_type = args.norm_type
      self.batch_size = args.batch_size
      self.u_dim = args.condition_dim
      self.x_dim = args.x_dim
      h_dim = args.h_dim
      z_dim = args.z_dim
      self.h_dim = h_dim
      self.z_dim = z_dim
      self.loss_type = args.loss_type
      self.beta = 1
      bu, cu, hu, wu = self.u_dim
      bx, cx, hx, wx = self.x_dim
      self.mse_criterion = nn.MSELoss(reduction='none')
      self.bits=args.n_bits
      self.dequantize = args.dequantize
      self.preprocess_range = args.preprocess_range
      self.n_logistics = args.n_logistics
      # Remember to downscale more when using 64x64. Overall the net should probably increase in size when using 
      # 64x64 images
      phi_x_t_channels = 256
      self.phi_x_t = nn.Sequential(
          nn.Conv2d(cx, 64, kernel_size=3, stride=2, padding=1),#32
          NormLayer(64, norm_type),
          nn.ReLU(),
          nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),#16
          NormLayer(128, norm_type),
          nn.ReLU(),
          nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),#8
          NormLayer(256, norm_type),
          nn.ReLU(),
          nn.Conv2d(256, phi_x_t_channels, kernel_size=3, stride=1, padding=1),
          NormLayer(phi_x_t_channels, norm_type),
          nn.ReLU(),
        )
      h, w = get_layer_size([hu, wu], kernels=[3, 3,3], paddings=[1, 1, 1], strides=[2, 2,2],
                        dilations=[1, 1,1])
      self.h = h
      self.w = w
      
      # Extractor of z
      phi_z_channels = 128
      self.phi_z = nn.Sequential(
        nn.Linear(z_dim, phi_z_channels*h*w),
        nn.ReLU(),
        nn.Linear(phi_z_channels*h*w, phi_z_channels*h*w),
        nn.ReLU(),
        UnFlatten(phi_z_channels, h, w),
        nn.Conv2d(phi_z_channels, phi_z_channels, kernel_size = 3, stride = 1, padding = 1),
        NormLayer(phi_z_channels, norm_type),
        nn.ReLU()
        ) #4x4
      
      # Encoder structure
      self.enc = nn.Sequential(
        nn.Conv2d(phi_x_t_channels + h_dim, 256,  kernel_size=3, stride=2, padding=1), #4
        NormLayer(256, norm_type),
        nn.ReLU(),
        Flatten(),
        )
      
      
      self.enc_mean =  nn.Sequential(
        nn.Linear((256)*h//2*w//2, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, z_dim),
        )
      
      self.enc_std = nn.Sequential(
        nn.Linear((256)*h//2*w//2, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, z_dim),
        nn.Softplus(),
        )

      # Prior structure
      self.prior = nn.Sequential( 
        nn.Conv2d(h_dim, 256, kernel_size = 3, stride = 2, padding = 1),
        NormLayer(256, norm_type),
        nn.ReLU(),
        Flatten(),
        )
      
      self.prior_mean = nn.Sequential(
        nn.Linear((256)*h//2*w//2, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, z_dim),
        )
      
      self.prior_std = nn.Sequential(
        nn.Linear((256)*h//2*w//2, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, z_dim),
        nn.Softplus()
        )
      
      # Decoder structure
      self.dec = nn.Sequential(
            nn.ConvTranspose2d(h_dim + phi_z_channels, 512, kernel_size=4, stride=2, padding=1, output_padding=0),
            NormLayer(512, norm_type),
            nn.ReLU(),
            nn.Conv2d(512, 256,  kernel_size=3, stride=1, padding=1),
            NormLayer(256, norm_type),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, output_padding=0), 
            NormLayer(64, norm_type),
            nn.ReLU(),
            nn.Conv2d(64, 64,  kernel_size=3, stride=1, padding=1),
            NormLayer(64, norm_type),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0), 
            NormLayer(32, norm_type),
            nn.ReLU(),
        )
      
      
        
      #self.dec_std = nn.Sequential(nn.Conv2d(32, cx,  kernel_size=3, stride=1, padding=1), 
       #                        nn.Softplus()
       #                        )

      if self.loss_type=="mol":
        if cx > 1:
          self.dec_mean = nn.Sequential(nn.Conv2d(32, self.n_logistics*10*cx,  kernel_size=3, stride=1, padding=1))
          self.likelihood = DiscretizedMixtureLogits(self.n_logistics)
        else:
          self.dec_mean = nn.Sequential(nn.Conv2d(32, self.n_logistics*3*cx,  kernel_size=3, stride=1, padding=1))
          self.likelihood = DiscretizedMixtureLogits_1d(self.n_logistics)
      else:
        self.variance = nn.Parameter(torch.ones([1]))
        if self.preprocess_range =='0.5':
          self.dec_mean = nn.Sequential(nn.Conv2d(32, cx,  kernel_size=3, stride=1, padding=1), 
                                  nn.Tanh()
                                  )
        elif self.preprocess_range =='1.0':
          self.dec_mean = nn.Sequential(nn.Conv2d(32, cx,  kernel_size=3, stride=1, padding=1), 
                                 nn.Sigmoid()
                                 )

      self.z_0 = nn.Parameter(torch.zeros(self.batch_size, z_dim))
      self.z_0x = nn.Parameter(torch.zeros(self.batch_size, z_dim))
      
      self.h_0 = nn.Parameter(torch.zeros(self.batch_size, self.h_dim, h, w))
      self.c_0 = nn.Parameter(torch.zeros(self.batch_size, self.h_dim, h, w))


      #LSTM
      self.lstm = ConvLSTM(in_channels = phi_x_t_channels + phi_z_channels, 
                           hidden_channels=self.h_dim, 
                           kernel_size=[3, 3], 
                           bias=True, 
                           peephole=True)


    def get_inits(self):
      loss = 0
      kl_loss = 0
      nll_loss = 0
      return self.h_0, self.c_0, self.z_0, self.z_0x, loss, kl_loss, nll_loss
    
    def uniform_binning_correction(self, x):
        n_bits = self.bits
        b, c, h, w = x.size()
        n_bins = 2 ** n_bits
        chw = c * h * w
        x_noise = x + torch.zeros_like(x).uniform_(0, 1.0 / n_bins)
        objective = -np.log(n_bins) * chw * torch.ones(b, device=x.device)
        return x_noise, objective
      
    def loss(self, xt):

      b, t, c, h, w = xt.shape
      hprev, cprev, zprev, zxprev, loss, kl_loss, nll_loss = self.get_inits()   

      for i in range(1, t):
        xt_features = self.phi_x_t(xt[:, i, :, :, :])
        ut = self.phi_x_t(xt[:, i-1, :, :, :])

        lstm_input = torch.cat([ut, self.phi_z(zxprev)], 1)
        _, ht, ct = self.lstm(lstm_input.unsqueeze(1), hprev, cprev)
        
        prior_t = self.prior(ht) 
        prior_mean_t = self.prior_mean(prior_t) 
        prior_std_t = self.prior_std(prior_t)
        prior_dist = td.Normal(prior_mean_t, prior_std_t)
        
        enc_t = self.enc(torch.cat([ht, xt_features], 1))
        enc_mean_t = self.enc_mean(enc_t)
        enc_std_t = self.enc_std(enc_t)
        enc_dist = td.Normal(enc_mean_t, enc_std_t)

        zx_t = enc_dist.rsample()

        dec_t = self.dec(torch.cat([ht, self.phi_z(zx_t)], 1))
        
        dec_mean_t = self.dec_mean(dec_t)
        
        zxprev = zx_t
        hprev = ht
        cprev = ct

        kl_loss = kl_loss + td.kl_divergence(enc_dist, prior_dist)
        if self.loss_type == "bernoulli":
            nll_loss = nll_loss - batch_reduce(td.Bernoulli(probs=dec_mean_t).log_prob(xt[:, i, :, :, :]))
        elif self.loss_type == "gaussian":
          
            if self.dequantize:
              x, nll_unif = self.uniform_binning_correction(xt[:, i, :, :, :])
            else:
              x = xt[:, i, :, :, :]
            
            nll_loss = nll_loss - batch_reduce(td.Normal(dec_mean_t, nn.Softplus()(self.variance)*torch.ones_like(dec_mean_t)).log_prob(x))
            nll_loss = nll_loss - nll_unif

        elif self.loss_type == "mse":
            nll_loss = nll_loss + batch_reduce(self.mse_criterion(dec_mean_t, xt[:, i, :, :, :]))
        elif self.loss_type == "mol":
            nll_loss = nll_loss - batch_reduce(self.likelihood(logits = dec_mean_t).log_prob(xt[:, i, :, :, :]))
        else:
            print("undefined loss")
        
        
      return batch_reduce(kl_loss).mean(), nll_loss.mean()

    # 3 plots, reconstruction, prediction og samples

    def predict(self, x, n_predictions, n_conditions):
      b, t, c, h, w = x.shape
      
      
      assert n_conditions <= t, "n_conditions > t, number of conditioned frames is greater than number of frames"
      predictions = torch.zeros((n_predictions, *x[:,0,:,:,:].shape))
      true_x = torch.zeros((n_conditions, *x[:,0,:,:,:].shape))
      hprev, cprev, zprev, zxprev,_,_,_ = self.get_inits()   
      
      true_x[0,:,:,:,:] = x[:, 0, :, :, :]
      for i in range(1, n_conditions):
        xt_features = self.phi_x_t(x[:, i, :, :, :])
        ut = self.phi_x_t(x[:, i-1, :, :, :])
        
        lstm_input = torch.cat([ut, self.phi_z(zxprev)], 1)
        _, ht, ct = self.lstm(lstm_input.unsqueeze(1), hprev, cprev)
        
        prior_t = self.prior(ht) 
        prior_mean_t = self.prior_mean(prior_t) 
        prior_std_t = self.prior_std(prior_t)
        z_t = td.Normal(prior_mean_t, prior_std_t).rsample()
        
        enc_t = self.enc(torch.cat([ht, xt_features], 1))
        enc_mean_t = self.enc_mean(enc_t) 
        enc_std_t = self.enc_std(enc_t)    
        zx_t = td.Normal(enc_mean_t, enc_std_t).rsample()
        
        zprev = z_t
        zxprev = zx_t
        hprev = ht
        cprev = ct
        
        true_x[i,:,:,:,:] = x[:, i, :, :, :].detach()
        
      prediction = x[:,n_conditions-1,:,:,:]
      
      for i in range(0, n_predictions):                   
        ut = self.phi_x_t(prediction)
        
        lstm_input = torch.cat([ut, self.phi_z(zprev)], 1)
        _, ht, ct = self.lstm(lstm_input.unsqueeze(1), hprev, cprev)
        
        prior_t = self.prior(ht) 
        prior_mean_t = self.prior_mean(prior_t) 
        prior_std_t = self.prior_std(prior_t)
        
        z_t = td.Normal(prior_mean_t, prior_std_t).rsample()
        
        dec_t = self.dec(torch.cat([ht, self.phi_z(z_t)], 1))
        dec_mean_t = self.dec_mean(dec_t)
        if self.loss_type == "mol":
          dec_mean_t=self.likelihood(logits = dec_mean_t).sample()
        prediction = dec_mean_t
        zprev = z_t
        hprev = ht
        cprev = ct
        
        predictions[i,:,:,:,:] = prediction.data
      return true_x, predictions

    def reconstruct(self, x):
      b, t, c, h, w = x.shape
      recons= torch.zeros((t, *x[:,0,:,:,:].shape))
      
      hprev, cprev, zprev, zxprev, _,_,_ = self.get_inits()

      for i in range(1, t):
        ut = self.phi_x_t(x[:, i-1, :, :, :])
        xt_features = self.phi_x_t(x[:, i, :, :, :])
        
        lstm_input = torch.cat([ut, self.phi_z(zxprev)], 1)
        _, ht, ct = self.lstm(lstm_input.unsqueeze(1), hprev, cprev)
        
        enc_t = self.enc(torch.cat([ht, xt_features], 1))
        enc_mean_t = self.enc_mean(enc_t) 
        enc_std_t = self.enc_std(enc_t)    
        zx_t = td.Normal(enc_mean_t, enc_std_t).rsample()
        
        
        dec_t = self.dec(torch.cat([ht, self.phi_z(zx_t)], 1))
        dec_mean_t = self.dec_mean(dec_t)
        if self.loss_type == "mol":
          dec_mean_t=self.likelihood(logits = dec_mean_t).sample()
        recons[i,:,:,:,:] = dec_mean_t.detach()
        zxprev = zx_t
        hprev = ht
        cprev = ct
        
      return recons
    
    def sample(self, xt, n_samples):
        
      b, c, h, w = self.x_dim
      samples = torch.zeros((n_samples, b,c,h,w))
      
      ht, ct, zprev, _, _,_,_ = self.get_inits()

      ut = self.phi_x_t(xt[:,0,:,:,:])
      for i in range(0, n_samples):
        
        lstm_input = torch.cat([ut, self.phi_z(zprev)], 1)
        _, ht, ct = self.lstm(lstm_input.unsqueeze(1), ht, ct)
          
        prior_t = self.prior(ht) 
        prior_mean_t = self.prior_mean(prior_t) 
        prior_std_t = self.prior_std(prior_t)
        z_t = td.Normal(prior_mean_t, prior_std_t).rsample()
        
        dec_t = self.dec(torch.cat([ht, self.phi_z(z_t)], 1))
        dec_mean_t = self.dec_mean(dec_t)
        if self.loss_type == "mol":
          dec_mean_t=self.likelihood(logits = dec_mean_t).sample()
        ut = self.phi_x_t(dec_mean_t)
        zprev = z_t
        
        samples[i,:,:,:,:] = dec_mean_t.detach()
      return samples
    
    def elbo_importance_weighting(self, xt, K):

      b, t, c, h, w = xt.shape
      hprev, cprev, zprev, zxprev, _, _, _ = self.get_inits()   
      loss=0
      for i in range(1, t):
        logpzs = torch.empty(size=(K, self.batch_size))
        logqzxs = torch.empty(size=(K, self.batch_size))
        lpx_Gz_obss = torch.empty(size=(K, self.batch_size))
        
        xt_features = self.phi_x_t(xt[:, i, :, :, :])
        ut = self.phi_x_t(xt[:, i-1, :, :, :])
        
        for k in range(0, K):
          lstm_input = torch.cat([ut, self.phi_z(zxprev)], 1)
          _, ht, ct = self.lstm(lstm_input.unsqueeze(1), hprev, cprev)
          
          prior_t = self.prior(ht) 
          prior_mean_t = self.prior_mean(prior_t) 
          prior_std_t = self.prior_std(prior_t)
          prior_dist = td.Normal(prior_mean_t, prior_std_t)
          
          enc_t = self.enc(torch.cat([ht, xt_features], 1))
          enc_mean_t = self.enc_mean(enc_t)
          enc_std_t = self.enc_std(enc_t)
          enc_dist = td.Normal(enc_mean_t, enc_std_t)
          
          zx_t = enc_dist.rsample()
          z_t = prior_dist.rsample()
          
          dec_t = self.dec(torch.cat([ht, self.phi_z(zx_t)], 1))
          dec_mean_t = self.dec_mean(dec_t)
          
  
          if self.loss_type == "bernoulli":
              lpx_Gz_obs = batch_reduce(td.Bernoulli(probs=dec_mean_t).log_prob(xt[:, i, :, :, :]))
          elif self.loss_type == "gaussian":
                        
              if self.dequantize:
                x, nll_unif = self.uniform_binning_correction(xt[:, i, :, :, :])
              else:
                x = xt[:, i, :, :, :]
              
              lpx_Gz_obs = batch_reduce(td.Normal(dec_mean_t, nn.Softplus()(self.variance)*torch.ones_like(dec_mean_t)).log_prob(x))
              lpx_Gz_obs = lpx_Gz_obs - nll_unif
            
          elif self.loss_type == "mse":
              lpx_Gz_obs =  - batch_reduce(self.mse_criterion(dec_mean_t, xt[:, i, :, :, :]))
          elif self.loss_type == "mol":
              lpx_Gz_obs = -batch_reduce(self.likelihood(logits = dec_mean_t).log_prob(xt[:, i, :, :, :]))
          else:
              print("undefined loss")
              
          logpzs[k] = prior_dist.log_prob(z_t).sum([-1])
          logqzxs[k] = enc_dist.log_prob(zx_t).sum([-1])
          lpx_Gz_obss[k] = lpx_Gz_obs
          zxprev = zx_t
          hprev = ht
          cprev = ct
        
        loss = loss - torch.mean(torch.logsumexp(lpx_Gz_obss + logpzs - logqzxs, 0)-torch.log(torch.tensor(K).float()))
        
      return loss
