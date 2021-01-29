import torch
import torch.nn as nn
from Utils import get_layer_size, Flatten, UnFlatten, set_gpu, batch_reduce
import torch.distributions as td
from Utils import ConvLSTMOld, NormLayer, vgg_layer, ActFun
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
device = set_gpu(True)

# add resq, 
#https://medium.com/@aminamollaysa/summary-of-the-recurrent-latent-variable-model-vrnn-4096b52e731


class Extractor(nn.Module):
    def __init__(self, L, xc = [8, 16, 32, 64, 128, 256, 512], nc=1):
        super(Extractor, self).__init__()
        self.L = L
        
        self.c0 = nn.Sequential(
                vgg_layer(nc, 8),
                vgg_layer(8, 8),
                )
        
        self.c1 = nn.Sequential(
                vgg_layer(8, 16),
                vgg_layer(16, 16),
                )
     
        self.c2 = nn.Sequential(
                vgg_layer(16, 32),
                vgg_layer(32, 32),
                )
     
        self.c3 = nn.Sequential(
                vgg_layer(32, 64),
                vgg_layer(64, 64),
                vgg_layer(64, 64),
                )
        
        self.c4 = nn.Sequential(
                vgg_layer(64, 128),
                vgg_layer(128, 128),
                vgg_layer(128, 128),
                )
        self.c5 = nn.Sequential(
                vgg_layer(128, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 256),
                )
 
        self.c6 = nn.Sequential(
                nn.Conv2d(256, 512, 1, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h0 = self.c0(input) # 64x64
        hs = []
        if self.L>=1:
            h1 = self.c1(self.mp(h0)) # 32x32
            hs.append(h1)
        if self.L>=2:
            h2 = self.c2(self.mp(h1)) # 16x16
            hs.append(h2)
        if self.L>=3:
            h3 = self.c3(self.mp(h2)) # 8x8
            hs.append(h3)
        if self.L>=4:
            h4 = self.c4(self.mp(h3)) # 4x4
            hs.append(h4)
        if self.L>=5:
            h5 = self.c5(self.mp(h4)) # 2x2
            hs.append(h5)
        if self.L>=6:
            h6 = self.c6(self.mp(h5)) # 1x1
            hs.append(h6)
        return hs




class Z_Net(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    super(Z_Net, self).__init__()
    self.conv = nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                           kernel_size = kernel_size, stride = stride, padding = padding),
            nn.ReLU())
    
    self.mu = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, 
                           kernel_size = kernel_size, stride = 1, padding = padding)
    
    self.var = nn.Sequential(nn.Conv2d(in_channels = out_channels, out_channels = out_channels, 
                           kernel_size = kernel_size, stride = 1, padding = padding), nn.Softplus())
    
  def gaussian_rsample(self, mean, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)

  def forward(self, input):
    output = self.conv(input)
    mu = self.mu(output)
    logvar = self.var(output).log()
    z_t = self.gaussian_rsample(mu, logvar)
    return z_t, mu, logvar
    
class Dense_Block_Prior(nn.Module):
  def __init__(self, hc, L, z_dim):
    super(Dense_Block_Prior, self).__init__()
    self.L = L
    kernel_sizes = [3,3,3,3,1,1]
    padding = [1,1,1,1,0,0]
    self.z_nets = []
    for i in range(0, L):
        z_net = Z_Net(in_channels = z_dim + hc[i]+z_dim*i, out_channels = z_dim, 
                           kernel_size = kernel_sizes[L-i-1], stride = 1, 
                           padding = padding[L-i-1])
        
        self.z_nets += [z_net]
        
    self.z_nets = nn.ModuleList(self.z_nets)  
    
  def scale_z(self, z, cur_res, prev_res):
      scaling_factor = cur_res//prev_res
      z = F.interpolate(z, scale_factor=scaling_factor)
      return z
      
  def forward(self, z_prev, h_states):
    prev_zs=[]
    mus = []
    lvs = []
    
    for i in range(0, self.L):
        input = [z_prev[i], h_states[i]]
        for j in range(0, len(prev_zs)):
            input += [self.scale_z(prev_zs[j], z_prev[i].shape[-1], prev_zs[j].shape[-1])]
        input = torch.cat(input, 1)
        z, mu, logvar = self.z_nets[i](input)
        prev_zs.append(z)
        mus.append(mu)
        lvs.append(logvar)
    #z_1 = 1x1...z_L = 32x32
    return prev_zs, mus, lvs

class Dense_Block_Posterior(nn.Module):
  def __init__(self, hc, xc, L, z_dim):
    super(Dense_Block_Posterior, self).__init__()
    self.L = L
    kernel_sizes = [3,3,3,3,1,1]
    padding = [1,1,1,1,0,0]
    
    self.z_nets = []
    for i in range(0, L):
        z_net = Z_Net(in_channels = z_dim + hc[i]+z_dim*i + xc[i], 
                           out_channels = z_dim, kernel_size = kernel_sizes[L-i-1], 
                           stride = 1, padding=padding[L-i-1])
        self.z_nets += [z_net]
    self.z_nets = nn.ModuleList(self.z_nets)  
    
    
  def scale_z(self, z, cur_res, prev_res):
      scaling_factor = cur_res//prev_res
      z = F.interpolate(z, scale_factor=scaling_factor)
      return z
      
  def forward(self, z_prev, h_states, x_features):
    prev_zs=[]
    mus = []
    lvs = []
    
    for i in range(0, self.L):
        input = [z_prev[i], h_states[i], x_features[i]]
        for j in range(0, len(prev_zs)):
            input += [self.scale_z(prev_zs[j], z_prev[i].shape[-1], prev_zs[j].shape[-1])]
        input = torch.cat(input, 1)
        z, mu, logvar = self.z_nets[i](input)
        prev_zs.append(z)
        mus.append(mu)
        lvs.append(logvar)
        
    return prev_zs, mus, lvs


class Decoder(nn.Module):
    def __init__(self, hc, zc, L, nc):
        super(Decoder, self).__init__()
        self.L=L
        #1x1->2x2
        if self.L>=1:
            self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(hc[0]+zc, 512, 1, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                vgg_layer(512, 512),
                )
        
        # 4 x 4
        if self.L>=2:
            self.upc2 = nn.Sequential(
                vgg_layer(512+hc[1]+zc, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 256)
                )
        # 8 x 8
        if self.L>=3:
            self.upc3 = nn.Sequential(
                vgg_layer(256+hc[2]+zc, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 128)
                )
        # 16 x 16
        if self.L>=4:
            self.upc4 = nn.Sequential(
                vgg_layer(128+hc[3]+zc, 128),
                vgg_layer(128, 128),
                vgg_layer(128, 64)
                )
        
        # 32 x 32
        if self.L>=5:
            self.upc5 = nn.Sequential(
                vgg_layer(64+hc[4]+zc, 64),
                vgg_layer(64, 32)
                )
        
        # 64 x 64
        if self.L>=6:
            self.upc6 = nn.Sequential(
                vgg_layer(32+hc[5]+zc, 32),
                vgg_layer(32, 16)
                )
        in_final = [512,256,128,64,32,16]
        self.final = nn.Sequential(
                vgg_layer(in_final[self.L-1], 8),
                nn.ConvTranspose2d(8, nc, 3, 1, 1),
                nn.Sigmoid()
                )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        
    def forward(self, zt, ht):
        if self.L>=1:
            output = self.upc1(torch.cat([zt[0], ht[0]],1))
        if self.L>=2:
            output = self.up(output)
            output = self.upc2(torch.cat([output, zt[1], ht[1]], 1)) 
        if self.L>=3:
            output = self.up(output) 
            output = self.upc3(torch.cat([output, zt[2], ht[2]], 1)) 
        if self.L>=4:
            output = self.up(output) 
            output = self.upc4(torch.cat([output, zt[3], ht[3]], 1))
        if self.L>=5:
            output = self.up(output) 
            output = self.upc5(torch.cat([output, zt[4], ht[4]], 1)) 
        if self.L>=6:
            output = self.up(output) 
            output = self.upc6(torch.cat([output, zt[5], ht[5]], 1)) 
        output = self.up(output)
        output = self.final(output) #64x64
        return output

class LSTM_Encoder(nn.Module):
  def __init__(self, L, hc = [32, 64, 96, 128, 256, 512], xc =[16, 32, 64, 128, 256, 512]):
    super(LSTM_Encoder, self).__init__()
    self.L = L
    ks = [5,5,3,3,2,1]
    
    self.lstms = []
    for i in range(0, L):
        self.lstms.append(ConvLSTMOld(in_channels = xc[i], hidden_channels = hc[i] , kernel_size=[ks[i], ks[i]]))
    
    self.lstms = nn.ModuleList(self.lstms)
    #32x32 -> 1x1
    
  def forward(self, input, h_prev, c_prev):
    ht = []
    ct = []
    #maybe make param here or merge into model
    for i in range(0, self.L):
        _, h_state, c_state=self.lstms[i](input[i].unsqueeze(1), h_prev[i], c_prev[i])
        ht.append(h_state)
        ct.append(c_state)
    
    return ht, ct

class SRNN_hier(nn.Module):
    def __init__(self, args):
      super(SRNN_hier, self).__init__()
      
      
      self.batch_size = args.batch_size
      self.u_dim = args.condition_dim
      self.x_dim = args.x_dim
      self.loss_type = args.loss_type
      bu, cu, hu, wu = self.u_dim
      bx, cx, hx, wx = self.x_dim
      self.mse_criterion = nn.MSELoss(reduction='none')
      self.L = 3
      self.z_dim = 1
      self.xc = [16, 32, 64, 128, 256, 512][0:self.L]
      self.hc = [32, 64, 96, 128, 256, 512][0:self.L]
      self.mse_criterion = nn.MSELoss(reduction="none")
      self.extractor = Extractor(L=self.L, xc=self.xc, nc=cx)
      self.lstm = LSTM_Encoder(L=self.L, hc=self.hc)
      
      hs = []
      cs = []
      zs = []
      zxs = []
      
      for i in range(0, self.L):
          hd = 32//2**(i)
          hs.append(nn.Parameter(torch.zeros(self.batch_size, self.hc[i], hd, hd)))
          cs.append(nn.Parameter(torch.zeros(self.batch_size, self.hc[i], hd, hd)))
          zd = 32//2**(self.L-i-1)
          zs.append(nn.Parameter(torch.zeros(self.batch_size, self.z_dim, zd, zd)))
          zxs.append(nn.Parameter(torch.zeros(self.batch_size, self.z_dim, zd, zd)))
      #this might be sketchy
      self.hs = nn.ParameterList(hs)
      self.cs = nn.ParameterList(cs)
      self.zs = nn.ParameterList(zs)
      self.zxs = nn.ParameterList(zxs)
      
      self.posterior = Dense_Block_Posterior(hc=self.hc[::-1], xc=self.xc[::-1], L=self.L, z_dim=self.z_dim)
      self.prior = Dense_Block_Prior(hc=self.hc[::-1], L=self.L, z_dim=self.z_dim)
      
      self.decoder = Decoder(hc = self.hc[::-1], zc=self.z_dim, L=self.L, nc=cx)
      
    def get_inits(self):
      kl_loss = 0
      nll_loss = 0
      h_states = self.hs
      c_states = self.cs
      z_prior = self.zs
      z_posterior = self.zxs
      return h_states, c_states, z_prior, z_posterior, kl_loss, nll_loss
     
    def gaussian_diag_logps(self, mean, logvar, sample):
        const = 2 * np.pi * torch.ones_like(mean).to(mean.device)
        return -0.5 * (torch.log(const) + logvar + (sample - mean)**2 / torch.exp(logvar))
    
    def calc_kl(self, mus, lvs, muxs, lvxs, ztxs):
        cur_kl=0
        
        for i in range(0, len(mus)):
            log_p = self.gaussian_diag_logps(mus[i], lvs[i], ztxs[i])
            log_q = self.gaussian_diag_logps(muxs[i], lvxs[i], ztxs[i])
            cur_kl += (batch_reduce(log_q)- batch_reduce(log_p))
        
        return cur_kl
            
    def loss(self, xt):

      b, t, c, h, w = xt.shape
      hprev, cprev, zprev, zprevx, kl_loss, nll_loss = self.get_inits()   
      
      nll = 0
      kl_loss = 0
      ut_features = self.extractor(xt[:, 0, :, :, :])
      for i in range(1, t):
        
        xt_features = self.extractor(xt[:, i, :, :, :])
        ht, ct = self.lstm(ut_features, hprev, cprev)
        
        
        zts, mus, lvs = self.prior(zprev, ht[::-1])
        ztxs, muxs, lvxs = self.posterior(zprevx, ht[::-1], xt_features[::-1])
        
        
        x_gen = self.decoder(ztxs, ht[::-1])
        
        kl_loss = kl_loss + self.calc_kl(mus, lvs, muxs, lvxs, ztxs)
        
        nll = nll + self.mse_criterion(x_gen, xt[:, i, :, :, :])
        
        ut_features = xt_features
        hprev = ht
        cprev = ct
        zprev = zts
        zprevx = ztxs
        
      return batch_reduce(nll).mean(), kl_loss.mean()
    
    def predict(self, xt, n_predictions, n_conditions):

      b, t, c, h, w = xt.shape
      hprev, cprev, zprev, zprevx, _, _ = self.get_inits()   
      predictions = torch.zeros((n_predictions, *xt[:,0,:,:,:].shape))
      true_x = torch.zeros((n_conditions, *xt[:,0,:,:,:].shape))
      
      ut_features = self.extractor(xt[:, 0, :, :, :])
      for i in range(1, n_conditions):
        
        xt_features = self.extractor(xt[:, i, :, :, :])
        ht, ct = self.lstm(ut_features, hprev, cprev)

        zts, mus, lvs = self.prior(zprev, ht[::-1])
        ztxs, muxs, lvxs = self.posterior(zprevx, ht[::-1], xt_features[::-1])

        self.decoder(ztxs, ht[::-1])

        ut_features = xt_features
        hprev = ht
        cprev = ct
        zprev = zts
        zprevx = ztxs
        true_x[i,:,:,:,:] = xt[:, i, :, :, :].detach()
        
      prediction = xt[:,n_conditions-1,:,:,:]
      for i in range(0, n_predictions):
          ut_features = self.extractor(prediction)
          ht, ct = self.lstm(ut_features, hprev, cprev)
          zts, mus, lvs = self.prior(zprev, ht[::-1])
          prediction = self.decoder(zts, ht[::-1])
          predictions[i,:,:,:,:] = prediction.data
          
          hprev = ht
          cprev = ct
          zprev = zts
      return true_x, predictions

    def reconstruct(self, xt):

      b, t, c, h, w = xt.shape
      hprev, cprev, _, zprevx, _, _ = self.get_inits()   
      recons= torch.zeros((t, *xt[:,0,:,:,:].shape))
      
      ut_features = self.extractor(xt[:, 0, :, :, :])
      for i in range(1, t):
        
        xt_features = self.extractor(xt[:, i, :, :, :])
        ht, ct = self.lstm(ut_features, hprev, cprev)

        ztxs, muxs, lvxs = self.posterior(zprevx, ht[::-1], xt_features[::-1])

        x_gen = self.decoder(ztxs, ht[::-1])

        ut_features = xt_features
        hprev = ht
        cprev = ct
        zprevx = ztxs
        recons[i,:,:,:,:] = x_gen.detach()
        

      return recons
  
    def sample(self, xt, n_samples):

      b, t, c, h, w = xt.shape
      hprev, cprev, zprev, _, _, _ = self.get_inits()   
      samples = torch.zeros((n_samples, b,c,h,w))
      
      ut_features = self.extractor(xt[:, 0, :, :, :])
      for i in range(1, t):
        
        ht, ct = self.lstm(ut_features, hprev, cprev)

        zts, mus, lvs = self.prior(zprev, ht[::-1])

        x_gen = self.decoder(zts, ht[::-1])

        ut_features = self.extractor(x_gen)
        hprev = ht
        cprev = ct
        zprev = zts
        samples[i,:,:,:,:] = x_gen.detach()
        

      return samples
    