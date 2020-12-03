import torch
import torch.nn as nn
from utils import get_layer_size, Flatten, UnFlatten, set_gpu
from torch.autograd import Variable
import torch.distributions as td
from modules import ConvLSTM, NormLayer

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
      self.mse_criterion = nn.MSELoss()
      
      # Remember to downscale more when using 64x64. Overall the net should probably increase in size when using 
      # 64x64 images
      phi_x_t_channels = 256
      self.phi_x_t = nn.Sequential(
          nn.Conv2d(cx, 64, kernel_size=3, stride=2, padding=1),
          NormLayer(64, norm_type),
          nn.ReLU(),
          nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
          NormLayer(128, norm_type),
          nn.ReLU(),
          nn.Conv2d(128, phi_x_t_channels, kernel_size=3, stride=1, padding=1),
          NormLayer(phi_x_t_channels, norm_type),
          nn.ReLU(),
        )
      h, w = get_layer_size([hu, wu], kernels=[3, 3], paddings=[1, 1], strides=[2, 2],
                        dilations=[1, 1])
      self.h = h
      self.w = w
      
      # Encoder structure
      self.enc = nn.Sequential(
        nn.Conv2d(phi_x_t_channels + h_dim, 128,  kernel_size=3, stride=2, padding=1),
        NormLayer(128, norm_type),
        nn.ReLU(),
        Flatten(),
        )
      self.enc_mean =  nn.Sequential(
        nn.Linear((128)*h//2*w//2, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, z_dim),
        )
      self.enc_std = nn.Sequential(
        nn.Linear((128)*h//2*w//2, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, z_dim),
        nn.Softplus(),
        )

      # Prior structure
      self.prior = nn.Sequential( 
        nn.Conv2d(h_dim, 128, kernel_size = 3, stride = 2, padding = 1),
        NormLayer(128, norm_type),
        nn.ReLU(),
        Flatten(),
        )
      
      self.prior_mean = nn.Sequential(
        nn.Linear((128)*h//2*w//2, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, z_dim),
        )
      
      self.prior_std = nn.Sequential(
        nn.Linear((128)*h//2*w//2, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, z_dim),
        nn.Softplus()
        )
      
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
        )
      
      # Decoder structure
      self.dec = nn.Sequential(
            nn.ConvTranspose2d(h_dim + phi_z_channels, h_dim // 2, kernel_size=4, stride=2, padding=1, output_padding=0),
            NormLayer(h_dim // 2, norm_type),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim // 2,  kernel_size=3, stride=1, padding=1),
            NormLayer(h_dim // 2, norm_type),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, h_dim // 4, kernel_size=4, stride=2, padding=1, output_padding=0), 
            NormLayer(h_dim // 4, norm_type),
            nn.ReLU(),
        )
      
      self.dec_mean = nn.Sequential(nn.Conv2d(h_dim // 4, cx,  kernel_size=3, stride=1, padding=1), 
                                nn.Sigmoid()
                                )
      self.dec_std = nn.Sequential(nn.Conv2d(h_dim // 4, cx,  kernel_size=3, stride=1, padding=1), 
                               nn.Softplus()
                               )

      #LSTM
      self.lstm = ConvLSTM(in_channels = phi_x_t_channels+phi_z_channels, 
                           hidden_channels=[h_dim], 
                           num_layers=1, 
                           kernel_size=(3, 3), 
                           bias=True, peephole=True, make_init=True)
      
      self.book = {"zt": 0, "enc_loc": 0, "ht": 0}

    def loss(self, xt):

      b, t, c, h, w = xt.shape
      kld_loss = 0
      nll_loss = 0
      
      ut_features = []
      for i in range(0, t):
        ut_features.append(self.phi_x_t(xt[:, i, :, :, :]))

      
      h = Variable(torch.zeros(b, self.h_dim, self.h, self.w).to(device))
      hidden_state = self.lstm._init_hidden(batch_size=self.batch_size, height=self.h, width=self.w)

      for i in range(0, t):
        xt_features = ut_features[i]
        
        
        prior_t = self.prior(h) 
        prior_mean_t = self.prior_mean(prior_t) 
        prior_std_t = self.prior_std(prior_t)

        enc_t = self.enc(torch.cat([h, xt_features], 1))
        enc_mean_t = self.enc_mean(enc_t) 
        enc_std_t = self.enc_std(enc_t)    

        z_t = td.Normal(enc_mean_t, enc_std_t).rsample()
        phi_z_t = self.phi_z(z_t)

        dec_t = self.dec(torch.cat([phi_z_t, h], 1))
        
        dec_mean_t = self.dec_mean(dec_t)
        

        lstm_input = torch.cat([xt_features, phi_z_t], 1)
        h, hidden_state = self.lstm(lstm_input.unsqueeze(1), hidden_state)

        enc_dist = td.Normal(enc_mean_t, enc_std_t)
        prior_dist = td.Normal(prior_mean_t, prior_std_t)
        kld_loss = kld_loss + self.beta * td.kl_divergence(enc_dist, prior_dist).sum([1]).mean()
        if self.loss_type == "bernoulli":
            nll_loss = nll_loss - td.Bernoulli(probs=dec_mean_t).log_prob(xt[:, i, :, :, :]).sum([1,2,3]).mean()
        elif self.loss_type == "gaussian":
            dec_std_t = self.dec_std(dec_t)
            nll_loss = nll_loss - td.Normal(dec_mean_t, dec_std_t).log_prob(xt[:, i, :, :, :]).sum([1,2,3]).mean()
        elif self.loss_type == "mse":
            nll_loss = nll_loss + self.mse_criterion(dec_mean_t, xt[:, i, :, :, :])
        else:
            print("undefined loss")
      
      self.book["zt"] = z_t.detach()
      self.book["enc_loc"] = enc_mean_t.detach()
      self.book["ht"] = h.detach()
      
      return kld_loss, nll_loss

    def predict(self, x, n_predictions, encoder_sample = False):
      b, t, c, h, w = x.shape
        
      predictions = torch.zeros((n_predictions, *x[:,0,:,:,:].shape))
      one_step = torch.zeros((t, *x[:,0,:,:,:].shape))
      
      ut_features = []
      for i in range(0, t):
        ut_features.append(self.phi_x_t(x[:, i, :, :, :]))
        
      h = Variable(torch.zeros(b, self.h_dim, self.h, self.w).to(device))
      hidden_state = self.lstm._init_hidden(batch_size=self.batch_size, height=self.h, width=self.w)


      for i in range(0, t):
        xt_features = ut_features[i]
        
        if encoder_sample == False:
            prior_t = self.prior(h) 
            prior_mean_t = self.prior_mean(prior_t) 
            prior_std_t = self.prior_std(prior_t)
            z_t = td.Normal(prior_mean_t, prior_std_t).rsample()
        else:
            enc_t = self.enc(torch.cat([h, xt_features], 1))
            enc_mean_t = self.enc_mean(enc_t) 
            enc_std_t = self.enc_std(enc_t)    
            z_t = td.Normal(enc_mean_t, enc_std_t).rsample()
        
        phi_z_t = self.phi_z(z_t)

        dec_t = self.dec(torch.cat([phi_z_t, h], 1))
        dec_mean_t = self.dec_mean(dec_t)



        lstm_input = torch.cat([xt_features, phi_z_t], 1)
        h, hidden_state = self.lstm(lstm_input.unsqueeze(1), hidden_state)
        
        one_step[i,:,:,:,:] = dec_mean_t.detach()
        
      prediction = x[:,-1,:,:,:]
      for i in range(0, n_predictions):                   
        xt_features = self.phi_x_t(prediction)
        prior_t = self.prior(h) 
        prior_mean_t = self.prior_mean(prior_t) 
        prior_std_t = self.prior_std(prior_t)
        
        z_t = td.Normal(prior_mean_t, prior_std_t).rsample()
        phi_z_t = self.phi_z(z_t)
        
          
        dec_t = self.dec(torch.cat([phi_z_t, h], 1))
        dec_mean_t = self.dec_mean(dec_t)
        #dec_std_t = self.dec_std(dec_t)

        # Recurrence
        lstm_input = torch.cat([xt_features, phi_z_t], 1)#.view(bu, -1)
        h, hidden_state = self.lstm(lstm_input.unsqueeze(1), hidden_state)
        
        prediction = dec_mean_t
        predictions[i,:,:,:,:] = prediction.data
      return one_step, predictions

    def reconstruct(self, x):
      b, t, c, h, w = x.shape
      recons= torch.zeros((t, *x[:,0,:,:,:].shape))
      
      ut_features = []
      for i in range(0, t):
        ut_features.append(self.phi_x_t(x[:, i, :, :, :]))
        
      h = Variable(torch.zeros(b, self.h_dim, self.h, self.w).to(device))

      for i in range(0, t):
        xt_features = ut_features[i]
        
        enc_t = self.enc(torch.cat([h, xt_features], 1))
        enc_mean_t = self.enc_mean(enc_t) 
        enc_std_t = self.enc_std(enc_t)    
        z_t = td.Normal(enc_mean_t, enc_std_t).rsample()
        
        phi_z_t = self.phi_z(z_t)

        dec_t = self.dec(torch.cat([phi_z_t, h], 1))
        dec_mean_t = self.dec_mean(dec_t)
        
        recons[i,:,:,:,:] = dec_mean_t.detach()
        
      return recons
    
    def sample(self, n_samples):
        
      b, c, h, w = self.x_dim
      samples = torch.zeros((n_samples, b,c,h,w))
      h = Variable(torch.zeros(b, self.h_dim, self.h, self.w).to(device))
      hidden_state = self.lstm._init_hidden(batch_size=self.batch_size, height=self.h, width=self.w)


      for i in range(0, n_samples):
        prior_t = self.prior(h) 
        prior_mean_t = self.prior_mean(prior_t) 
        prior_std_t = self.prior_std(prior_t)
        z_t = td.Normal(prior_mean_t, prior_std_t).rsample()
        
        phi_z_t = self.phi_z(z_t)

        dec_t = self.dec(torch.cat([phi_z_t, h], 1))
        dec_mean_t = self.dec_mean(dec_t)

        xt_features = self.phi_x_t(dec_mean_t)

        lstm_input = torch.cat([xt_features, phi_z_t], 1)#.view(b, -1)
        h, hidden_state = self.lstm(lstm_input.unsqueeze(1), hidden_state)
        
        samples[i,:,:,:,:] = dec_mean_t.detach()
      return samples
    

