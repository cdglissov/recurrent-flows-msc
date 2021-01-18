import torch
import torch.nn as nn
from Utils import get_layer_size, Flatten, UnFlatten, set_gpu, batch_reduce
import torch.distributions as td
from Utils import ConvLSTMOld, NormLayer

device = set_gpu(True)

# add resq, 
#https://medium.com/@aminamollaysa/summary-of-the-recurrent-latent-variable-model-vrnn-4096b52e731
class SRNN(nn.Module):
    def __init__(self, args):
      super(SRNN, self).__init__()
      
      norm_type = args.norm_type
      self.batch_size = args.batch_size
      self.u_dim = args.condition_dim
      self.a_dim = args.a_dim
      self.x_dim = args.x_dim
      self.enable_smoothing = args.enable_smoothing
      h_dim = args.h_dim
      z_dim = args.z_dim
      self.h_dim = h_dim
      self.z_dim = z_dim
      self.loss_type = args.loss_type
      self.beta = 1
      bu, cu, hu, wu = self.u_dim
      bx, cx, hx, wx = self.x_dim
      self.mse_criterion = nn.MSELoss(reduction='none')
      self.res_q = args.res_q
      
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
      if self.enable_smoothing:
          self.enc = nn.Sequential(
            nn.Conv2d(phi_z_channels + self.a_dim, 256,  kernel_size=3, stride=2, padding=1),
            NormLayer(256, norm_type),
            nn.ReLU(),
            Flatten(),
            )
      else:
          self.enc = nn.Sequential(
            nn.Conv2d(phi_z_channels + self.h_dim + phi_x_t_channels, 256,  kernel_size=3, stride=2, padding=1),
            NormLayer(256, norm_type),
            nn.ReLU(),
            Flatten(),
            )
      
      self.enc_mean =  nn.Sequential(
        nn.Linear((256)*h//2*w//2, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, z_dim), #maybe tanh here?
        )
      
      self.enc_std = nn.Sequential(
        nn.Linear((256)*h//2*w//2, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, z_dim),
        nn.Softplus()
        )

      # Prior structure
      self.prior = nn.Sequential( 
        nn.Conv2d(h_dim + phi_z_channels, 256, kernel_size = 3, stride = 2, padding = 1),
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
      
      
      self.dec_mean = nn.Sequential(nn.Conv2d(32, cx,  kernel_size=3, stride=1, padding=1), 
                                nn.Sigmoid()
                                )
      self.dec_std = nn.Sequential(nn.Conv2d(32, cx,  kernel_size=3, stride=1, padding=1), 
                               nn.Softplus()
                               )

      self.z_0 = nn.Parameter(torch.zeros(self.batch_size, z_dim))
      self.z_0x = nn.Parameter(torch.zeros(self.batch_size, z_dim))
      
      self.h_0 = nn.Parameter(torch.zeros(self.batch_size, self.h_dim, h, w))
      self.c_0 = nn.Parameter(torch.zeros(self.batch_size, self.h_dim, h, w))
      
      self.a_0 = nn.Parameter(torch.zeros(self.batch_size, self.a_dim, h, w))
      self.ca_0 = nn.Parameter(torch.zeros(self.batch_size, self.a_dim, h, w))

      #LSTM
      self.lstm_h = ConvLSTMOld(in_channels = phi_x_t_channels, 
                           hidden_channels=self.h_dim, 
                           kernel_size=[3, 3], 
                           bias=True, 
                           peephole=True)
      
      self.lstm_a = ConvLSTMOld(in_channels = phi_x_t_channels + self.h_dim, 
                           hidden_channels=self.h_dim, 
                           kernel_size=[3, 3], 
                           bias=True, 
                           peephole=True)
      
      self.D = args.num_shots + 1 # Plus one as that is more intuative
      self.overshot_w = args.overshot_w #Weight for overshoots.
    def get_inits(self):
      loss = 0
      kl_loss = 0
      nll_loss = 0
      return self.h_0, self.c_0, self.z_0, self.z_0x, self.a_0, self.ca_0, loss, kl_loss, nll_loss

    def loss(self, xt):

      b, t, c, h, w = xt.shape
      hprev, cprev, zprev, zprevx, aprev, caprev, loss, kl_loss, nll_loss = self.get_inits()   
      
      store_ht = torch.zeros((t-1, *hprev.shape)).cuda()
      store_at = torch.zeros((t-1, *hprev.shape)).cuda()
      
      #Find ht
      for i in range(1, t):
        ut = self.phi_x_t(xt[:, i-1, :, :, :])
        _, ht, ct = self.lstm_h(ut.unsqueeze(1), hprev, cprev)
        store_ht[i-1,:,:,:,:] = ht
        hprev = ht
        cprev = ct
    
      if self.enable_smoothing:
          #Find at
          for i in range(1, t):
            xt_features = self.phi_x_t(xt[:, t-i, :, :, :])
            lstm_a_input = torch.cat([store_ht[t-i-1,:,:,:,:], xt_features], 1)
            _, at, c_at = self.lstm_a(lstm_a_input.unsqueeze(1), aprev, caprev)
            aprev = at
            caprev = c_at
            store_at[t-i-1,:,:,:,:] = at
      
      store_ztx_mean = torch.zeros((t-1, *zprevx.shape)).cuda()
      store_ztx_std = torch.zeros((t-1, *zprevx.shape)).cuda()
      store_zt = torch.zeros((t-1, *zprev.shape)).cuda()
      for i in range(1, t):
        ht=store_ht[i-1,:,:,:,:]
        prior_t = self.prior(torch.cat([ht, self.phi_z(zprev)],1))
        prior_mean_t = self.prior_mean(prior_t) 
        prior_std_t = self.prior_std(prior_t)
        prior_dist = td.Normal(prior_mean_t, prior_std_t)
        
        if self.enable_smoothing:
            at = store_at[i-1,:,:,:,:]
            enc_t = self.enc(torch.cat([at, self.phi_z(zprevx)], 1))
        else:
            xt_features = self.phi_x_t(xt[:, i, :, :, :])
            enc_t = self.enc(torch.cat([ht, self.phi_z(zprevx), xt_features], 1))
        enc_std_t = self.enc_std(enc_t)
        
        if self.res_q:
            enc_mean_t = prior_mean_t + self.enc_mean(enc_t)
        else:
            enc_mean_t = self.enc_mean(enc_t)
        
        enc_dist = td.Normal(enc_mean_t, enc_std_t)

        z_tx = enc_dist.rsample()
        z_t = prior_dist.rsample()
        
        store_ztx_mean[i-1,:,:] = enc_mean_t
        store_ztx_std[i-1,:,:] = enc_std_t
        
        store_zt[i-1,:,:] = z_t

        dec_t = self.dec(torch.cat([ht, self.phi_z(z_tx)], 1))
        dec_mean_t = self.dec_mean(dec_t)
        
        zprevx = z_tx
        zprev = z_t
        
        kl_loss = kl_loss + self.beta * td.kl_divergence(enc_dist, prior_dist)

 
        if self.loss_type == "bernoulli":
            nll_loss = nll_loss - td.Bernoulli(probs=dec_mean_t).log_prob(xt[:, i, :, :, :])
        elif self.loss_type == "gaussian":
            dec_std_t = self.dec_std(dec_t)
            nll_loss = nll_loss - td.Normal(dec_mean_t, dec_std_t).log_prob(xt[:, i, :, :, :])
        elif self.loss_type == "mse":
            nll_loss = nll_loss + self.mse_criterion(dec_mean_t, xt[:, i, :, :, :])
        else:
            print("undefined loss")
      
      if self.D > 1: ## If true able overshoots
          overshot_loss = 0
          overshot_w = self.overshot_w # Weight of overshot.
          Dinit = self.D # is the number of over samples, if D=1 no over shooting will happen.
          # Given we have calculated for D = 1 as the above, we continue from there
          for i in range(1, t):
              idt = i-1 # index t, Does this to make index less confusing
              zprev = store_zt[idt, : , :]
              D = min(t-i, Dinit) #Do this so that for ts at t-D still do overshoting but with less overshooting # Dont know if this is the correct way to do it but makes sense
              for d in range(1, D): # D is the number of overshootes
                  
                  ht = store_ht[idt + d, :, :, :, :] # So find the ht for t + d
                  prior_t = self.prior(torch.cat([ht, self.phi_z(zprev)],1))
                  prior_mean_t = self.prior_mean(prior_t)
                  prior_std_t = self.prior_std(prior_t)
                  zprev = prior_dist.rsample()
                  prior_dist = td.Normal(prior_mean_t, prior_std_t)

                  enc_dist = td.Normal(store_ztx_mean[idt + d, :, :], store_ztx_std[idt + d, :, :] ) #Encoded values is matching ht so t + d
                  overshot_loss = overshot_loss + overshot_w * self.beta * td.kl_divergence(enc_dist, prior_dist)
          kl_loss = 1/Dinit*(kl_loss + overshot_loss) # The first loss is for D=1, 
      

      return batch_reduce(kl_loss).mean(), batch_reduce(nll_loss).mean()


    def predict(self, xt, n_predictions, n_conditions):
      b, t, c, h, w = xt.shape
      
      assert n_conditions <= t, "n_conditions > t, number of conditioned frames is greater than number of frames"
      
      predictions = torch.zeros((n_predictions, *xt[:,0,:,:,:].shape))
      true_x = torch.zeros((n_conditions, *xt[:,0,:,:,:].shape))
      hprev, cprev, zprev, zprevx, aprev, caprev, _, _, _ = self.get_inits()     

      store_ht = torch.zeros((t-1, *hprev.shape)).cuda()
      true_x[0,:,:,:,:] = xt[:, 0, :, :, :].detach()
      
      #Find ht
      for i in range(1, n_conditions):
        ut = self.phi_x_t(xt[:, i-1, :, :, :])
        _, ht, ct = self.lstm_h(ut.unsqueeze(1), hprev, cprev)
        store_ht[i-1,:,:,:,:] = ht
        hprev = ht
        cprev = ct

      #Find encoder samples, should we add res_q here? #add warmup with resq
      for i in range(1, n_conditions):
        ht = store_ht[i-1,:,:,:,:]
        prior_t = self.prior(torch.cat([ht, self.phi_z(zprev)],1)) 
        prior_mean_t = self.prior_mean(prior_t) 
        prior_std_t = self.prior_std(prior_t)
        prior_dist = td.Normal(prior_mean_t, prior_std_t)
        z_t = prior_dist.rsample()
        zprev = z_t
        true_x[i,:,:,:,:] = xt[:, i, :, :, :].detach()
      
      prediction = xt[:,n_conditions-1,:,:,:]
      
      for i in range(0, n_predictions):  
        ut = self.phi_x_t(prediction)                 

        _, ht, ct = self.lstm_h(ut.unsqueeze(1), hprev, cprev)
        
        prior_t = self.prior(torch.cat([ht, self.phi_z(zprev)],1)) 
        prior_mean_t = self.prior_mean(prior_t) 
        prior_std_t = self.prior_std(prior_t)
        prior_dist = td.Normal(prior_mean_t, prior_std_t)
        
        z_t = prior_dist.rsample()
        
        dec_t = self.dec(torch.cat([ht, self.phi_z(z_t)], 1))
        dec_mean_t = self.dec_mean(dec_t)
        
        prediction = dec_mean_t
        zprev = z_t
        hprev = ht
        cprev = ct
        predictions[i,:,:,:,:] = prediction.data
      return true_x, predictions

    def reconstruct(self, xt):
      b, t, c, h, w = xt.shape
      
      recons = torch.zeros((t, *xt[:,0,:,:,:].shape))
      hprev, cprev, zprev, zprevx, aprev, caprev, _, _, _ = self.get_inits()     

      store_ht = torch.zeros((t-1, *hprev.shape)).cuda()
      store_at = torch.zeros((t-1, *hprev.shape)).cuda()
      
      #Find ht
      for i in range(1, t):
        ut = self.phi_x_t(xt[:, i-1, :, :, :])
        _, ht, ct = self.lstm_h(ut.unsqueeze(1), hprev, cprev)
        store_ht[i-1,:,:,:,:] = ht
        hprev = ht
        cprev = ct
    
      #Find at
      if self.enable_smoothing:
          for i in range(1, t):
            xt_features = self.phi_x_t(xt[:, t-i, :, :, :])
            lstm_a_input = torch.cat([store_ht[t-i-1,:,:,:,:], xt_features], 1)
            _, at, c_at = self.lstm_a(lstm_a_input.unsqueeze(1), aprev, caprev)
            aprev = at
            caprev = c_at
            store_at[t-i-1,:,:,:,:] = at
        
      for i in range(1, t):
        if self.enable_smoothing:
            at = store_at[i-1,:,:,:,:]
            enc_t = self.enc(torch.cat([at, self.phi_z(zprevx)], 1))
        else:
            xt_features = self.phi_x_t(xt[:, i, :, :, :])
            enc_t = self.enc(torch.cat([ht, self.phi_z(zprevx), xt_features], 1))
        
        if self.res_q:
            ht = store_ht[i-1,:,:,:,:]
            prior_t = self.prior(torch.cat([ht, self.phi_z(zprev)],1)) 
            prior_mean_t = self.prior_mean(prior_t) 
            prior_std_t = self.prior_std(prior_t)
            prior_dist = td.Normal(prior_mean_t, prior_std_t)
            z_t = prior_dist.rsample()
            zprev = z_t
            enc_mean_t = prior_mean_t + self.enc_mean(enc_t)
        else:
            enc_mean_t = self.enc_mean(enc_t)
            
        enc_std_t = self.enc_std(enc_t)
        enc_dist = td.Normal(enc_mean_t, enc_std_t)
        z_tx = enc_dist.rsample()

        ht = store_ht[i-1,:,:,:,:]
        dec_t = self.dec(torch.cat([ht, self.phi_z(z_tx)], 1))
        dec_mean_t = self.dec_mean(dec_t)
        zprevx = z_tx
        recons[i,:,:,:,:] = dec_mean_t.detach()
        
      return recons
    
    def sample(self, xt, n_samples):
      b, t, c, h, w = xt.shape
      
      samples = torch.zeros((n_samples, b,c,h,w))
      hprev, cprev, zprev, zprevx, _, _, _, _, _ = self.get_inits()     
      condition = xt[:, 0, :, :, :]
    
      for i in range(0, n_samples):                   
        ut = self.phi_x_t(condition)
        _, ht, ct = self.lstm_h(ut.unsqueeze(1), hprev, cprev)
        prior_t = self.prior(torch.cat([ht, self.phi_z(zprev)],1)) 
        prior_mean_t = self.prior_mean(prior_t) 
        prior_std_t = self.prior_std(prior_t)
        prior_dist = td.Normal(prior_mean_t, prior_std_t)
        
        z_t = prior_dist.rsample()
        
        dec_t = self.dec(torch.cat([ht, self.phi_z(z_t)], 1))
        dec_mean_t = self.dec_mean(dec_t)
        condition = dec_mean_t
        zprev = z_t
        hprev = ht
        cprev=ct
        samples[i,:,:,:,:] = dec_mean_t.data
      
      return samples
    

