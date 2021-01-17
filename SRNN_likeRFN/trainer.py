
import torch
import torch.nn as nn
from Utils import VGG_upscaler, VGG_downscaler, SimpleParamNet, ConvLSTMOld, free_bits_kl, batch_reduce, Squeeze2dDecoder
import torch.distributions as td
from Utils import set_gpu

device = set_gpu(True)
class SRNN_RFN(nn.Module):
    def __init__(self, args):
      super(SRNN_RFN, self).__init__()
      self.params = args
      batch_size = args.batch_size
      self.u_dim = args.x_dim
      self.x_dim = args.condition_dim
      self.h_dim = args.h_dim
      self.z_dim = args.z_dim
      self.beta = 1
      scaler = args.structure_scaler

      norm_type = args.norm_type
      norm_type_features = args.norm_type_features
      self.prior_structure = args.prior_structure
      self.encoder_structure = args.encoder_structure

      self.free_bits = args.free_bits

      self.downscaler_tanh = args.downscaler_tanh
      
      self.skip_connection_features = args.skip_connection_features
      self.upscaler_tanh = args.upscaler_tanh
      
      down_structure = args.extractor_structure 
      up_structure = args.upscaler_structure
      self.L =len(down_structure) ## Do this Hax, but then we can use the exact same net as in RFN.

      if not self.skip_connection_features:
          skip = False
      else:
          skip = True
          
      self.extractor = VGG_downscaler(down_structure,L=self.L, in_channels = self.x_dim[1], 
                                      norm_type=norm_type_features, non_lin = "relu", scale = scaler, 
                                      skip_con=skip, tanh = self.downscaler_tanh)
      
      # adjust channel dims to match up_structure. Reversed.

      dims_skip = self.extractor.get_layer_size(down_structure, self.x_dim)
      hu, wu = (self.u_dim[2], self.u_dim[3])
      for i in range(0, self.L):
        hu, wu = (hu//2, hu//2)
      c_features = dims_skip[-1][1]
      
      self.z_0 = nn.Parameter(torch.zeros(batch_size, self.z_dim, hu, wu))
      self.z_0x = nn.Parameter(torch.zeros(batch_size, self.z_dim, hu, wu))
      
      self.h_0 = nn.Parameter(torch.zeros(batch_size, self.h_dim, hu, wu))
      self.c_0 = nn.Parameter(torch.zeros(batch_size, self.h_dim, hu, wu))

      # Feature extractor and upscaler for flow
      up_structure[-1] = up_structure[-1]+[self.x_dim[1]*2*4] # Add x_mean and x_std for different losses

      self.squeeze_fun = Squeeze2dDecoder(undo_squeeze = True)
      self.upscaler = VGG_upscaler(up_structure, L=self.L, in_channels = self.h_dim + self.z_dim, 
                                   norm_type = norm_type_features, non_lin = "leakyrelu",
                                   scale = scaler, skips = self.skip_connection_features,
                                   size_skips = dims_skip, tanh = self.upscaler_tanh)

      # ConvLSTM
      self.lstm = ConvLSTMOld(in_channels = c_features, hidden_channels=self.h_dim, 
                           kernel_size=[3, 3], bias=True, peephole=True)

      # Prior
      prior_struct = self.prior_structure
      self.prior = SimpleParamNet(prior_struct, in_channels = self.h_dim + self.z_dim, 
                                  out_channels = self.z_dim, norm_type = norm_type, non_lin = "leakyrelu")
      

      # Variational encoder
      enc_struct = self.encoder_structure
      self.encoder = SimpleParamNet(enc_struct, in_channels = c_features + self.h_dim + self.z_dim, 
                                    out_channels = self.z_dim, norm_type = norm_type, non_lin = "leakyrelu")
      self.loss_select = args.loss_select
      #self.loss_select = 'prob_loss'#'MSE'
      self.mse_criterion = nn.MSELoss(reduction='none')
      self.eps = 1e-6
    def get_inits(self):
      loss = 0
      kl_loss = 0
      nll_loss = 0
      return self.h_0, self.c_0, self.z_0, self.z_0x, loss, kl_loss, nll_loss

    def loss(self, x, logdet):
      assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
      ht, ct, zt, zxt, loss, kl_loss, nll_loss = self.get_inits()
      t = x.shape[1]

      condition_list = self.extractor(x[:, 0, :, :, :])
      #condition_list.insert(0,dummy)
      for i in range(1, t):
        
        x_feature_list = self.extractor(x[:, i, :, :, :])
        #x_feature_list.insert(0,dummy)


        if not self.skip_connection_features:
            condition = condition_list
            x_feature = x_feature_list
        else:
            condition = condition_list[-1]
            x_feature = x_feature_list[-1]


        _, ht, ct = self.lstm(condition.unsqueeze(1), ht, ct) 
        
        prior_mean, prior_std = self.prior(torch.cat((ht, zt), dim=1))
        dist_prior = td.Normal(prior_mean, prior_std + self.eps)
        zt = dist_prior.rsample()

        enc_mean, enc_std = self.encoder(torch.cat((ht, zxt, x_feature), dim = 1))
        dist_enc = td.Normal(enc_mean, enc_std + self.eps)
        zxt = dist_enc.rsample()

        if self.skip_connection_features:
            x_hat = self.upscaler(torch.cat((ht, zxt), dim = 1), skip_list = condition_list)
        else:
            x_hat = self.upscaler(torch.cat((ht, zxt), dim = 1))
        
        x_hat = x_hat[0] # Get only first input

        x_hat = self.squeeze_fun(x_hat) # Here we make a flow undo squeeze to make the correct dimensions. 

        x_mean, x_std  = x_hat.chunk(2, dim=1) 

        if self.loss_select=='MSE':
            nll = self.mse_criterion(x_mean,x[:, i, :, :, :])
        elif self.loss_select=='gaussian':
            x_std = nn.Softplus()(x_std) + self.eps
            nll = batch_reduce(-td.Normal(x_mean, x_std).log_prob(x[:, i, :, :, :]))+logdet


        kl_loss = kl_loss + td.kl_divergence(dist_enc, dist_prior)
        
        if self.free_bits>0:
            kl_free_bit = free_bits_kl(kl_loss, free_bits = self.free_bits)
        else:
            kl_free_bit = kl_loss
        
        nll_loss = nll_loss + nll 

        condition_list = x_feature_list
        
      return batch_reduce(kl_free_bit).mean(), batch_reduce(kl_loss).mean(), nll_loss.mean()
    

  
    def predict(self, x, n_conditions, n_predictions):
        #sample(self, x, n_predictions=6, encoder_sample = False, start_predictions = None):

      assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
      ht, ct, zt, zxt, _, _, _ = self.get_inits()

      predictions = torch.zeros((n_predictions, *x[:,0,:,:,:].shape))
      true_x = torch.zeros((n_conditions, *x[:,0,:,:,:].shape))
      condition_list = self.extractor(x[:, 0, :, :, :])


      for i in range(1, n_conditions):
        
        condition_list = self.extractor(x[:, i-1, :, :, :])
        if not self.skip_connection_features:
            condition = condition_list
        else:
            condition = condition_list[-1]
                
        _, ht, ct = self.lstm(condition.unsqueeze(1), ht, ct) 
        
        # We do not need to update the encoder due to LSTM not being conditioned on it
        prior_mean, prior_std = self.prior(torch.cat((ht, zt), dim=1))
        dist_prior = td.Normal(prior_mean, prior_std + self.eps)
        zt = dist_prior.sample()
        
        true_x[i,:,:,:,:] = x[:, i, :, :, :].detach()


      
      # Make predictions
      prediction = x[:,n_conditions-1,:,:,:]
      for i in range(0, n_predictions):
          if not self.skip_connection_features:
              condition = self.extractor(prediction)
          else:
              condition_list = self.extractor(prediction)
              condition = condition_list[-1]
          _, ht, ct = self.lstm(condition.unsqueeze(1), ht, ct)  
          prior_mean, prior_std = self.prior(torch.cat((ht, zt), dim=1))
          dist_prior = td.Normal(prior_mean, prior_std + self.eps)
          zt = dist_prior.sample()
          if self.skip_connection_features:
              x_hat = self.upscaler(torch.cat((ht, zt), dim = 1), skip_list = condition_list)
          else:
              x_hat = self.upscaler(torch.cat((ht, zt), dim = 1))
          x_hat = x_hat[0] # Get only first input
          x_hat = self.squeeze_fun(x_hat) # Here we make a flow undo squeeze to make the correct dimensions. 
          x_mean, x_std  = x_hat.chunk(2, dim=1)
          if self.loss_select=='MSE':
              prediction = x_mean
          elif self.loss_select=='gaussian':
              x_std = nn.Softplus()(x_std)
              prediction =td.Normal(x_mean, x_std + self.eps).sample()
          predictions[i,:,:,:,:] = prediction.detach()
      return true_x, predictions
  
    def reconstruct(self, x):
      assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
      ht, ct, zt, zxt, loss, kl_loss, nll_loss = self.get_inits()
      t = x.shape[1]

      condition_list = self.extractor(x[:, 0, :, :, :])
      recons = torch.zeros((t, *x[:,0,:,:,:].shape))
      #condition_list.insert(0,dummy)
      for i in range(1, t):
        
        x_feature_list = self.extractor(x[:, i, :, :, :])
        #x_feature_list.insert(0,dummy)


        if not self.skip_connection_features:
            condition = condition_list
            x_feature = x_feature_list
        else:
            condition = condition_list[-1]
            x_feature = x_feature_list[-1]


        _, ht, ct = self.lstm(condition.unsqueeze(1), ht, ct) 
        
        enc_mean, enc_std = self.encoder(torch.cat((ht, zxt, x_feature), dim = 1))
        dist_enc = td.Normal(enc_mean, enc_std + self.eps)
        zxt = dist_enc.rsample()

        if self.skip_connection_features:
            x_hat = self.upscaler(torch.cat((ht, zxt), dim = 1), skip_list = condition_list)
        else:
            x_hat = self.upscaler(torch.cat((ht, zxt), dim = 1))
        
        x_hat = x_hat[0] # Get only first input

        x_hat = self.squeeze_fun(x_hat) # Here we make a flow undo squeeze to make the correct dimensions. 

        x_mean, x_std  = x_hat.chunk(2, dim=1) 

        if self.loss_select=='MSE':
            recon_sample = x_mean
        elif self.loss_select=='gaussian':
            x_std = nn.Softplus()(x_std)
            recon_sample =td.Normal(x_mean, x_std + self.eps).sample()
        recons[i,:,:,:,:] = recon_sample.detach()

        condition_list = x_feature_list
      return recons

    
    def sample(self, x, n_samples):
        assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
        ht, ct, zt, _, _, _, _ = self.get_inits()
        
        samples = torch.zeros((n_samples, *x[:,0,:,:,:].shape))
        
        condition_list = self.extractor(x[:, 0, :, :, :])
        # 
        for i in range(0, n_samples):
            
            if not self.skip_connection_features:
                condition = condition_list
            else:
                condition = condition_list[-1]
                
            _, ht, ct = self.lstm(condition.unsqueeze(1), ht, ct) 
            
            # We do not need to update the encoder due to LSTM not being conditioned on it
            prior_mean, prior_std = self.prior(torch.cat((ht, zt), dim=1))
            dist_prior = td.Normal(prior_mean, prior_std)
            zt = dist_prior.sample()
            
            if self.skip_connection_features:
                x_hat = self.upscaler(torch.cat((ht, zt), dim = 1), skip_list = condition_list)
            else:
                x_hat = self.upscaler(torch.cat((ht, zt), dim = 1))
            
            x_hat = x_hat[0] # Get only first input

            x_hat = self.squeeze_fun(x_hat) # Here we make a flow undo squeeze to make the correct dimensions. 
    
            x_mean, x_std  = x_hat.chunk(2, dim=1) 
    
            if self.loss_select=='MSE':
                sample = x_mean
            elif self.loss_select=='gaussian':
                x_std = nn.Softplus()(x_std)
                sample = td.Normal(x_mean, x_std + self.eps).sample()
            
            
            samples[i,:,:,:,:] = sample.detach()
            condition_list = self.extractor(sample)
            
        return samples
