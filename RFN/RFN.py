from Flow import ListGlow
import torch
import torch.nn as nn
from Utils import VGG_upscaler, VGG_downscaler, SimpleParamNet, ConvLSTM, ConvLSTMOld
import torch.distributions as td

# Actually seems very promising to only give input from extractor, test against skip connections.
# Stronger prior seems to be beneficial. Also there is some indication a stronger split is helpful.
# 

class RFN(nn.Module):
    def __init__(self, args):
      super(RFN, self).__init__()
      self.params = args
      batch_size = args.batch_size
      self.u_dim = args.x_dim
      self.x_dim = args.condition_dim
      self.h_dim = args.h_dim
      self.z_dim = args.z_dim
      self.beta = 1
      scaler = args.structure_scaler
      self.L = args.L
      self.K = args.K
      norm_type = args.norm_type
      norm_type_features = args.norm_type_features
      self.temperature = args.temperature
      self.prior_structure = args.prior_structure
      self.encoder_structure = args.encoder_structure
      condition_size_list = []
      self.skip_connection=args.skip_connection
      down_structure = args.extractor_structure 
      up_structure = args.upscaler_structure
      
      
      self.extractor = VGG_downscaler(down_structure,L=self.L, in_channels = self.x_dim[1], 
                                      norm_type=norm_type_features, non_lin = "leakyrelu", scale = scaler, 
                                      skip_con=self.skip_connection)
      # adjust channel dims to match up_structure. Reversed.
      channel_dims = [i[-1] for i in up_structure][::-1]

      dims_skip = self.extractor.get_layer_size(down_structure, self.x_dim)
      hu, wu = (self.u_dim[2], self.u_dim[3])
      
      for i in range(0, self.L):
        hu, wu = (hu//2, hu//2)
        if self.skip_connection == "with_skip":
            condition_size_list.append([batch_size, channel_dims[i]+dims_skip[i][1], hu, wu])
        elif self.skip_connection =="without_skip":
            condition_size_list.append([batch_size, channel_dims[i], hu, wu])
        elif self.skip_connection == "only_skip":
            condition_size_list.append([batch_size, dims_skip[i][1], hu, wu])
        else:
            print("choose skip setting")
            
      c_features = dims_skip[-1][1]
      
      self.z_0 = nn.Parameter(torch.zeros(batch_size, self.z_dim, hu, wu))
      self.z_0x = nn.Parameter(torch.zeros(batch_size, self.z_dim, hu, wu))
      
      self.h_0 = nn.Parameter(torch.zeros(batch_size, self.h_dim, hu, wu))
      self.c_0 = nn.Parameter(torch.zeros(batch_size, self.h_dim, hu, wu))
      
      # Feature extractor and upscaler for flow
      self.upscaler = VGG_upscaler(up_structure, L=self.L, in_channels = self.h_dim + self.z_dim, 
                                   norm_type = norm_type_features, non_lin = "leakyrelu", scale = scaler)

      # ConvLSTM
      self.lstm = ConvLSTMOld(in_channels = c_features, hidden_channels=self.h_dim, 
                           kernel_size=[3, 3], bias=True, peephole=True)

      # Prior
      prior_struct = self.prior_structure
      self.prior = SimpleParamNet(prior_struct, in_channels = self.h_dim + self.z_dim, 
                                  out_channels = self.z_dim, norm_type = norm_type, non_lin = "leakyrelu")
      
      # Flow
      base_dim = (batch_size, self.h_dim + self.z_dim, hu, wu)
      self.flow = ListGlow(self.x_dim, condition_size_list, base_dim, 
                           args=self.params)

      # Variational encoder
      enc_struct = self.encoder_structure
      self.encoder = SimpleParamNet(enc_struct, in_channels = c_features + self.h_dim + self.z_dim, 
                                    out_channels = self.z_dim, norm_type = norm_type, non_lin = "leakyrelu")
      

    def get_inits(self):
      loss = 0
      kl_loss = 0
      nll_loss = 0
      return self.h_0, self.c_0, self.z_0, self.z_0x, loss, kl_loss, nll_loss

    def loss(self, x, logdet):
      assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
      hprev, cprev, zprev, zxprev, loss, kl_loss, nll_loss = self.get_inits()
      t = x.shape[1]
      condition_list = self.extractor(x[:, 0, :, :, :])
      
      for i in range(1, t):
        
        x_feature_list = self.extractor(x[:, i, :, :, :])
        
        if self.skip_connection == "without_skip":
            condition = condition_list
            x_feature = x_feature_list
        else:
            condition = condition_list[-1]
            x_feature = x_feature_list[-1]

        
        _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev) 
        
        # TODO: maybe try to make another LSTM but only for the prior.
        prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
        dist_prior = td.Normal(prior_mean, prior_std)
        zt = dist_prior.rsample()

        enc_mean, enc_std = self.encoder(torch.cat((ht, zxprev, x_feature), dim = 1))
        dist_enc = td.Normal(enc_mean, enc_std)
        zxt = dist_enc.rsample()
        
        flow_conditions = self.upscaler(torch.cat((ht, zxt), dim = 1))
        base_conditions = torch.cat((ht, zxt), dim = 1)
        
        if self.skip_connection=="with_skip":
            flow_conditions = self.combineconditions(flow_conditions, condition_list)
        elif self.skip_connection=="only_skip":
            flow_conditions = condition_list

        b, nll = self.flow.log_prob(x[:, i, :, :, :], flow_conditions, base_conditions, logdet)
        
        kl_loss = kl_loss + td.kl_divergence(dist_enc, dist_prior).sum([1,2,3]).mean()
        nll_loss = nll_loss + nll 

        zprev, zxprev, condition_list = zt, zxt, x_feature_list
        hprev, cprev = ht ,ct
        
      return kl_loss, nll_loss
    
    def combineconditions(self,flow_conditions, skip_conditions):
      flow_conditions_combined = []
      
      for k in range(0,len(flow_conditions)):
        flow_conditions_combined.append(torch.cat((flow_conditions[k], skip_conditions[k]),dim=1))
        
      return flow_conditions_combined
  
    def sample(self, x, n_predictions=6, encoder_sample = False, start_predictions = None):
      assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
      hprev, cprev, zprev, zxprev, _, _, _ = self.get_inits()
      if start_predictions is not None:
          t = start_predictions
      else:
          t = x.shape[1]

      samples = torch.zeros((t-1, *x[:,0,:,:,:].shape))
      samples_recon = torch.zeros((t-1, *x[:,0,:,:,:].shape))

      condition_list = self.extractor(x[:, 0, :, :, :])

      for i in range(1, t):
        
        x_feature_list = self.extractor(x[:, i, :, :, :])
        
        if self.skip_connection == "without_skip":
            condition = condition_list
            x_feature = x_feature_list
        else:
            condition = condition_list[-1]
            x_feature = x_feature_list[-1]
        _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev) 
        
        enc_mean, enc_std = self.encoder(torch.cat((ht, zxprev, x_feature), dim = 1))
        dist_enc = td.Normal(enc_mean, enc_std)
        zxt = dist_enc.sample()
        prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
        dist_prior = td.Normal(prior_mean, prior_std)
        zt = dist_prior.sample()
        
        if encoder_sample:
          flow_conditions = self.upscaler(torch.cat((ht, zxt), dim = 1))
          base_conditions = torch.cat((ht, zxt), dim = 1)
        else:
          flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1))
          base_conditions = torch.cat((ht, zt), dim = 1)
        
        if self.skip_connection=="with_skip":
            flow_conditions = self.combineconditions(flow_conditions, condition_list)
        elif self.skip_connection=="only_skip":
            flow_conditions = condition_list

        sample = self.flow.sample(None, flow_conditions, base_conditions, self.temperature)
        z, _ = self.flow.log_prob(x[:, i, :, :, :], flow_conditions, base_conditions, 0.0)
        sample_recon = self.flow.sample(z, flow_conditions, base_conditions, self.temperature)
        
        condition_list, zxprev, zprev = x_feature_list, zxt, zt
        hprev, cprev = ct.detach(), ct.detach()
        samples[i-1,:,:,:,:] = sample.detach()
        samples_recon[i-1,:,:,:,:] = sample_recon.detach()
      
      # Make predictions
      predictions = torch.zeros((n_predictions, *x[:,0,:,:,:].shape))
      prediction = sample
      for i in range(0, n_predictions):
          
        if self.skip_connection == "without_skip":
            condition = self.extractor(prediction)
        else:
            condition_list = self.extractor(prediction)
            condition = condition_list[-1]
        _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)  
        
        prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
        dist_prior = td.Normal(prior_mean, prior_std)
        zt = dist_prior.sample()

        flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1))
        
        if self.skip_connection=="with_skip":
            flow_conditions = self.combineconditions(flow_conditions, condition_list)
        elif self.skip_connection=="only_skip":
            flow_conditions = condition_list
            
        base_conditions = torch.cat((ht, zt), dim = 1)
        
        predictions[i,:,:,:,:] = prediction.detach()
        prediction = self.flow.sample(None, flow_conditions, base_conditions, self.temperature)
        hprev, cprev = ht.detach(), ct.detach()
        zprev = zt
      return samples, samples_recon, predictions

