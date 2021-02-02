from Flow import ListGlow
import torch
import torch.nn as nn
from Utils import VGG_upscaler, VGG_downscaler, SimpleParamNet, ConvLSTM, free_bits_kl, batch_reduce
import torch.distributions as td

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
      self.free_bits = args.free_bits
      self.skip_connection_flow = args.skip_connection_flow
      self.downscaler_tanh = args.downscaler_tanh
      self.skip_connection_features = args.skip_connection_features
      self.upscaler_tanh = args.upscaler_tanh
      #new:
      self.a_dim = args.a_dim
      self.enable_smoothing=args.enable_smoothing
      self.res_q = args.res_q
      self.D = args.D + 1
      self.overshot_w = args.overshot_w
      
      down_structure = args.extractor_structure
      up_structure = args.upscaler_structure

      if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
          skip = False
      else:
          skip = True
      self.extractor = VGG_downscaler(down_structure,L=self.L, in_channels = self.x_dim[1],
                                      norm_type=norm_type_features, non_lin = "relu", scale = scaler,
                                      skip_con=skip, tanh = self.downscaler_tanh)

      # adjust channel dims to match up_structure. Reversed.
      channel_dims = [i[-1] for i in up_structure][::-1]

      dims_skip = self.extractor.get_layer_size(down_structure, self.x_dim)
      hu, wu = (self.u_dim[2], self.u_dim[3])

      for i in range(0, self.L):
        hu, wu = (hu//2, hu//2)
        if self.skip_connection_flow == "with_skip":
            condition_size_list.append([batch_size, channel_dims[i]+dims_skip[i][1], hu, wu])
        elif self.skip_connection_flow =="without_skip":
            condition_size_list.append([batch_size, channel_dims[i], hu, wu])
        elif self.skip_connection_flow == "only_skip":
            condition_size_list.append([batch_size, dims_skip[i][1], hu, wu])
        else:
            print("choose skip setting")

      c_features = dims_skip[-1][1]

      self.z_0 = nn.Parameter(torch.zeros(batch_size, self.z_dim, hu, wu))
      self.z_0x = nn.Parameter(torch.zeros(batch_size, self.z_dim, hu, wu))

      self.h_0 = nn.Parameter(torch.zeros(batch_size, self.h_dim, hu, wu))
      self.c_0 = nn.Parameter(torch.zeros(batch_size, self.h_dim, hu, wu))

      self.a_0 = nn.Parameter(torch.zeros(batch_size, self.a_dim, hu, wu))
      self.ca_0 = nn.Parameter(torch.zeros(batch_size, self.a_dim, hu, wu))

      # Feature extractor and upscaler for flow
      self.upscaler = VGG_upscaler(up_structure, L=self.L, in_channels = self.h_dim + self.z_dim,
                                   norm_type = norm_type_features, non_lin = "leakyrelu",
                                   scale = scaler, skips = self.skip_connection_features,
                                   size_skips = dims_skip, tanh = self.upscaler_tanh)

      # ConvLSTM
      self.lstm = ConvLSTM(in_channels = c_features, hidden_channels=self.h_dim,
                           kernel_size=[3, 3], bias=True, peephole=True)
      
      self.a_lstm = ConvLSTM(in_channels = c_features + self.h_dim, hidden_channels=self.a_dim,
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
      if self.enable_smoothing:
          self.encoder = SimpleParamNet(enc_struct, in_channels = self.a_dim + self.z_dim,
                                    out_channels = self.z_dim, norm_type = norm_type, non_lin = "leakyrelu")
      else:
          self.encoder = SimpleParamNet(enc_struct, in_channels = c_features + self.h_dim + self.z_dim,
                           out_channels = self.z_dim, norm_type = norm_type, non_lin = "leakyrelu")


    def get_inits(self):
      loss = 0
      kl_loss = 0
      nll_loss = 0
      return self.h_0, self.c_0,self.a_0, self.ca_0, self.z_0, self.z_0x, loss, kl_loss, nll_loss

    def loss(self, x, logdet):
      assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
      hprev, cprev, aprev, caprev, zprev, zxprev, loss, kl_loss, nll_loss = self.get_inits()
      t = x.shape[1]
      store_ht = torch.zeros((t-1, *hprev.shape)).cuda()
      store_at = torch.zeros((t-1, *aprev.shape)).cuda()
      #need to debug if this breaks backprop?! Don't think it does: https://github.com/pytorch/pytorch/issues/23653
      store_x_features = []
      
      #x
      for i in range(0,t):
          x_feature_list = self.extractor(x[:, i, :, :, :])
          store_x_features.append(x_feature_list)
          
      #h
      for i in range(1, t):
        if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
            condition = store_x_features[i-1]
        else:
            condition = store_x_features[i-1][-1]
        _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)
        store_ht[i-1,:,:,:,:] = ht
        hprev=ht
        cprev=ct
        
     
      if self.enable_smoothing:
          #Find at
          for i in range(1, t):
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                x_feature = store_x_features[t-i]
            else:
                x_feature = store_x_features[t-i][-1]
            lstm_a_input = torch.cat([store_ht[t-i-1,:,:,:,:], x_feature], 1)
            _, at, c_at = self.a_lstm(lstm_a_input.unsqueeze(1), aprev, caprev)
            aprev = at
            caprev = c_at
            store_at[t-i-1,:,:,:,:] = at
      
      store_ztx_mean = torch.zeros((t-1, *zxprev.shape)).cuda()
      store_ztx_std = torch.zeros((t-1, *zxprev.shape)).cuda()
      store_ztx = torch.zeros((t-1, *zprev.shape)).cuda()
      for i in range(1, t):
        if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
            x_feature = store_x_features[i]
            
        else:
            x_feature = store_x_features[i][-1]
            
        ht = store_ht[i-1,:,:,:,:]
        
        if self.enable_smoothing:
            at = store_at[i-1,:,:,:,:]
            enc_mean, enc_std = self.encoder(torch.cat((at, zxprev), dim = 1))
        else:
            x_feature = store_x_features[i]
            enc_mean, enc_std = self.encoder(torch.cat((ht, zxprev, x_feature), dim = 1))
        
        if self.res_q:
            prior_mean, prior_std = self.prior(torch.cat((ht, zxprev), dim=1))

            enc_mean = prior_mean + enc_mean
        else:
            prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))

        dist_prior = td.Normal(prior_mean, prior_std)
        zt = dist_prior.rsample()
        
        dist_enc = td.Normal(enc_mean, enc_std)
        zxt = dist_enc.rsample()
        
        store_ztx_mean[i-1,...] = enc_mean
        store_ztx_std[i-1,...] = enc_std
        store_ztx[i-1,...] = zxt
        store_ztx[i-1,...] = zxprev

        if self.skip_connection_features:
            flow_conditions = self.upscaler(torch.cat((ht, zxt), dim = 1), skip_list = store_x_features[i-1])
        else:
            flow_conditions = self.upscaler(torch.cat((ht, zxt), dim = 1))

        base_conditions = torch.cat((ht, zxt), dim = 1)

        if self.skip_connection_flow == "with_skip":
            flow_conditions = self.combineconditions(flow_conditions, store_x_features[i-1])
        elif self.skip_connection_flow == "only_skip":
            flow_conditions = store_x_features[i-1]

        b, nll = self.flow.log_prob(x[:, i, :, :, :], flow_conditions, base_conditions, logdet)


        if self.D == 1:
            kl_loss = kl_loss + td.kl_divergence(dist_enc, dist_prior)

        nll_loss = nll_loss + nll

        zprev, zxprev = zt, zxt, 
        
      if self.D > 1: # is the number of over samples, if D=1 no over shooting will happen.

          overshot_w = self.overshot_w # Weight of overshot.
          Dinit = self.D
          kl_loss = 0

          for i in range(1, t):
              overshot_loss = 0
              idt = i-1 # index t, Does this to make index less confusing
              zprev = store_ztx[idt, : , :]
              D = min(t-i, Dinit) #Do this so that for ts at t-D still do overshoting but with less overshooting # Dont know if this is the correct way to do it but makes sense
              for d in range(0, D): # D is the number of overshootes in paper 1=< d <= D. We do 0<=d<D so now a index offset, but is still the same..
                  ht = store_ht[idt + d, :, :, :, :] # So find the ht for t + d
                  prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
                  dist_prior = td.Normal(prior_mean, prior_std)
                  zprev = dist_prior.rsample()
                  enc_mean = store_ztx_mean[idt + d, :, :]
                  enc_std = store_ztx_std[idt + d, :, :]
                  if d > 0:
                      # .detach() to stop gradients from encoder, such that the encoder does not conform to the prior, but still to recon loss
                      # They do this in the paper for d>0.
                      enc_mean = enc_mean.detach().clone()
                      enc_std = enc_std.detach().clone()
                  
                  enc_dist = td.Normal(enc_mean, enc_std)
                    
                  overshot_loss = overshot_loss + overshot_w * td.kl_divergence(enc_dist, dist_prior)
              kl_loss = kl_loss + 1/D * overshot_loss
      
      if self.free_bits>0:
          kl_free_bit = free_bits_kl(kl_loss, free_bits = self.free_bits)
      else:
          kl_free_bit = kl_loss

      return batch_reduce(kl_free_bit).mean(), batch_reduce(kl_loss).mean(), nll_loss.mean()

    def combineconditions(self, flow_conditions, skip_conditions):
      flow_conditions_combined = []

      for k in range(0, len(flow_conditions)):
        flow_conditions_combined.append(torch.cat((flow_conditions[k], skip_conditions[k]),dim=1))
      return flow_conditions_combined

    def predict(self, x, n_predictions, n_conditions):
        assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
        hprev, cprev, aprev, caprev, zprev, zxprev, _, _, _ = self.get_inits()
        t = x.shape[1]
        
        store_ht = torch.zeros((t-1, *hprev.shape)).cuda()
        store_at = torch.zeros((t-1, *aprev.shape)).cuda()
        store_x_features = []
        
        predictions = torch.zeros((n_predictions, *x[:,0,:,:,:].shape))
        true_x = torch.zeros((n_conditions, *x[:,0,:,:,:].shape))

        # Warm-up
        true_x[0,:,:,:,:] = x[:, 0, :, :, :].detach()
        #x
        for i in range(0,n_conditions):
            x_feature_list = self.extractor(x[:, i, :, :, :])
            store_x_features.append(x_feature_list)
            
        #h
        for i in range(1, n_conditions):
          if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
              condition = store_x_features[i-1]
          else:
              condition = store_x_features[i-1][-1]
          _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)
          store_ht[i-1,:,:,:,:] = ht
          hprev=ht
          cprev=ct
          
       
        if self.enable_smoothing:
            #Find at
            for i in range(1, n_conditions):
              if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                  x_feature = store_x_features[n_conditions-i]
              else:
                  x_feature = store_x_features[n_conditions-i][-1]
              lstm_a_input = torch.cat([store_ht[n_conditions-i-1,:,:,:,:], x_feature], 1)
              _, at, c_at = self.a_lstm(lstm_a_input.unsqueeze(1), aprev, caprev)
              aprev = at
              caprev = c_at
              store_at[n_conditions-i-1,:,:,:,:] = at
        
        for i in range(1, n_conditions):
          if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
              x_feature = store_x_features[i]
          else:
              x_feature = store_x_features[i][-1]
              
          ht = store_ht[i-1,:,:,:,:]
          
          if self.enable_smoothing:
              at = store_at[i-1,:,:,:,:]
              enc_mean, enc_std = self.encoder(torch.cat((at, zxprev), dim = 1))
          else:
              enc_mean, enc_std = self.encoder(torch.cat((ht, zxprev, x_feature), dim = 1))
          
          if self.res_q:
              prior_mean, prior_std = self.prior(torch.cat((ht, zxprev), dim=1))
              enc_mean = prior_mean + enc_mean
          else:
              prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
  
          dist_prior = td.Normal(prior_mean, prior_std)
          zt = dist_prior.rsample()
          
          dist_enc = td.Normal(enc_mean, enc_std)
          zxt = dist_enc.rsample()

          true_x[i,:,:,:,:] = x[:, i, :, :, :].detach()
          zprev = zt
          zxprev = zxt
            

        prediction = x[:,n_conditions-1,:,:,:]
        for i in range(0, n_predictions):
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                condition = self.extractor(prediction)
            else:
                condition_list = self.extractor(prediction)
                condition = condition_list[-1]
            _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)

            prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
            dist_prior = td.Normal(prior_mean, prior_std)
            zt = dist_prior.sample()

            if self.skip_connection_features:
                flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1), skip_list = condition_list)
            else:
                flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1))

            if self.skip_connection_flow == "with_skip":
                flow_conditions = self.combineconditions(flow_conditions, condition_list)
            elif self.skip_connection_flow == "only_skip":
                flow_conditions = condition_list

            base_conditions = torch.cat((ht, zt), dim = 1)
            prediction = self.flow.sample(None, flow_conditions, base_conditions, self.temperature)
            predictions[i,:,:,:,:] = prediction.detach()
            hprev, cprev = ht, ct
            zprev = zt

        return true_x, predictions

    def reconstruct(self, x):
        assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
        hprev, cprev,aprev,caprev, _, zxprev, _, _, _ = self.get_inits()

        t = x.shape[1]
        recons = torch.zeros((t, *x[:,0,:,:,:].shape))
        recons_flow = torch.zeros((t, *x[:,0,:,:,:].shape))
        store_ht = torch.zeros((t-1, *hprev.shape)).cuda()
        store_at = torch.zeros((t-1, *aprev.shape)).cuda()
        store_x_features = []
        
        #x
        for i in range(0,t):
            x_feature_list = self.extractor(x[:, i, :, :, :])
            store_x_features.append(x_feature_list)
              
          #h
        for i in range(1, t):
          if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
              condition = store_x_features[i-1]
          else:
              condition = store_x_features[i-1][-1]
          _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)
          store_ht[i-1,:,:,:,:] = ht
          hprev=ht
          cprev=ct
            
         
        if self.enable_smoothing:
            #Find at
            for i in range(1, t):
              if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                  x_feature = store_x_features[t-i]
              else:
                  x_feature = store_x_features[t-i][-1]
              lstm_a_input = torch.cat([store_ht[t-i-1,:,:,:,:], x_feature], 1)
              _, at, c_at = self.a_lstm(lstm_a_input.unsqueeze(1), aprev, caprev)
              aprev = at
              caprev = c_at
              store_at[t-i-1,:,:,:,:] = at
        
        for i in range(1, t):
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                x_feature = store_x_features[i]
            else:
                x_feature = store_x_features[i][-1]
            
            ht = store_ht[i-1,:,:,:,:]

            if self.enable_smoothing:
                at = store_at[i-1,:,:,:,:]
                enc_mean, enc_std = self.encoder(torch.cat((at, zxprev), dim = 1))
            else:
                x_feature = store_x_features[i]
                enc_mean, enc_std = self.encoder(torch.cat((ht, zxprev, x_feature), dim = 1))
            
            if self.res_q:
                prior_mean, _ = self.prior(torch.cat((ht, zxprev), dim=1))
    
                enc_mean = prior_mean + enc_mean
    
            dist_enc = td.Normal(enc_mean, enc_std)
            zxt = dist_enc.rsample()

            if self.skip_connection_features:
                flow_conditions = self.upscaler(torch.cat((ht, zxt), dim = 1), skip_list = store_x_features[i-1])
            else:
                flow_conditions = self.upscaler(torch.cat((ht, zxt), dim = 1))

            if self.skip_connection_flow == "with_skip":
                flow_conditions = self.combineconditions(flow_conditions, store_x_features[i-1])
            elif self.skip_connection_flow == "only_skip":
                flow_conditions = store_x_features[i-1]

            base_conditions = torch.cat((ht, zxt), dim = 1)

            # To check bijection of the flow we use z to reconstruct the image
            z, _ = self.flow.log_prob(x[:, i, :, :, :], flow_conditions, base_conditions, 0.0)
            recon_flow_sample = self.flow.sample(z, flow_conditions, base_conditions, self.temperature)

            # To look at a normal reconstruction we just use the posterior distribution
            recon_sample = self.flow.sample(None, flow_conditions, base_conditions, self.temperature)

            recons[i,:,:,:,:] = recon_sample.detach()
            recons_flow[i,:,:,:,:] = recon_flow_sample.detach()

            hprev = ht
            cprev = ct
            zxprev = zxt
        return recons, recons_flow


    def sample(self, x, n_samples):
        assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
        hprev, cprev,_,_, zprev, _, _, _, _ = self.get_inits()

        samples = torch.zeros((n_samples, *x[:,0,:,:,:].shape))

        condition_list = self.extractor(x[:, 0, :, :, :])
        #
        for i in range(0, n_samples):

            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                condition = condition_list
            else:
                condition = condition_list[-1]

            _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)

            # We do not need to update the encoder due to LSTM not being conditioned on it
            prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
            dist_prior = td.Normal(prior_mean, prior_std)
            zt = dist_prior.sample()

            if self.skip_connection_features:
                flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1), skip_list = condition_list)
            else:
                flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1))

            if self.skip_connection_flow == "with_skip":
                flow_conditions = self.combineconditions(flow_conditions, condition_list)
            elif self.skip_connection_flow == "only_skip":
                flow_conditions = condition_list

            base_conditions = torch.cat((ht, zt), dim = 1)
            sample = self.flow.sample(None, flow_conditions, base_conditions, self.temperature)

            samples[i,:,:,:,:] = sample.detach()
            zprev = zt
            hprev = ht
            cprev = ct
            condition_list = self.extractor(sample)

        return samples

    def param_analysis(self, x, n_predictions, n_conditions):
        assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
        hprev, cprev,aprev,caprev, zprev, zxprev, _, _, _ = self.get_inits()  
        hw = x.shape[-1]//(2**self.L)
        std_p = torch.zeros((n_predictions+n_conditions-1, x.shape[0], self.z_dim, hw, hw))
        mu_p = torch.zeros((n_predictions+n_conditions-1, x.shape[0], self.z_dim, hw, hw))
        std_q = torch.zeros((n_predictions+n_conditions-1, x.shape[0], self.z_dim, hw, hw))
        mu_q = torch.zeros((n_predictions+n_conditions-1, x.shape[0], self.z_dim, hw, hw))
        t = n_conditions + n_predictions
        store_ht = torch.zeros((t-1, *hprev.shape)).cuda()
        store_at = torch.zeros((t-1, *aprev.shape)).cuda()
        store_x_features = []
        std_flow = []
        mu_flow = []
        
        for i in range(0,t):
            x_feature_list = self.extractor(x[:, i, :, :, :])
            store_x_features.append(x_feature_list)
        
        for i in range(1, t):
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                condition = store_x_features[i-1]
            else:
                condition = store_x_features[i-1][-1]
            _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)
            store_ht[i-1,:,:,:,:] = ht
            hprev=ht
            cprev=ct
        
        if self.enable_smoothing:
            #Find at
            for i in range(1, t):
              if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                  x_feature = store_x_features[t-i]
              else:
                  x_feature = store_x_features[t-i][-1]
              lstm_a_input = torch.cat([store_ht[t-i-1,:,:,:,:], x_feature], 1)
              _, at, c_at = self.a_lstm(lstm_a_input.unsqueeze(1), aprev, caprev)
              aprev = at
              caprev = c_at
              store_at[t-i-1,:,:,:,:] = at
        
        for i in range(1, n_conditions + n_predictions):
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                x_feature = store_x_features[i]
            else:
                x_feature = store_x_features[i][-1]
            
            ht = store_ht[i-1,:,:,:,:]
            
            if self.enable_smoothing:
                at = store_at[i-1,:,:,:,:]
                enc_mean, enc_std = self.encoder(torch.cat((at, zxprev), dim = 1))
            else:
                x_feature = store_x_features[i]
                enc_mean, enc_std = self.encoder(torch.cat((ht, zxprev, x_feature), dim = 1))
            
            if self.res_q:
                prior_mean, prior_std = self.prior(torch.cat((ht, zxprev), dim=1))
    
                enc_mean = prior_mean + enc_mean
            else:
                prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
            
            dist_prior = td.Normal(prior_mean, prior_std)
            zt = dist_prior.rsample()
            
            dist_enc = td.Normal(enc_mean, enc_std)
            zxt = dist_enc.rsample()
            
            std_p[i-1,:,:,:,:] = prior_std.detach()
            mu_p[i-1,:,:,:,:] = prior_mean.detach()
            std_q[i-1,:,:,:,:] = enc_std.detach()
            mu_q[i-1,:,:,:,:] = enc_mean.detach()
            
            if self.skip_connection_features:
                flow_conditions = self.upscaler(torch.cat((ht, zxt), dim = 1), skip_list = store_x_features[i-1])
            else:
                flow_conditions = self.upscaler(torch.cat((ht, zxt), dim = 1))
            
            if self.skip_connection_flow == "with_skip":
                flow_conditions = self.combineconditions(flow_conditions, store_x_features[i-1])
            elif self.skip_connection_flow == "only_skip":
                flow_conditions = store_x_features[i-1]
            
            base_conditions = torch.cat((ht, zt), dim = 1)
            prediction, params = self.flow.sample(None, flow_conditions, base_conditions, 1.0, eval_params = True)
            std_flow.append(params[1].detach())
            mu_flow.append(params[0].detach())
            
            zxprev = zxt
            zprev = zt
            
        return mu_p, std_p, mu_q, std_q, torch.stack(mu_flow), torch.stack(std_flow)
