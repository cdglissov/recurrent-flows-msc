from cFlow import ListGlow
import torch
import torch.nn as nn
from Utils import VGG_downscaler

class cGlow(nn.Module):
    def __init__(self, args):
      super(cGlow, self).__init__()
      self.params = args
      batch_size = args.batch_size
      self.u_dim = args.x_dim
      self.x_dim = args.condition_dim
      scaler = args.structure_scaler
      self.one_condition = args.one_condition
      self.L = args.L
      self.K = args.K
      norm_type_features = args.norm_type_features
      
      self.downscaler_tanh = args.downscaler_tanh
      self.temperature = args.temperature
      condition_size_list = []
      
      
      cu, hu, wu = (self.u_dim[1], self.u_dim[2], self.u_dim[3])
      down_structure = args.extractor_structure
      if not self.one_condition:
          self.extractor = VGG_downscaler(down_structure,L=self.L, in_channels = self.x_dim[1], 
                                          norm_type=norm_type_features, non_lin = "relu", scale = scaler, 
                                          skip_con=True, tanh = self.downscaler_tanh)
      
        # adjust channel dims to match up_structure. Reversed.
          dims_skip = self.extractor.get_layer_size(down_structure, self.x_dim)
          for i in range(0, self.L):
            hu, wu = (hu//2, hu//2)
            condition_size_list.append([batch_size, dims_skip[i][1], hu, wu])
          cu = dims_skip[-1][1]
      else:
          for i in range(0, self.L):
            condition_size_list.append([batch_size, cu, hu, wu]) # So it is just the same size for all.
       # Flow
      base_dim = (batch_size, cu, hu, wu)
      self.flow = ListGlow(self.x_dim, condition_size_list, base_dim, 
                           args=self.params)

      # Variational encoder


    def loss(self, x, condition, logdet):
      assert len(x.shape) == 4, "x must be [bs, c, h, w]"
      if not self.one_condition:
          condition_list = self.extractor(condition)
          base_conditions = condition_list[-1]
          condition = condition_list
      else:
          condition_list = []
          for i in range(0,self.L):
              condition_list.append(condition)
          base_conditions = condition
          condition = condition_list
    
      
      b, nll = self.flow.log_prob(x, condition, base_conditions, logdet)
        
      return nll
    
  
    def sample(self, x, condition, n_samples=2):
      assert len(x.shape) == 4, "x must be [bs, c, h, w]"

      samples = torch.zeros((n_samples, *x[:,:,:,:].shape))
      samples_recon = torch.zeros((n_samples, *x[:,:,:,:].shape))

      if not self.one_condition:
          condition_list = self.extractor(condition)
          base_conditions = condition_list[-1]
          condition = condition_list
      else:
          condition_list = []
          for i in range(0,self.L):
              condition_list.append(condition)
          base_conditions = condition
          condition = condition_list
    
      for i in range(0,n_samples):
          sample = self.flow.sample(None, condition, base_conditions, self.temperature)
          z, _ = self.flow.log_prob(x[:, :, :, :], condition, base_conditions, 0.0)
          sample_recon = self.flow.sample(z, condition, base_conditions, self.temperature)
          samples[i-1,:,:,:,:] = sample.detach()
          samples_recon[i-1,:,:,:,:] = sample_recon.detach()
      return samples, samples_recon
