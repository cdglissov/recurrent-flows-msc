from Flow import ListGlow
import torch
import torch.nn as nn
from Utils import Encoder, Decoder, init_weights, gaussian_lstm, lstm_svg
import torch.distributions as td
from Utils import set_gpu

device = set_gpu(True)



class SVG(nn.Module):
    def __init__(self, args):
        super(SVG, self).__init__()
        z_dim=args.z_dim 
        c_features = args.c_features
        h_dim = args.h_dim
        posterior_rnn_layers = args.posterior_rnn_layers
        predictor_rnn_layers = args.predictor_rnn_layers
        prior_rnn_layers = args.prior_rnn_layers
        
        x_dim = args.x_dim
        self.batch_size, channels, hx, wx = x_dim
        self.n_past = args.n_past
        self.n_future = args.n_future
        self.beta = 1 
        self.temperature = args.temperature 
        
        ### Flow settings
        L=args.L 
        channels_flow = [32, 64, 128, 256, 512]
        condition_size_list = []
        for i in range(0, L):
          hx, wx = (hx//2, hx//2)
          condition_size_list.append([self.batch_size, channels_flow[i], hx, wx])
        condition_size_list

        base_dim = (self.batch_size, 512, hx, wx)
        self.flow = ListGlow(x_dim, condition_size_list, base_dim, args)
        ###
        
        self.encoder = Encoder(c_features, channels)
        self.decoder = Decoder(c_features, channels)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        
        self.frame_predictor = lstm_svg(c_features+z_dim, c_features, h_dim, predictor_rnn_layers, self.batch_size)
        self.posterior = gaussian_lstm(c_features, z_dim, h_dim, posterior_rnn_layers, self.batch_size)
        self.prior = gaussian_lstm(c_features, z_dim, h_dim, prior_rnn_layers, self.batch_size)
        self.frame_predictor.apply(init_weights)
        self.posterior.apply(init_weights)
        self.prior.apply(init_weights)
        
        learning_rate = args.learning_rate
        if args.optimizer == "rmsprop":
            c_optimizer = torch.optim.RMSprop
        else:
            c_optimizer = torch.optim.Adam
        betas_low = args.betas_low
        self.frame_predictor_optimizer = c_optimizer(self.frame_predictor.parameters(), lr=learning_rate, betas=(betas_low, 0.999))
        self.posterior_optimizer = c_optimizer(self.posterior.parameters(), lr=learning_rate, betas=(betas_low, 0.999))
        self.prior_optimizer = c_optimizer(self.prior.parameters(), lr=learning_rate, betas=(betas_low, 0.999))
        self.encoder_optimizer = c_optimizer(self.encoder.parameters(), lr=learning_rate, betas=(betas_low, 0.999))
        self.decoder_optimizer = c_optimizer(self.decoder.parameters(), lr=learning_rate, betas=(betas_low, 0.999))
        self.flow_optimizer = c_optimizer(self.flow.parameters(), lr=learning_rate, betas=(betas_low, 0.999))
        
        self.frame_predictor.to(device)
        self.posterior.to(device)
        self.prior.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.flow.to(device)
    
        # Schedulers
        patience_lr = args.patience_lr
        factor_lr = args.factor_lr
        min_lr = args.min_lr
        self.sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.frame_predictor_optimizer, 'min', 
                            patience=patience_lr, 
                            factor=factor_lr, 
                            min_lr=min_lr)
        self.sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.posterior_optimizer, 'min', 
                            patience=patience_lr, 
                            factor=factor_lr, 
                            min_lr=min_lr)
        self.sched3 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.prior_optimizer, 'min', 
                            patience=patience_lr, 
                            factor=factor_lr, 
                            min_lr=min_lr)
        self.sched4 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.encoder_optimizer, 'min', 
                            patience=patience_lr, 
                            factor=factor_lr, 
                            min_lr=min_lr)
        self.sched5 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, 'min', 
                            patience=patience_lr, 
                            factor=factor_lr, 
                            min_lr=min_lr)
        self.sched6 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.flow_optimizer, 'min', 
                            patience=patience_lr, 
                            factor=factor_lr, 
                            min_lr=min_lr)
        
    def loss(self, x, logdet):
      self.frame_predictor.zero_grad()
      self.posterior.zero_grad()
      self.prior.zero_grad()
      self.encoder.zero_grad()
      self.decoder.zero_grad()
      self.flow.zero_grad()
    
      self.frame_predictor.hidden = self.frame_predictor.init_hidden()
      self.posterior.hidden = self.posterior.init_hidden()
      self.prior.hidden = self.prior.init_hidden()
    
      nll = 0
      kld = 0
      
    
      for i in range(1, self.n_past + self.n_future):
          h, skip = self.encoder(x[:,i-1,:,:,:])
          h_target = self.encoder(x[:,i,:,:,:])[0]
          
          z_t, mu, logvar = self.posterior(h_target)
          _, mu_p, logvar_p = self.prior(h)
          h_pred = self.frame_predictor(torch.cat([h, z_t], 1))
          
          x_upscaled = self.decoder([h_pred, skip])[::-1]
          base_conditions = x_upscaled[-1]#h_pred.view(self.batch_size, -1, 1, 1)
          _, nll = self.flow.log_prob(x[:, i, :, :, :], x_upscaled, base_conditions, logdet)
    
          dist_enc = td.Normal(mu, torch.exp(logvar))
          dist_prior = td.Normal(mu_p, torch.exp(logvar_p))
          nll += nll
          kld += (td.kl_divergence(dist_enc, dist_prior).sum() / self.batch_size)
      
      loss = nll + kld*self.beta
      loss.backward()
      
    
      self.frame_predictor_optimizer.step()
      self.posterior_optimizer.step()
      self.prior_optimizer.step()
      self.encoder_optimizer.step()
      self.decoder_optimizer.step()
      self.flow_optimizer.step()
      
      self.sched1.step(loss)
      self.sched2.step(loss)
      self.sched3.step(loss)
      self.sched4.step(loss)
      self.sched5.step(loss)
      self.sched6.step(loss)
      
      return kld.data, nll.data, loss.data
    
    
    def plot_rec(self, x, encoder_sample):
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        gen_seq = []
        gen_seq.append(x[:, 0,:,:,:])
        
        for i in range(1, self.n_past + self.n_future):
            h, skip  = self.encoder(x[:,i-1,:,:,:])
            h_target = self.encoder(x[:,i,:,:,:])[0]
            
            h = h.detach()
            h_target = h_target.detach()
            if encoder_sample:
                z_t, _, _= self.posterior(h_target)
            else:
                z_t, _, _ = self.prior(h)
            if i < self.n_past: 
                gen_seq.append(x[:,i,:,:,:])
            else:
                h_pred = self.frame_predictor(torch.cat([h, z_t], 1))
                x_upscaled = self.decoder([h_pred, skip])[::-1]
                base_conditions = x_upscaled[-1]#h_pred.view(self.batch_size, -1, 1, 1)
                sample = self.flow.sample(None, x_upscaled, base_conditions, temperature = self.temperature)
                gen_seq.append(sample)
        
        return torch.stack(gen_seq, dim=0)

          