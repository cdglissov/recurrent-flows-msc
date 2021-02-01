''' 
!!! CREDITS !!!
This code is mainly taken from and inspired by https://github.com/edenton/svg
Which is the official SVG implementation
'''

import torch
import torch.nn as nn
from Utils import get_layer_size, Flatten, UnFlatten, set_gpu, batch_reduce
from torch.autograd import Variable
import torch.distributions as td
device = set_gpu(True)


class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, input):
        return self.main(input)

class Encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(Encoder, self).__init__()
        self.dim = dim
        # 64 x 64
        self.c1 = nn.Sequential(
                vgg_layer(nc, 64),
                vgg_layer(64, 64),
                )
        
        # 32 x 32
        self.c2 = nn.Sequential(
                vgg_layer(64, 128),
                vgg_layer(128, 128),
                )
        
        # 16 x 16 
        self.c3 = nn.Sequential(
                vgg_layer(128, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 256),
                )
        
        # 8 x 8
        self.c4 = nn.Sequential(
                vgg_layer(256, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 512),
                )
        
        # 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h1 = self.c1(input) # 64x64
        h2 = self.c2(self.mp(h1)) # 16x16
        h3 = self.c3(self.mp(h2)) # 8x8
        h4 = self.c4(self.mp(h3)) # 4x4
        h5 = self.c5(self.mp(h4)) # 1x1
        return h5.view(-1, self.dim), [h1, h2, h3, h4]

class Decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(Decoder, self).__init__()
        self.dim = dim
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # 8 x 8
        self.upc2 = nn.Sequential(
                vgg_layer(512*2, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 256)
                )
        # 16 x 16
        self.upc3 = nn.Sequential(
                vgg_layer(256*2, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 128)
                )
        # 32 x 32
        self.upc4 = nn.Sequential(
                vgg_layer(128*2, 128),
                vgg_layer(128, 64)
                )

        self.out = nn.Sequential(
                vgg_layer(64*2, 64),
                nn.ConvTranspose2d(64, nc, 3, 1, 1),
                nn.Sigmoid()
                )
        
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        vec, skip = input 
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 2
        up1 = self.up(d1) # 2 -> 4
        d2 = self.upc2(torch.cat([up1, skip[3]], 1)) # 4 x 4
        up2 = self.up(d2) # 4 -> 8 
        d3 = self.upc3(torch.cat([up2, skip[2]], 1)) # 8 x 8 
        up3 = self.up(d3) # 8 -> 16 
        d4 = self.upc4(torch.cat([up3, skip[1]], 1)) # 16 x 16
        up4 = self.up(d4) # 16 -> 32
        output = self.out(torch.cat([up4, skip[0]], 1)) # 64 x 64
        return output

class lstm_svg(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(lstm_svg, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)

class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.std_net = nn.Sequential(nn.Linear(hidden_size, output_size),
                                        nn.Softplus())
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        stdp = self.std_net(h_in)
        z = self.reparameterize(mu, stdp)
        return z, mu, stdp

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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
        self.loss_type = args.loss_type
        self.batch_size, channels, hx, wx = x_dim
        self.n_conditions = args.n_conditions
        self.n_predictions = args.n_predictions
        self.mse_criterion = nn.MSELoss(reduction="none")
        self.encoder = Encoder(c_features, channels)
        self.decoder = Decoder(c_features, channels)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        
        self.frame_predictor = lstm_svg(c_features+z_dim, c_features, h_dim, predictor_rnn_layers, self.batch_size).to(device)
        self.posterior = gaussian_lstm(c_features, z_dim, h_dim, posterior_rnn_layers, self.batch_size).to(device)
        self.prior = gaussian_lstm(c_features, z_dim, h_dim, prior_rnn_layers, self.batch_size).to(device)
        
        self.frame_predictor.apply(init_weights).to(device)
        self.posterior.apply(init_weights).to(device)
        self.prior.apply(init_weights).to(device)
          

        
    def loss(self, x):
      self.frame_predictor.hidden = self.frame_predictor.init_hidden()
      self.posterior.hidden = self.posterior.init_hidden()
      self.prior.hidden = self.prior.init_hidden()
      t=x.shape[1]
      nll = 0
      kl = 0
    
      for i in range(1, t):
          h, skip = self.encoder(x[:,i-1,:,:,:])
          h_target = self.encoder(x[:,i,:,:,:])[0]
          z_t, mu_q, logvar_q = self.posterior(h_target)
          _, mu_p, logvar_p = self.prior(h)
          h_pred = self.frame_predictor(torch.cat([h, z_t], 1))
          
          x_pred = self.decoder([h_pred, skip])
 
    
          if self.loss_type == "bernoulli":
              nll = nll - td.Bernoulli(probs=x_pred).log_prob(x[:, i, :, :, :])
          elif self.loss_type == "mse":
              nll = nll + self.mse_criterion(x_pred, x[:, i, :, :, :])
          else:
              print("undefined loss")
          

          kl = kl + self.kl_criterion(mu_q, logvar_q, mu_p, logvar_p)
          
      
      return kl, batch_reduce(nll).mean()
    
    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
    
        sigma1 = logvar1.mul(0.5).exp() 
        sigma2 = logvar2.mul(0.5).exp() 
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld.sum() / self.batch_size
    
    def reconstruct(self, x):
        b,t,c,h,w=x.shape
        
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()
        
        recons = torch.zeros((t, *x[:,0,:,:,:].shape))
        
        for i in range(1, t):
            condition, skip  = self.encoder(x[:,i-1,:,:,:])
            target = self.encoder(x[:,i,:,:,:])[0]
            target = target.detach()
            condition = condition.detach()
            z_t, _, _= self.posterior(target)
            h_pred = self.frame_predictor(torch.cat([condition, z_t], 1))
            x_pred = self.decoder([h_pred, skip])
            recons[i,:,:,:,:] = x_pred.detach()
        return recons
    
    def sample(self, x, n_samples):
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()
        b,t,c,h,w=x.shape
        
        samples = torch.zeros((n_samples, *x[:,0,:,:,:].shape))
        
        condition_x = x[:,0,:,:,:]
        
        for i in range(1, n_samples):
            condition, skip  = self.encoder(condition_x)
            condition = condition.detach()
            
            z_t, _, _ = self.prior(condition)
            h_pred = self.frame_predictor(torch.cat([condition, z_t], 1))
            x_pred = self.decoder([h_pred, skip])
            samples[i,:,:,:,:] = x_pred.detach()
            condition_x = x_pred
        return samples

    def predict(self, x, n_predictions, n_conditions):
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()
        
        b,t,c,h,w=x.shape
        assert n_conditions <= t, "n_conditions > t, number of conditioned frames is greater than number of frames"
      
        predictions = torch.zeros((n_predictions, *x[:,0,:,:,:].shape))
        true_x = torch.zeros((n_conditions, *x[:,0,:,:,:].shape))
        true_x[0,:,:,:,:]=x[:,0,:,:,:]
        x_in = x[:,0,:,:,:]
        for i in range(1, n_predictions+n_conditions):
            condition, skip  = self.encoder(x_in)
            condition=condition.detach()
            if i < n_conditions:
                target = self.encoder(x[:,i,:,:,:])[0]
                target=target.detach()
                z_t, _, _ = self.posterior(target)
                self.prior(condition)
                self.frame_predictor(torch.cat([condition, z_t], 1))
                true_x[i,:,:,:,:]=x[:,i,:,:,:]
                x_in=x[:,i,:,:,:]
            else:
                z_t, _, _ = self.prior(condition)
                h_pred = self.frame_predictor(torch.cat([condition, z_t], 1))
                x_in = self.decoder([h_pred, skip])
                predictions[i-n_conditions,:,:,:,:] = x_in.data
        return true_x, predictions



    

