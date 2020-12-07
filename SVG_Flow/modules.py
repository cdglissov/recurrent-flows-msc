import torch
import torch.nn as nn
from utils import set_gpu
device = set_gpu(True)
from torch.autograd import Variable

class ActFun(nn.Module):
  def __init__(self, non_lin, in_place = False):
    super(ActFun, self).__init__()
    if non_lin=='relu':
      self.net=nn.ReLU(inplace = in_place)
    elif non_lin=='leakyrelu':
      self.net=nn.LeakyReLU(negative_slope=0.20, inplace = in_place)
    else:
      assert False, 'Please specify a activation type from the set {relu,leakyrelu}'
 
  def forward(self,x):
    return self.net(x)

class NoNorm(nn.Module):
    def __init__(self):
      super(NoNorm, self).__init__()
    
    def forward(self, x):
      return x

class NormLayer(nn.Module):
    def __init__(self, in_channels, norm_type):
      super(NormLayer, self).__init__()
      if norm_type =='batchnorm':
        self.norm = nn.BatchNorm2d(in_channels)
      elif norm_type =='instancenorm':
        self.norm = nn.InstanceNorm2d(in_channels)
      elif norm_type =='none':
        self.norm = NoNorm()
      else:
        assert False, 'Please specify a norm type from the set {batchnorm, instancenorm, none}'
    
    def forward(self, x):
      return self.norm(x)

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
        self.c0 = nn.Sequential(
                vgg_layer(nc, 32),
                vgg_layer(32, 32),
                )
        self.c1 = nn.Sequential(
                vgg_layer(32, 64),
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
                nn.Conv2d(512, dim, 2, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h0 = self.c0(input) # 64x64
        h1 = self.c1(self.mp(h0)) # 32x32
        h2 = self.c2(self.mp(h1)) # 16x16
        h3 = self.c3(self.mp(h2)) # 8x8
        h4 = self.c4(self.mp(h3)) # 4x4
        h5 = self.c5(self.mp(h4)) # 1x1
        return h5.view(-1, self.dim), [h0, h1, h2, h3, h4]


class Decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(Decoder, self).__init__()
        self.dim = dim
        #1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 2, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                vgg_layer(512, 512),
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
        self.upc5 = nn.Sequential(
                vgg_layer(64*2, 64),
                vgg_layer(64, 32)
                )
        
        self.tanh = nn.Tanh()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        vec, skip = input 
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 2
        up1 = self.up(d1) # 2 -> 4
        d2 = self.upc2(torch.cat([up1, skip[4]], 1)) # 4 x 4
        up2 = self.up(d2) # 4 -> 8 
        d3 = self.upc3(torch.cat([up2, skip[3]], 1)) # 8 x 8 
        up3 = self.up(d3) # 8 -> 16 
        d4 = self.upc4(torch.cat([up3, skip[2]], 1)) # 16 x 16
        up4 = self.up(d4) # 16 -> 32
        d5 = self.upc5(torch.cat([up4, skip[1]], 1)) # 32 x 32
        
        return [d1,d2,d3,d4,d5]


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
                nn.Tanh()
                )
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
        self.mu_net = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Tanh())
        self.logvar_net = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Tanh())
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
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
