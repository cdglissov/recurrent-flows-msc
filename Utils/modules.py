import torch
import torch.nn as nn
from .utils import set_gpu
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
  
class VGG_downscaler(nn.Module):
  def __init__(self, structures, L, in_channels, norm_type = "batchnorm", non_lin = "relu", scale = 2, skip_con=False, tanh = False):
    super(VGG_downscaler, self).__init__()
    assert len(structures) == L, "Please specify number of blocks = L"
    self.l_nets = nn.ModuleList([])
    self.L = L
    self.skip_con = skip_con
    self.scale = scale
    for l in range(0, L):
      structure = structures[l]
      layers = []
      count = 0
      for i in structure:
          count = count + 1
          # Insured tanh activation in last layer, this is done to avoid exploding gradients.
          if l == L-1 and len(structure) == count:
              ActivationFun = nn.Tanh()
          elif len(structure) == count and tanh:
              ActivationFun = tanh0_5()
          else:
              ActivationFun = ActFun(non_lin, in_place=True)
          if i == 'pool':
              layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
          elif i == "conv":
              conv_channels = int(in_channels*scale)
              conv2d = nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride = 2, padding=1,bias=False)
              layers += [conv2d,
                           NormLayer(conv_channels, norm_type = norm_type), 
                           ActivationFun]
              in_channels = conv_channels
          elif i == "squeeze":
              conv_channels = in_channels * 4
              conv = Squeeze2dDecoder(undo_squeeze=False)
              layers += [conv, 
                        NormLayer(conv_channels, norm_type = norm_type),
                        ActivationFun]
              in_channels = conv_channels
          else:
              conv2d = nn.Conv2d(in_channels, i, kernel_size=3, stride=1, padding=1,bias=False)
              layers += [conv2d,  NormLayer(i, norm_type = norm_type), ActivationFun]
              in_channels = i
        

      self.net = nn.Sequential(*layers).to(device)
      self.l_nets.append(self.net)
      
  def get_layer_size(self,structures,x_size):
      bs, c, hx, wx = x_size
      layerdims = []
      for l in range(0, len(structures)):
          structure = structures[l]
          for i in structure:
              if i == 'pool':
                  hx = hx // 2
                  wx = wx // 2
                  c = c
              elif i == "conv":
                  hx = hx // 2
                  wx = wx // 2
                  c = int(c*self.scale)
              elif i == "squeeze":
                  hx = hx // 2
                  wx = wx // 2
                  c = c * 4
              else:
                  c = i
          layerdims.append([bs, c, hx, wx])
      return layerdims
  
  def forward(self, x, block_size=None):
    outputs = []
    for i in range(0, self.L):
      x = self.l_nets[i](x)
      if self.skip_con:
          outputs.append(x)
      else:
          outputs=x
    return outputs
  
class Squeeze2dDecoder(nn.Module):
    def __init__(self,undo_squeeze = False):
        super(Squeeze2dDecoder, self).__init__()
        self.undo_squeeze=undo_squeeze
    def forward(self, x):
      B, C, H, W = x.shape
      if self.undo_squeeze == False:
        # C x H x W -> 4C x H/2 x W/2
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.reshape(B, C * 4, H // 2, W // 2)
      else:
        # 4C x H/2 x W/2  ->  C x H x W
        x = x.reshape(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.reshape(B, C // 4, H * 2, W * 2)
      return x

class tanh0_5(nn.Module):
    def __init__(self):
        super(tanh0_5, self).__init__()
        self.tanh=nn.Tanh()
    def forward(self,x):
        return 0.5*self.tanh(x)

class VGG_upscaler(nn.Module):
  def __init__(self, structures, L, in_channels, norm_type = "batchnorm", non_lin = "relu", scale = 2, skips = False, size_skips = None, tanh = False):
    super(VGG_upscaler, self).__init__()
    assert len(structures) == L, "Please specify number of blocks = L"
    self.l_nets = nn.ModuleList([])
    self.upscales_nets = nn.ModuleList([])
    self.L = L
    self.skips = skips
    size_skips.reverse()
    for l in range(0, L):
      structure = structures[l]
      layers = []
      count = 0
      for i in structure:
          count = count + 1
          if len(structure) == count and tanh:
              ActivationFun = tanh0_5()
          else:
              ActivationFun = ActFun(non_lin, in_place=True)
          if (skips and count == 1 and l == 0) or (skips and count == 2 and not l == 0):  
              """ So for the lowest layer of L it is the first it is connected to, and the other the second.
              Kinda weird but is easier to structure the nets."""
              skip_channels = size_skips[l][1]
          else:
              skip_channels = 0
          if i == 'upsample':
              layer_up = [nn.Upsample(scale_factor=2, mode='nearest')]
          elif i == "deconv":
              deconv_channels = in_channels // scale
              deconv = nn.ConvTranspose2d(in_channels, deconv_channels, kernel_size=4, stride = 2, padding=1, bias=False)
              layer_up = [deconv, 
                        NormLayer(deconv_channels, norm_type = norm_type),
                        ActivationFun]
              in_channels = deconv_channels
          elif i == "squeeze":
              deconv_channels = in_channels // 4
              deconv = Squeeze2dDecoder(undo_squeeze=True)
              layer_up = [deconv, 
                        NormLayer(deconv_channels, norm_type = norm_type),
                        ActivationFun]
              in_channels = deconv_channels
          else:
              conv2d = nn.Conv2d(in_channels + skip_channels, i, kernel_size=3, stride=1, padding=1,bias=False)
              layers += [conv2d,  NormLayer(i, norm_type = norm_type), ActivationFun]
              in_channels = i
      if l > 0:
          self.upscales_nets.append(nn.Sequential(*layer_up).to(device))
      self.net = nn.Sequential(*layers).to(device)
      self.l_nets.append(self.net)

  def forward(self, x, skip_list = None):
    outputs = []
    if self.skips:
        skip_list.reverse()
    for i in range(0, self.L):
      
      if i > 0:
          x = self.upscales_nets[i-1](x)
      if self.skips:
          x = self.l_nets[i](torch.cat((x, skip_list[i]), dim = 1))
      else:
          x = self.l_nets[i](x)
      outputs.append(x)
    # This needs to be done twice as it is calling the place in memorty
    if self.skips:
        skip_list.reverse()
    outputs.reverse()
    return outputs

class SimpleParamNet(nn.Module):
  def __init__(self, structure, in_channels, out_channels, norm_type = "batchnorm", non_lin = "leakyrelu", scale = 2):
    super(SimpleParamNet, self).__init__()
 
    layers = []
    for i in structure:
        if i == 'pool':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif i == "conv":
            conv_channels = int(scale*in_channels)
            conv2d = nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride = 2, padding=1).to(device)
            layers += [conv2d,
                       NormLayer(conv_channels, norm_type = norm_type), 
                       ActFun(non_lin, in_place=True)]
            in_channels = conv_channels
        else:
            conv2d = nn.Conv2d(in_channels, i, kernel_size=3, padding=1)
            layers += [conv2d, NormLayer(i, norm_type = norm_type), ActFun(non_lin, in_place=True)]
            in_channels = i
 
    self.net = nn.Sequential(*layers)
    self.param_net = nn.Conv2d(in_channels, 2*out_channels,  kernel_size=3, stride=1, padding=1)
    self.softplus = nn.Softplus()
 
  def forward(self, x):
    output = self.net(x)
    loc, log_scale = self.param_net(output).chunk(2, 1)
    scale = self.softplus(log_scale)
    return loc, scale

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
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).to(device)),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).to(device))))
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
        self.mu_net = nn.Sequential(nn.Linear(hidden_size, output_size))
        self.logvar_net = nn.Sequential(nn.Linear(hidden_size, output_size))
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).to(device)),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).to(device))))
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

####
class ConvLSTMLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias, dropout = 0, peephole=True, norm = False):
        super(ConvLSTMLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size 
        self.peephole = peephole
        self.padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.bias = bias
        layers = []

        layers.append(nn.Conv2d(in_channels = self.in_channels + self.hidden_channels,
                              out_channels = 4 * self.hidden_channels,
                              kernel_size = self.kernel_size,
                              stride = 1,
                              padding = self.padding,
                              bias = self.bias))

        if norm == True:
          # TODO: Groupnorm might not work, specify groups.
          layers.append(nn.GroupNorm(4 * self.hidden_channels // 32, 4 * self.hidden_channels))
        if dropout != 0:
          layers.append(nn.Dropout2d(p = dropout))

        self.conv = nn.Sequential(*layers)

        self.init_done = False
        self.apply(self.initialize_weights)

    def forward(self, input_tensor, cur_state):
        b, c, h, w = input_tensor.shape
        if cur_state[0] == None:
          h_cur = nn.Parameter(torch.zeros(b, self.hidden_channels, h, w)).to(device)
          c_cur = nn.Parameter(torch.zeros(b, self.hidden_channels, h, w)).to(device)
        else:
          h_cur, c_cur = cur_state

        if self.init_done == False:
          self.initialize_peephole(h, w)
          self.init_done = True

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i + self.Wci * c_cur)
        f = torch.sigmoid(cc_f + self.Wcf * c_cur)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        o = torch.sigmoid(cc_o + self.Wco*c_next)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
    

    def initialize_weights(self, layer):
      if type(layer) == nn.Conv2d:
        nn.init.xavier_normal_(layer.weight)
        nn.init.uniform_(layer.bias)
    
    def initialize_peephole(self, height, width):
      if self.peephole:
        self.Wci = nn.Parameter(torch.zeros(1, self.hidden_channels, height, width)).to(device)
        self.Wcf = nn.Parameter(torch.zeros(1, self.hidden_channels, height, width)).to(device)
        self.Wco = nn.Parameter(torch.zeros(1, self.hidden_channels, height, width)).to(device)
      else:
        self.Wci = 0
        self.Wcf = 0
        self.Wco = 0

        
class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias=True, dropout = 0, peephole=True, norm = False):
        super(ConvLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        self.LSTMlayer = ConvLSTMLayer(in_channels=in_channels,
                                          hidden_channels=hidden_channels,
                                          kernel_size=kernel_size,
                                          bias=bias, dropout=dropout, peephole = peephole,
                                          norm = norm)
    
    def forward(self, x, ht=None, ct=None):
        b, seq_len, channel, h, w = x.size()
        output = []

        for t in range(seq_len):
            ht, ct = self.LSTMlayer(input_tensor=x[:, t, :, :, :],
                                              cur_state=[ht, ct])
            output.append(ht)
        return torch.stack(output,1), ht, ct
