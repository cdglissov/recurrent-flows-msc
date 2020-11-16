import torch
import torch.nn as nn


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
  def __init__(self, structure, in_channels = 1, norm_type = "batchnorm", non_lin = "leakyrelu", scale=2):
    super(VGG_downscaler, self).__init__()

    layers = []
    for i in structure:
        if i == 'pool':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif i == "conv":
            conv_channels = int(in_channels*scale)
            conv2d = nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride = 2, padding=1)
            layers += [conv2d,
                       NormLayer(conv_channels, norm_type = norm_type), 
                       ActFun(non_lin, in_place=True)]
            in_channels = conv_channels
        else:
            conv2d = nn.Conv2d(in_channels, i, kernel_size=3, padding=1)
            layers += [conv2d, NormLayer(i, norm_type = norm_type), ActFun(non_lin, in_place=True)]
            in_channels = i
    
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)


class VGG_upscaler(nn.Module):
  def __init__(self, structures, L, in_channels, norm_type = "batchnorm", non_lin = "relu", scale = 2):
    super(VGG_upscaler, self).__init__()
    assert len(structures) == L, "Please specify number of blocks = L"
    self.l_nets = []
    self.L = L

    for l in range(0, L):
      structure = structures[l]
      layers = []
      for i in structure:
          if i == 'upsample':
              layers += [nn.Upsample(scale_factor=2, mode='nearest')]
          elif i == "deconv":
              deconv_channels = in_channels // scale
              deconv = nn.ConvTranspose2d(in_channels, deconv_channels, kernel_size=4, stride = 2, padding=1, bias=False)
              layers += [deconv, 
                        NormLayer(deconv_channels, norm_type = norm_type),
                        ActFun(non_lin, in_place=True)]
              in_channels = deconv_channels
          else:
              conv2d = nn.Conv2d(in_channels, i, kernel_size=3, stride=1, padding=1)
              layers += [conv2d,  NormLayer(i, norm_type = norm_type), ActFun(non_lin, in_place=True)]
              in_channels = i
      
          self.net = nn.Sequential(*layers).to(device)
          # for m in self.modules():
          #   if isinstance(m, nn.Conv2d):
          #       m.weight.data.normal_(0, 0.05)
          #       m.bias.data.zero_()
          #   elif isinstance(m, nn.ConvTranspose2d):
          #       m.weight.data.normal_(0, 0.05)

      self.l_nets.append(self.net)

  def forward(self, x, block_size=None):
    outputs = []
    for i in range(0, self.L):
      x = self.l_nets[i](x)
      outputs.append(x)
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
            conv2d = nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride = 2, padding=1)
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
