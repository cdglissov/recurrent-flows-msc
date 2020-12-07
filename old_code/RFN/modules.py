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
class VGG_downscaler_v2(nn.Module):
  def __init__(self, structures, L, in_channels, norm_type = "batchnorm", non_lin = "relu", scale = 2,skip_con=False):
    super(VGG_downscaler_v2, self).__init__()
    assert len(structures) == L, "Please specify number of blocks = L"
    self.l_nets = []
    self.L = L
    self.skip_con = skip_con
    self.scale = scale
    for l in range(0, L):
      structure = structures[l]
      layers = []
      count = 0
      for i in structure:
          count = count + 1
          if i == 'pool':
              layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
          elif i == "conv":
              conv_channels = int(in_channels*scale)
              conv2d = nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride = 2, padding=1,bias=False)
              layers += [conv2d,
                           NormLayer(conv_channels, norm_type = norm_type), 
                           ActFun(non_lin, in_place=True)]
              in_channels = conv_channels
          elif i == "squeeze":
              conv_channels = in_channels * 4
              conv = Squeeze2dDecoder(undo_squeeze=False)
              layers += [conv, 
                        NormLayer(conv_channels, norm_type = norm_type),
                        ActFun(non_lin, in_place=False)]
              in_channels = conv_channels
          else:
              conv2d = nn.Conv2d(in_channels, i, kernel_size=3, stride=1, padding=1,bias=False)
              if l == L-1 and len(structure) == count:
                  layers += [conv2d, NormLayer(i, norm_type = norm_type), nn.Tanh()]
              else:
                  layers += [conv2d,  NormLayer(i, norm_type = norm_type), ActFun(non_lin, in_place=True)]
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

class VGG_downscaler(nn.Module):
  def __init__(self, structure, in_channels = 1, norm_type = "batchnorm", non_lin = "leakyrelu", scale=2):
    super(VGG_downscaler, self).__init__()

    layers = []
    for i in structure:
        if i == 'pool':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif i == "conv":
            conv_channels = int(in_channels*scale)
            conv2d = nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride = 2, padding=1,bias=False)
            layers += [conv2d,
                       NormLayer(conv_channels, norm_type = norm_type), 
                       ActFun(non_lin, in_place=True)]
            in_channels = conv_channels
        else:
            conv2d = nn.Conv2d(in_channels, i, kernel_size=3, padding=1,bias=False)
            layers += [conv2d, NormLayer(i, norm_type = norm_type), ActFun(non_lin, in_place=False)]
            in_channels = i
    # Last layer
    conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,bias=False)
    layers += [conv2d, nn.Tanh()]
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)

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
          elif i == "squeeze":
              deconv_channels = in_channels // 4
              deconv = Squeeze2dDecoder(undo_squeeze=True)
              layers += [deconv, 
                        NormLayer(deconv_channels, norm_type = norm_type),
                        ActFun(non_lin, in_place=False)]
              in_channels = deconv_channels
          else:
              conv2d = nn.Conv2d(in_channels, i, kernel_size=3, stride=1, padding=1,bias=False)
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


class ConvLSTMLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias, 
                 dropout = 0, peephole=True, norm = False, make_init = True):
        super(ConvLSTMLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size 
        self.peephole = peephole
        self.make_init = make_init
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
        if self.make_init:
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
        if self.bias:
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
    
    def init_hidden(self, batch_size, height, width):
        return (Variable(torch.zeros(batch_size, self.hidden_channels, height, width)).to(device),
                Variable(torch.zeros(batch_size, self.hidden_channels, height, width)).to(device))
        
class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias=True, 
                 dropout = 0, peephole=True, norm = False, make_init = True, num_layers = 1):
        super(ConvLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        
        self.num_layers = num_layers
        
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels  = self._extend_for_multilayer(hidden_channels, num_layers)
        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length')
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_in_channels = in_channels if i == 0 else hidden_channels[i-1]
            cell_list.append(ConvLSTMLayer(in_channels=cur_in_channels,
                                          hidden_channels=hidden_channels[i],
                                          kernel_size=kernel_size[i],
                                          bias=bias, dropout=dropout, peephole = peephole,
                                          norm = norm, make_init = make_init))
        
        self.LSTMlayers = nn.ModuleList(cell_list)


    def forward(self, x, hidden_states):
        b, seq_len, channel, h, w = x.size()
        x = x.view(b*seq_len, channel, h, w)
        cur_layer_input = x
        for layer in range(self.num_layers):
            ht, ct = hidden_states[layer]
            ht, ct = self.LSTMlayers[layer](input_tensor=cur_layer_input,
                                          cur_state=[ht, ct])
            hidden_states[layer] = ht, ct
            cur_layer_input = ht
            
        return ht, hidden_states
    
    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.LSTMlayers[i].init_hidden(batch_size, height, width))
        return init_states
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
