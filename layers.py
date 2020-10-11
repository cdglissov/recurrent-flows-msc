import torch
import torch.nn as nn

class LinearNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.normal_(mean=0.0, std=0.1)
        self.linear.bias.data.normal_(mean=0.0, std=0.1)
  
    def forward(self, input):
      output = self.linear(input)
      return output 


class LinearZeros(nn.Module):
    def __init__(self, in_channels, out_channels, use_logscale=False):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        
        self.use_logscale = use_logscale

        if use_logscale:
          self.logscale_factor = 3
          self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):
      if self.use_logscale:
        output = self.linear(input) * torch.exp(self.logs * self.logscale_factor)
      else:
        output = self.linear(input)
      return output 


class Conv2dResize(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        
        stride = [in_size[1]//out_size[1], in_size[2]//out_size[2]]
        kernel_size = Conv2dResize.compute_kernel_size(in_size, out_size, stride)
        
        self.conv = nn.Conv2d(in_channels=in_size[0], out_channels=out_size[0], kernel_size=kernel_size, stride=stride)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    @staticmethod
    def compute_kernel_size(in_size, out_size, stride):
        k0 = in_size[1] - (out_size[1] - 1) * stride[0]
        k1 = in_size[2] - (out_size[2] - 1) * stride[1]
        return[k0,k1]

    def forward(self, input):
      output = self.conv(input)
      return output 

class Conv2dZeros(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3,3], stride=[1,1]):
        super().__init__()
        
        padding = (kernel_size[0] - 1) // 2
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
    
    def forward(self, input):
      output = self.conv(input)
      return output 


class ActNorm(nn.Module):

    def __init__(self, num_channels):
        super().__init__()

        size = [1, num_channels, 1, 1]

        bias = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size)*0.05)
        logs = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size)*0.05)
        self.register_parameter("bias", nn.Parameter(torch.Tensor(bias), requires_grad=True))
        self.register_parameter("logs", nn.Parameter(torch.Tensor(logs), requires_grad=True))


    def forward(self, x, logdet=0, reverse=False):
        dims = x.size(2) * x.size(3)
        if reverse == False:
            x = x + self.bias
            x = x * torch.exp(self.logs)
            dlogdet = torch.sum(self.logs) * dims
            logdet = logdet + dlogdet
        else:
            x = x * torch.exp(-self.logs)
            x = x - self.bias
            dlogdet = - torch.sum(self.logs) * dims
            logdet = logdet + dlogdet

        return x, logdet


class Conv2dNormy(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1]):
        super().__init__()

        padding = [(kernel_size[0]-1)//2, (kernel_size[1]-1)//2]

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv.weight.data.normal_(mean=0.0, std=0.05)
        self.actnorm = ActNorm(out_channels)

    def forward(self, input):
        output = self.conv(input)
        output, _ = self.actnorm(output)
        return output

class Conv2dZerosy(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1], use_logscale = True):
        super().__init__()

        padding = [(kernel_size[0]-1)//2, (kernel_size[1]-1)//2]
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.use_logscale = use_logscale

        if use_logscale:
          self.logscale_factor = 3.0
          self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
          #self.register_parameter("newbias", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, input):
      if self.use_logscale:
        #output = output + self.newbias
        output = self.conv(input)* torch.exp(self.logs * self.logscale_factor)
      else:
        output = self.conv(input)
      return output



#### RESNETS ####
class WeightNormConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0,
                 bias=True):
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_dim, out_dim, kernel_size,
                      stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)



class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_type):
        super(ResnetBlock, self).__init__()

        if norm_type == "wn":
          layers = [WeightNormConv2d(dim, dim, (1, 1), stride=1, padding=0),
              nn.LeakyReLU(),
              WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1),
              nn.LeakyReLU(),
              WeightNormConv2d(dim, dim, (1, 1), stride=1, padding=0)
          ]
        elif norm_type == "bn":
          layers = [nn.Conv2d(dim, dim, (1, 1), stride=1, padding=0),
              nn.BatchNorm2d(dim),
              nn.LeakyReLU(),
              nn.Conv2d(dim, dim, (3, 3), stride=1, padding=1),
              nn.BatchNorm2d(dim),
              nn.LeakyReLU(),
              nn.Conv2d(dim, dim, (1, 1), stride=1, padding=0),
              nn.BatchNorm2d(dim)
          ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class SimpleResnet(nn.Module):
    '''
    SimpleResnet can take two types of normalization, weightnormalization often used in Glow and batchnormalization for more general usage.
    Downsample will cause height/2, width/2. While upsampling will do the opposite. If no type is specified the same dimensions will be output.
    '''
    def __init__(self, in_channels, out_channels, n_filters=32, n_blocks=2, sample_type = None, norm_type = "wn"):
        super(SimpleResnet, self).__init__()
        
        if sample_type == "up":
          self.pre_conv = nn.ConvTranspose2d(in_channels, n_filters, 3, stride=2, padding=1, output_padding=1)
        elif sample_type == "down":
          self.pre_conv = nn.Conv2d(in_channels, n_filters, 3, stride=2, padding=1)
        else:
          self.pre_conv = nn.Conv2d(in_channels, n_filters, 3, stride=1, padding=1)

  
        if norm_type == "wn":
            layers = [nn.utils.weight_norm(self.pre_conv),
                      nn.ReLU()]
            for _ in range(n_blocks):
                layers.append(ResnetBlock(n_filters, norm_type))
            layers.append(nn.ReLU())

        elif norm_type == "bn":
            layers = [self.pre_conv,
                      nn.BatchNorm2d(n_filters),
                      nn.ReLU()]
            for _ in range(n_blocks):
                layers.append(ResnetBlock(n_filters, norm_type))
            layers.append(nn.ReLU())
        else:
            print("Specify normalization type")

        self.resnet = nn.Sequential(*layers)
        self.post_conv = nn.Conv2d(n_filters, out_channels, 1, stride=1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.post_conv(x)
        return x

#### LSTM ####
class ConvLSTMLayer(nn.Module):
    # Only works with 3x3 kernels
    def __init__(self, in_channels, hidden_channels, kernel_size, bias, dropout = 0, peephole=True):
        super(ConvLSTMLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.peephole = peephole
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias
        self.conv = nn.Sequential(nn.Conv2d(in_channels = self.in_channels + self.hidden_channels,
                              out_channels = 4 * self.hidden_channels,
                              kernel_size = self.kernel_size,
                              padding = self.padding,
                              bias = self.bias))
        
        self.dropout = nn.Dropout2d(p = dropout)
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
        combined_conv = self.dropout(combined_conv)

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
    def __init__(self, in_channels, hidden_channels, kernel_size, bias=True, dropout = 0, peephole=True):
        super(ConvLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        self.LSTMlayer = ConvLSTMLayer(in_channels=in_channels,
                                          hidden_channels=hidden_channels,
                                          kernel_size=kernel_size,
                                          bias=bias, dropout=dropout, peephole = peephole)
    
    def forward(self, x, ht=None, ct=None):
        b, seq_len, channel, h, w = x.size()
        output = []
        for t in range(seq_len):
            ht, ct = self.LSTMlayer(input_tensor=x[:, t, :, :, :],
                                              cur_state=[ht, ct])
            output.append(ht)
        return torch.stack(output), ht, ct
