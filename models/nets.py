import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, n_layers, hidden_dim, output_dim):
        super(MLP, self).__init__()
        layers = []
        for _ in range(n_layers):
          layers.append(nn.Linear(input_dim, hidden_dim))
          layers.append(nn.ReLU())
          input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
      return self.layers(x)