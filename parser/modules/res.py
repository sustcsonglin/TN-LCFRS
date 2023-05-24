import torch.nn as nn

class ResLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.linear(x) + x


class MultiResidualLayer(nn.Module):
  def __init__(self, in_dim=100, res_dim = 100, out_dim=None, num_layers=3):
    super(MultiResidualLayer, self).__init__()
    self.num_layers = num_layers
    if in_dim is not None:
      self.in_linear = nn.Linear(in_dim, res_dim)
    else:
      self.in_linear = None
    if out_dim is not None:
      self.out_linear = nn.Linear(res_dim, out_dim)
    else:
      self.out_linear = None
    self.res_blocks = nn.ModuleList([ResLayer(res_dim, res_dim) for _ in range(num_layers)])

  def forward(self, x):
    if self.in_linear is not None:
      out = self.in_linear(x)
    else:
      out = x
    for i in range(self.num_layers):
      out = self.res_blocks[i](out)
    if self.out_linear is not None:
      out = self.out_linear(out)
    return out


class ResLayerTanh(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayerTanh, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.linear(x) + x

