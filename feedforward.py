import torch.nn as nn

class FFN(nn.Module):
  def __init__(self,C):
    super().__init__()
    self.l1_layer = nn.Linear(C,4*C)
    self.l2_layer = nn.Linear(4*C,C)
    self.gelu = nn.GELU()
    self.dropout  = nn.Dropout(0.1)

  def forward(self,x):

    x = self.l1_layer(x)
    x = self.gelu(x)
    x = self.l2_layer(x)
    x = self.dropout(x)
    return x