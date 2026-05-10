import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import FFN

class TransformerBlock(nn.Module):
  def __init__(self,head_count,T,C):
    super().__init__()
    self.layernorm1 = nn.LayerNorm(C)
    self.layernorm2 = nn.LayerNorm(C)
    self.attention = MultiHeadAttention(T,C,head_count)
    self.ffn = FFN(C)

  def forward(self,x):   #input eg:(20,100,384)
    B,T,C = x.shape
    x = x + self.attention(self.layernorm1(x))
    x = x + self.ffn(self.layernorm2(x))
    return x