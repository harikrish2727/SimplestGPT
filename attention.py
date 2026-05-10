import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
  def __init__(self, T,C,head_count):
    super().__init__()
    self.head_count = head_count
    self.k_d = C//head_count
    self.register_buffer("mask",torch.tril(torch.ones(T,T)))

    self.wq = nn.Linear(C,C,bias=False)
    self.wk = nn.Linear(C,C,bias=False)
    self.wv = nn.Linear(C,C,bias=False)
    self.w_o = nn.Linear(C,C)
    self.dropout1 = nn.Dropout(0.05)

  def forward(self,x):
    B,T,C = x.shape   #if (20,100,384) then

    q = self.wq(x)    #(20,100,384)
    k = self.wk(x)
    v = self.wv(x)

    q = q.view(B,T,self.head_count,self.k_d)    #if 12 heads->(20,100,12,384//12=32)
    k = k.view(B,T,self.head_count,self.k_d)    #(20,100,12,32)
    v = v.view(B,T,self.head_count,self.k_d)

    q = q.transpose(1,2)                     #(20,12,100,32)
    k = k.transpose(1,2)
    v = v.transpose(1,2) #B,head,T,k_d

    score = self.k_d**0.5
    attn = q@k.transpose(-2,-1)/score    #(20,12,100,100)

    mask = self.mask[:T,:T]
    attn = attn.masked_fill(mask==0,float('-inf'))
    softmax_out = F.softmax(attn,dim=-1,)  #(20,12,100,100)
    attn_scores = self.dropout1(softmax_out)
    attn_scores = attn_scores@v          #(20,12,100,32)

    attn_scores = attn_scores.transpose(1,2)   #(20,100,12,32)

    attn_scores = attn_scores.contiguous().view(B,T,C)   #(20,100,384)
    out = self.w_o(attn_scores)
    # print(out)
    return out
