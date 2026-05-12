import torch
import torch.nn as nn
import torch.nn.functional as F

from block import TransformerBlock

class SimplestGPT(nn.Module):
  def __init__(self,vocabulary,emb_dim,seq_length,head_count,n_blocks):
    super().__init__()
    self.blocks = nn.Sequential(*[TransformerBlock(head_count,seq_length,emb_dim) for _ in range(n_blocks)])
    self.embedding = nn.Embedding(vocabulary,emb_dim)
    self.pos_emb = nn.Embedding(seq_length,emb_dim)
    self.lm_head = nn.Linear(emb_dim,vocabulary,bias=False)
    self.layernorm = nn.LayerNorm(emb_dim)
    self.T = seq_length
    self.lm_head.weight = self.embedding.weight


  def forward(self,x):
    B,T = x.shape
    token_emb = self.embedding(x)
    pos = torch.arange(T,device=x.device)
    pos_emb = self.pos_emb(pos)
    x = token_emb+pos_emb

    x = self.blocks(x)
    x = self.layernorm(x)
    x = self.lm_head(x)
    return x

  @torch.no_grad()
  def test_prediction(self,x):
    print(f"input:{x}")
    logits = self(x)
    probs = F.softmax(logits,dim=-1)
    out_ids = torch.multinomial(probs[-1],num_samples=1)
    return out_ids

  @torch.no_grad()
  def generate(self,idx,max_tokens,temp,top_k=None):
    tok_length = len(idx[-1])
    for i in range(max_tokens):
      idx = idx[:,-self.T:]
      logits = self(idx)
      logits = logits[:,-1,:]
      if temp==0:
        out_idx = torch.argmax(logits,dim=-1,keepdim=True)
      else:
        logits = logits/temp
        if top_k is not None:
          val,id = torch.topk(logits,top_k)
          min_val = val[:,-1].unsqueeze(-1)
          logits = logits.masked_fill(logits<min_val,float('-inf'))
        probs = F.softmax(logits,dim=-1)
        out_idx = torch.multinomial(probs,num_samples=1)
      idx = torch.cat([idx,out_idx],dim=1)
    return idx
