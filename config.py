from dataclasses import dataclass

@dataclass
class LLMConfig():
  seq_len:int = 256
  batch_size:int = 64
  emb_dim:int = 256
  vocabulary:int = 65
  head_count:int = 8
  n_block:int = 8

  def __post_init__(self):
    assert self.d_model % self.n_heads ==0
    
  @property
  def head_dim(self):
    return self.emb_dim//self.haed_count

llm_config = LLMConfig()
