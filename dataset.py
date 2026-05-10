from torch.utils.data import Dataset
import torch



class ShakespereDataset(Dataset):
  def __init__(self,data,seq_len):
    super().__init__()
    self.data = torch.tensor(data)
    self.seq_len = seq_len

  def __len__(self):
    return len(self.data)-self.seq_len

  def __getitem__(self, index):
    inp_seq = self.data[index:index+self.seq_len]
    target_seq = self.data[index+1:index+self.seq_len+1]
    return inp_seq,target_seq