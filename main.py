import torch
from gpt import SimplestGPT
from transformers import get_cosine_schedule_with_warmup
from dataset import ShakespereDataset
from torch.utils.data import DataLoader
from data_preparation import read_data
from train import training
from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv("url")



def stream_data(input_data):
  for char in input_data:
    yield char


def main(read_data_func,data_set,stream_data_func):
    input_text = read_data_func()
    chars = sorted(list(set(input_text)))
    char2i = {ch:i for i,ch in enumerate(chars)}
    i2char = {i:ch for i,ch in enumerate(chars)}

    n = len(input_text)
    train_data = input_text[:int(n*.90)]
    val_data = input_text[int(n*.90):]
    
    train_ids = [char2i[i] for i in stream_data_func(train_data)]
    val_ids = [char2i[i] for i in stream_data_func(val_data)]

    train_data = data_set(train_ids,64)
    val_data = data_set(val_ids,64)
    
    train_loader = DataLoader(dataset=train_data,batch_size=64,shuffle=True)
    val_loader = DataLoader(dataset=val_data,batch_size=64,shuffle=False)

    return train_loader, val_loader, char2i, i2char










if __name__ == "__main__":
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  train_loader, val_loader, char2i, i2char = main(read_data, ShakespereDataset, stream_data)

  gpt = SimplestGPT(vocabulary=65, emb_dim=256, seq_length=256, head_count=8, n_blocks=8)

  optim = torch.optim.AdamW(gpt.parameters(),lr=3e-4,betas=[0.9,0.95],eps=1e-8,weight_decay=0.1)
  scheduler = get_cosine_schedule_with_warmup(optim,num_warmup_steps=1000,num_training_steps=10000)

  training(gpt,num_epochs=2,train_loader=train_loader,val_loader=val_loader,optimizer=optim,lr_scheduler=scheduler,device=device)

  gpt.load_state_dict(torch.load("best_model.pt"))

  torch.save({"model_state": gpt.state_dict(),
            "model_config": {"vocabulary":65,"emb_dim":256,"seq_length":256,"head_count":8,"n_blocks":8},
            "char2i":char2i,
            "i2char":i2char},"shakespere_gpt.pt"
           )