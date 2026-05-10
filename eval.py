import torch.nn.functional as F


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(model,val_loader,device):
  model.eval()
  tot_loss = 0
  step=0
  model = model.to(device)
  for x,y in val_loader:
    x,y = x.to(device),y.to(device)
    # B,T = x.shape
    logits = model(x)
    B,T,V = logits.shape
    loss = F.cross_entropy(logits.view(B*T,V),y.view(B*T))
    tot_loss+=loss.item()
    step+=1
  return tot_loss/step