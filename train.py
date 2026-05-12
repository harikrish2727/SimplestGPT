import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from eval import evaluate



def training(model, num_epochs, train_loader, val_loader, optimizer, lr_scheduler,device):
    step = 0
    eval_every = 100
    model = model.to(device)

    for i in range(num_epochs):
        model.train()
        epoch_loss = []

        for input_seq, tar_seq in tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}"):
            input_seq, tar_seq = input_seq.to(device), tar_seq.to(device)
            logits = model(input_seq)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), tar_seq.view(B*T))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            epoch_loss.append(loss.item())
            step += 1

            if step % eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                avg_train = sum(epoch_loss[-eval_every:]) / eval_every
                print(f"step {step} | train_loss: {avg_train:.4f} | val_loss: {val_loss:.4f} | grad_norm: {grad_norm:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}")
                model.train()