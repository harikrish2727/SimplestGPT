import torch
from gpt import SimplestGPT

def load_trained_model(path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ckpt = torch.load(path,map_location=device)
  config = ckpt["model_config"]
  model = SimplestGPT(**config)
  model.load_state_dict(ckpt["model_state"])

  char2i = ckpt["char2i"]
  i2char = ckpt["i2char"]
  return model,char2i,i2char



def test_generation(model,char2i,i2char,query,device,max_tokens=20,temp=0,top_k=None):
  l=len(query)
  inp_token = torch.tensor([char2i[i] for i in query]).reshape(1,l)
  model = model.to(device)
  idx = inp_token.to(device)
  out_tokens = model.generate(idx,max_tokens,temp,top_k)
  return "".join([i2char[i] for i in out_tokens.tolist()[0]])

if __name__ == "__main__":
  import pathlib
  model_path = "./shakespeare_gpt.pt"
  if not pathlib.Path(model_path).exists():
    print(f"Model file not found at {model_path}. Please ensure the model is trained and saved correctly.")
  else:
    model,char2i,i2char = load_trained_model(model_path)
    print("Model loaded successfully. Testing generation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    query = "To be, or not to be: that is the question:"
    generated_text = test_generation(model, char2i, i2char, query, device, max_tokens=20, temp=0.8, top_k=10)
    print(f"Generated continuation:\n{generated_text}")

