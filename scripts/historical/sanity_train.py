import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from inference.model_loader import load_weights_into_gpt
from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer

# Mock tensorflow and tqdm so gpt_download3 can be imported without crashing PyTorch dynamo
import sys
import math
sys.modules['tensorflow'] = math
sys.modules['tqdm'] = math

import numpy as np
from models.transformer.config import GPT2_124M_CONFIG
from models.transformer.model import GPTModel

def create_dummy_params():
    params = {"wpe": np.random.normal(size=(1024, 768)).astype(np.float32), 
              "wte": np.random.normal(size=(50257, 768)).astype(np.float32)}
    params["blocks"] = []
    for _ in range(12):
        block = {
            "attn": {
                "c_attn": {"w": np.random.normal(size=(768, 768*3)).astype(np.float32), 
                           "b": np.random.normal(size=(768*3,)).astype(np.float32)},
                "c_proj": {"w": np.random.normal(size=(768, 768)).astype(np.float32), 
                           "b": np.random.normal(size=(768,)).astype(np.float32)}
            },
            "mlp": {
                "c_fc": {"w": np.random.normal(size=(768, 3072)).astype(np.float32), 
                         "b": np.random.normal(size=(3072,)).astype(np.float32)},
                "c_proj": {"w": np.random.normal(size=(3072, 768)).astype(np.float32), 
                           "b": np.random.normal(size=(768,)).astype(np.float32)}
            },
            "ln_1": {"g": np.random.normal(size=(768,)).astype(np.float32), 
                     "b": np.random.normal(size=(768,)).astype(np.float32)},
            "ln_2": {"g": np.random.normal(size=(768,)).astype(np.float32), 
                     "b": np.random.normal(size=(768,)).astype(np.float32)}
        }
        params["blocks"].append(block)
    params["g"] = np.random.normal(size=(768,)).astype(np.float32)
    params["b"] = np.random.normal(size=(768,)).astype(np.float32)
    return params

class LoRADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_id = 50256  # GPT-2 EOS
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                self.data.append(item["text"])
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.encode(text)
        tokens.append(self.eos_id)
        
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            padding = [self.eos_id] * (self.max_length - len(tokens))
            tokens.extend(padding)
            
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

def apply_lora_freezing(model):
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze LoRA
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            
def sanity_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    tokenizer = GPT2Tokenizer()
    
    # 1. & 2. Load the model using existing loader logic
    print("Loading pretrained model...")
    model = GPTModel(GPT2_124M_CONFIG)
    params = create_dummy_params()
    load_weights_into_gpt(model, params)
    model.to(device)
        
    # 3. & 4. Apply freezing
    apply_lora_freezing(model)
    
    # 5. Verify trainable parameters = 294,912
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    assert trainable_params == 294912, f"Expected 294912, got {trainable_params}"
    
    # 6. Verify base model parameter doesn't have requires_grad
    print(f"Base param requires_grad: {model.trf_blocks[0].att.W_key.weight.requires_grad}")
    assert not model.trf_blocks[0].att.W_key.weight.requires_grad
    
    # Setup dataset
    data_path = Path("data/training/lora/lora_sft.jsonl")
    dataset = LoRADataset(data_path, tokenizer, max_length=128) # small length for sanity
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    # Capture initial states to verify changes
    initial_lora_b = model.trf_blocks[0].att.W_query.lora_B.clone().detach()
    initial_w_key = model.trf_blocks[0].att.W_key.weight.clone().detach()
    
    # Training Loop
    model.train()
    print("Starting sanity training loop over 50 steps...")
    batch_x, batch_y = next(iter(dataloader))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    
    initial_loss = None
    final_loss = None
    
    for step in range(50):
        optimizer.zero_grad()
        logits = model(batch_x)
        
        # Shifted implicit in dataset generation: x is tokens[:-1], y is tokens[1:]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        if step == 49:
            final_loss = loss.item()
            
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss:   {final_loss:.4f}")
    assert initial_loss is not None and np.isfinite(initial_loss)
    assert final_loss < initial_loss
    
    # Verify parameter updates
    final_lora_b = model.trf_blocks[0].att.W_query.lora_B.detach()
    final_w_key = model.trf_blocks[0].att.W_key.weight.detach()
    
    lora_changed = not torch.equal(initial_lora_b, final_lora_b)
    base_changed = not torch.equal(initial_w_key, final_w_key)
    
    print(f"LoRA parameter changed: {lora_changed}")
    print(f"Base parameter changed: {base_changed}")
    assert lora_changed
    assert not base_changed
    
    # Verify forward pass still works
    model.eval()
    with torch.no_grad():
        out = model(batch_x)
        print("Forward pass shape post-training:", out.shape)
        
    # Verify save works
    checkpoint = {
        "model_state_dict": {k: v for k, v in model.state_dict().items() if "lora_" in k}
    }
    torch.save(checkpoint, "sanity_lora.pt")
    print("LoRA checkpoint saved successfully to sanity_lora.pt")

if __name__ == "__main__":
    sanity_test()
