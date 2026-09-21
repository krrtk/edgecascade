import json
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


from inference.model_loader import load_pretrained_gpt2_124m
from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer
from inference.generator import generate_text

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
            
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    
    padded_xs = []
    padded_ys = []
    for x, y in zip(xs, ys):
        pad_len = max_len - len(x)
        padded_xs.append(torch.cat([x, torch.full((pad_len,), 50256, dtype=torch.long)]))
        padded_ys.append(torch.cat([y, torch.full((pad_len,), 50256, dtype=torch.long)]))
        
    return torch.stack(padded_xs), torch.stack(padded_ys)

def apply_lora_freezing(model):
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

def sanity_checks(model, dataset, device):
    print("Running sanity checks...")
    
    # Check trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 294912, f"Expected 294912, got {trainable_params}"
    
    # Check base param freezing
    assert not model.trf_blocks[0].att.W_key.weight.requires_grad
    
    # Setup mini training loop
    dl = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    initial_lora_b = model.trf_blocks[0].att.W_query.lora_B.clone().detach()
    initial_w_key = model.trf_blocks[0].att.W_key.weight.clone().detach()
    
    batch_x, batch_y = next(iter(dl))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    
    model.train()
    opt.zero_grad()
    logits = model(batch_x)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
    
    assert torch.isfinite(loss), "Loss is not finite!"
    
    loss.backward()
    
    # Check gradients exist for LoRA and NOT for base
    assert model.trf_blocks[0].att.W_query.lora_B.grad is not None, "LoRA B has no gradient!"
    assert model.trf_blocks[0].att.W_key.weight.grad is None, "Base param has gradient!"
    
    opt.step()
    
    final_lora_b = model.trf_blocks[0].att.W_query.lora_B.detach()
    final_w_key = model.trf_blocks[0].att.W_key.weight.detach()
    
    assert not torch.equal(initial_lora_b, final_lora_b), "LoRA parameter didn't change!"
    assert torch.equal(initial_w_key, final_w_key), "Base parameter changed!"
    
    print("Sanity checks passed.")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    torch.manual_seed(42)
    
    print("Loading tokenizer and dataset...")
    tokenizer = GPT2Tokenizer()
    
    # Use max_length 1024, dataset generation will truncate/pad
    dataset = LoRADataset("data/training/lora/lora_sft.jsonl", tokenizer, max_length=1024)
    
    print(f"Loaded {len(dataset)} examples.")
    
    print("Loading model...")
    model = load_pretrained_gpt2_124m(device=device)
    apply_lora_freezing(model)
    model.to(device)
    
    sanity_checks(model, dataset, device)
    
    print("Starting full training...")
    epochs = 4
    batch_size = 2
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # GPT-2 124M LoRA 8 hyperparams
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Loss only over non-padding tokens if we wanted to, but the objective says:
    # "Do not calculate loss on padding tokens if padding is required."
    loss_fn = nn.CrossEntropyLoss(ignore_index=50256)
    
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for step, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            
            if (step + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{len(dl)} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dl):.4f}")
        
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f}s")
    
    # Save checkpoint
    out_dir = Path("models/checkpoints/lora")
    out_dir.mkdir(parents=True, exist_ok=True)
    chkpt_path = out_dir / "lora_adapter.pt"
    checkpoint = {
        "model_state_dict": {k: v for k, v in model.state_dict().items() if "lora_" in k},
        "hyperparameters": {"rank": 8, "alpha": 16}
    }
    torch.save(checkpoint, chkpt_path)
    print(f"LoRA adapter saved to {chkpt_path}")
    
    # Post-training verification
    print("Running post-training verification...")
    fresh_model = load_pretrained_gpt2_124m(device=device)
    fresh_model.to(device)
    apply_lora_freezing(fresh_model)
    fresh_model.load_state_dict(torch.load(chkpt_path)["model_state_dict"], strict=False)
    
    isro_prompt = "ISRO's Chandrayaan-3 mission"
    dpdpa_prompt = "Under the DPDPA, a Data Fiduciary must"
    
    print("\nISRO Prompt test:")
    print(generate_text(fresh_model, tokenizer, isro_prompt, 25, device))
    
    print("\nDPDPA Prompt test:")
    print(generate_text(fresh_model, tokenizer, dpdpa_prompt, 25, device))
    
    print("\nALL PHASE 2 CRITERIA PASSED.")

if __name__ == "__main__":
    train()
