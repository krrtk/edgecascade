import json
import time
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.model_loader import load_pretrained_gpt2_124m
from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer
from inference.generator import generate_text

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def apply_lora_freezing(model):
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = GPT2Tokenizer()
    questions = load_jsonl("data/evaluation/evaluation_questions.jsonl")
    
    model = load_pretrained_gpt2_124m(device=device)
    apply_lora_freezing(model)
    
    lora_path = "models/checkpoints/lora/lora_adapter.pt"
    if not Path(lora_path).exists():
        print(f"LoRA adapter not found at {lora_path}")
        return
        
    print(f"Loading LoRA adapter from {lora_path}...")
    model.load_state_dict(torch.load(lora_path, map_location=device)["model_state_dict"], strict=False)
    model.eval()
    
    print("\n--- LoRA Quality Check ---")
    
    for i, q in enumerate(questions):
        # We test just a few for visual inspection
        if i % 5 != 0:
            continue
            
        prompt = q["question"] + "\nAnswer:"
        print(f"\nQ: {q['question']}")
        print(f"Domain: {q['domain']} | Category: {q['category']}")
        print(f"Ground Truth: {q['ground_truth']}")
        
        start = time.time()
        # Max new tokens 30 is enough for these answers
        out = generate_text(model, tokenizer, prompt, max_new_tokens=30, device=device)
        elapsed = time.time() - start
        
        # Extract just the newly generated text if possible, or print whole
        print(f"Generated: {out.strip()}")
        print(f"Time: {elapsed:.2f}s")
        
    print("\nEvaluation complete.")

if __name__ == "__main__":
    evaluate()
