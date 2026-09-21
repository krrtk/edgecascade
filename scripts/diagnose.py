import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

print("1. Importing model loader...")
from inference.model_loader import load_pretrained_gpt2_124m
print("2. Importing tokenizer...")
from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer
print("3. Importing pipeline...")
from inference.pipeline import EdgeCascadePipeline

print("4. Setting up device...")
device = torch.device("cpu")

print("5. Loading tokenizer...")
tokenizer = GPT2Tokenizer()

print("6. Loading model...")
model = load_pretrained_gpt2_124m(device=device)

print("7. Loading LoRA...")
lora_path = "models/checkpoints/lora/lora_adapter.pt"
model.load_state_dict(torch.load(lora_path, map_location=device)["model_state_dict"], strict=False)

print("8. Initializing Pipeline...")
rag_paths = [
    "data/retrieval_final/isro/cartosat1_chunks.jsonl",
    "data/retrieval_final/isro/resourcesat2_chunks.jsonl",
    "data/retrieval_final/dpdpa/dpdpa_chunks.jsonl"
]
pipeline = EdgeCascadePipeline(model, tokenizer, device, rag_paths)

print("9. Running test query...")
res = pipeline.run("Test query", "isro", mode="local")
print(res)
print("10. Done!")
