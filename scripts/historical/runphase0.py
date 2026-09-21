import torch

from inference.model_loader import load_pretrained_gpt2_124m
from inference.generator import generate_text
from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

model = load_pretrained_gpt2_124m(device=device)

tokenizer = GPT2Tokenizer()

prompt = "Every effort moves you"

output = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=25,
    device=device,
)

print("\nGenerated text:")
print(output)