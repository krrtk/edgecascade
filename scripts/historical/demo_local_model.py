import torch
import sys
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.model_loader import load_pretrained_gpt2_124m
from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer
from inference.generator import generate_text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model and tokenizer for Demo...")
    
    tokenizer = GPT2Tokenizer()
    model = load_pretrained_gpt2_124m(device=device)
    
    lora_path = "models/checkpoints/lora/lora_adapter_sft.pt"
    if Path(lora_path).exists():
        model.load_state_dict(torch.load(lora_path, map_location=device)["model_state_dict"], strict=False)
    else:
        print(f"Error: Could not find {lora_path}")
        return
        
    model.to(device)
    model.eval()
    
    questions = [
        "What is remote sensing?",
        "What is the difference between active and passive remote sensing?",
        "What is a Data Principal under the DPDPA?",
        "What factors must the Board consider when determining a monetary penalty?"
    ]
    
    print("\n" + "="*50)
    print("EdgeCascade Local SFT Model Demo")
    print("="*50)
    
    for q in questions:
        print(f"\nQUESTION: {q}")
        prompt = f"### Question:\n{q}\n\n### Answer:\n"
        
        # Using top-k sampling for better generation
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=40,
            device=device,
            temperature=0.8,
            top_k=40
        )
        
        # Extract just the generated answer
        answer = output[len(prompt):].strip().split('### Question:')[0].strip()
        print(f"LOCAL MODEL ANSWER: {answer}")
        
if __name__ == "__main__":
    main()
