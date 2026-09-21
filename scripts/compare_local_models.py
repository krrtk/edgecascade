import json
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.model_loader import load_pretrained_gpt2_124m
from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer
from inference.generator import generate_text

def get_answer(model, tokenizer, question, device):
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=40,
        device=device,
        temperature=0.8,
        top_k=40
    )
    answer = output[len(prompt):].strip().split('### Question:')[0].strip()
    return answer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer()
    
    questions = [
        "What is remote sensing?",
        "What is spatial resolution?",
        "What is the difference between active and passive remote sensing?",
        "What is a Data Principal under the DPDPA?",
        "What is a Data Fiduciary?",
        "Why is electromagnetic radiation useful in remote sensing?"
    ]
    
    results = []
    
    cpt_path = "models/checkpoints/lora/lora_adapter_cpt.pt"
    sft_path = "models/checkpoints/lora/lora_adapter_sft.pt"
    
    print("Evaluating CPT Model...")
    cpt_model = load_pretrained_gpt2_124m(device=device)
    if Path(cpt_path).exists():
        cpt_model.load_state_dict(torch.load(cpt_path, map_location=device)["model_state_dict"], strict=False)
    cpt_model.eval()
    
    print("Evaluating SFT Model...")
    sft_model = load_pretrained_gpt2_124m(device=device)
    if Path(sft_path).exists():
        sft_model.load_state_dict(torch.load(sft_path, map_location=device)["model_state_dict"], strict=False)
    sft_model.eval()
    
    for q in questions:
        print(f"Q: {q}")
        old_ans = get_answer(cpt_model, tokenizer, q, device)
        new_ans = get_answer(sft_model, tokenizer, q, device)
        
        results.append({
            "question": q,
            "old_answer": old_ans,
            "new_answer": new_ans
        })
        
    out_dir = Path("results/final")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "local_model_before_after.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nComparison saved to {out_file}")

if __name__ == "__main__":
    main()
