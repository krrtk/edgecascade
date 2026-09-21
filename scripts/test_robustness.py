import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from inference.model_loader import load_pretrained_gpt2_124m
from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer
from inference.pipeline import EdgeCascadePipeline

def test_robustness():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = GPT2Tokenizer()
    model = load_pretrained_gpt2_124m(device=device)
    for param in model.parameters(): param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name: param.requires_grad = True
        
    lora_path = "models/checkpoints/lora/lora_adapter.pt"
    if Path(lora_path).exists():
        model.load_state_dict(torch.load(lora_path, map_location=device)["model_state_dict"], strict=False)
    
    model.eval()
    
    rag_paths = [
        "data/retrieval_final/isro/cartosat1_chunks.jsonl",
        "data/retrieval_final/isro/resourcesat2_chunks.jsonl",
        "data/retrieval_final/dpdpa/dpdpa_chunks.jsonl"
    ]
    
    pipeline = EdgeCascadePipeline(model, tokenizer, device, rag_paths)
    
    sanity_prompts = [
        ("ISRO", "What are the main applications of Cartosat-1?"),
        ("ISRO", "Can you explain how Resourcesat-2 helps with agriculture?"),
        ("DPDPA", "Who is considered a Data Fiduciary?"),
        ("DPDPA", "What are the rights of a child under the DPDPA?")
    ]
    
    print("\n--- ROBUSTNESS SANITY TESTS ---")
    results = []
    for domain, prompt in sanity_prompts:
        print(f"\nQ: {prompt}")
        stats = pipeline.run(prompt, domain=domain.lower(), mode="cascade", ground_truth="N/A")
        print(f"Tier Reached: {stats['tier']}")
        print(f"Answer: {stats['answer'][:200]}")
        print(f"Escalations: {stats['escalation_reason']}")
        print("-" * 50)
        results.append({
            "domain": domain,
            "question": prompt,
            "tier_reached": stats["tier"],
            "answer": stats["answer"],
            "escalation_reason": stats["escalation_reason"]
        })
        
    import json
    import os
    os.makedirs("results/final", exist_ok=True)
    with open("results/final/robustness_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    test_robustness()
