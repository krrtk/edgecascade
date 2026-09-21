import os
import torch
from inference.practical_local_model import PracticalLocalModel
from inference.pipeline import EdgeCascadePipeline

def main():
    print("==================================================")
    print("EdgeCascade - Interactive Recruiter Demo")
    print("==================================================")

    # Initialize components
    print("\n[1/3] Loading Practical Local Model (Qwen-0.5B)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_model = PracticalLocalModel(device=device)

    print("[2/3] Initializing TF-IDF Retriever...")
    rag_paths = {
        "isro": "data/retrieval_final/isro_index.pkl",
        "dpdpa": "data/retrieval_final/dpdpa_index.pkl"
    }
    
    print("[3/3] Initializing Pipeline and Semantic Judge...")
    pipeline = EdgeCascadePipeline(
        local_model=local_model,
        tokenizer=None, # PracticalLocalModel handles its own tokenizer
        device=device,
        rag_paths=rag_paths
    )

    print("\nSetup complete. Running demo cases...\n")

    cases = [
        {
            "desc": "Case 1: Simple Local Knowledge (Expected: Tier 1)",
            "query": "Can a Data Principal manage consent through a Consent Manager?",
            "domain": "dpdpa",
            "ground_truth": "Yes, a Data Principal may give, manage, review, or withdraw consent through a Consent Manager."
        },
        {
            "desc": "Case 2: RAG Rescue (Expected: Tier 2)",
            "query": "What is the OBSSR capacity?",
            "domain": "isro",
            "ground_truth": "The OBSSR capacity is 120 GB."
        }
    ]

    for case in cases:
        print("-" * 50)
        print(f"{case['desc']}")
        print(f"QUESTION: {case['query']}")
        
        stats = pipeline.run(case['query'], case['domain'], case['ground_truth'], mode="cascade")
        
        print(f"SELECTED TIER: {stats['tier'].upper()}")
        print(f"ANSWER: {stats['answer']}")
        print(f"LATENCY: {stats['latency_total']:.2f}s")
        if stats['tier'] == 'tier2':
            print(f"-> RAG rescued query after Tier 1 Judge returned: {stats['local_judge_result']}")

    print("-" * 50)
    print("\nDemo completed successfully. Note: To see Tier 3 escalation, please configure an API Judge in .env.")

if __name__ == "__main__":
    main()
