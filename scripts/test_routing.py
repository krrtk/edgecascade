import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.practical_local_model import PracticalLocalModel
from inference.pipeline import EdgeCascadePipeline

def load_jsonl(path, limit=5):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records

def main():
    print("Loading 5 questions for routing smoke test...")
    questions = load_jsonl("data/evaluation/evaluation_questions.jsonl", limit=5)

    print("Loading Practical Local Model...")
    model = PracticalLocalModel(device="cpu")
    
    rag_paths = [
        "data/retrieval_final/isro/cartosat1_chunks.jsonl",
        "data/retrieval_final/isro/resourcesat2_chunks.jsonl",
        "data/retrieval_final/dpdpa/dpdpa_chunks.jsonl"
    ]
    
    pipeline = EdgeCascadePipeline(model, None, "cpu", rag_paths)
    
    print("\n--- Routing Smoke Test ---")
    
    for i, q in enumerate(questions):
        print(f"\nQ{i+1}: {q['question']}")
        stats = pipeline.run(q["question"], q["domain"], ground_truth=q["ground_truth"], mode="cascade")
        
        print(f"Final Tier: {stats['tier']}")
        print(f"Gate Reason T1: {stats['gate_reason_t1']}")
        print(f"Gate Reason T2: {stats['gate_reason_t2']}")
        print(f"Escalation Reason: {stats.get('escalation_reason', 'N/A')}")
        print(f"Answer: {stats['answer'][:100]}...")

    print("\n--- Smoke Test Complete ---")

if __name__ == "__main__":
    main()
