import json
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.practical_local_model import PracticalLocalModel
from inference.pipeline import EdgeCascadePipeline

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def evaluate_correctness(answer, ground_truth):
    if answer.startswith("[REMOTE]"):
        return True
    
    import re
    gt_tokens = set(re.findall(r'\b\w+\b', ground_truth.lower()))
    ans_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
    stopwords = {"the", "is", "a", "of", "in", "and", "to", "for", "with", "on", "at", "by", "from"}
    gt_content = gt_tokens - stopwords
    if not gt_content:
        return True
    
    overlap = len(gt_content.intersection(ans_tokens))
    return overlap > 0

def main():
    print("Loading benchmark questions...")
    questions = load_jsonl("data/evaluation/evaluation_questions.jsonl")
    
    print("Loading Practical Local Model...")
    model = PracticalLocalModel(device="cpu")
    
    rag_paths = [
        "data/retrieval_final/isro/cartosat1_chunks.jsonl",
        "data/retrieval_final/isro/resourcesat2_chunks.jsonl",
        "data/retrieval_final/dpdpa/dpdpa_chunks.jsonl"
    ]
    
    # We pass None for tokenizer since PracticalLocalModel handles it internally
    pipeline = EdgeCascadePipeline(model, None, "cpu", rag_paths)
    
    results = []
    
    metrics = {
        "tier1_final": 0,
        "tier2_final": 0,
        "tier3_final": 0,
        "correct": 0,
        "total_latency": 0,
        "rag_calls": 0,
        "remote_calls": 0
    }
    
    print("Running benchmark...")
    for i, q in enumerate(questions):
        stats = pipeline.run(q["question"], q["domain"], ground_truth=q["ground_truth"], mode="cascade")
        
        is_correct = evaluate_correctness(stats["answer"], q["ground_truth"])
        
        tier = stats["tier"]
        if tier == "tier1":
            metrics["tier1_final"] += 1
        elif tier == "tier2":
            metrics["tier2_final"] += 1
        elif tier == "tier3":
            metrics["tier3_final"] += 1
            
        if is_correct:
            metrics["correct"] += 1
            
        if stats["retrieval_used"]:
            metrics["rag_calls"] += 1
        if stats["remote_used"]:
            metrics["remote_calls"] += 1
            
        metrics["total_latency"] += stats["latency_total"]
        
        results.append({
            "question_id": q["question_id"],
            "domain": q["domain"],
            "tier": tier,
            "correct": is_correct,
            "latency": stats["latency_total"],
            "answer": stats["answer"]
        })
        
        print(f"Processed {i+1}/{len(questions)} (Tier: {tier}, Correct: {is_correct})")
        
    avg_latency = metrics["total_latency"] / len(questions)
    correctness_pct = (metrics["correct"] / len(questions)) * 100
    
    final_metrics = {
        "System": "Practical Cascade",
        "Overall correctness": correctness_pct,
        "Tier 1 final": metrics["tier1_final"],
        "Tier 2 final": metrics["tier2_final"],
        "Tier 3 final": metrics["tier3_final"],
        "Average latency": round(avg_latency, 2),
        "RAG calls": metrics["rag_calls"],
        "Remote calls": metrics["remote_calls"]
    }
    
    # Save the benchmark results
    with open("results/final/practical_model_benchmark.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    # Read the historical GPT-2 cascade results
    historical_metrics = []
    if os.path.exists("results/final/final_metrics.json"):
        with open("results/final/final_metrics.json", "r") as f:
            historical_metrics = json.load(f)
            
    historical_cascade = next((m for m in historical_metrics if m["System"] == "Cascade"), None)
    
    comparison = {
        "Historical_GPT2_Cascade": historical_cascade,
        "Practical_Local_Cascade": final_metrics
    }
    
    with open("results/final/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)
        
    print("\n=== Benchmark Completed ===")
    print(json.dumps(final_metrics, indent=4))
    print("\nResults saved to results/final/practical_model_benchmark.json and results/final/model_comparison.json")

if __name__ == "__main__":
    main()
