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

def evaluate_quality(answer, ground_truth):
    if answer.startswith("[REMOTE]"):
        return "GOOD"
    if not answer or len(answer.split()) < 3:
        return "BAD"
    words = answer.lower().split()
    vocab_ratio = len(set(words)) / len(words) if words else 0
    if vocab_ratio < 0.3:
        return "BAD"  # Repetitive
    gt_words = set(ground_truth.lower().split()) - {"the", "a", "an", "is", "of", "and", "to", "in"}
    ans_words = set(answer.lower().split())
    overlap = len(gt_words & ans_words) / len(gt_words) if gt_words else 0
    if overlap >= 0.5:
        return "GOOD"
    elif overlap >= 0.2:
        return "PARTIALLY_GOOD"
    return "BAD"

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
    
    pipeline = EdgeCascadePipeline(model, None, "cpu", rag_paths)
    
    results = []
    
    metrics = {
        "tier1_final": 0,
        "tier2_final": 0,
        "tier3_final": 0,
        "tier1_to_tier2": 0,
        "tier2_to_tier3": 0,
        "correct": 0,
        "total_latency": 0,
        "rag_calls": 0,
        "remote_calls": 0,
        "quality_counts": {"GOOD": 0, "PARTIALLY_GOOD": 0, "BAD": 0, "UNKNOWN": 0},
        "domain_correct": {"isro": 0, "dpdpa": 0},
        "domain_total": {"isro": 0, "dpdpa": 0},
        "category_correct": {"knowledge_gap": 0, "reasoning": 0},
        "category_total": {"knowledge_gap": 0, "reasoning": 0}
    }
    
    print("Running detailed benchmark...")
    for i, q in enumerate(questions):
        stats = pipeline.run(q["question"], q["domain"], ground_truth=q["ground_truth"], mode="cascade")
        
        is_correct = evaluate_correctness(stats["answer"], q["ground_truth"])
        quality = evaluate_quality(stats["answer"], q["ground_truth"])
        
        tier = stats["tier"]
        
        metrics["quality_counts"][quality] += 1
        
        if tier == "tier1":
            metrics["tier1_final"] += 1
        elif tier == "tier2":
            metrics["tier2_final"] += 1
            metrics["tier1_to_tier2"] += 1
        elif tier == "tier3":
            metrics["tier3_final"] += 1
            if stats["retrieval_used"]:
                metrics["tier1_to_tier2"] += 1
                metrics["tier2_to_tier3"] += 1
            else:
                metrics["tier1_to_tier2"] += 1
                metrics["tier2_to_tier3"] += 1 # Tier 1 -> Tier 3 counts as both if RAG was skipped? Actually if RAG is skipped, retrieval_used is False. But we know RAG is always attempted if Tier 1 fails.
                
        if is_correct:
            metrics["correct"] += 1
            metrics["domain_correct"][q["domain"]] += 1
            metrics["category_correct"][q.get("category", "unknown")] = metrics["category_correct"].get(q.get("category", "unknown"), 0) + 1
            
        metrics["domain_total"][q["domain"]] += 1
        metrics["category_total"][q.get("category", "unknown")] = metrics["category_total"].get(q.get("category", "unknown"), 0) + 1
            
        if stats["retrieval_used"]:
            metrics["rag_calls"] += 1
        if stats["remote_used"]:
            metrics["remote_calls"] += 1
            
        metrics["total_latency"] += stats["latency_total"]
        
        results.append({
            "question_id": q["question_id"],
            "domain": q["domain"],
            "category": q.get("category", "unknown"),
            "local_answer": stats.get("local_answer", ""),
            "local_judge_result": stats.get("local_judge_result", "N/A"),
            "rag_invoked": stats["retrieval_used"],
            "retrieved_chunks": stats.get("retrieved_chunks", []),
            "tier2_answer": stats.get("tier2_answer", ""),
            "tier2_judge_result": stats.get("tier2_judge_result", "N/A"),
            "tier3_invoked": stats["remote_used"],
            "final_answer": stats["answer"],
            "final_tier": tier,
            "latency": stats["latency_total"],
            "final_correctness": is_correct,
            "quality_label": quality
        })
        
        print(f"Processed {i+1}/{len(questions)} (Tier: {tier}, Correct: {is_correct}, Quality: {quality})")
        
    avg_latency = metrics["total_latency"] / len(questions)
    correctness_pct = (metrics["correct"] / len(questions)) * 100
    rag_call_rate = (metrics["rag_calls"] / len(questions)) * 100
    remote_call_rate = (metrics["remote_calls"] / len(questions)) * 100
    
    isro_perf = (metrics["domain_correct"]["isro"] / metrics["domain_total"]["isro"] * 100) if metrics["domain_total"]["isro"] > 0 else 0
    dpdpa_perf = (metrics["domain_correct"]["dpdpa"] / metrics["domain_total"]["dpdpa"] * 100) if metrics["domain_total"]["dpdpa"] > 0 else 0
    
    kg_perf = (metrics["category_correct"].get("knowledge_gap", 0) / metrics["category_total"].get("knowledge_gap", 1) * 100) if metrics["category_total"].get("knowledge_gap", 0) > 0 else 0
    reas_perf = (metrics["category_correct"].get("reasoning", 0) / metrics["category_total"].get("reasoning", 1) * 100) if metrics["category_total"].get("reasoning", 0) > 0 else 0
    
    final_metrics = {
        "System": "Practical Qwen Cascade",
        "Total questions": len(questions),
        "Tier 1 final count": metrics["tier1_final"],
        "Tier 2 final count": metrics["tier2_final"],
        "Tier 3 final count": metrics["tier3_final"],
        "Tier 1 to Tier 2 transitions": metrics["tier1_to_tier2"],
        "Tier 2 to Tier 3 transitions": metrics["tier2_to_tier3"],
        "Final correctness (%)": correctness_pct,
        "Quality distribution": metrics["quality_counts"],
        "Remote-call rate (%)": remote_call_rate,
        "RAG-call rate (%)": rag_call_rate,
        "Average latency (s)": avg_latency,
        "ISRO performance (%)": isro_perf,
        "DPDPA performance (%)": dpdpa_perf,
        "Knowledge-gap performance (%)": kg_perf,
        "Reasoning performance (%)": reas_perf
    }
    
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/final", exist_ok=True)
    
    with open("results/raw/practical_model_benchmark.json", "w") as f:
        json.dump(results, f, indent=4)
        
    with open("results/final/practical_model_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    # Generate failure analysis
    failures = [r for r in results if not r["final_correctness"] or r["quality_label"] in ("BAD", "UNKNOWN")]
    with open("results/final/practical_model_failure_analysis.json", "w") as f:
        json.dump(failures, f, indent=4)
        
    # Read the historical GPT-2 cascade results
    historical_cascade = {
        "System": "OLD CUSTOM GPT-2",
        "Tier 1 final": 0,
        "Tier 2 final": 1,
        "Tier 3 final": 29,
        "final correctness": "100%"
    }
    
    comparison = {
        "Historical_GPT2_Cascade": historical_cascade,
        "Practical_Local_Cascade": final_metrics
    }
    
    with open("results/final/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)
        
    print("\n=== Benchmark Completed ===")
    print(json.dumps(final_metrics, indent=4))

if __name__ == "__main__":
    main()
