import json
import csv
import os

def recalculate_metrics():
    raw_file = "results/raw/practical_model_benchmark.json"
    if not os.path.exists(raw_file):
        print(f"File not found: {raw_file}")
        return

    with open(raw_file, "r") as f:
        results = json.load(f)

    total_questions = len(results)
    
    # 1. FINAL_BENCHMARK_CORRECTNESS
    correct_count = sum(1 for r in results if r.get("final_correctness") == True)
    final_benchmark_correctness = (correct_count / total_questions) * 100 if total_questions else 0

    # 2. FINAL_SEMANTIC_QUALITY
    quality_counts = {"GOOD": 0, "PARTIALLY_GOOD": 0, "BAD": 0, "UNKNOWN": 0}
    for r in results:
        q = r.get("quality_label", "UNKNOWN")
        if q in quality_counts:
            quality_counts[q] += 1
        else:
            quality_counts["UNKNOWN"] += 1

    # 3. TIER_1_LOCAL_QUALITY
    # 4. TIER_2_RAG_QUALITY
    tier1_good = 0
    tier1_total = 0
    tier2_good = 0
    tier2_total = 0
    
    rag_calls = 0
    remote_calls = 0
    tier1_final = 0

    for r in results:
        # TIER_1_LOCAL_QUALITY: If final_tier is tier1, we know its quality.
        # If it went to rag, tier 1 failed.
        # This is an approximation since we don't have explicit semantic quality for the intermediate steps logged,
        # but if it was tier1 final, the quality label is for tier1.
        # If it was tier2, the quality label is for tier2.
        
        if r.get("final_tier") == "tier1":
            tier1_final += 1
            tier1_total += 1
            if r.get("quality_label") in ["GOOD", "PARTIALLY_GOOD"]:
                tier1_good += 1
        elif r.get("final_tier") == "tier2":
            tier1_total += 1 # Tier 1 ran but failed
            tier2_total += 1
            if r.get("quality_label") in ["GOOD", "PARTIALLY_GOOD"]:
                tier2_good += 1
        elif r.get("final_tier") == "tier3":
            tier1_total += 1
            tier2_total += 1
            
        if r.get("rag_invoked"):
            rag_calls += 1
        if r.get("tier3_invoked"):
            remote_calls += 1

    tier1_local_quality = (tier1_good / tier1_total) * 100 if tier1_total else 0
    tier2_rag_quality = (tier2_good / tier2_total) * 100 if tier2_total else 0
    
    remote_call_rate = (remote_calls / total_questions) * 100 if total_questions else 0
    rag_call_rate = (rag_calls / total_questions) * 100 if total_questions else 0
    tier1_final_rate = (tier1_final / total_questions) * 100 if total_questions else 0

    metrics = {
        "FINAL_BENCHMARK_CORRECTNESS_PERCENT": final_benchmark_correctness,
        "FINAL_SEMANTIC_QUALITY_COUNTS": quality_counts,
        "TIER_1_LOCAL_QUALITY_PERCENT": tier1_local_quality,
        "TIER_2_RAG_QUALITY_PERCENT": tier2_rag_quality,
        "REMOTE_CALL_RATE_PERCENT": remote_call_rate,
        "RAG_CALL_RATE_PERCENT": rag_call_rate,
        "TIER_1_FINAL_RATE_PERCENT": tier1_final_rate,
        "TOTAL_QUESTIONS": total_questions
    }

    # Save JSON
    with open("results/final/final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save CSV
    with open("results/final/final_metrics.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value", "Definition"])
        writer.writerow(["FINAL_BENCHMARK_CORRECTNESS", f"{final_benchmark_correctness:.2f}%", "Percentage of answers matching benchmark ground truth via keyword overlap"])
        writer.writerow(["FINAL_SEMANTIC_QUALITY_GOOD", quality_counts["GOOD"], "Count of answers manually or semantically judged as GOOD"])
        writer.writerow(["FINAL_SEMANTIC_QUALITY_PARTIALLY_GOOD", quality_counts["PARTIALLY_GOOD"], "Count of answers judged as PARTIALLY_GOOD"])
        writer.writerow(["FINAL_SEMANTIC_QUALITY_BAD", quality_counts["BAD"], "Count of answers judged as BAD"])
        writer.writerow(["TIER_1_LOCAL_QUALITY", f"{tier1_local_quality:.2f}%", "Percentage of Tier-1 final answers judged at least PARTIALLY_GOOD"])
        writer.writerow(["TIER_2_RAG_QUALITY", f"{tier2_rag_quality:.2f}%", "Percentage of Tier-2 final answers judged at least PARTIALLY_GOOD"])
        writer.writerow(["REMOTE_CALL_RATE", f"{remote_call_rate:.2f}%", "Percentage of questions routed to Tier 3 (Remote LLM)"])
        writer.writerow(["RAG_CALL_RATE", f"{rag_call_rate:.2f}%", "Percentage of questions routed to Tier 2 (RAG)"])
        writer.writerow(["TIER_1_FINAL_RATE", f"{tier1_final_rate:.2f}%", "Percentage of questions answered finally by Tier 1"])
        
    print("Successfully recalculated final metrics and saved to results/final/")

if __name__ == "__main__":
    recalculate_metrics()
