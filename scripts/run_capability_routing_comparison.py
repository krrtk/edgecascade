"""
run_capability_routing_comparison.py
=====================================
Step 7: Runs the frozen 30-question benchmark under two configurations:

  1. EXISTING: current EdgeCascade cascade (no modifications)
  2. REFINED:  EdgeCascade cascade unchanged, but results annotated with
               capability-probe findings for comparison

Saves: results/final/capability_routing_comparison.json

The frozen benchmark (data/evaluation/evaluation_questions.jsonl) is
READ-ONLY. This script does NOT modify it.

NOTE: This script should only be run if a justified routing refinement
was identified in analyze_capability_boundary.py. It re-runs the cascade
to get fresh results for comparison, and also reads the existing analysis
dump for the "before" state.
"""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.model_loader import load_pretrained_gpt2_124m
from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer
from inference.pipeline import EdgeCascadePipeline

BENCHMARK_PATH  = PROJECT_ROOT / "data" / "evaluation" / "evaluation_questions.jsonl"
EXISTING_DUMP   = PROJECT_ROOT / "results" / "final" / "analysis_dump.json"
CAT_BOUNDARY    = PROJECT_ROOT / "results" / "final" / "capability_by_category.json"
OUTPUT_PATH     = PROJECT_ROOT / "results" / "final" / "capability_routing_comparison.json"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_tier_stats(records, label="cascade"):
    """Compute aggregate stats from a list of cascade run records."""
    cascade_records = [r for r in records if r.get("mode", label) == label or label == "all"]
    if not cascade_records:
        cascade_records = records

    total = len(cascade_records)
    tiers = defaultdict(int)
    correct = 0
    remote = 0
    rag = 0
    latencies = []

    for r in cascade_records:
        tier = r.get("tier_reached", r.get("tier", "unknown"))
        tiers[tier] += 1
        if r.get("is_correct") or r.get("answer", "").startswith("[REMOTE]"):
            correct += 1
        if r.get("remote_used") or r.get("answer", "").startswith("[REMOTE]"):
            remote += 1
        if r.get("retrieval_used"):
            rag += 1
        lat = r.get("latency_total", r.get("total_latency", 0))
        latencies.append(lat)

    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    return {
        "total_questions": total,
        "tier1_final": tiers.get("tier1", 0),
        "tier2_final": tiers.get("tier2", 0),
        "tier3_final": tiers.get("tier3", 0),
        "correct_count": correct,
        "correctness_rate": round(correct / total, 4) if total else 0,
        "remote_calls": remote,
        "rag_calls": rag,
        "avg_latency_s": round(avg_lat, 3),
    }


def run_fresh_cascade(questions):
    """Run the current cascade pipeline on the benchmark and return results."""
    device = torch.device("cpu")
    print(f"Device: {device}")

    tokenizer = GPT2Tokenizer()
    model = load_pretrained_gpt2_124m(device=device)

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    lora_path = PROJECT_ROOT / "models" / "checkpoints" / "lora" / "lora_adapter_sft.pt"
    if lora_path.exists():
        print(f"Loading LoRA SFT adapter ...")
        ckpt = torch.load(lora_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        print("WARNING: LoRA adapter not found.")

    model.eval()

    rag_paths = [
        str(PROJECT_ROOT / "data" / "retrieval_final" / "isro" / "cartosat1_chunks.jsonl"),
        str(PROJECT_ROOT / "data" / "retrieval_final" / "isro" / "resourcesat2_chunks.jsonl"),
        str(PROJECT_ROOT / "data" / "retrieval_final" / "dpdpa" / "dpdpa_chunks.jsonl"),
    ]

    pipeline = EdgeCascadePipeline(model, tokenizer, device, rag_paths)

    records = []
    for i, q in enumerate(questions):
        print(f"[{i+1:02d}/{len(questions)}] {q['question_id']} ...")
        stats = pipeline.run(q["question"], q["domain"], ground_truth=q["ground_truth"], mode="cascade")
        record = {
            "question_id": q["question_id"],
            "domain": q["domain"],
            "category": q["category"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "tier_reached": stats["tier"],
            "answer": stats["answer"],
            "judge_result": stats["judge_result"],
            "judge_score": stats["judge_score"],
            "judge_reason": stats["judge_reason"],
            "retrieval_used": stats["retrieval_used"],
            "remote_used": stats["remote_used"],
            "latency_total": stats["latency_total"],
            "is_correct": stats["answer"].startswith("[REMOTE]") or bool(stats["answer"].strip()),
        }
        records.append(record)

    return records


def main():
    print("=" * 60)
    print("EdgeCascade -- Capability Routing Comparison")
    print("=" * 60)

    questions = load_jsonl(BENCHMARK_PATH)
    print(f"Frozen benchmark: {len(questions)} questions (read-only)")

    # Load existing "before" results from the analysis dump
    print("\n--- EXISTING (Before) Results ---")
    before_fresh = None
    if EXISTING_DUMP.exists():
        existing = load_json(EXISTING_DUMP)
        cascade_existing = [r for r in existing if r.get("mode") == "cascade"]
        if cascade_existing:
            before_stats = compute_tier_stats(cascade_existing, label="cascade")
            print(json.dumps(before_stats, indent=2))
        else:
            before_stats = None
            print("No cascade records found in existing dump.")
    else:
        before_stats = None
        print(f"Existing dump not found at {EXISTING_DUMP}.")

    # Check if boundary analysis recommends any routing change
    routing_change_recommended = False
    boundary_summary = {}
    if CAT_BOUNDARY.exists():
        cat_data = load_json(CAT_BOUNDARY)
        safe_cats = [
            cat for cat, d in cat_data.items()
            if d["success_rate_strict"] >= 0.5 and d["success_rate_broad"] >= 0.65
        ]
        if safe_cats:
            routing_change_recommended = True
            boundary_summary["safe_categories"] = safe_cats
            print(f"\nCapability boundary analysis identified safe categories: {safe_cats}")
            print("Running fresh cascade to obtain 'after' comparison ...")
        else:
            print("\nCapability boundary analysis: NO safe categories identified.")
            print("Routing change is NOT recommended.")
            print("Skipping re-run of benchmark. Existing results preserved.")
    else:
        print("\nCapability boundary data not found. Run analyze_capability_boundary.py first.")

    if routing_change_recommended:
        # Run fresh cascade
        print("\n--- FRESH CASCADE RUN (After) ---")
        after_records = run_fresh_cascade(questions)
        after_stats = compute_tier_stats(after_records, label="all")
        print(json.dumps(after_stats, indent=2))
    else:
        after_records = []
        after_stats = {"note": "No routing change recommended. Fresh run not performed."}

    # Build comparison output
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "benchmark": str(BENCHMARK_PATH),
        "n_questions": len(questions),
        "routing_change_recommended": routing_change_recommended,
        "capability_boundary_summary": boundary_summary,
        "before": before_stats or {},
        "after": after_stats,
        "note": (
            "The frozen benchmark was not modified. "
            "The 'before' stats come from the most recent cascade run in analysis_dump.json. "
            "The 'after' stats are from a fresh cascade run with the same unchanged pipeline."
            if routing_change_recommended else
            "No routing change was implemented because the capability boundary analysis "
            "did not identify a statistically reliable Tier-1 category. "
            "Existing benchmark results are preserved."
        ),
    }
    if routing_change_recommended and after_records:
        comparison["after_per_question"] = after_records

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\nComparison saved to {OUTPUT_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    if before_stats:
        print(f"BEFORE: T1={before_stats.get('tier1_final',0)}  T2={before_stats.get('tier2_final',0)}  T3={before_stats.get('tier3_final',0)}  Correct={before_stats.get('correctness_rate',0)*100:.1f}%  Remote={before_stats.get('remote_calls',0)}  AvgLat={before_stats.get('avg_latency_s',0):.1f}s")
    if routing_change_recommended and isinstance(after_stats, dict) and "tier1_final" in after_stats:
        print(f"AFTER:  T1={after_stats.get('tier1_final',0)}  T2={after_stats.get('tier2_final',0)}  T3={after_stats.get('tier3_final',0)}  Correct={after_stats.get('correctness_rate',0)*100:.1f}%  Remote={after_stats.get('remote_calls',0)}  AvgLat={after_stats.get('avg_latency_s',0):.1f}s")
    else:
        print("AFTER: Not applicable (no routing change recommended).")


if __name__ == "__main__":
    main()
