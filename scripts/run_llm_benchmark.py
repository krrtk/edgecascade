import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.practical_local_model import PracticalLocalModel
from inference.pipeline import EdgeCascadePipeline
from inference.judge import LLMEvaluationJudge, get_judge

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def main():
    print("Initializing LLM Evaluation Judge...")
    try:
        eval_judge = LLMEvaluationJudge()
        print(f"LLM Judge initialized using {eval_judge.provider} ({eval_judge.model})")
    except ValueError:
        print("\n" + "="*60)
        print("ERROR: API key not found. The semantic benchmark could not be completed.")
        print("Please set GROQ_API_KEY or OPENAI_API_KEY environment variable.")
        print("="*60 + "\n")
        return

    # Ensure routing defaults to semantic judge, not API judge, unless explicitly requested
    if not os.environ.get("ROUTING_JUDGE"):
        os.environ["ROUTING_JUDGE"] = "semantic"

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
    
    systems = ["local", "rag", "remote", "cascade"]
    
    all_results = {s: [] for s in systems}
    system_metrics = {
        s: {
            "correct": 0,
            "partially_correct": 0,
            "incorrect": 0,
            "api_error": 0,
            "rag_calls": 0,
            "local_generation_calls": 0, # Tier 1 & 2 calls
            "remote_generation_calls": 0, # Tier 3 calls
            "judge_calls": 0, # LLM Evaluation Judge calls
            "total_latency": 0,
            "tier1_count": 0,
            "tier2_count": 0,
            "tier3_count": 0
        } for s in systems
    }
    
    for i, q in enumerate(questions):
        print(f"\n--- Processing Question {i+1}/{len(questions)} ---")
        question_stats = {}
        candidates_for_judge = {}
        
        for system in systems:
            print(f"  [DEBUG] Starting pipeline.run for mode={system}...")
            stats = pipeline.run(q["question"], q["domain"], ground_truth=q["ground_truth"], mode=system)
            print(f"  [DEBUG] Finished pipeline.run for mode={system}")
            question_stats[system] = stats
            
            # Record metrics
            m = system_metrics[system]
            if stats["retrieval_used"]: m["rag_calls"] += 1
            if stats["remote_used"]: m["remote_generation_calls"] += 1
            
            local_calls = 0
            if stats.get("local_answer"): local_calls += 1
            if stats.get("tier2_answer"): local_calls += 1
            m["local_generation_calls"] += local_calls
            m["total_latency"] += stats["latency_total"]
            
            tier = stats.get("tier", "unknown")
            if tier == "tier1": m["tier1_count"] += 1
            elif tier == "tier2": m["tier2_count"] += 1
            elif tier == "tier3": m["tier3_count"] += 1
            
            answer_to_evaluate = stats["answer"]
            if answer_to_evaluate.startswith("[REMOTE] "):
                answer_to_evaluate = answer_to_evaluate[9:]
            elif answer_to_evaluate.startswith("[REMOTE_ERROR]"):
                answer_to_evaluate = answer_to_evaluate
                
            candidates_for_judge[system] = answer_to_evaluate

        # Batched evaluation
        print("  [DEBUG] Starting evaluate_batch...")
        judge_results = eval_judge.evaluate_batch(q["question"], q["ground_truth"], candidates_for_judge)
        print("  [DEBUG] Finished evaluate_batch")
        
        for system in systems:
            m = system_metrics[system]
            m["judge_calls"] += 1 # We count 1 logically per system, though batched physically
            
            j_res = judge_results.get(system, {"status": "error", "score": 0.0, "reason": "Missing result", "correct": False})
            
            status = j_res["status"]
            if candidates_for_judge[system].startswith("[REMOTE_ERROR]") or status == "api_error":
                m["api_error"] += 1
                j_res["status"] = "API_ERROR"
            elif status == "correct":
                m["correct"] += 1
            elif status == "partially correct":
                m["partially_correct"] += 1
            else:
                m["incorrect"] += 1
                
            stats = question_stats[system]
            all_results[system].append({
                "question_id": q["question_id"],
                "answer": stats["answer"],
                "tier_reached": stats["tier"],
                "correctness": j_res["correct"],
                "judge_score": j_res["score"],
                "judge_reason": j_res["reason"],
                "judge_status": j_res["status"],
                "local_generation_calls_for_this_q": local_calls,
                "latency": stats["latency_total"]
            })
            
            print(f"[{system.upper()}] Tier: {stats['tier']} | Eval: {j_res['status'].upper()} | Ans: {stats['answer'][:40]}...")

    # Calculate final metrics
    for system in systems:
        m = system_metrics[system]
        total = len(questions)
        valid = total - m["api_error"]
        
        # Accuracy over total questions (including API errors as 0 score) or just valid?
        # The prompt says: "Do not treat API errors as incorrect." but accuracy usually divides by total.
        # We will compute both.
        m["accuracy"] = (m["correct"] + 0.5 * m["partially_correct"]) / total * 100
        m["accuracy_strict"] = (m["correct"]) / total * 100

    cascade = system_metrics["cascade"]
    remote = system_metrics["remote"]
    rag = system_metrics["rag"]
    
    comparisons = {
        "remote_generation_reduction_vs_always_remote": remote["remote_generation_calls"] - cascade["remote_generation_calls"],
        "rag_call_reduction_vs_always_rag": rag["rag_calls"] - cascade["rag_calls"],
        "remote_calls_per_correct_answer": cascade["remote_generation_calls"] / cascade["correct"] if cascade["correct"] > 0 else 0,
        "physical_judge_api_calls": eval_judge.call_count
    }
    
    final_report = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "judge_provider": eval_judge.provider,
            "judge_model": eval_judge.model,
            "benchmark_size": len(questions),
            "systems_evaluated": systems,
            "routing_method": "EvidenceGate (Deterministic)",
            "evaluation_method": "Batched LLM Semantic Evaluation"
        },
        "system_metrics": system_metrics,
        "comparisons": comparisons
    }

    os.makedirs("results/final", exist_ok=True)
    
    with open("results/final/routing_benchmark.json", "w") as f:
        json.dump(final_report, f, indent=4)
        
    with open("results/final/routing_raw_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # Generate Markdown Report
    md_content = f"# EdgeCascade Routing & Evaluation Benchmark\n\n"
    md_content += f"**Date:** {final_report['metadata']['date']}\n"
    md_content += f"**Judge Provider:** {final_report['metadata']['judge_provider']} ({final_report['metadata']['judge_model']})\n"
    md_content += f"**Routing Method:** {final_report['metadata']['routing_method']}\n"
    md_content += f"**Evaluation Method:** {final_report['metadata']['evaluation_method']}\n"
    md_content += f"**Benchmark Size:** {final_report['metadata']['benchmark_size']} questions\n\n"
    
    md_content += f"## System Performance\n\n"
    md_content += "| System | Correct | Partial | Incorrect | API Error | Accuracy (Strict) | RAG Calls | Tier-3 Calls | Judge Calls |\n"
    md_content += "|---|---|---|---|---|---|---|---|---|\n"
    
    for sys_name, m in system_metrics.items():
        md_content += f"| {sys_name.capitalize()} | {m['correct']} | {m['partially_correct']} | {m['incorrect']} | {m['api_error']} | {m['accuracy_strict']:.2f}% | {m['rag_calls']} | {m['remote_generation_calls']} | {m['judge_calls']} |\n"

    md_content += f"\n## EdgeCascade Tier Distribution\n\n"
    md_content += f"- **Tier 1 (Local):** {cascade['tier1_count']}\n"
    md_content += f"- **Tier 2 (RAG):** {cascade['tier2_count']}\n"
    md_content += f"- **Tier 3 (Remote):** {cascade['tier3_count']}\n"
    md_content += f"- **Percentage answered without Tier 3:** {((cascade['tier1_count'] + cascade['tier2_count']) / len(questions)) * 100:.2f}%\n"

    md_content += f"\n## Key Comparisons\n\n"
    for k, v in comparisons.items():
        if isinstance(v, float):
            md_content += f"* **{k.replace('_', ' ').title()}:** {v:.2f}\n"
        else:
            md_content += f"* **{k.replace('_', ' ').title()}:** {v}\n"
            
    md_content += "\n## Architecture Note\n"
    md_content += "The LLM Judge is an **EVALUATION ORACLE ONLY**. It is NOT used for runtime routing. Runtime routing uses a deterministic `EvidenceGate`.\n"
    
    with open("results/final/routing_benchmark.md", "w") as f:
        f.write(md_content)
        
    print("\n=== Benchmark Completed ===")
    print(f"Results saved to results/final/routing_benchmark.json and .md")

if __name__ == "__main__":
    main()
