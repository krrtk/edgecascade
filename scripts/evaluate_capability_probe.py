"""
evaluate_capability_probe.py
============================
Independent evaluator: reads capability_probe_results.json,
calls the API judge with (question, ground_truth, generated_answer)
and produces:
  - results/final/capability_probe_evaluation.json
  - results/final/capability_by_category.json

The judge is used ONLY as an offline analysis tool.
It does NOT influence routing.

The judge is told:
  - the question
  - the ground truth
  - the generated answer

It is NOT told:
  - expected_tier
  - routing labels
  - retrieval context
  - question_id as semantic information

Quality labels: GOOD / PARTIALLY_GOOD / BAD / UNKNOWN
"""

import json
import os
import sys
import time
import requests
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_RESULTS   = PROJECT_ROOT / "results" / "raw" / "capability_probe_results.json"
EVAL_OUTPUT   = PROJECT_ROOT / "results" / "final" / "capability_probe_evaluation.json"
CAT_OUTPUT    = PROJECT_ROOT / "results" / "final" / "capability_by_category.json"

QUALITY_LABELS = ["GOOD", "PARTIALLY_GOOD", "BAD", "UNKNOWN"]


# ---------------------------------------------------------------------------
# Load API credentials (same .env loading as judge.py)
# ---------------------------------------------------------------------------

def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    os.environ[k] = v

load_env()


# ---------------------------------------------------------------------------
# Offline capability judge
# ---------------------------------------------------------------------------

def judge_answer_quality(question, ground_truth, generated_answer, api_key, endpoint, model_id, retry=3):
    """
    Calls the LLM API to assess whether the generated_answer is
    GOOD / PARTIALLY_GOOD / BAD / UNKNOWN relative to the ground_truth.

    This is STRICTLY an offline analysis call.
    No routing label or retrieval context is passed.
    """
    ans = generated_answer.strip()

    system_prompt = (
        "You are an independent evaluator assessing the factual quality of a model-generated answer. "
        "You are given a question, a reference ground truth answer, and a generated answer. "
        "Your task is to determine the quality of the generated answer.\n\n"
        "Quality categories:\n"
        "  GOOD: The generated answer is factually correct, relevant, and substantially agrees with the ground truth. "
        "Minor phrasing differences are acceptable.\n"
        "  PARTIALLY_GOOD: The answer contains some correct information but is incomplete, partially incorrect, or includes "
        "unsupported claims alongside correct ones.\n"
        "  BAD: The answer is factually incorrect, hallucinates content, is circular/repetitive, is irrelevant, "
        "or fails to address the question at all.\n"
        "  UNKNOWN: The answer is too ambiguous to classify, or contains both correct and incorrect elements in an "
        "indistinguishable way.\n\n"
        "IMPORTANT: Do NOT reward fluency alone. A fluent answer that is factually wrong must be rated BAD.\n"
        "Do NOT penalize brevity if the core answer is correct.\n\n"
        "You MUST output exactly valid JSON:\n"
        "{\n"
        "  \"quality\": \"GOOD\" | \"PARTIALLY_GOOD\" | \"BAD\" | \"UNKNOWN\",\n"
        "  \"score\": int (1-5),\n"
        "  \"reason\": \"Brief explanation (1-2 sentences)\",\n"
        "  \"failure_mode\": \"none\" | \"hallucination\" | \"incompleteness\" | \"repetition\" | "
        "\"instruction_following\" | \"circular\" | \"irrelevant\" | \"other\"\n"
        "}\n"
        "Score guide: 1=clearly bad, 2=weak, 3=borderline, 4=good, 5=strong/highly accurate."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Ground Truth: {ground_truth}\n\n"
        f"Generated Answer: {ans}"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(retry):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            quality = result.get("quality", "UNKNOWN")
            if quality not in QUALITY_LABELS:
                quality = "UNKNOWN"
            return {
                "quality": quality,
                "score": int(result.get("score", 1)),
                "reason": str(result.get("reason", "")),
                "failure_mode": str(result.get("failure_mode", "other")),
            }
        except Exception as e:
            if attempt == retry - 1:
                return {
                    "quality": "UNKNOWN",
                    "score": 1,
                    "reason": f"API Error: {str(e)}",
                    "failure_mode": "other",
                }
            time.sleep(2)


# ---------------------------------------------------------------------------
# Aggregate by category
# ---------------------------------------------------------------------------

def aggregate_by_category(evaluated):
    cats = defaultdict(lambda: {
        "n": 0,
        "GOOD": 0,
        "PARTIALLY_GOOD": 0,
        "BAD": 0,
        "UNKNOWN": 0,
        "latencies": [],
        "log_probs": [],
    })

    for r in evaluated:
        cat = r["category"]
        cats[cat]["n"] += 1
        cats[cat][r["quality"]] += 1
        cats[cat]["latencies"].append(r["generation_latency_s"])
        if r.get("avg_log_prob") is not None:
            cats[cat]["log_probs"].append(r["avg_log_prob"])

    summary = {}
    for cat, d in sorted(cats.items()):
        n = d["n"]
        good = d["GOOD"]
        partial = d["PARTIALLY_GOOD"]
        bad = d["BAD"]
        unk = d["UNKNOWN"]
        success_rate = (good + partial) / n if n else 0.0
        strict_rate = good / n if n else 0.0
        avg_lat = sum(d["latencies"]) / len(d["latencies"]) if d["latencies"] else 0.0
        avg_lp = sum(d["log_probs"]) / len(d["log_probs"]) if d["log_probs"] else None

        summary[cat] = {
            "n_questions": n,
            "GOOD": good,
            "PARTIALLY_GOOD": partial,
            "BAD": bad,
            "UNKNOWN": unk,
            "success_rate_broad": round(success_rate, 4),
            "success_rate_strict": round(strict_rate, 4),
            "avg_latency_s": round(avg_lat, 4),
            "avg_log_prob": round(avg_lp, 4) if avg_lp is not None else None,
        }

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("EdgeCascade -- Capability Probe Offline Evaluator")
    print("=" * 60)

    # Check API key
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found in environment. Set GROQ_API_KEY or OPENAI_API_KEY.")

    if os.environ.get("GROQ_API_KEY"):
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        model_id = "qwen/qwen3.8-27b"
    else:
        endpoint = "https://api.openai.com/v1/chat/completions"
        model_id = "gpt-4o-mini"

    print(f"API endpoint: {endpoint}")
    print(f"Model: {model_id}")

    # Load raw probe results
    if not RAW_RESULTS.exists():
        print(f"ERROR: Raw results not found at {RAW_RESULTS}")
        print("Run scripts/run_capability_probe.py first.")
        sys.exit(1)

    with open(RAW_RESULTS, "r", encoding="utf-8") as f:
        probe_results = json.load(f)

    print(f"\nEvaluating {len(probe_results)} answers ...")
    print()

    evaluated = []
    for i, r in enumerate(probe_results):
        qid = r["question_id"]
        question = r["question"]
        ground_truth = r["ground_truth"]
        generated_answer = r["generated_answer"]
        category = r["category"]
        domain = r["domain"]

        print(f"[{i+1:02d}/{len(probe_results)}] {qid} ({category}) ...")

        eval_result = judge_answer_quality(
            question=question,
            ground_truth=ground_truth,
            generated_answer=generated_answer,
            api_key=api_key,
            endpoint=endpoint,
            model_id=model_id,
        )

        record = {
            "question_id": qid,
            "domain": domain,
            "category": category,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "quality": eval_result["quality"],
            "score": eval_result["score"],
            "reason": eval_result["reason"],
            "failure_mode": eval_result["failure_mode"],
            "generation_latency_s": r.get("generation_latency_s"),
            "avg_log_prob": r.get("avg_log_prob"),
            "num_generated_tokens": r.get("num_generated_tokens"),
        }
        evaluated.append(record)

        print(f"  Quality: {eval_result['quality']} (score={eval_result['score']})")
        print(f"  Reason: {eval_result['reason'][:100]}")
        print(f"  Failure mode: {eval_result['failure_mode']}")
        print()

        # Rate limiting guard
        time.sleep(0.5)

    # Save evaluation
    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(evaluated, f, indent=2, ensure_ascii=False)
    print(f"Evaluation saved to {EVAL_OUTPUT}")

    # Aggregate by category
    cat_summary = aggregate_by_category(evaluated)
    with open(CAT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(cat_summary, f, indent=2, ensure_ascii=False)
    print(f"Category summary saved to {CAT_OUTPUT}")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS BY CATEGORY")
    print("=" * 60)
    total_q = len(evaluated)
    total_good = sum(1 for r in evaluated if r["quality"] == "GOOD")
    total_partial = sum(1 for r in evaluated if r["quality"] == "PARTIALLY_GOOD")
    total_bad = sum(1 for r in evaluated if r["quality"] == "BAD")
    total_unk = sum(1 for r in evaluated if r["quality"] == "UNKNOWN")

    print(f"{'Category':<20} {'N':>4} {'GOOD':>6} {'PART':>6} {'BAD':>6} {'UNK':>5} {'Strict%':>8} {'Broad%':>8}")
    print("-" * 72)
    for cat, d in cat_summary.items():
        print(
            f"{cat:<20} {d['n_questions']:>4} {d['GOOD']:>6} {d['PARTIALLY_GOOD']:>6} "
            f"{d['BAD']:>6} {d['UNKNOWN']:>5} "
            f"{d['success_rate_strict']*100:>7.1f}% {d['success_rate_broad']*100:>7.1f}%"
        )
    print("-" * 72)
    broad = (total_good + total_partial) / total_q if total_q else 0
    strict = total_good / total_q if total_q else 0
    print(f"{'TOTAL':<20} {total_q:>4} {total_good:>6} {total_partial:>6} {total_bad:>6} {total_unk:>5} {strict*100:>7.1f}% {broad*100:>7.1f}%")


if __name__ == "__main__":
    main()
