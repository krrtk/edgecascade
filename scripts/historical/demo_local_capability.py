"""
demo_local_capability.py
=========================
Recruiter-facing demonstration of the 124M local model's capability boundary.

Shows:
  - Successful local examples (if any)
  - Borderline examples
  - Clear failure examples

Reads capability_probe_evaluation.json if available for pre-computed quality
labels. Otherwise runs inference live and labels answers heuristically.

Usage:
    python scripts/demo_local_capability.py [--live]

    --live  Force live inference even if evaluation results exist.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.model_loader import load_pretrained_gpt2_124m
from inference.generator import generate_text
from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer

EVAL_PATH = PROJECT_ROOT / "results" / "final" / "capability_probe_evaluation.json"
LORA_PATH = PROJECT_ROOT / "models" / "checkpoints" / "lora" / "lora_adapter_sft.pt"

GEN_CONFIG = {
    "max_new_tokens": 60,
    "temperature": 0.8,
    "top_k": 40,
}

# Curated demo questions that span the capability spectrum
DEMO_QUESTIONS = [
    # Expected: potential success (trained definitions)
    {
        "question": "What is remote sensing?",
        "category": "BASIC_DEFINITION",
        "domain": "isro",
        "ground_truth": "Remote sensing is the science and art of obtaining information about an object, area, or phenomenon through the analysis of data acquired by a device not in contact with the object.",
        "expected_outcome": "Success candidate (trained definition)",
    },
    {
        "question": "Who is a Data Principal under the DPDPA?",
        "category": "BASIC_DEFINITION",
        "domain": "dpdpa",
        "ground_truth": "A Data Principal is the individual to whom the personal data relates.",
        "expected_outcome": "Success candidate (trained definition)",
    },
    # Borderline
    {
        "question": "What is the difference between active and passive remote sensing?",
        "category": "SIMPLE_CONCEPT",
        "domain": "isro",
        "ground_truth": "Passive remote sensing measures reflected sunlight; active remote sensing provides its own energy source like radar.",
        "expected_outcome": "Borderline (trained concept, may hallucinate details)",
    },
    {
        "question": "What are the requirements for valid consent under the DPDPA?",
        "category": "SIMPLE_RELATIONSHIP",
        "domain": "dpdpa",
        "ground_truth": "Consent must be free, specific, informed, unconditional and unambiguous with a clear affirmative action.",
        "expected_outcome": "Borderline (multi-element answer)",
    },
    # Failure expected
    {
        "question": "Why would a passive remote sensing system be unable to collect data at night?",
        "category": "REASONING",
        "domain": "isro",
        "ground_truth": "Passive sensors rely on reflected sunlight; at night there is no sunlight to reflect.",
        "expected_outcome": "Failure expected (requires reasoning)",
    },
    {
        "question": "What is the NavIC satellite system used for?",
        "category": "SPECIFIC_FACT",
        "domain": "isro",
        "ground_truth": "NavIC (IRNSS) provides accurate position information to users in India and the region extending up to 1500 km from its boundary.",
        "expected_outcome": "Failure likely (specific facts, hallucination risk)",
    },
    {
        "question": "How does a Data Fiduciary prove that consent was validly obtained in a dispute?",
        "category": "REASONING",
        "domain": "dpdpa",
        "ground_truth": "The Data Fiduciary bears the burden of proof and must show that a notice was given and consent was obtained in accordance with the Act.",
        "expected_outcome": "Failure expected (legal reasoning + multi-fact)",
    },
]


def extract_answer(full_output, prompt):
    if full_output.startswith(prompt):
        text = full_output[len(prompt):]
    else:
        marker = "### Answer:"
        idx = full_output.find(marker)
        if idx != -1:
            text = full_output[idx + len(marker):]
        else:
            text = full_output
    text = text.split("### Question:")[0].split("### Answer:")[0]
    return text.strip()


def heuristic_quality(answer, ground_truth):
    """Simple heuristic quality label for demo when API judge is unavailable."""
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


def load_precomputed_quality(eval_path, question_text):
    """Look up precomputed quality from evaluation results."""
    if not eval_path.exists():
        return None
    with open(eval_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        if r["question"] == question_text:
            return {
                "quality": r["quality"],
                "reason": r.get("reason", ""),
                "failure_mode": r.get("failure_mode", ""),
                "avg_log_prob": r.get("avg_log_prob"),
                "generated_answer": r["generated_answer"],
            }
    return None


QUALITY_COLORS = {
    "GOOD": "\033[92m",          # green
    "PARTIALLY_GOOD": "\033[93m",# yellow
    "BAD": "\033[91m",           # red
    "UNKNOWN": "\033[90m",       # grey
}
RESET = "\033[0m"


def print_demo_result(q, answer, quality, reason, avg_log_prob, latency, source):
    color = QUALITY_COLORS.get(quality, "")
    print(f"\n{'='*68}")
    print(f"QUESTION   : {q['question']}")
    print(f"CATEGORY   : {q['category']}  |  DOMAIN: {q['domain']}")
    print(f"EXPECTED   : {q['expected_outcome']}")
    print(f"")
    print(f"LOCAL ANSWER   :")
    print(f"  {answer[:300]}")
    print(f"")
    print(f"GROUND TRUTH   :")
    print(f"  {q['ground_truth']}")
    print(f"")
    lp_str = f"{avg_log_prob:.3f}" if avg_log_prob is not None else "N/A"
    lat_str = f"{latency:.2f}s" if latency is not None else "N/A (precomputed)"
    print(f"LOCAL CONFIDENCE (avg log-prob): {lp_str}")
    print(f"LATENCY        : {lat_str}")
    print(f"{color}QUALITY RESULT : {quality}{RESET}")
    if reason:
        print(f"JUDGE REASON   : {reason[:150]}")
    print(f"SOURCE         : {source}")
    print(f"{'='*68}")


def main():
    parser = argparse.ArgumentParser(description="EdgeCascade Local Capability Demo")
    parser.add_argument("--live", action="store_true", help="Force live inference (ignore precomputed results)")
    args = parser.parse_args()

    print("=" * 68)
    print("  EdgeCascade -- Local Model Capability Demonstration")
    print("  Model: Custom GPT-2 124M + Q/V LoRA (rank=8) + SFT adapter")
    print("  Purpose: Show what the local model CAN and CANNOT do")
    print("=" * 68)

    # Check if we have precomputed evaluation results
    has_precomputed = EVAL_PATH.exists() and not args.live

    model = None
    tokenizer = None

    if not has_precomputed or args.live:
        print("\nLoading model for live inference ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        tokenizer = GPT2Tokenizer()
        model = load_pretrained_gpt2_124m(device=device)

        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        if LORA_PATH.exists():
            ckpt = torch.load(LORA_PATH, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"SFT LoRA adapter loaded from {LORA_PATH}")
        else:
            print(f"WARNING: No SFT LoRA adapter found. Running base GPT-2.")

        model.to(device)
        model.eval()
    else:
        print(f"\nUsing precomputed evaluation results from {EVAL_PATH}")
        device = "cpu"

    # Demo loop
    results = []
    for q in DEMO_QUESTIONS:
        precomputed = None
        if has_precomputed:
            precomputed = load_precomputed_quality(EVAL_PATH, q["question"])

        if precomputed:
            answer = precomputed["generated_answer"]
            quality = precomputed["quality"]
            reason = precomputed.get("reason", "")
            avg_log_prob = precomputed.get("avg_log_prob")
            latency = None
            source = "precomputed (capability_probe_evaluation.json)"
        elif model is not None:
            # Live inference
            prompt = f"### Question:\n{q['question']}\n\n### Answer:\n"
            t0 = time.time()
            full_output, log_probs = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=GEN_CONFIG["max_new_tokens"],
                device=device,
                return_probs=True,
                temperature=GEN_CONFIG["temperature"],
                top_k=GEN_CONFIG["top_k"],
            )
            latency = time.time() - t0
            answer = extract_answer(full_output, prompt)
            avg_log_prob = sum(log_probs)/len(log_probs) if log_probs else None
            quality = heuristic_quality(answer, q["ground_truth"])
            reason = "(heuristic label — run evaluate_capability_probe.py for API-judged labels)"
            source = "live inference"
        else:
            # No precomputed result and no model loaded; skip gracefully
            answer = "(question not in probe dataset; model not loaded for this demo run)"
            quality = "UNKNOWN"
            reason = "Not in precomputed probe evaluation"
            avg_log_prob = None
            latency = None
            source = "skipped"

        print_demo_result(q, answer, quality, reason, avg_log_prob, latency, source)
        results.append({
            "question": q["question"],
            "category": q["category"],
            "generated_answer": answer,
            "quality": quality,
            "avg_log_prob": avg_log_prob,
        })

    # Summary
    print("\n" + "=" * 68)
    print("DEMO SUMMARY")
    print("=" * 68)
    from collections import Counter
    counts = Counter(r["quality"] for r in results)
    total = len(results)
    for label in ["GOOD", "PARTIALLY_GOOD", "BAD", "UNKNOWN"]:
        n = counts.get(label, 0)
        pct = n / total * 100
        color = QUALITY_COLORS.get(label, "")
        print(f"  {color}{label:<16}{RESET}: {n}/{total} ({pct:.0f}%)")

    print()
    good_count = counts.get("GOOD", 0)
    if good_count >= 3:
        print("  >> Some capability demonstrated. Local model useful for Tier-1 on specific categories.")
    elif good_count >= 1:
        print("  >> Very limited capability. Tier-1 usage only in narrow, highly specific cases.")
    else:
        print("  >> No reliable capability demonstrated on this demo set.")
        print("     The cascade correctly routes most queries to Tier 2 or Tier 3.")

    print()
    print("  Note: This is an honest capability demonstration. The local 124M model")
    print("  is architecturally present and trainable, but its knowledge capacity")
    print("  is inherently limited by model size (124M parameters, rank-8 LoRA).")
    print("  The EdgeCascade design correctly handles this by escalating to better tiers.")
    print("=" * 68)


if __name__ == "__main__":
    main()
