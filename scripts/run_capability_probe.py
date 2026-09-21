"""
run_capability_probe.py
=======================
Strictly local-model capability experiment.
Loads the SFT-tuned LoRA GPT-2 124M and generates answers for every
question in data/evaluation/capability_probe.jsonl.

NO judge calls. NO RAG. NO remote LLM.

Results are saved to results/raw/capability_probe_results.json
"""

import io
import json
import sys
import time
import warnings
from pathlib import Path

# Force stdout to UTF-8 so special chars in generated text don't crash
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch

warnings.filterwarnings("ignore")

# project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.model_loader import load_pretrained_gpt2_124m
from inference.generator import generate_text
from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROBE_DATASET = PROJECT_ROOT / "data" / "evaluation" / "capability_probe.jsonl"
OUTPUT_PATH   = PROJECT_ROOT / "results" / "raw" / "capability_probe_results.json"
LORA_PATH     = PROJECT_ROOT / "models" / "checkpoints" / "lora" / "lora_adapter_sft.pt"

GEN_CONFIG = {
    "max_new_tokens": 60,
    "temperature": 0.8,
    "top_k": 40,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_probe_dataset(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_answer(full_output, prompt):
    """Strip the prompt prefix and clean up the generated answer."""
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


def compute_avg_log_prob(log_probs):
    if not log_probs:
        return None
    return sum(log_probs) / len(log_probs)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("EdgeCascade -- Local Capability Probe Runner")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer ...")
    tokenizer = GPT2Tokenizer()

    print("Loading pretrained GPT-2 124M ...")
    model = load_pretrained_gpt2_124m(device=device)

    # Freeze base weights, mark LoRA trainable for state_dict compatibility
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    lora_hyperparams = {}
    if LORA_PATH.exists():
        print(f"Loading SFT LoRA adapter from {LORA_PATH} ...")
        ckpt = torch.load(LORA_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        lora_hyperparams = ckpt.get("hyperparameters", {})
        print(f"  LoRA hyperparameters: {lora_hyperparams}")
    else:
        print(f"WARNING: SFT LoRA adapter not found at {LORA_PATH}. Running base model.")

    model.to(device)
    model.eval()

    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA parameters active: {lora_params:,}")

    print(f"\nLoading capability probe dataset from {PROBE_DATASET} ...")
    questions = load_probe_dataset(PROBE_DATASET)
    print(f"  {len(questions)} questions loaded.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Run probe
    # -----------------------------------------------------------------------
    results = []
    total_start = time.time()

    for i, q in enumerate(questions):
        qid = q["question_id"]
        question_text = q["question"]
        domain = q["domain"]
        category = q["category"]
        ground_truth = q.get("ground_truth", "")

        prompt = f"### Question:\n{question_text}\n\n### Answer:\n"

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

        generated_answer = extract_answer(full_output, prompt)
        avg_log_prob = compute_avg_log_prob(log_probs)

        record = {
            "question_id": qid,
            "domain": domain,
            "category": category,
            "question": question_text,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "generation_latency_s": round(latency, 4),
            "avg_log_prob": round(avg_log_prob, 6) if avg_log_prob is not None else None,
            "num_generated_tokens": len(log_probs),
            "generation_config": GEN_CONFIG,
            "lora_hyperparameters": lora_hyperparams,
        }
        results.append(record)

        logp_str = f"{avg_log_prob:.3f}" if avg_log_prob is not None else "N/A"
        # Sanitize text for printing (replace non-ascii safely)
        q_safe = question_text[:80].encode('ascii', 'replace').decode('ascii')
        a_safe = generated_answer[:120].encode('ascii', 'replace').decode('ascii')
        print(f"[{i+1:02d}/{len(questions)}] {qid} | {latency:.2f}s | logp={logp_str}")
        print(f"  Q: {q_safe}")
        print(f"  A: {a_safe}")
        print()

        # Incremental save after each question (in case of crash)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    total_time = time.time() - total_start

    # Final save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nProbe complete. {len(results)} records saved to {OUTPUT_PATH}")
    print(f"Total wall-clock time: {total_time:.1f}s")
    print(f"Average latency per question: {total_time/len(results):.2f}s")


if __name__ == "__main__":
    main()
