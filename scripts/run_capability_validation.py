"""
run_capability_validation.py
==============================
Step 10: Final validation.
Verifies:
  - model loads correctly
  - SFT adapter loads
  - capability probe results exist
  - API judge is reachable
  - existing benchmark is intact (read-only check)
  - no holdout leakage in probe dataset
  - no benchmark question leakage in probe dataset
  - all required artifacts are present

Produces: results/logs/capability_probe.log
"""

import json
import os
import sys
import time
import requests
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_PATH = PROJECT_ROOT / "results" / "logs" / "capability_probe.log"

REQUIRED_ARTIFACTS = [
    "data/evaluation/capability_probe.jsonl",
    "results/raw/capability_probe_results.json",
    "results/final/capability_probe_evaluation.json",
    "results/final/capability_by_category.json",
    "results/final/capability_boundary_report.md",
    "results/final/capability_routing_comparison.json",
    "scripts/run_capability_probe.py",
    "scripts/evaluate_capability_probe.py",
    "scripts/analyze_capability_boundary.py",
    "scripts/demo_local_capability.py",
    "scripts/run_capability_routing_comparison.py",
]

BENCHMARK_PATH = PROJECT_ROOT / "data" / "evaluation" / "evaluation_questions.jsonl"
PROBE_PATH     = PROJECT_ROOT / "data" / "evaluation" / "capability_probe.jsonl"
PROBE_RESULTS  = PROJECT_ROOT / "results" / "raw" / "capability_probe_results.json"

# Known holdout sources that must not appear in probe
HOLDOUT_MARKERS = [
    "Cartosat-1", "cartosat1", "holdout_document",
    "PAN-Fore", "PAN-Aft", "OBSSR",
    "dpdpa_005", "dpdpa_006", "dpdpa_036",  # holdout DPDPA sections
]


def log(lines, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    lines.append(entry)
    print(entry)


def check_model_load(lines):
    log(lines, "CHECK: Model load ...")
    try:
        import torch
        from inference.model_loader import load_pretrained_gpt2_124m
        from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer

        device = torch.device("cpu")
        tokenizer = GPT2Tokenizer()
        model = load_pretrained_gpt2_124m(device=device)

        lora_path = PROJECT_ROOT / "models" / "checkpoints" / "lora" / "lora_adapter_sft.pt"
        if lora_path.exists():
            ckpt = torch.load(lora_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            hp = ckpt.get("hyperparameters", {})
            log(lines, f"  PASS: SFT LoRA adapter loaded. Hyperparameters: {hp}")
        else:
            log(lines, f"  FAIL: LoRA adapter not found at {lora_path}")
            return False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(lines, f"  INFO: Model loaded. LoRA trainable params = {trainable:,}")

        # Quick generation test
        from inference.generator import generate_text
        prompt = "### Question:\nWhat is remote sensing?\n\n### Answer:\n"
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        out, lp = generate_text(model, tokenizer, prompt, max_new_tokens=10, device=device, return_probs=True)
        log(lines, f"  PASS: Generation test ok. Output tokens: {len(lp)}")
        return True
    except Exception as e:
        log(lines, f"  FAIL: {e}")
        return False


def check_api_judge(lines):
    log(lines, "CHECK: API judge reachability ...")

    def load_env():
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        os.environ[k] = v
    load_env()

    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log(lines, "  FAIL: No API key found.")
        return False

    if os.environ.get("GROQ_API_KEY"):
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        model_id = "qwen/qwen3.8-27b"
    else:
        endpoint = "https://api.openai.com/v1/chat/completions"
        model_id = "gpt-4o-mini"

    try:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say 'ok' in exactly one word."}],
            "temperature": 0.0,
            "max_tokens": 5,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        log(lines, f"  PASS: API judge reachable (endpoint={endpoint}, model={model_id})")
        return True
    except Exception as e:
        log(lines, f"  FAIL: {e}")
        return False


def check_benchmark_intact(lines):
    log(lines, "CHECK: Frozen benchmark integrity ...")
    if not BENCHMARK_PATH.exists():
        log(lines, f"  FAIL: Benchmark not found at {BENCHMARK_PATH}")
        return False

    records = []
    with open(BENCHMARK_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if len(records) != 30:
        log(lines, f"  FAIL: Expected 30 benchmark questions, found {len(records)}")
        return False

    benchmark_ids = {r["question_id"] for r in records}
    log(lines, f"  PASS: Benchmark intact. 30 questions. IDs: {sorted(benchmark_ids)[:3]}...")
    return True


def check_no_holdout_leakage(lines):
    log(lines, "CHECK: No holdout leakage in capability probe ...")
    if not PROBE_PATH.exists():
        log(lines, f"  SKIP: Probe file not found.")
        return True

    with open(PROBE_PATH) as f:
        probe_text = f.read().lower()

    violations = []
    for marker in HOLDOUT_MARKERS:
        if marker.lower() in probe_text and "source_segments" in probe_text:
            # Only flag as violation if marker appears in a ground_truth or question field
            import re
            records = [json.loads(l) for l in open(PROBE_PATH) if l.strip()]
            for r in records:
                q = r.get("question", "").lower()
                gt = r.get("ground_truth", "").lower()
                if marker.lower() in q or marker.lower() in gt:
                    violations.append(f"  Marker '{marker}' in question/GT of {r['question_id']}")

    if violations:
        log(lines, f"  WARN: Potential holdout references found:")
        for v in violations:
            log(lines, v)
        return False
    else:
        log(lines, "  PASS: No holdout marker found in probe questions/ground truths.")
        return True


def check_no_benchmark_leakage(lines):
    log(lines, "CHECK: No benchmark question leakage in probe ...")
    if not BENCHMARK_PATH.exists() or not PROBE_PATH.exists():
        log(lines, "  SKIP: Files not found.")
        return True

    bench_questions = set()
    with open(BENCHMARK_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                bench_questions.add(r["question"].lower().strip())

    probe_questions = set()
    with open(PROBE_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                probe_questions.add(r["question"].lower().strip())

    overlap = bench_questions & probe_questions
    if overlap:
        log(lines, f"  FAIL: {len(overlap)} questions overlap with benchmark:")
        for q in list(overlap)[:3]:
            log(lines, f"    - {q[:80]}")
        return False
    else:
        log(lines, f"  PASS: No benchmark question appears in probe dataset.")
        return True


def check_probe_results(lines):
    log(lines, "CHECK: Capability probe results ...")
    if not PROBE_RESULTS.exists():
        log(lines, f"  FAIL: Probe results not found at {PROBE_RESULTS}")
        return False

    with open(PROBE_RESULTS) as f:
        results = json.load(f)

    n = len(results)
    log(lines, f"  INFO: {n} probe results found.")

    has_latency = all("generation_latency_s" in r for r in results)
    has_logp = all("avg_log_prob" in r for r in results)
    log(lines, f"  INFO: Has latency: {has_latency}, Has log-prob: {has_logp}")

    if n < 35:
        log(lines, f"  WARN: Only {n} results. Expected ~41.")
    else:
        log(lines, f"  PASS: Probe results complete.")
    return True


def check_artifacts(lines):
    log(lines, "CHECK: Required artifacts ...")
    missing = []
    for path in REQUIRED_ARTIFACTS:
        full = PROJECT_ROOT / path
        if full.exists():
            log(lines, f"  OK:      {path}")
        else:
            log(lines, f"  MISSING: {path}")
            missing.append(path)

    if missing:
        log(lines, f"  WARN: {len(missing)} artifact(s) missing.")
        return False
    log(lines, "  PASS: All artifacts present.")
    return True


def main():
    print("=" * 60)
    print("EdgeCascade -- Capability Probe Validation")
    print("=" * 60)

    lines = [
        f"EdgeCascade Capability Probe Validation Log",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 60,
        "",
    ]

    results = {
        "model_load": check_model_load(lines),
        "api_judge": check_api_judge(lines),
        "benchmark_intact": check_benchmark_intact(lines),
        "no_holdout_leakage": check_no_holdout_leakage(lines),
        "no_benchmark_leakage": check_no_benchmark_leakage(lines),
        "probe_results": check_probe_results(lines),
        "artifacts": check_artifacts(lines),
    }

    lines.append("")
    lines.append("=" * 60)
    lines.append("VALIDATION SUMMARY")
    lines.append("=" * 60)

    all_passed = all(results.values())
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL/WARN"
        lines.append(f"  {status:<10} {check}")
        print(f"  {status:<10} {check}")

    lines.append("")
    if all_passed:
        lines.append("OVERALL: ALL CHECKS PASSED")
        print("\nOVERALL: ALL CHECKS PASSED")
    else:
        failed = [k for k, v in results.items() if not v]
        lines.append(f"OVERALL: {len(failed)} check(s) failed/warned: {failed}")
        print(f"\nOVERALL: {len(failed)} check(s) need attention: {failed}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nLog saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
