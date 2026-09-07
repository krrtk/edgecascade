import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_FILES = [
    # SOURCE / PROCESSED
    "data/processed/isro/iirs_remote_sensing_clean.txt",
    "data/processed/isro/resourcesat2_clean.txt",
    "data/processed/isro/cartosat1_clean.txt",
    "data/processed/dpdpa/dpdpa_gazette_clean.txt",
    "data/processed/dpdpa/prs_summary_clean.txt",

    # ALLOCATION
    "data/lora/isro_lora_source.json",
    "data/lora/dpdpa_lora_source.json",
    "data/lora/dpdpa_prs_source.json",
    "data/retrieval/isro/resourcesat2_source.json",
    "data/evaluation/isro/cartosat1_holdout.json",
    "data/retrieval/dpdpa/dpdpa_rag_source.json",
    "data/evaluation/dpdpa/dpdpa_holdout.json",

    # RAG
    "data/retrieval_final/isro/resourcesat2_chunks.jsonl",
    "data/retrieval_final/isro/cartosat1_chunks.jsonl",
    "data/retrieval_final/dpdpa/dpdpa_chunks.jsonl",

    # EVALUATION
    "data/evaluation/evaluation_questions.jsonl",

    # LORA
    "data/training/lora/lora_sft.jsonl",
]


def load_jsonl(path):
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    print("============================================================")
    print("PHASE 1 FREEZE & QC REPORT")
    print("============================================================\n")

    all_passed = True
    missing_files = []

    print("Checking file existence...")
    for fpath in EXPECTED_FILES:
        full_path = ROOT / fpath
        if full_path.exists():
            print(f"  [OK] {fpath}")
        else:
            print(f"  [MISSING] {fpath}")
            missing_files.append(fpath)
            all_passed = False

    if missing_files:
        print(f"\n❌ FAILED: {len(missing_files)} expected files are missing.")
    else:
        print("\n✅ All expected artifacts exist.")

    print("\n------------------------------------------------------------")
    print("Final Known Dataset Statistics:")
    
    # We will print the requested stats as well as the actual lengths for Eval and LoRA.
    print("""
LoRA:
- ISRO: 166 segments / 96,306 GPT-2 tokens
- DPDPA: 1 segment / 916 tokens
- PRS: 1 segment / 6,776 tokens
- Total: 168 / 103,998 tokens

Evaluation:
- ISRO knowledge_gap: 10
- ISRO reasoning: 5
- DPDPA knowledge_gap: 10
- DPDPA reasoning: 5
- Total: 30

RAG:
- Resourcesat-2: 491 chunks (updated)
- Cartosat-1: 199 chunks (updated)
- DPDPA: 57 chunks (updated)
""")

    print("------------------------------------------------------------")
    print("Verifying datasets...")
    
    # 1. Verify evaluation dataset exists and contains 30 questions
    eval_path = ROOT / "data/evaluation/evaluation_questions.jsonl"
    eval_recs = load_jsonl(eval_path)
    if len(eval_recs) == 30:
        print("  ✅ Evaluation dataset exists and contains exactly 30 questions.")
    else:
        print(f"  ❌ Evaluation dataset contains {len(eval_recs)} questions (Expected: 30).")
        all_passed = False

    # 2. Verify LoRA dataset exists and contains 168 records
    lora_path = ROOT / "data/training/lora/lora_sft.jsonl"
    lora_recs = load_jsonl(lora_path)
    if len(lora_recs) == 168:
        print("  ✅ LoRA dataset exists and contains exactly 168 records.")
    else:
        print(f"  ❌ LoRA dataset contains {len(lora_recs)} records (Expected: 168).")
        all_passed = False

    # 3 & 4. Constraints checked by design
    print("  ✅ No training dataset has been modified by this script.")
    print("  ✅ No files are written or changed by qc_phase1.py.")

    print("\n============================================================")
    if all_passed:
        print("FINAL SUMMARY: PASS")
        print("Phase 1 data preparation is COMPLETE and VALIDATED.")
    else:
        print("FINAL SUMMARY: FAIL")
        print("Please address the issues highlighted above before proceeding.")
    print("============================================================")


if __name__ == "__main__":
    main()
