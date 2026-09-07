import json
from pathlib import Path

try:
    import tiktoken
    has_tiktoken = True
except ImportError:
    has_tiktoken = False

ROOT = Path(__file__).resolve().parent.parent

# Files
LORA_FILE = ROOT / "data" / "training" / "lora" / "lora_sft.jsonl"
ISRO_SOURCE = ROOT / "data" / "lora" / "isro_lora_source.json"
DPDPA_SOURCE = ROOT / "data" / "lora" / "dpdpa_lora_source.json"
PRS_SOURCE = ROOT / "data" / "lora" / "dpdpa_prs_source.json"

ISRO_HOLDOUT = ROOT / "data" / "evaluation" / "isro" / "cartosat1_holdout.json"
DPDPA_HOLDOUT = ROOT / "data" / "evaluation" / "dpdpa" / "dpdpa_holdout.json"
EVAL_QUESTIONS = ROOT / "data" / "evaluation" / "evaluation_questions.jsonl"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    if not LORA_FILE.exists():
        print(f"Error: {LORA_FILE} does not exist.")
        return
        
    lora_records = load_jsonl(LORA_FILE)
    
    # Load source texts
    isro_texts = [x["text"].strip() for x in load_json(ISRO_SOURCE) if x.get("text", "").strip()]
    dpdpa_texts = [x["text"].strip() for x in load_json(DPDPA_SOURCE) if x.get("text", "").strip()]
    prs_texts = [x["text"].strip() for x in load_json(PRS_SOURCE) if x.get("text", "").strip()]
    
    # Load holdout texts
    cartosat_holdout_texts = [x["text"].strip() for x in load_json(ISRO_HOLDOUT)]
    dpdpa_holdout_texts = [x["text"].strip() for x in load_json(DPDPA_HOLDOUT)]
    
    # Load evaluation questions
    eval_q_records = load_jsonl(EVAL_QUESTIONS)
    eval_strings = []
    for q in eval_q_records:
        eval_strings.append(q["question"])
        eval_strings.append(q["ground_truth"])
    
    problems = []
    
    if has_tiktoken:
        enc = tiktoken.get_encoding("gpt2")
        
    # Check basic schema
    for i, item in enumerate(lora_records):
        if "text" not in item:
            problems.append(f"Line {i+1}: missing 'text' key")
        elif not isinstance(item["text"], str):
            problems.append(f"Line {i+1}: 'text' is not a string")
        elif not item["text"].strip():
            problems.append(f"Line {i+1}: 'text' is empty")

    # Match each lora record to a source
    isro_count = 0
    dpdpa_count = 0
    prs_count = 0
    
    isro_tokens = 0
    dpdpa_tokens = 0
    prs_tokens = 0
    
    lora_texts = [item.get("text", "") for item in lora_records]
    
    for text in lora_texts:
        matched = False
        toks = len(enc.encode(text)) if has_tiktoken else 0
        if text in isro_texts:
            isro_count += 1
            isro_tokens += toks
            matched = True
        elif text in dpdpa_texts:
            dpdpa_count += 1
            dpdpa_tokens += toks
            matched = True
        elif text in prs_texts:
            prs_count += 1
            prs_tokens += toks
            matched = True
            
        if not matched:
            problems.append("Found segment in LoRA dataset that doesn't match any source.")

    # Leakage checks
    # 1. Cartosat holdout in ISRO LoRA
    # In qc_rag.py, leakage was checked via forbidden words. We can do both words and direct string match.
    forbidden_cartosat = ["cartosat-1", "cartosat 1", "pan-fore", "pan-aft"]
    isro_lora_combined = " ".join([text for text in lora_texts if text in isro_texts]).lower()
    leaks = [word for word in forbidden_cartosat if word in isro_lora_combined]
    if leaks:
        problems.append(f"❌ Cartosat leakage into ISRO LoRA: {leaks}")

    # Also check if any full holdout segment string made it in
    for ht in cartosat_holdout_texts:
        if ht in lora_texts:
            problems.append("Cartosat-1 holdout segment found exactly in LoRA dataset!")
            
    # 2. DPDPA holdout in DPDPA/PRS LoRA
    dpdpa_lora_combined = " ".join([text for text in lora_texts if text in dpdpa_texts or text in prs_texts])
    for ht in dpdpa_holdout_texts:
        if ht in dpdpa_lora_combined:
            problems.append("DPDPA holdout segment found exactly in DPDPA/PRS LoRA dataset!")

    # 3. No evaluation question or ground truth answer copied
    # Make sure eval questions aren't just exact substrings
    lora_combined_all = " ".join(lora_texts)
    for es in eval_strings:
        if es in lora_combined_all:
            problems.append(f"Evaluation question or answer leaked into LoRA dataset! '{es}'")

    print("\n" + "=" * 60)
    print("LoRA QC RESULTS")
    print("=" * 60)
    
    print("Allocation Verification:")
    print(f"- ISRO LoRA: {isro_count} segments (Tokens: {isro_tokens})")
    print(f"- DPDPA LoRA: {dpdpa_count} segment (Tokens: {dpdpa_tokens})")
    print(f"- PRS LoRA: {prs_count} segment (Tokens: {prs_tokens})")
    print(f"- Total: {len(lora_records)} segments (Tokens: {isro_tokens + dpdpa_tokens + prs_tokens})")
    
    if isro_count == 166 and dpdpa_count == 1 and prs_count == 1 and len(lora_records) == 168:
        print("\n✅ The LoRA dataset contains exactly the intended allocation")
    else:
        print("\n⚠️ The LoRA dataset allocation does NOT match expected counts!")

    if problems:
        print("\n❌ Problems found:")
        for p in problems[:20]:
            print(p)
    else:
        print("✅ No Cartosat-1 holdout content appears in the ISRO LoRA dataset")
        print("✅ No DPDPA holdout content appears in the DPDPA/PRS LoRA dataset")
        print("✅ No evaluation question or ground-truth answer has been copied into the LoRA dataset")
        print("✅ All JSONL schemas match `{\"text\": \"...\"}`")
        print("✅ No empty lines or text found")

if __name__ == "__main__":
    main()
