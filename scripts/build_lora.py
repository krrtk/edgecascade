import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LORA_DIR = ROOT / "data" / "lora"
OUTPUT_DIR = ROOT / "data" / "training" / "lora"

SOURCES = [
    LORA_DIR / "isro_lora_source.json",
    LORA_DIR / "dpdpa_lora_source.json",
    LORA_DIR / "dpdpa_prs_source.json"
]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "lora_sft.jsonl"
    
    total = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for source in SOURCES:
            if not source.exists():
                print(f"Warning: {source} does not exist.")
                continue
            
            data = load_json(source)
            for item in data:
                text = item.get("text", "").strip()
                if text:
                    record = {"text": text}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                    
    print(f"LoRA SFT dataset built at {out_file} with {total} segments.")

if __name__ == "__main__":
    main()
