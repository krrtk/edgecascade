from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent.parent

LORA = ROOT / "data" / "lora"
RETRIEVAL = ROOT / "data" / "retrieval"
EVAL = ROOT / "data" / "evaluation"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_cartosat_leakage():
    path = LORA / "isro_lora_source.json"
    data = load(path)

    forbidden = [
        "cartosat-1",
        "cartosat 1",
        "pan-fore",
        "pan-aft"
    ]

    leaks = []

    for item in data:
        text = item["text"].lower()

        for word in forbidden:
            if word in text:
                leaks.append((item["segment_id"], word))

    print("\nISRO Cartosat-1 leakage:")

    if leaks:
        print("❌ LEAK DETECTED")
        for leak in leaks[:20]:
            print(leak)
    else:
        print("✅ No Cartosat-1 leakage")


def check_dpdpa_leakage():
    paths = [
        LORA / "dpdpa_lora_source.json",
        LORA / "dpdpa_prs_source.json"
    ]

    forbidden_patterns = [
        r"\bsection\s+9\b",
        r"\bsection\s+17\b",
        r"two hundred and fifty crore",
        r"₹\s*250",
        r"250\s*crore"
    ]

    leaks = []

    for path in paths:

        data = load(path)

        for item in data:

            text = item["text"].lower()

            for pattern in forbidden_patterns:

                if re.search(pattern, text):
                    leaks.append(
                        (path.name, item["segment_id"], pattern)
                    )

    print("\nDPDPA holdout leakage:")

    if leaks:
        print("❌ LEAK DETECTED")
        for leak in leaks[:20]:
            print(leak)
    else:
        print("✅ No DPDPA holdout leakage")


def check_empty_segments():

    files = [
        LORA / "isro_lora_source.json",
        LORA / "dpdpa_lora_source.json",
        LORA / "dpdpa_prs_source.json",
        RETRIEVAL / "isro" / "resourcesat2_source.json",
        RETRIEVAL / "dpdpa" / "dpdpa_rag_source.json",
        EVAL / "isro" / "cartosat1_holdout.json",
        EVAL / "dpdpa" / "dpdpa_holdout.json",
    ]

    print("\nEmpty segments:")

    problems = []

    for path in files:

        if not path.exists():
            print(f"❌ Missing: {path}")
            continue

        data = load(path)

        for item in data:

            if not item.get("text", "").strip():
                problems.append(
                    (path.name, item.get("segment_id"))
                )

    if problems:
        print("❌ Empty segments found:")
        for problem in problems:
            print(problem)
    else:
        print("✅ No empty segments")


def check_duplicates():

    files = [
        LORA / "isro_lora_source.json",
        LORA / "dpdpa_lora_source.json",
        LORA / "dpdpa_prs_source.json",
    ]

    print("\nDuplicate LoRA segments:")

    seen = {}
    duplicates = []

    for path in files:

        data = load(path)

        for item in data:

            text = item["text"].strip()

            if text in seen:
                duplicates.append(
                    (path.name, item["segment_id"], seen[text])
                )
            else:
                seen[text] = item["segment_id"]

    if duplicates:
        print("⚠️ Duplicates found:")
        for duplicate in duplicates[:20]:
            print(duplicate)
    else:
        print("✅ No exact duplicate LoRA segments")


def print_statistics():

    print("\nDataset statistics:")

    files = {
        "ISRO LoRA":
            LORA / "isro_lora_source.json",

        "DPDPA LoRA":
            LORA / "dpdpa_lora_source.json",

        "PRS LoRA":
            LORA / "dpdpa_prs_source.json",

        "Resourcesat RAG":
            RETRIEVAL / "isro" / "resourcesat2_source.json",

        "DPDPA RAG":
            RETRIEVAL / "dpdpa" / "dpdpa_rag_source.json",

        "Cartosat Holdout":
            EVAL / "isro" / "cartosat1_holdout.json",

        "DPDPA Holdout":
            EVAL / "dpdpa" / "dpdpa_holdout.json",
    }

    for name, path in files.items():

        if not path.exists():
            print(f"{name}: MISSING")
            continue

        data = load(path)

        total_chars = sum(
            len(item.get("text", ""))
            for item in data
        )

        print(
            f"{name:20} "
            f"segments={len(data):4} "
            f"characters={total_chars:,}"
        )


def main():

    check_cartosat_leakage()
    check_dpdpa_leakage()
    check_empty_segments()
    check_duplicates()
    print_statistics()

    print("\nQC complete.")


if __name__ == "__main__":
    main()