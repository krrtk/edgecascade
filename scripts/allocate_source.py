from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent.parent

SEGMENTS = ROOT / "data" / "processed" / "segments"
LORA = ROOT / "data" / "lora"
RETRIEVAL = ROOT / "data" / "retrieval"
EVAL = ROOT / "data" / "evaluation"

for path in [LORA, RETRIEVAL, EVAL]:
    path.mkdir(parents=True, exist_ok=True)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def allocate_iirs():

    data = load(
        SEGMENTS / "isro" / "iirs_segments.json"
    )

    lora = []
    exclude = []

    for item in data:

        text = item["text"].lower()

        # Protect Cartosat-1 holdout from accidental leakage
        if any(x in text for x in [
            "cartosat-1",
            "cartosat 1",
            "pan-fore",
            "pan-aft"
        ]):
            exclude.append(item)
        else:
            lora.append(item)

    save(LORA / "isro_lora_source.json", lora)

    print(f"IIRS → LoRA: {len(lora)}")
    print(f"IIRS → Excluded: {len(exclude)}")


def allocate_resourcesat():

    data = load(
        SEGMENTS / "isro" / "resourcesat2_segments.json"
    )

    # Resourcesat specifications belong to RAG.
    for item in data:
        item["destination"] = "rag"

    save(
        RETRIEVAL / "isro" / "resourcesat2_source.json",
        data
    )

    print(f"Resourcesat-2 → RAG: {len(data)}")


def allocate_cartosat():

    data = load(
        SEGMENTS / "isro" / "cartosat1_holdout.json"
    )

    # Entire mission is holdout.
    for item in data:
        item["destination"] = "rag_holdout"

    save(
        EVAL / "isro" / "cartosat1_holdout.json",
        data
    )

    print("Cartosat-1 → HOLDOUT")


def allocate_dpdpa():

    data = load(
        SEGMENTS / "dpdpa" / "dpdpa_segments.json"
    )

    lora = []
    rag = []
    holdout = []

    for item in data:

        text = item["text"].lower()

        # Critical holdouts
        if (
            re.search(r"\bsection\s+9\b", text)
            or re.search(r"\bsection\s+17\b", text)
            or "schedule" in text
        ):
            item["destination"] = "holdout"
            holdout.append(item)

        # Section 2 = definitions → LoRA
        elif re.search(r"\bsection\s+2\b", text):
            item["destination"] = "lora"
            lora.append(item)

        # Everything else → RAG
        else:
            item["destination"] = "rag"
            rag.append(item)

    save(
        LORA / "dpdpa_lora_source.json",
        lora
    )

    save(
        RETRIEVAL / "dpdpa" / "dpdpa_rag_source.json",
        rag
    )

    save(
        EVAL / "dpdpa" / "dpdpa_holdout.json",
        holdout
    )

    print(f"DPDPA → LoRA: {len(lora)}")
    print(f"DPDPA → RAG: {len(rag)}")
    print(f"DPDPA → HOLDOUT: {len(holdout)}")


def allocate_prs():

    data = load(
        SEGMENTS / "dpdpa" / "prs_segments.json"
    )

    cleaned = []

    for item in data:

        text = item["text"]

        # Remove exact penalty figures from the conceptual LoRA source.
        text = re.sub(
            r".{0,200}(?:250\s*crore|two hundred and fifty crore).{0,200}",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        item["text"] = text
        item["destination"] = "lora"

        cleaned.append(item)

    save(
        LORA / "dpdpa_prs_source.json",
        cleaned
    )

    print(f"PRS → LoRA: {len(cleaned)}")


def main():

    (RETRIEVAL / "isro").mkdir(exist_ok=True)
    (RETRIEVAL / "dpdpa").mkdir(exist_ok=True)

    allocate_iirs()
    allocate_resourcesat()
    allocate_cartosat()

    allocate_dpdpa()
    allocate_prs()

    print("\nAllocation complete.")


if __name__ == "__main__":
    main()