import json
import re
from pathlib import Path
import tiktoken


# ============================================================
# CONFIG
# ============================================================

BASE = Path("data")

RESOURCESAT_SOURCE = BASE / "retrieval/isro/resourcesat2_source.json"
CARTOSAT_SOURCE = BASE / "evaluation/isro/cartosat1_holdout.json"
DPDPA_SOURCE = BASE / "retrieval/dpdpa/dpdpa_rag_source.json"

OUTPUT_DIR = BASE / "retrieval_final"

TARGET_TOKENS = 300
MAX_TOKENS = 445

enc = tiktoken.get_encoding("gpt2")
print("🔥 NEW TOKEN-AWARE BUILD_RAG.PY IS RUNNING 🔥")

# ============================================================
# HELPERS
# ============================================================

def tokenize(text):
    return enc.encode(text)


def detokenize(tokens):
    return enc.decode(tokens)


def contains_cartosat(text):
    """
    Detect Cartosat-1 references in Resourcesat material.
    """
    patterns = [
        r"\bcartosat[\s\-]?1\b",
        r"\bcartosat[\s\-]?i\b",
    ]

    text_lower = text.lower()

    return any(re.search(pattern, text_lower) for pattern in patterns)


def split_into_paragraphs(text):
    """
    Preserve paragraph boundaries where possible.
    """
    paragraphs = re.split(r"\n\s*\n+", text)

    return [
        p.strip()
        for p in paragraphs
        if p.strip()
    ]


def chunk_text(text, target_tokens=TARGET_TOKENS, max_tokens=MAX_TOKENS):
    """
    Token-aware chunking.

    - Prefer paragraph boundaries.
    - Aim for ~300 tokens.
    - Never allow a chunk above 450 tokens.
    - If a paragraph itself is too large, hard-split it.
    """

    paragraphs = split_into_paragraphs(text)

    chunks = []
    current_tokens = []

    for paragraph in paragraphs:

        paragraph_tokens = tokenize(paragraph)

        # ----------------------------------------------------
        # Paragraph itself is larger than maximum.
        # Hard split it.
        # ----------------------------------------------------
        if len(paragraph_tokens) > max_tokens:

            # Flush current chunk first.
            if current_tokens:
                chunks.append(detokenize(current_tokens))
                current_tokens = []

            for i in range(0, len(paragraph_tokens), max_tokens):
                piece = paragraph_tokens[i:i + max_tokens]
                chunks.append(detokenize(piece))

            continue

        # ----------------------------------------------------
        # Adding paragraph would exceed target.
        # Finish current chunk.
        # ----------------------------------------------------
        if current_tokens and len(current_tokens) + len(paragraph_tokens) > target_tokens:
            chunks.append(detokenize(current_tokens))
            current_tokens = []

        current_tokens.extend(paragraph_tokens)

    # Final chunk
    if current_tokens:
        chunks.append(detokenize(current_tokens))

    return [c.strip() for c in chunks if c.strip()]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# GENERIC SOURCE → CHUNKS
# ============================================================

def build_chunks(source_records, source_name):
    output = []

    for record in source_records:

        text = record.get("text", "").strip()

        if not text:
            continue

        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):

            new_record = {
                "chunk_id": f"{source_name}_{len(output):05d}",
                "domain": record.get("domain") or ("isro" if "sat" in source_name else "dpdpa"),
                "source_doc": record.get("source_doc") or record.get("source") or "unknown",
                "source_type": record.get("source_type") or ("tech_handbook" if "sat" in source_name else "legal_act"),
                "title": record.get("title") or record.get("heading") or "Unknown",
                "section_ref": record.get("section_ref") or record.get("heading") or "Unknown",
                "page_num": record.get("page_num"),
                "version_year": record.get("version_year"),
                "mission": record.get("mission"),
                "text": chunk,
            }

            output.append(new_record)

    return output


# ============================================================
# RESOURCESAT
# ============================================================

def build_resourcesat():

    source = load_json(RESOURCESAT_SOURCE)

    chunks = build_chunks(
        source,
        "resourcesat2"
    )

    # --------------------------------------------------------
    # Remove Cartosat leakage.
    # --------------------------------------------------------

    before = len(chunks)

    chunks = [
        chunk
        for chunk in chunks
        if not contains_cartosat(chunk["text"])
    ]

    removed = before - len(chunks)

    print(f"Resourcesat chunks removed for Cartosat leakage: {removed}")

    return chunks


# ============================================================
# CARTOSAT
# ============================================================

def build_cartosat():

    source = load_json(CARTOSAT_SOURCE)

    return build_chunks(
        source,
        "cartosat1"
    )


# ============================================================
# DPDPA
# ============================================================

def build_dpdpa():

    source = load_json(DPDPA_SOURCE)

    return build_chunks(
        source,
        "dpdpa"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    (OUTPUT_DIR / "isro").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "dpdpa").mkdir(parents=True, exist_ok=True)

    resourcesat = build_resourcesat()
    cartosat = build_cartosat()
    dpdpa = build_dpdpa()

    write_jsonl(
        OUTPUT_DIR / "isro" / "resourcesat2_chunks.jsonl",
        resourcesat
    )

    write_jsonl(
        OUTPUT_DIR / "isro" / "cartosat1_chunks.jsonl",
        cartosat
    )

    write_jsonl(
        OUTPUT_DIR / "dpdpa" / "dpdpa_chunks.jsonl",
        dpdpa
    )

    print()
    print(f"Resourcesat RAG chunks: {len(resourcesat)}")
    print(f"Cartosat-1 RAG chunks: {len(cartosat)}")
    print(f"DPDPA RAG chunks: {len(dpdpa)}")
    print()
    print("RAG corpus construction complete.")


if __name__ == "__main__":
    main()