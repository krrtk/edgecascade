from pathlib import Path
import json
import statistics

import tiktoken


ROOT = Path(__file__).resolve().parent.parent
RAG = ROOT / "data" / "retrieval_final"


# GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def check_file(path):

    records = load_jsonl(path)

    token_counts = []
    problems = []

    required_fields = [
        "chunk_id",
        "domain",
        "source_doc",
        "source_type",
        "title",
        "text"
    ]

    seen_ids = set()

    for item in records:

        # Required metadata
        for field in required_fields:
            if field not in item or not item[field]:
                problems.append(
                    f"{item.get('chunk_id', 'UNKNOWN')}: missing {field}"
                )

        # Duplicate IDs
        chunk_id = item.get("chunk_id")

        if chunk_id in seen_ids:
            problems.append(
                f"{chunk_id}: duplicate chunk_id"
            )

        seen_ids.add(chunk_id)

        # Token count
        tokens = len(enc.encode(item["text"]))
        token_counts.append(tokens)

    print("\n" + "=" * 60)
    print(path.name)
    print("=" * 60)

    print(f"Chunks: {len(records)}")
    print(f"Min tokens: {min(token_counts)}")
    print(f"Max tokens: {max(token_counts)}")
    print(f"Mean tokens: {statistics.mean(token_counts):.1f}")
    print(f"Median tokens: {statistics.median(token_counts):.1f}")

    too_small = sum(t < 100 for t in token_counts)
    too_large = sum(t > 450 for t in token_counts)

    print(f"<100 tokens: {too_small}")
    print(f">450 tokens: {too_large}")

    if problems:
        print("\n❌ Metadata problems:")
        for problem in problems[:20]:
            print(problem)
    else:
        print("\n✅ Metadata valid")

    if too_large:
        print("⚠️ Some chunks exceed 450 tokens")
    else:
        print("✅ No chunks exceed 450 tokens")

    return records


def check_holdout_separation():

    cartosat = RAG / "isro" / "cartosat1_chunks.jsonl"
    resourcesat = RAG / "isro" / "resourcesat2_chunks.jsonl"

    carto_records = load_jsonl(cartosat)
    resource_records = load_jsonl(resourcesat)

    print("\n" + "=" * 60)
    print("HOLDOUT SEPARATION")
    print("=" * 60)

    resource_text = " ".join(
        x["text"].lower()
        for x in resource_records
    )

    forbidden = [
        "cartosat-1",
        "cartosat 1",
        "pan-fore",
        "pan-aft"
    ]

    leaks = [
        word for word in forbidden
        if word in resource_text
    ]

    if leaks:
        print("❌ Cartosat leakage into Resourcesat RAG:")
        print(leaks)
    else:
        print("✅ Cartosat-1 not present in Resourcesat RAG")


def main():

    files = [
        RAG / "isro" / "resourcesat2_chunks.jsonl",
        RAG / "isro" / "cartosat1_chunks.jsonl",
        RAG / "dpdpa" / "dpdpa_chunks.jsonl",
    ]

    for path in files:
        check_file(path)

    check_holdout_separation()

    print("\nRAG QC complete.")


if __name__ == "__main__":
    main()