from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent

PROCESSED = ROOT / "data" / "processed"

ISRO = PROCESSED / "isro"
DPDPA = PROCESSED / "dpdpa"


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def clean_text(text: str):

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove our page markers
    text = re.sub(
        r"===== PAGE \d+ =====",
        "",
        text
    )

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip completely empty lines for now
        if not line:
            cleaned_lines.append("")
            continue

        # Collapse repeated spaces/tabs
        line = re.sub(r"[ \t]+", " ", line)

        cleaned_lines.append(line)

    # Reconstruct while preserving line boundaries
    text = "\n".join(cleaned_lines)

    # Fix words broken across lines.
    # IMPORTANT:
    # Do this only when a word is explicitly hyphenated.
    text = re.sub(
        r"(\w)-\n(\w)",
        r"\1-\2",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def process_file(input_path: Path, output_path: Path):

    print(f"Cleaning: {input_path.name}")

    text = read_text(input_path)
    cleaned = clean_text(text)

    write_text(output_path, cleaned)

    print(f"Saved: {output_path}")


def main():

    # ---------------- ISRO ----------------

    process_file(
        ISRO / "iirs_remote_sensing_raw.txt",
        ISRO / "iirs_remote_sensing_clean.txt"
    )

    process_file(
        ISRO / "resourcesat2_raw.txt",
        ISRO / "resourcesat2_clean.txt"
    )

    process_file(
        ISRO / "cartosat1_raw.txt",
        ISRO / "cartosat1_clean.txt"
    )

    # ---------------- DPDPA ----------------

    process_file(
        DPDPA / "dpdpa_gazette_raw.txt",
        DPDPA / "dpdpa_gazette_clean.txt"
    )

    process_file(
        DPDPA / "prs_summary_raw.txt",
        DPDPA / "prs_summary_clean.txt"
    )


if __name__ == "__main__":
    main()