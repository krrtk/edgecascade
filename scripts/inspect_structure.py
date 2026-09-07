from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

ISRO = PROCESSED / "isro"
DPDPA = PROCESSED / "dpdpa"


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def find_headings(text: str):
    """
    Look for lines that appear to represent structural headings.
    This is inspection only — nothing is modified.
    """

    patterns = [
        # Chapter 1 / CHAPTER 1
        r"^\s*chapter\s+\d+.*$",

        # Section 1. / Section 1:
        r"^\s*section\s+\d+[.:]?.*$",

        # 1. Introduction
        r"^\s*\d+\.\s+[A-Z][A-Za-z0-9 ,&()\-]{3,}$",

        # 1.1 Something
        r"^\s*\d+\.\d+\s+.*$",

        # SCHEDULE
        r"^\s*schedule\s*$",
    ]

    matches = []

    for line_number, line in enumerate(text.splitlines(), start=1):

        stripped = line.strip()

        if not stripped:
            continue

        for pattern in patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                matches.append((line_number, stripped))
                break

    return matches


def inspect_file(path: Path):
    print("\n" + "=" * 80)
    print(path.name)
    print("=" * 80)

    text = read_text(path)
    headings = find_headings(text)

    print(f"Detected {len(headings)} possible structural headings.\n")

    for line_number, heading in headings[:150]:
        print(f"{line_number:>6}: {heading}")

    if len(headings) > 150:
        print(f"\n... {len(headings) - 150} additional matches omitted.")


def main():

    # ISRO
    inspect_file(ISRO / "iirs_remote_sensing_clean.txt")
    inspect_file(ISRO / "resourcesat2_clean.txt")
    inspect_file(ISRO / "cartosat1_clean.txt")

    # DPDPA
    inspect_file(DPDPA / "dpdpa_gazette_clean.txt")
    inspect_file(DPDPA / "prs_summary_clean.txt")


if __name__ == "__main__":
    main()