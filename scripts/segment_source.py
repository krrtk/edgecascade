from pathlib import Path
import re
import json

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SEGMENTS = PROCESSED / "segments"

ISRO = PROCESSED / "isro"
DPDPA = PROCESSED / "dpdpa"

SEGMENTS.mkdir(parents=True, exist_ok=True)
(SEGMENTS / "isro").mkdir(exist_ok=True)
(SEGMENTS / "dpdpa").mkdir(exist_ok=True)


def read(path):
    return path.read_text(encoding="utf-8")


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# ---------------------------------------------------------
# Generic heading-based segmentation
# ---------------------------------------------------------

def segment_by_heading(text, pattern):
    lines = text.splitlines()

    matches = []

    for i, line in enumerate(lines):
        line = line.strip()

        if re.match(pattern, line, re.IGNORECASE):
            matches.append(i)

    segments = []

    for idx, start in enumerate(matches):
        end = matches[idx + 1] if idx + 1 < len(matches) else len(lines)

        content = "\n".join(lines[start:end]).strip()

        if content:
            segments.append(content)

    return segments


# ---------------------------------------------------------
# DPDPA
# ---------------------------------------------------------

def segment_dpdpa():

    path = DPDPA / "dpdpa_gazette_clean.txt"
    text = read(path)

    pattern = (
        r"^(?:section\s+\d+[\.:]?.*|"
        r"\d+\.\s+.*|"
        r"schedule)$"
    )

    segments = segment_by_heading(text, pattern)

    output = []

    for i, segment in enumerate(segments):
        first_line = segment.splitlines()[0].strip()

        output.append({
            "segment_id": f"dpdpa_{i:03d}",
            "source": "dpdpa_2023_gazette_official.pdf",
            "heading": first_line,
            "text": segment
        })

    write_json(
        SEGMENTS / "dpdpa" / "dpdpa_segments.json",
        output
    )

    print(f"DPDPA: {len(output)} segments")


# ---------------------------------------------------------
# IIRS
# ---------------------------------------------------------

def segment_iirs():

    path = ISRO / "iirs_remote_sensing_clean.txt"
    text = read(path)

    pattern = r"^(?:chapter\s+\d+.*|\d+\.\s+.*)$"

    segments = segment_by_heading(text, pattern)

    output = []

    for i, segment in enumerate(segments):
        first_line = segment.splitlines()[0].strip()

        output.append({
            "segment_id": f"iirs_{i:03d}",
            "source": "isro_iirs_remote_sensing_ebook_2023.pdf",
            "heading": first_line,
            "text": segment
        })

    write_json(
        SEGMENTS / "isro" / "iirs_segments.json",
        output
    )

    print(f"IIRS: {len(output)} segments")


# ---------------------------------------------------------
# Resourcesat-2
# ---------------------------------------------------------

def segment_resourcesat():

    path = ISRO / "resourcesat2_clean.txt"
    text = read(path)

    pattern = r"^(?:chapter\s+\d+.*|\d+(?:\.\d+)*[\.:]?\s+.*)$"

    segments = segment_by_heading(text, pattern)

    output = []

    for i, segment in enumerate(segments):
        first_line = segment.splitlines()[0].strip()

        output.append({
            "segment_id": f"resourcesat2_{i:03d}",
            "source": "isro_nrsc_resourcesat2_handbook.pdf",
            "heading": first_line,
            "mission": "resourcesat-2",
            "text": segment
        })

    write_json(
        SEGMENTS / "isro" / "resourcesat2_segments.json",
        output
    )

    print(f"Resourcesat-2: {len(output)} segments")


# ---------------------------------------------------------
# Cartosat-1
# ---------------------------------------------------------

def segment_cartosat():

    path = ISRO / "cartosat1_clean.txt"
    text = read(path)

    # We intentionally keep the whole document isolated.
    output = [{
        "segment_id": "cartosat1_full",
        "source": "isro_nrsc_cartosat1_handbook.pdf",
        "mission": "cartosat-1",
        "holdout": True,
        "text": text
    }]

    write_json(
        SEGMENTS / "isro" / "cartosat1_holdout.json",
        output
    )

    print("Cartosat-1: isolated as complete holdout")


# ---------------------------------------------------------
# PRS
# ---------------------------------------------------------

def segment_prs():

    path = DPDPA / "prs_summary_clean.txt"
    text = read(path)

    # Keep conceptual PRS material together for now.
    output = [{
        "segment_id": "prs_dpdpa_full",
        "source": "dpdpa_2023_prs_summary.pdf",
        "text": text
    }]

    write_json(
        SEGMENTS / "dpdpa" / "prs_segments.json",
        output
    )

    print("PRS: 1 conceptual segment")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    segment_iirs()
    segment_resourcesat()
    segment_cartosat()

    segment_dpdpa()
    segment_prs()

    print("\nSegmentation complete.")