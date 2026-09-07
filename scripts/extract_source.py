from pathlib import Path
import csv
import fitz  # PyMuPDF


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

ISRO_RAW = RAW / "isro"
DPDPA_RAW = RAW / "dpdpa"

ISRO_PROCESSED = PROCESSED / "isro"
DPDPA_PROCESSED = PROCESSED / "dpdpa"


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

ISRO_PROCESSED.mkdir(parents=True, exist_ok=True)
DPDPA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------

def extract_pdf(pdf_path: Path, output_path: Path):
    print(f"Extracting: {pdf_path.name}")

    doc = fitz.open(pdf_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")

            f.write(f"\n\n===== PAGE {page_number} =====\n\n")
            f.write(text)

    doc.close()

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# WMO CSV filtering
# ---------------------------------------------------------

def process_wmo_csv(csv_path: Path, output_path: Path):
    print(f"Processing: {csv_path.name}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        rows = []

        for row in reader:
            agencies = row.get("Agencies", "")

            # OSCAR export may contain multiple agencies
            # separated by newlines.
            if "ISRO" in agencies.upper():
                rows.append(row)

    if not rows:
        print("WARNING: No rows found containing ISRO in Agencies")
        return

    fieldnames = rows[0].keys()

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} ISRO records to: {output_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # ---------------- ISRO ----------------

    extract_pdf(
        ISRO_RAW / "isro_iirs_remote_sensing_ebook_2023.pdf",
        ISRO_PROCESSED / "iirs_remote_sensing_raw.txt"
    )

    extract_pdf(
        ISRO_RAW / "isro_nrsc_resourcesat2_handbook.pdf",
        ISRO_PROCESSED / "resourcesat2_raw.txt"
    )

    extract_pdf(
        ISRO_RAW / "isro_nrsc_cartosat1_handbook.pdf",
        ISRO_PROCESSED / "cartosat1_raw.txt"
    )

    process_wmo_csv(
        ISRO_RAW / "isro_wmo_oscar_satellites.csv",
        ISRO_PROCESSED / "oscar_isro_normalized.csv"
    )

    # ---------------- DPDPA ----------------

    extract_pdf(
        DPDPA_RAW / "dpdpa_2023_gazette_official.pdf",
        DPDPA_PROCESSED / "dpdpa_gazette_raw.txt"
    )

    extract_pdf(
        DPDPA_RAW / "dpdpa_2023_prs_summary.pdf",
        DPDPA_PROCESSED / "prs_summary_raw.txt"
    )

    print("\nExtraction complete.")


if __name__ == "__main__":
    main()