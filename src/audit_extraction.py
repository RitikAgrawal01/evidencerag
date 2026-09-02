"""
audit_extraction.py
--------------------
Produces reports/extraction_audit.txt: for every paper, the detected
layout ("single"/"double") of every page, plus a readable text sample
from a few sampled BODY pages so a human can eyeball whether the final
reading order actually makes sense.

This exists because extraction correctness cannot be unit-tested into
existence -- there is no ground truth for "correct reading order" other
than a person reading the paper. Run this once after any change to
extract.py, read the samples against the real PDF, and only then trust
the corpus built on top of it.

Usage:  python src/audit_extraction.py
Output: reports/extraction_audit.txt
"""

import glob
from pathlib import Path

from extract import extract_paper

# Pages to sample per paper for the human-readable text dump. 0-indexed
# would be confusing next to 1-indexed page numbers elsewhere in this
# project, so these are 1-indexed like everything the reader sees.
SAMPLE_PAGES_1_INDEXED = [3, 8, 15]
SAMPLE_CHARS = 500


def audit_paper(pdf_path: str, out) -> None:
    paper = extract_paper(pdf_path)
    out.write(f"=== {paper.source_file} ({len(paper.pages)} pages) ===\n")

    # 1. Per-page layout call -- lets you spot-check the detector itself,
    #    not just its downstream effect on text order.
    from extract import _detect_layout
    import pymupdf

    doc = pymupdf.open(pdf_path)
    layouts = []
    for i, page_obj in enumerate(doc):
        blocks = [b for b in page_obj.get_text("blocks") if b[6] == 0 and b[4].strip()]
        if len(blocks) < 2:
            layouts.append("?")
            continue
        layouts.append(_detect_layout(blocks, page_obj.rect.width)[0].upper())  # "S" or "D"
    doc.close()
    out.write("  layout per page: " + "".join(layouts) + "\n")
    single_count = layouts.count("S")
    double_count = layouts.count("D")
    out.write(f"  -> {single_count} single-column pages, {double_count} two-column pages\n\n")

    # 2. Readable text samples from a few body pages, in final reading order.
    for page_no in SAMPLE_PAGES_1_INDEXED:
        if page_no > len(paper.pages):
            continue
        page = paper.pages[page_no - 1]
        if page.is_likely_toc:
            out.write(f"  -- page {page_no}: flagged as ToC, skipping sample --\n\n")
            continue
        sample = page.text[:SAMPLE_CHARS].replace("\n", " \\n ")
        out.write(f"  -- page {page_no} sample (first {SAMPLE_CHARS} chars, \\n shown explicitly) --\n")
        out.write(f"  {sample}\n\n")

    out.write("\n")


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "extraction_audit.txt"

    with open(out_path, "w", encoding="utf-8") as out:
        for pdf_path in sorted(glob.glob("data/papers/*.pdf")):
            audit_paper(pdf_path, out)

    print(f"Wrote {out_path}")
    print("Now open each PDF and read the sampled pages against these dumps.")


if __name__ == "__main__":
    main()
