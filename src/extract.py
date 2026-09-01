"""
extract.py
----------
Extracts text from academic PDFs, page by page, correcting for the
two-column layout that trips up naive text extraction.

Why this matters: arXiv papers are laid out in two columns. A plain
page.get_text() call can interleave lines from both columns, scrambling
sentences. Instead we pull text as positioned "blocks" (each with
x0,y0,x1,y1 coordinates) and sort them left-column-top-to-bottom, then
right-column-top-to-bottom.

Known limitation: full-width titles/figures that straddle the column
midline can still get misordered. Acceptable for a first pass.
"""

import re
import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

def _is_likely_toc_page(page_text: str, page_number: int) -> bool:
    """
    A Table-of-Contents page has near-zero substantive content (just
    section titles + page numbers), so it's pure noise for retrieval.

    Deliberately conservative: only flag a page that literally opens
    with "Contents"/"Table of Contents", AND only within the first few
    pages of the paper. An earlier, broader version of this heuristic
    (flagging pages with a high fraction of bare-number lines) also
    caught results tables and reference lists -- both have lots of
    short numeric tokens too -- which would have silently deleted real
    content. Under-catching a rare multi-page ToC continuation is a
    much safer failure mode than over-catching real content.
    """
    if page_number > 6:
        return False
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    if not lines:
        return False
    return lines[0].lower() in ("contents", "table of contents")


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str
    is_likely_toc: bool = False


@dataclass
class PaperText:
    source_file: str
    pages: List[PageText] = field(default_factory=list)


def _sort_blocks_by_reading_order(blocks, page_width: float):
    """
    Classify each block as left-column (0) or right-column (1) by its
    horizontal center, then sort by (column, vertical position).
    """
    midpoint = page_width / 2

    def block_key(b):
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        center_x = (x0 + x1) / 2
        column = 0 if center_x < midpoint else 1
        return (column, y0)

    return sorted(blocks, key=block_key)


def extract_paper(pdf_path: str) -> PaperText:
    doc = fitz.open(pdf_path)
    paper = PaperText(source_file=Path(pdf_path).name)

    for page_index in range(doc.page_count):
        page = doc[page_index]
        blocks = page.get_text("blocks")
        # block_type == 0 means a text block (1 means image); drop empties
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
        ordered = _sort_blocks_by_reading_order(text_blocks, page.rect.width)
        page_text = "\n".join(b[4].strip() for b in ordered)
        paper.pages.append(PageText(
            page_number=page_index + 1,
            text=page_text,
            is_likely_toc=_is_likely_toc_page(page_text, page_index + 1),
        ))

    doc.close()
    return paper


if __name__ == "__main__":
    import glob
    for pdf_path in sorted(glob.glob("data/papers/*.pdf")):
        paper = extract_paper(pdf_path)
        total_chars = sum(len(p.text) for p in paper.pages)
        print(f"{paper.source_file}: {len(paper.pages)} pages, {total_chars} chars extracted")
