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

CORRECTNESS FIX (was a bug in the first version of this file): not every
paper in this corpus is two-column. Measured directly against the PDFs --
sam3, conformal_prediction and depth_anything_3 are single-column;
sages_cvs_challenge, surgicalsam and murali_latent_graph are two-column.
Sorting a single-column page with the two-column rule scrambles it: every
block spans nearly the full page width, so its horizontal centre sits
right on the midpoint, and sub-pixel width differences flip blocks
between the "left" and "right" bucket essentially at random. On sam3
this reordered ~68% of body blocks on the pages checked.

Fix: detect layout per PAGE (not per paper -- a paper's own front matter
or a wide table can differ from its body), then sort accordingly:
  - single-column page  -> sort by y0 alone (pure top-to-bottom)
  - two-column page     -> sort column-then-row, but full-width blocks
                           (headers, wide figure captions, wide tables)
                           are pulled out first and re-inserted as page-
                           width "bands" that split the column flow at
                           their vertical position, instead of being
                           forced into an arbitrary left/right bucket.
"""

import re
import pymupdf  # PyMuPDF
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


def _is_full_width(block, page_width: float, threshold: float = 0.65) -> bool:
    """
    A block wider than `threshold` of the page width cannot be a genuine
    column of two-column body text (a real column tops out around 45-48%
    of the page, once margins and the gutter are accounted for). It is
    either the whole story on a single-column page, or a header / wide
    figure caption / wide table straddling both columns on a two-column
    page. 0.65 leaves headroom above real column width while staying
    clearly below "spans (almost) the whole page".
    """
    x0, x1 = block[0], block[2]
    return (x1 - x0) > threshold * page_width


def _detect_layout(blocks, page_width: float,
                    min_block_width_frac: float = 0.15,
                    majority_threshold: float = 0.5) -> str:
    """
    Classify a PAGE (not the whole paper) as "single" or "double" column.

    Only "qualifying" blocks (wider than min_block_width_frac of the page)
    are considered -- tiny blocks like stray page numbers or single
    characters are noise for this decision either way and would dilute
    the ratio. Among those, if most are full-width (see _is_full_width),
    the page reads as one column.

    Deliberately per-page rather than a single flag for the whole paper:
    a paper's title page, abstract, or a wide table can differ from its
    two-column body, and vice versa.
    """
    qualifying = [b for b in blocks if (b[2] - b[0]) > min_block_width_frac * page_width]
    if not qualifying:
        return "double"  # arbitrary; sort order barely matters with <1 real block
    full_width = [b for b in qualifying if _is_full_width(b, page_width)]
    return "single" if len(full_width) / len(qualifying) > majority_threshold else "double"


def _sort_blocks_single_column(blocks):
    """A single-column page is just read top-to-bottom."""
    return sorted(blocks, key=lambda b: b[1])


def _sort_blocks_two_column(blocks, page_width: float):
    """
    Standard two-column academic reading order: all of column 0
    (left) top-to-bottom, then all of column 1 (right) top-to-bottom.

    Full-width blocks (headers, wide figure captions, wide tables) are
    pulled out first -- forcing one into a column bucket by its centre
    x-coordinate is arbitrary and was the pre-fix behaviour's residual
    weakness even on genuinely two-column pages. Instead each full-width
    block splits the page into a vertical "band": every column block
    above it is read (left-column-then-right-column) before the
    full-width block, then the remaining column blocks after it are
    read the same way, and so on down the page.
    """
    midpoint = page_width / 2

    def column_key(b):
        center_x = (b[0] + b[2]) / 2
        column = 0 if center_x < midpoint else 1
        return (column, b[1])

    def column_sort(bs):
        return sorted(bs, key=column_key)

    straddlers = sorted(
        [b for b in blocks if _is_full_width(b, page_width)],
        key=lambda b: b[1],
    )
    columned = [b for b in blocks if not _is_full_width(b, page_width)]

    if not straddlers:
        return column_sort(columned)

    ordered = []
    remaining = columned
    for s in straddlers:
        band = [b for b in remaining if b[1] < s[1]]
        remaining = [b for b in remaining if b[1] >= s[1]]
        ordered.extend(column_sort(band))
        ordered.append(s)
    ordered.extend(column_sort(remaining))
    return ordered


def _sort_blocks_by_reading_order(blocks, page_width: float):
    """
    Detect this page's layout and sort accordingly. See _detect_layout
    and the module docstring for why this can no longer be a single
    fixed rule for the whole paper.
    """
    layout = _detect_layout(blocks, page_width)
    if layout == "single":
        return _sort_blocks_single_column(blocks)
    return _sort_blocks_two_column(blocks, page_width)


def extract_paper(pdf_path: str) -> PaperText:
    doc = pymupdf.open(pdf_path)
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
