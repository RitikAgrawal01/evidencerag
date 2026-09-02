"""
chunking.py
-----------
Three chunking strategies over extracted paper text -- deliberately
built to be compared empirically (Recall@K, MRR) rather than assumed
to rank in any particular order.

1. fixed_size    -- naive raw-character sliding window (baseline)
2. recursive     -- splits on the largest available semantic boundary
                     first (paragraphs, then lines, then sentences),
                     only falling back to a hard character cut if
                     nothing else fits
3. section_aware -- splits along detected paper section headers
                     (Introduction, Method, Results, ...) instead of
                     arbitrary character counts

All three attach page-number metadata to every chunk, computed from a
single shared character-offset index -- built once, alongside the text
itself, rather than reverse-searched afterwards (reverse string search
is fragile if a phrase repeats). This is what lets later stages cite
"page 4" instead of just a bare paragraph of text.

ToC pages (flagged in extract.py) are dropped before chunking -- they
carry no retrievable content. That heuristic only catches a ToC's FIRST
page (it looks for the literal word "Contents" at the top); multi-page
ToC continuations, dense results tables, and complex figures all slip
through as ordinary chunks. Rather than chase each of these as a new
extract.py special case, every chunk carries `single_token_line_ratio`
(see _single_token_line_ratio below) -- one measurable signal for "this
looks like linearized non-prose", found to cover all three cases without
being told which one it's looking at. It is stored as metadata, not used
to drop anything here.

Every chunk also carries a stable `chunk_id` and a `content_hash` of its
own text -- required for later stages to tell "this chunk's content
changed" from "only its index shifted", and to label evaluation ground
truth against something more durable than a position in a list.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from extract import PaperText


@dataclass
class Chunk:
    text: str
    source_file: str
    strategy: str
    chunk_index: int
    start_page: int
    end_page: int
    char_start: int
    char_end: int
    chunk_id: str
    content_hash: str
    single_token_line_ratio: float
    section: Optional[str] = None


def _content_hash(text: str) -> str:
    """
    Short, stable fingerprint of a chunk's own (whitespace-normalised)
    text. Lets a later run tell "this chunk's content actually changed"
    from "only its position in the list shifted" -- chunk_index alone
    can't distinguish those, and that distinction is what a regression
    check (Day 6) or a "did re-extraction change anything" check needs.
    """
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _single_token_line_ratio(text: str) -> float:
    """
    Fraction of non-empty lines that are a single whitespace-separated
    token. Measured empirically across this corpus (n=210 non-ToC-flagged
    pages): median 0.25, p75 0.48, p90 0.71 -- a CONTINUUM, not a clean
    bimodal split. By hand inspection, the top decile (>0.7) is almost
    entirely results tables, dense figures, and (for conformal_prediction
    specifically) ToC continuation pages the page-1 heuristic in
    extract.py doesn't catch. There is no single threshold that cleanly
    separates "prose" from "not prose", so this is stored as a per-chunk
    signal for later, evidence-based filtering (eval-question sampling,
    failure analysis) rather than used to drop anything at chunking time.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    single = sum(1 for l in lines if len(l.split()) == 1)
    return single / len(lines)


def _make_chunk(text, source_file, strategy, idx, start_page, end_page,
                 char_start, char_end, section=None):
    stripped = text.strip()
    paper_slug = Path(source_file).stem
    return Chunk(
        text=stripped,
        source_file=source_file,
        strategy=strategy,
        chunk_index=idx,
        start_page=start_page,
        end_page=end_page,
        char_start=char_start,
        char_end=char_end,
        section=section,
        chunk_id=f"{paper_slug}::{strategy}::{idx:04d}",
        content_hash=_content_hash(stripped),
        single_token_line_ratio=_single_token_line_ratio(stripped),
    )


# ---------------------------------------------------------------------------
# Shared: build one character-offset index that both the joined text and
# every chunk's page attribution derive from -- guarantees they agree.
# ---------------------------------------------------------------------------

def _paper_lines_with_pages(paper: PaperText) -> List[Tuple[int, str]]:
    out = []
    for page in paper.pages:
        if page.is_likely_toc:
            continue
        for line in page.text.split("\n"):
            out.append((page.page_number, line))
    return out


def _build_char_index(tagged_lines):
    pieces, spans = [], []
    offset = 0
    for page_num, line in tagged_lines:
        pieces.append(line)
        start = offset
        end = start + len(line)
        spans.append((page_num, start, end))
        offset = end + 1  # +1 for the "\n" joiner used below
    full_text = "\n".join(pieces)
    return full_text, spans


def _page_for_offset(spans, char_offset):
    for page_num, start, end in spans:
        if start <= char_offset < end:
            return page_num
    return spans[-1][0] if spans else 1


def _page_range(spans, start, end):
    p_start = _page_for_offset(spans, start)
    p_end = _page_for_offset(spans, max(start, end - 1))
    return (p_start, p_end)


# ---------------------------------------------------------------------------
# Strategy 1: fixed-size character windows (naive baseline)
# ---------------------------------------------------------------------------

def fixed_size_chunks(paper: PaperText, chunk_size=1000, overlap=100) -> List[Chunk]:
    tagged_lines = _paper_lines_with_pages(paper)
    full_text, spans = _build_char_index(tagged_lines)

    chunks = []
    start, idx = 0, 0
    n = len(full_text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = full_text[start:end]
        start_page, end_page = _page_range(spans, start, end)
        chunks.append(_make_chunk(piece, paper.source_file, "fixed_size", idx, start_page, end_page,
                                   char_start=start, char_end=end))
        idx += 1
        if end == n:
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: recursive chunking (paragraph/sentence-aware)
# ---------------------------------------------------------------------------

def _recursive_split(text: str, chunk_size: int, separators=None) -> List[str]:
    """
    Try the largest semantic boundary first (paragraph breaks), then
    progressively finer ones (line breaks, sentence ends), only
    falling back to a hard character cut if nothing else fits.
    Reimplemented directly rather than imported so the mechanism isn't
    a black box.

    Produces boundary-respecting, NON-overlapping pieces only -- overlap
    used to be bolted on here (only in the hard-character-cut fallback
    below, which meant paragraph-bounded pieces got none at all: see
    DECISIONS.md). It now lives in one place, `_stitch_overlap`, applied
    uniformly by the caller to the final flat piece list regardless of
    which rule produced any given piece.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        pieces, start = [], 0
        while start < len(text):
            pieces.append(text[start:start + chunk_size])
            start += chunk_size
        return pieces

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)

    chunks, current = [], ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_recursive_split(part, chunk_size, rest))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def _stitch_overlap(located, overlap: int):
    """
    Give consecutive, non-overlapping, boundary-respecting pieces real
    shared context at their boundary -- what `overlap` always claimed to
    provide (see DECISIONS.md for the bug this replaces).

    `located` is a list of (piece_text, approx_char_start) pairs, in
    document order. Every piece after the first gets a word-boundary-
    trimmed tail of up to `overlap` characters from the PRECEDING
    ORIGINAL piece prepended -- "original" meaning we always borrow from
    what that neighbour looked like before any stitching, so overlap
    doesn't compound across a long run of chunks. The borrowed tail is
    trimmed back to the nearest following space so a chunk never starts
    mid-word (unless the tail itself has no space in it at all, e.g. a
    long unbroken token -- an accepted rare edge case, same spirit as
    _recursive_split's own hard-cut fallback).

    Returns a new list of (stitched_text, char_start) pairs, same length
    and order as `located`.
    """
    if overlap <= 0 or len(located) < 2:
        return located

    out = [located[0]]
    for i in range(1, len(located)):
        prev_text, prev_pos = located[i - 1]
        piece, pos = located[i]
        tail = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
        space_idx = tail.find(" ")
        if space_idx != -1:
            tail = tail[space_idx + 1:]
        if tail:
            out.append((f"{tail} {piece}", prev_pos + len(prev_text) - len(tail)))
        else:
            out.append((piece, pos))
    return out


def _locate_pieces(pieces, container_text: str):
    """
    Approximately locate each boundary-respecting piece's start offset
    within `container_text` by searching for a probe of its own first 40
    (stripped) characters, moving the search cursor forward each time so
    later pieces can't match an earlier position.

    This is a probe-based search, not exact offset tracking through the
    split -- a real (if usually small) approximation. Good enough for
    page attribution and now for overlap stitching too; see DECISIONS.md
    for why a full exact-offset rewrite wasn't judged worth it here.
    """
    located, search_from = [], 0
    for piece in pieces:
        probe = piece.strip()[:40]
        pos = container_text.find(probe, search_from) if probe else search_from
        if pos == -1:
            pos = search_from
        located.append((piece, pos))
        search_from = pos
    return located


def recursive_chunks(paper: PaperText, chunk_size=1000, overlap=100) -> List[Chunk]:
    tagged_lines = _paper_lines_with_pages(paper)
    full_text, spans = _build_char_index(tagged_lines)
    pieces = _recursive_split(full_text, chunk_size)
    stitched = _stitch_overlap(_locate_pieces(pieces, full_text), overlap)

    chunks = []
    for idx, (text, char_start) in enumerate(stitched):
        char_end = char_start + len(text)
        start_page, end_page = _page_range(spans, char_start, char_end)
        chunks.append(_make_chunk(text, paper.source_file, "recursive", idx, start_page, end_page,
                                   char_start=char_start, char_end=char_end))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: section-aware chunking
# ---------------------------------------------------------------------------

_NUMBERED_HEADER = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z][A-Za-z\s\-:]{2,60}$")
_NAMED_HEADER = re.compile(
    r"^(abstract|introduction|related work|background|method(s|ology)?|"
    r"approach|experiments?|results?|discussion|conclusion(s)?|"
    r"acknowledgm?ents?|references|appendix)\s*$", re.IGNORECASE)


def _looks_like_header(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 80:
        return False
    return bool(_NUMBERED_HEADER.match(line) or _NAMED_HEADER.match(line))


def _merge_tiny_chunks(chunks: List[Chunk], min_size=80) -> List[Chunk]:
    """
    A header immediately followed by almost no body text (or a false
    header match inside a table) produces a near-empty, useless chunk.
    Merge anything under min_size into the next chunk (or the previous
    one, if it's the last chunk in the paper) rather than leaving it as
    standalone junk that could get retrieved on its own.
    """
    if not chunks:
        return chunks
    merged = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if len(c.text) < min_size and i + 1 < len(chunks):
            nxt = chunks[i + 1]
            combined_text = c.text + "\n" + nxt.text
            merged.append(_make_chunk(
                combined_text, c.source_file, c.strategy, len(merged),
                min(c.start_page, nxt.start_page), max(c.end_page, nxt.end_page),
                char_start=min(c.char_start, nxt.char_start),
                char_end=max(c.char_end, nxt.char_end),
                section=c.section or nxt.section,
            ))
            i += 2
        elif len(c.text) < min_size and merged:
            prev = merged.pop()
            combined_text = prev.text + "\n" + c.text
            merged.append(_make_chunk(
                combined_text, c.source_file, c.strategy, len(merged),
                min(prev.start_page, c.start_page), max(prev.end_page, c.end_page),
                char_start=min(prev.char_start, c.char_start),
                char_end=max(prev.char_end, c.char_end),
                section=prev.section or c.section,
            ))
            i += 1
        else:
            merged.append(_make_chunk(c.text, c.source_file, c.strategy, len(merged), c.start_page, c.end_page,
                                       char_start=c.char_start, char_end=c.char_end, section=c.section))
            i += 1
    return merged


def section_aware_chunks(paper: PaperText, max_chunk_size=1000, overlap=100) -> List[Chunk]:
    """
    Split along detected section headers instead of raw character
    counts. Header detection is regex-based and heuristic -- it will
    miss unusually formatted headers. Any section still longer than
    max_chunk_size is subdivided with the same recursive logic as
    strategy 2.
    """
    tagged_lines = _paper_lines_with_pages(paper)
    full_text, spans = _build_char_index(tagged_lines)
    lines = full_text.split("\n")

    sections, current_header, current_lines = [], "Preamble", []
    for line in lines:
        if _looks_like_header(line):
            if current_lines:
                sections.append((current_header, current_lines))
            current_header, current_lines = line.strip(), []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_header, current_lines))

    chunks, idx, search_from = [], 0, 0
    for header, body_lines in sections:
        body_text = "\n".join(body_lines).strip()
        section_full = f"{header}\n{body_text}" if body_text else header
        probe = body_text.strip()[:40] if body_text else header[:40]
        pos = full_text.find(probe, search_from) if probe else search_from
        if pos == -1:
            pos = search_from
        end_pos = pos + len(section_full)
        search_from = pos

        if len(section_full) <= max_chunk_size:
            start_page, end_page = _page_range(spans, pos, end_pos)
            chunks.append(_make_chunk(section_full, paper.source_file, "section_aware", idx, start_page, end_page,
                                       char_start=pos, char_end=end_pos, section=header))
            idx += 1
        else:
            # Oversized section: fall back to the same boundary-respecting
            # splitter as strategy 2, but overlap here only ever bridges
            # sub-pieces WITHIN this one section, never across a section
            # boundary -- section_aware's whole point is chunks that don't
            # straddle two different arguments, and overlap across
            # sections would quietly undo that.
            sub_pieces = _recursive_split(section_full, max_chunk_size)
            sub_located = _locate_pieces(sub_pieces, section_full)
            sub_stitched = _stitch_overlap(sub_located, overlap)
            for piece, local_pos in sub_stitched:
                abs_start = pos + local_pos
                abs_end = abs_start + len(piece)
                sp, ep = _page_range(spans, abs_start, abs_end)
                chunks.append(_make_chunk(piece, paper.source_file, "section_aware", idx, sp, ep,
                                           char_start=abs_start, char_end=abs_end, section=header))
                idx += 1
    return _merge_tiny_chunks(chunks)


if __name__ == "__main__":
    import glob
    from extract import extract_paper

    strategies = {
        "fixed_size": fixed_size_chunks,
        "recursive": recursive_chunks,
        "section_aware": section_aware_chunks,
    }

    with open("chunking_report.txt", "w", encoding="utf-8") as out:
        for pdf_path in sorted(glob.glob("data/papers/*.pdf")):
            paper = extract_paper(pdf_path)
            out.write(f"=== {paper.source_file} ===\n")
            for name, fn in strategies.items():
                chunks = fn(paper)
                lengths = [len(c.text) for c in chunks]
                avg_len = sum(lengths) / len(lengths) if lengths else 0
                out.write(f"  {name}: {len(chunks)} chunks, avg {avg_len:.0f} chars, "
                          f"min {min(lengths) if lengths else 0}, max {max(lengths) if lengths else 0}\n")
            out.write("\n")
