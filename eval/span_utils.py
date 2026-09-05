"""
eval/span_utils.py
-------------------
ONE normalization function, shared between draft_qa.py (today: verifying a
drafted gold_span is really a substring of its source chunk before the item
is even offered for human review) and ir_metrics.py (Day 4: the actual
Recall@K hit rule, span-in-retrieved-chunk). These two checks are the exact
same operation -- "is this span really in this text" -- performed at two
different times against two different texts (the source chunk now, a
retrieved chunk later). If they used two independently-written normalization
routines, the two could quietly drift apart: a span verified as a clean
substring today could fail to match at eval time for a reason that has
nothing to do with retrieval quality (e.g. one strips a ligature the other
doesn't), and that would corrupt the eval metrics for a reason invisible to
anyone reading them. Importing the same function in both places makes that
class of bug impossible rather than merely unlikely.

PLAN.md's own spec for the hit rule (Part III): "collapse whitespace, strip
soft hyphens and ligatures, lowercase, then test substring containment."
This is exactly that, no more:

  - unicodedata.normalize("NFKC", text) folds ligatures (the real corpus
    genuinely contains them -- e.g. chunks_section_aware.jsonl has the
    literal single-codepoint "fi" ligature glyph, U+FB01, in the word
    "Quantification" from conformal_prediction's PDF -- not the two
    separate letters f-i. NFKC maps U+FB01 to the two-character "fi") and a
    long tail of other compatibility variants (fancy quotes, full-width
    forms) for free, not just the specific ligatures anyone happens to have
    seen.
  - explicit soft-hyphen (U+00AD) removal: NFKC does NOT strip this on its
    own (it's a formatting-control codepoint, not a compatibility variant),
    and PDF text extraction is exactly where it turns up, marking
    line-break hyphenation points that the extracted text itself does not
    visually render as a hyphen.
  - whitespace collapse + lowercase, applied last, after the above two have
    had a chance to change what characters are actually present.
"""

import re
import unicodedata

_SOFT_HYPHEN = "­"


def normalize_for_span_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_SOFT_HYPHEN, "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def span_in_text(span: str, text: str) -> bool:
    """True if `span` is a substring of `text` after both are normalized
    identically. This is the ONE check both draft-time verification and
    Day 4's Recall@K hit rule must use -- never re-derive it inline."""
    return normalize_for_span_match(span) in normalize_for_span_match(text)
