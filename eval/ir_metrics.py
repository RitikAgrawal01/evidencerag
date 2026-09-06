"""
eval/ir_metrics.py
------------------
Day 4: the actual Recall@K / MRR@10 / context-length hit-rule logic, shared
by every stage's run_retrieval.py invocation. Nothing in this file talks to
Chroma, a retriever, or an embedding model -- it only ever sees plain
strings (retrieved chunk texts, in rank order) and a question's ground
truth (gold_span / gold_pages / paper from qa_gold.jsonl). That separation
is deliberate: it is what lets this file be fully unit-tested with
hand-verifiable synthetic examples, with zero external dependencies beyond
span_utils, in ANY environment -- including one with no chromadb, no
torch, and no network (see DECISIONS.md).

PLAN.md Part III's hit rule, exactly as specified:
  PRIMARY:   normalise both sides, then substring-test gold_span in
             chunk.text (span_utils.span_in_text -- the SAME normalization
             draft_qa.py used to verify the span in the first place).
  SECONDARY (diagnostic only, never counted in Recall@K): same paper AND
             page overlap with gold_pages. This exists to distinguish, at
             failure-analysis time, "the right region of the corpus was
             retrieved but the exact sentence didn't survive being split
             across a chunk boundary" from "a completely unrelated part of
             the corpus was retrieved" -- see PLAN.md's Trap 2. It is
             NEVER used to award a Recall@K hit; a rank is only ever real
             if the literal span was found.

COMPARATIVE QUESTIONS (qtype == "comparative", per draft_qa.py's schema
extension -- "paper", "gold_pages", "gold_span" are all 2-element lists,
one element per source paper): a comparative question is only answerable
from retrieval if BOTH evidence spans are present among the retrieved
chunks. Modeled as "hit rank = the LATER of the two individual spans'
ranks" (i.e. max, not min, not average): the retriever hasn't actually
supplied everything the question needs until the SECOND of the two spans
has also shown up. If either span never appears in the retrieved list at
all, the whole question is a miss (rank=None), regardless of how early the
other one was found. This is a design decision PLAN.md leaves unspecified
-- draft_qa.py's own docstring flagged that this file would have to make
it (point 4 there); recorded here and in DECISIONS.md so it isn't
rediscovered as a surprise later.
"""

from typing import List, Optional, Sequence

from span_utils import span_in_text

RANK_CUTOFFS = (1, 3, 5, 10)


def find_rank(retrieved_texts: Sequence[str], gold_span: str) -> Optional[int]:
    """1-indexed rank of the first retrieved text containing gold_span as a
    (normalized) substring, or None if no retrieved text contains it."""
    for rank, text in enumerate(retrieved_texts, start=1):
        if span_in_text(gold_span, text):
            return rank
    return None


def find_rank_comparative(
    retrieved_texts: Sequence[str], gold_spans: Sequence[str]
) -> Optional[int]:
    """Comparative-question rank: both gold_spans must be found among
    retrieved_texts. Returns the LATER (max) of the two individual ranks --
    that's the point at which the retriever has supplied everything the
    question needs. None if either span is never found at all."""
    ranks = [find_rank(retrieved_texts, span) for span in gold_spans]
    if any(r is None for r in ranks):
        return None
    return max(ranks)


def recall_at_k(ranks: Sequence[Optional[int]], k: int) -> float:
    """Fraction of questions whose rank was found AND is <= k. An empty
    `ranks` returns 0.0 rather than raising -- a run with zero questions is
    a config-plumbing bug to catch elsewhere, not a metrics-module crash."""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def mrr_at_k(ranks: Sequence[Optional[int]], k: int = 10) -> float:
    """Mean reciprocal rank, counting only ranks <= k (a hit found at rank
    11 contributes 0 to MRR@10, exactly like it contributes 0 to
    Recall@10 -- the two metrics must agree on what counts as 'found
    within the window' or comparing them side by side is meaningless)."""
    if not ranks:
        return 0.0
    total = 0.0
    for r in ranks:
        if r is not None and r <= k:
            total += 1.0 / r
    return total / len(ranks)


def mean_context_chars_at_k(text_lists: Sequence[Sequence[str]], k: int = 5) -> float:
    """PLAN.md Trap 1: 'longer chunks win for free' -- never report
    Recall@5 alone. This is the number to report next to it: the average
    total character count a human (or an LLM) would actually have to read
    across the top-k retrieved chunks, per question, averaged over all
    questions. text_lists[i] is question i's retrieved texts in rank
    order; only the first k of each are counted."""
    if not text_lists:
        return 0.0
    totals = [sum(len(t) for t in texts[:k]) for texts in text_lists]
    return sum(totals) / len(totals)


def _hit_paper(metadata: dict) -> Optional[str]:
    chunk_id = metadata.get("chunk_id", "")
    return chunk_id.split("::")[0] if "::" in chunk_id else None


def page_overlap_hit(
    paper: str, gold_pages: Optional[List[int]], hit_metadatas: Sequence[dict]
) -> bool:
    """Secondary DIAGNOSTIC rule only (see module docstring) -- never
    counted toward Recall@K. True if any retrieved hit's metadata names
    the same paper (via its chunk_id's `paper::strategy::index` prefix)
    AND that hit's [start_page, end_page] range includes at least one of
    gold_pages."""
    if not gold_pages:
        return False
    gold_page_set = set(gold_pages)
    for meta in hit_metadatas:
        if _hit_paper(meta) != paper:
            continue
        start, end = meta.get("start_page"), meta.get("end_page")
        if start is None or end is None:
            continue
        # A small number of real chunks (found during Stage A failure
        # analysis, 5 Sep 2026 -- see DECISIONS.md) have start_page >
        # end_page: a malformed range from extract.py's page attribution,
        # not something this diagnostic should quietly under-report on.
        # Normalizing here doesn't fix the source data, but it does mean
        # THIS check judges page overlap correctly regardless of which
        # order the two ended up in.
        lo, hi = min(start, end), max(start, end)
        if any(lo <= p <= hi for p in gold_page_set):
            return True
    return False


def page_overlap_hit_comparative(
    papers: Sequence[str],
    gold_pages_list: Sequence[Optional[List[int]]],
    hit_metadatas: Sequence[dict],
) -> bool:
    """Comparative form of page_overlap_hit: BOTH papers' page ranges must
    independently show up somewhere in hit_metadatas (not necessarily the
    same hit)."""
    return all(
        page_overlap_hit(p, gp, hit_metadatas)
        for p, gp in zip(papers, gold_pages_list)
    )


if __name__ == "__main__":
    # Hand-verifiable self-test -- no chromadb/torch/network required,
    # matching the project's established testing philosophy (see
    # retrievers.py / rerank.py docstrings in DECISIONS.md).
    failures = []

    def check(label, got, expected):
        ok = got == expected
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: got {got!r}, expected {expected!r}")
        if not ok:
            failures.append(label)

    print("find_rank:")
    texts = ["irrelevant chunk one", "the CAT sat on the mat", "another irrelevant chunk"]
    check("exact substring at rank 2", find_rank(texts, "cat sat on the mat"), 2)
    check("not present -> None", find_rank(texts, "dog ran in the park"), None)
    check("fi-ligature (U+FB01) normalized to plain 'fi'",
          find_rank(["conformal prediction needs quantiﬁcation of uncertainty"],
                     "quantification of uncertainty"),
          1)
    check("soft hyphen (U+00AD) stripped",
          find_rank(["a distribution-free ap­proach to prediction"],
                     "distribution-free approach to prediction"),
          1)

    print("\nfind_rank_comparative:")
    texts_c = ["span A lives here", "irrelevant", "span B lives here too"]
    check("both found, later rank wins (3, not 1)",
          find_rank_comparative(texts_c, ["span A lives here", "span B lives here too"]), 3)
    check("one missing -> None",
          find_rank_comparative(texts_c, ["span A lives here", "span C never appears"]), None)

    print("\nrecall_at_k / mrr_at_k:")
    ranks = [1, 3, None, 7, 2]
    check("recall@1 = 1/5", recall_at_k(ranks, 1), 1 / 5)
    check("recall@3 = 3/5 (ranks 1,3,2 <=3)", recall_at_k(ranks, 3), 3 / 5)
    check("recall@10 = 4/5 (None never counts)", recall_at_k(ranks, 10), 4 / 5)
    check("recall_at_k([], 5) == 0.0", recall_at_k([], 5), 0.0)
    expected_mrr10 = (1 / 1 + 1 / 3 + 0 + 1 / 7 + 1 / 2) / 5
    check("mrr@10 matches hand computation", round(mrr_at_k(ranks, 10), 10), round(expected_mrr10, 10))
    check("mrr@2 excludes ranks 3 and 7", round(mrr_at_k(ranks, 2), 10), round((1 / 1 + 0 + 0 + 0 + 1 / 2) / 5, 10))

    print("\nmean_context_chars_at_k:")
    text_lists = [["a" * 100, "b" * 200, "c" * 50], ["d" * 300]]
    # q1 top-2: 100+200=300 ; q2 top-2: 300 (only one chunk available)
    check("mean top-2 chars", mean_context_chars_at_k(text_lists, 2), (300 + 300) / 2)
    check("empty list -> 0.0", mean_context_chars_at_k([], 5), 0.0)

    print("\npage_overlap_hit / page_overlap_hit_comparative:")
    metas = [
        {"chunk_id": "sam3::section_aware::0010", "start_page": 8, "end_page": 9},
        {"chunk_id": "surgicalsam::section_aware::0005", "start_page": 4, "end_page": 5},
    ]
    check("same paper, page overlaps", page_overlap_hit("sam3", [9, 10], metas), True)
    check("same paper, no page overlap", page_overlap_hit("sam3", [20], metas), False)
    check("wrong paper entirely", page_overlap_hit("depth_anything_3", [8], metas), False)
    check("gold_pages None -> False", page_overlap_hit("sam3", None, metas), False)
    check("comparative: both sides overlap",
          page_overlap_hit_comparative(["sam3", "surgicalsam"], [[8], [4, 5]], metas), True)
    check("comparative: one side fails -> False",
          page_overlap_hit_comparative(["sam3", "surgicalsam"], [[8], [99]], metas), False)
    check("reversed start/end page range (real corpus defect) still overlaps correctly",
          page_overlap_hit("sam3", [70], [{"chunk_id": "sam3::section_aware::0246", "start_page": 78, "end_page": 68}]),
          True)

    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED: ' + ', '.join(failures)}")
    if failures:
        raise SystemExit(1)
