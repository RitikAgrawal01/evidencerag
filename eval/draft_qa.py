"""
eval/draft_qa.py
-----------------
Day 3 Step 1: drafts eval/qa_draft.jsonl -- GPT-4o-mini-proposed QA candidates,
sampled from the SECTION-AWARE chunk set. This script's only job is to produce
well-formed, PROGRAMMATICALLY VERIFIED candidates for a human to accept or
reject. It never writes qa_gold.jsonl itself: Step 2 (PLAN.md, "entirely
manual, non-negotiable") is what turns a trusted subset of this file's output
into the actual locked benchmark. Running this script twice, or deleting half
its output, changes nothing that matters -- it is scratch, by design.

Only the 90 ANSWERABLE questions across five types are drafted here (factual,
numerical, method, comparative, limitation). The 15 unanswerable questions are
Step 3, hand-written by Ritik -- an LLM asked to write "a question this corpus
can't answer" reliably produces obviously-absurd ones, not the plausible-but-
absent questions that make an abstention measurement meaningful.

DESIGN DECISIONS MADE HERE THAT PLAN.md LEAVES UNSPECIFIED (flagged, not
hidden -- see DECISIONS.md for the full writeup):

1. PER-PAPER QUOTA. PLAN.md gives total counts (30 factual / 20 numerical /
   20 method / 10 comparative / 10 limitation) but not how those split across
   six papers of very different size. Split proportional to each paper's
   SHARE OF SECTION-AWARE CHUNKS -- the exact population being sampled from,
   not a proxy like page count -- via the largest-remainder method (see
   `_allocate_per_paper_quota`), so a paper with more section-aware chunks
   earns more questions without any paper being shut out.

2. WHICH CHUNK GETS WHICH TYPE. "Numerical" and "limitation" questions need
   chunks that can actually support them. A chunk with no digits cannot
   honestly yield a numerical question no matter how good the prompt is.
   `_looks_numerical` / `_looks_limitation` are cheap regex heuristics that
   bias SAMPLING ORDER toward promising chunks first, falling back to the
   rest of that paper's pool only if the promising ones run out before quota
   is met. They do not gate acceptance -- GPT can still decline (or the span
   check can still fail) on a heuristically-promising chunk, in which case
   the next candidate is tried. Step 2's human review is the real filter;
   this only makes Step 1 not waste most of its calls on hopeless chunks.

3. COMPARATIVE QUESTIONS NEED TWO CHUNKS FROM TWO PAPERS THAT ARE ACTUALLY
   RELATED. Rather than hand-pick topic pairs (which would silently bake in
   my own assumptions about which papers relate, not a measured fact) or
   compare raw vocabulary overlap (which is exactly the "reuses the query's
   words without answering it" trap rerank.py's docstring describes),
   `_build_comparative_candidates` reuses Day 2's ALREADY-BUILT
   `section_aware__minilm` Chroma collection: pulls every chunk's own
   embedding, computes cosine similarity between every cross-paper pair, and
   ranks candidates by that similarity -- the same dense-similarity signal
   DenseRetriever already uses for retrieval, applied here to find which
   chunks from different papers are topically close enough to support a real
   comparative question. Capped per paper-pair (default 3) so all 10 don't
   come from a single pair of papers that happen to be very similar.

4. SCHEMA EXTENSION FOR COMPARATIVE. The eval schema in PLAN.md's Part III
   assumes one paper / one gold_span per question. A comparative question
   needs evidence from two different papers, so for qtype=="comparative"
   only: "paper" is a 2-element list, "gold_pages" a 2-element list of
   per-paper page lists, "gold_span" a 2-element list of per-paper spans --
   EACH span still verified as a literal substring of its OWN paper's chunk,
   independently. Every other qtype keeps the exact single-value schema from
   PLAN.md. Day 4's ir_metrics.py will need to know about this -- flagged
   here and in DECISIONS.md so it doesn't get discovered as a surprise then.

5. SPAN VERIFICATION NORMALIZATION lives in eval/span_utils.py, shared with
   Day 4's ir_metrics.py, so "is this span really in this text" is answered
   identically at draft time and at eval time. See that file's docstring.

6. NON-CONTENT SECTIONS ARE EXCLUDED ENTIRELY, FOR EVERY QUESTION TYPE. Found
   after the first real smoke-test run: a "factual" item drafted cleanly from
   conformal_prediction::section_aware::0113 turned out to ask "who are the
   authors of the paper titled 'Cautious deep learning'" -- span-verified
   correctly (the citation text really is a verbatim substring of the chunk)
   but the chunk's own `section` field is "References". It is a bibliography
   entry, not a claim the paper itself makes -- span verification cannot
   catch this, because the span genuinely IS in the chunk; the problem is
   that the chunk isn't "content" at all. Checked how common this is across
   the real corpus: 186 of 843 section-aware chunks (~22%) carry a non-content
   section label (171 "References", 12 "REFERENCES", 3 "Acknowledgment") --
   roughly 1 in 5 chunks. `_looks_numerical`
   / `_looks_limitation` only ever biased ordering for two of five types;
   this instead filters non-content sections (references, bibliography,
   acknowledgments) out of `_load_section_aware_chunks`'s output entirely, so
   every question type and the comparative-candidate builder inherit the same
   exclusion automatically rather than each needing to remember it.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from span_utils import span_in_text

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
EVAL_DIR = REPO_ROOT / "eval"

MODEL = "gpt-4o-mini"
SEED = 20260902  # fixed -- makes candidate sampling order reproducible

TYPE_QUOTA = {"factual": 30, "numerical": 20, "method": 20, "limitation": 10}
COMPARATIVE_QUOTA = 10
MAX_COMPARATIVE_PER_PAPER_PAIR = 3

QTYPE_GUIDANCE = {
    "factual": "a general lookup fact directly stated in the excerpt (a name, "
               "a term's definition, what something is called or does)",
    "numerical": "a question whose answer is a specific number, quantity, "
                 "count, percentage, or measurement stated in the excerpt",
    "method": "a question about HOW something is done -- a procedure, "
              "algorithm step, architectural choice, or design decision "
              "described in the excerpt",
    "limitation": "a question about a stated weakness, failure mode, "
                  "constraint, or open problem of the approach, as described "
                  "in the excerpt",
}

SYSTEM_PROMPT_SINGLE = """You are building a retrieval-evaluation benchmark for a RAG system. You will be given one excerpt from an academic paper. Write exactly one question of the requested type, answerable using ONLY the given excerpt -- not general knowledge, not anything from outside this excerpt.

Respond with a JSON object with these exact keys:
{"question": "...", "gold_span": "...", "gold_answer": "..."}

"gold_span" MUST be copied character-for-character from the excerpt below -- same spelling, same capitalization, same punctuation, no paraphrasing, no fixing typos. It should be a single sentence or clause (well under 200 characters) that directly contains the answer. "gold_answer" is a short reference answer (a few words), not a restatement of the whole span.

If you cannot honestly write a good question of the requested type from this excerpt, respond instead with:
{"skip": true, "reason": "..."}
Do not force a bad question rather than skipping."""

SYSTEM_PROMPT_COMPARATIVE = """You are building a retrieval-evaluation benchmark for a RAG system. You will be given two excerpts, from two DIFFERENT papers, that appear topically related. Write exactly one question that genuinely requires BOTH excerpts to answer -- not a question either excerpt alone would answer. A good comparative question asks how the two relate, contrasts a design choice or number between them, or asks about something common to both.

Respond with a JSON object with these exact keys:
{"question": "...", "gold_span_a": "...", "gold_span_b": "...", "gold_answer": "..."}

"gold_span_a" MUST be copied character-for-character from Excerpt A, "gold_span_b" character-for-character from Excerpt B -- no paraphrasing. Each should be a single sentence or clause, well under 200 characters. "gold_answer" is a short reference answer synthesizing both.

If these two excerpts do not actually support a genuine comparative question -- e.g. they only share surface vocabulary without a real conceptual link -- respond instead with:
{"skip": true, "reason": "..."}"""


def _paper_slug(chunk_id: str) -> str:
    return chunk_id.split("::")[0]


# Exact (case-insensitive, whitespace-stripped) section-label matches only --
# deliberately NOT a substring/regex match. A section actually titled e.g.
# "Acknowledgments and Funding" would need judgment calls a plain substring
# test can't make safely, and no such label was observed in the real corpus
# (see module docstring, point 6, for the Counter run this was checked with).
# If one ever turns up, extend this set explicitly rather than loosening the
# match.
_NON_CONTENT_SECTIONS = {
    "references",
    "reference",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "acknowledgment",
    "acknowledgement",
}


def _is_non_content_section(chunk: dict) -> bool:
    section = (chunk.get("section") or "").strip().lower()
    return section in _NON_CONTENT_SECTIONS


def _load_section_aware_chunks():
    records = []
    with open(PROCESSED_DIR / "chunks_section_aware.jsonl", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    total = len(records)
    records = [c for c in records if not _is_non_content_section(c)]
    excluded = total - len(records)
    print(f"_load_section_aware_chunks: excluded {excluded}/{total} non-content "
          f"(references/acknowledgments) chunks from the sampling pool")
    return records


_CITATION_BRACKET = re.compile(r"\[\d+(?:,\s*\d+)*\]")


def _looks_numerical(text: str) -> bool:
    """Cheap bias, not a gate (see module docstring, point 2)."""
    stripped = _CITATION_BRACKET.sub("", text)
    if re.search(r"\d+(\.\d+)?\s?%", stripped):
        return True
    return bool(re.search(r"\b\d{2,}\b", stripped))


_LIMITATION_SECTION_RE = re.compile(r"limitation|discussion|conclusion|future work", re.IGNORECASE)


def _looks_limitation(chunk: dict) -> bool:
    section = chunk.get("section") or ""
    if _LIMITATION_SECTION_RE.search(section):
        return True
    return bool(re.search(r"\blimitation", chunk["text"], re.IGNORECASE))


def _allocate_per_paper_quota(chunks_by_paper: dict, type_quota: dict) -> dict:
    """
    Largest-remainder (Hare-Niemeyer) apportionment: split each type's total
    across papers proportional to that paper's share of section-aware chunks.
    Floor each paper's exact share, then hand the few leftover slots to
    whichever papers had the largest fractional remainder -- ties broken by
    paper name so a re-run is deterministic, not by dict/insertion order.
    """
    total_chunks = sum(len(v) for v in chunks_by_paper.values())
    papers = list(chunks_by_paper.keys())
    allocation = {p: {} for p in papers}
    for qtype, total in type_quota.items():
        exact = {p: total * len(chunks_by_paper[p]) / total_chunks for p in papers}
        floors = {p: int(exact[p]) for p in papers}
        remainder = total - sum(floors.values())
        order = sorted(papers, key=lambda p: (-(exact[p] - floors[p]), p))
        for p in order[:remainder]:
            floors[p] += 1
        for p in papers:
            allocation[p][qtype] = floors[p]
    return allocation


def _candidate_pool(rng, chunks_by_paper, paper, qtype, used_chunk_ids):
    pool = [c for c in chunks_by_paper[paper] if c["chunk_id"] not in used_chunk_ids]
    if qtype == "numerical":
        primary = [c for c in pool if _looks_numerical(c["text"])]
    elif qtype == "limitation":
        primary = [c for c in pool if _looks_limitation(c)]
    else:
        primary = list(pool)
    primary_ids = {c["chunk_id"] for c in primary}
    fallback = [c for c in pool if c["chunk_id"] not in primary_ids]
    rng.shuffle(primary)
    rng.shuffle(fallback)
    return primary + fallback


def _call_openai(client, system_prompt, user_prompt, model=MODEL, max_retries=3):
    import time
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001 -- deliberately broad: network,
            last_err = e        # rate-limit, and malformed-JSON all land here
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_err}")


def _gold_pages(chunk: dict):
    return sorted(set([chunk["start_page"], chunk["end_page"]]))


def _draft_single_paper_item(client, chunk, qtype):
    paper = _paper_slug(chunk["chunk_id"])
    section = chunk.get("section") or "(no section header)"
    user_prompt = (
        f"Paper: {paper}\n"
        f"Section: {section}\n"
        f"Requested question type: {qtype} -- {QTYPE_GUIDANCE[qtype]}\n\n"
        f"Excerpt:\n\"\"\"\n{chunk['text']}\n\"\"\""
    )
    try:
        result = _call_openai(client, SYSTEM_PROMPT_SINGLE, user_prompt)
    except RuntimeError as e:
        return None, f"api_error: {e}"

    if not isinstance(result, dict):
        return None, "malformed_response"
    if result.get("skip"):
        return None, f"model_skipped: {result.get('reason', '')}"

    required = {"question", "gold_span", "gold_answer"}
    if not required.issubset(result.keys()):
        return None, f"missing_keys: {required - result.keys()}"
    if not all(isinstance(result[k], str) and result[k].strip() for k in required):
        return None, "empty_or_non_string_field"
    if not span_in_text(result["gold_span"], chunk["text"]):
        return None, "span_not_verbatim"

    record = {
        "question": result["question"].strip(),
        "qtype": qtype,
        "paper": paper,
        "gold_pages": _gold_pages(chunk),
        "gold_span": result["gold_span"],
        "gold_answer": result["gold_answer"].strip(),
        "answerable": True,
        "reviewed_by_human": False,
        "source_chunk_id": chunk["chunk_id"],
    }
    return record, None


def _draft_comparative_item(client, chunk_a, chunk_b):
    paper_a, paper_b = _paper_slug(chunk_a["chunk_id"]), _paper_slug(chunk_b["chunk_id"])
    user_prompt = (
        f"Excerpt A -- paper: {paper_a}, section: {chunk_a.get('section') or '(no section header)'}\n"
        f"\"\"\"\n{chunk_a['text']}\n\"\"\"\n\n"
        f"Excerpt B -- paper: {paper_b}, section: {chunk_b.get('section') or '(no section header)'}\n"
        f"\"\"\"\n{chunk_b['text']}\n\"\"\""
    )
    try:
        result = _call_openai(client, SYSTEM_PROMPT_COMPARATIVE, user_prompt)
    except RuntimeError as e:
        return None, f"api_error: {e}"

    if not isinstance(result, dict):
        return None, "malformed_response"
    if result.get("skip"):
        return None, f"model_skipped: {result.get('reason', '')}"

    required = {"question", "gold_span_a", "gold_span_b", "gold_answer"}
    if not required.issubset(result.keys()):
        return None, f"missing_keys: {required - result.keys()}"
    if not all(isinstance(result[k], str) and result[k].strip() for k in required):
        return None, "empty_or_non_string_field"
    if not span_in_text(result["gold_span_a"], chunk_a["text"]):
        return None, "span_a_not_verbatim"
    if not span_in_text(result["gold_span_b"], chunk_b["text"]):
        return None, "span_b_not_verbatim"

    record = {
        "question": result["question"].strip(),
        "qtype": "comparative",
        "paper": [paper_a, paper_b],
        "gold_pages": [_gold_pages(chunk_a), _gold_pages(chunk_b)],
        "gold_span": [result["gold_span_a"], result["gold_span_b"]],
        "gold_answer": result["gold_answer"].strip(),
        "answerable": True,
        "reviewed_by_human": False,
        "source_chunk_id": [chunk_a["chunk_id"], chunk_b["chunk_id"]],
    }
    return record, None


def _build_comparative_candidates(chroma_client, chunks_by_id, max_per_paper_pair=MAX_COMPARATIVE_PER_PAPER_PAIR):
    """
    Ranks cross-paper chunk pairs by cosine similarity of their ALREADY-BUILT
    section_aware__minilm embeddings (Day 2's store.py) -- reusing the exact
    signal DenseRetriever uses for retrieval, not a fresh heuristic. Returns
    (score, chunk_a, chunk_b) tuples, highest similarity first, with no chunk
    reused across pairs and at most `max_per_paper_pair` pairs contributed by
    any single pair of papers (so 10 comparative questions can't all come
    from the two most similar papers in the corpus).

    Day 2's store was built over the FULL 843-chunk set, before the
    non-content-section exclusion (module docstring, point 6) existed --
    so the collection still has embeddings for the ~183 References chunks
    that `chunks_by_id` (built from the now-filtered `_load_section_aware_chunks`
    output) no longer contains. Any id absent from `chunks_by_id` is dropped
    here, upfront, rather than surfacing as a KeyError deep in the ranking
    loop below the first time a scored pair happens to touch one.
    """
    import numpy as np

    collection = chroma_client.get_collection(name="section_aware__minilm")
    got = collection.get(include=["embeddings"])
    keep = [idx for idx, cid in enumerate(got["ids"]) if cid in chunks_by_id]
    dropped = len(got["ids"]) - len(keep)
    if dropped:
        print(f"_build_comparative_candidates: dropped {dropped} embedded chunk(s) "
              f"absent from the filtered chunk set (non-content sections)")
    ids = [got["ids"][idx] for idx in keep]
    embeddings = np.asarray([got["embeddings"][idx] for idx in keep], dtype=float)
    papers = [_paper_slug(cid) for cid in ids]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms
    sim = unit @ unit.T

    n = len(ids)
    scored_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if papers[i] == papers[j]:
                continue
            scored_pairs.append((float(sim[i, j]), i, j))
    scored_pairs.sort(key=lambda t: t[0], reverse=True)

    used_chunks = set()
    pair_counts = {}
    ranked = []
    for score, i, j in scored_pairs:
        if ids[i] in used_chunks or ids[j] in used_chunks:
            continue
        key = tuple(sorted((papers[i], papers[j])))
        if pair_counts.get(key, 0) >= max_per_paper_pair:
            continue
        ranked.append((score, chunks_by_id[ids[i]], chunks_by_id[ids[j]]))
        used_chunks.add(ids[i])
        used_chunks.add(ids[j])
        pair_counts[key] = pair_counts.get(key, 0) + 1
    return ranked


def draft_single_paper_quota(client, chunks_by_paper, quota_by_paper, rng):
    used_chunk_ids = set()
    all_accepted = []
    shortfalls = []
    for paper in sorted(chunks_by_paper):
        for qtype, quota in quota_by_paper[paper].items():
            if quota == 0:
                continue
            accepted = []
            rejections = []
            for chunk in _candidate_pool(rng, chunks_by_paper, paper, qtype, used_chunk_ids):
                if len(accepted) >= quota:
                    break
                record, reason = _draft_single_paper_item(client, chunk, qtype)
                if record is not None:
                    accepted.append(record)
                    used_chunk_ids.add(chunk["chunk_id"])
                else:
                    rejections.append((chunk["chunk_id"], reason))
            shortfall = quota - len(accepted)
            print(f"  [{paper}/{qtype}] drafted {len(accepted)}/{quota}"
                  + (f"  -- SHORTFALL {shortfall}" if shortfall else ""))
            all_accepted.extend(accepted)
            if shortfall:
                shortfalls.append((paper, qtype, shortfall, rejections))
    return all_accepted, used_chunk_ids, shortfalls


def draft_comparative_quota(client, chroma_client, chunks_by_id, used_chunk_ids):
    candidates = _build_comparative_candidates(chroma_client, chunks_by_id)
    print(f"\ncomparative: {len(candidates)} cross-paper candidate pairs ranked by cosine similarity")
    accepted = []
    rejections = []
    for score, chunk_a, chunk_b in candidates:
        if len(accepted) >= COMPARATIVE_QUOTA:
            break
        if chunk_a["chunk_id"] in used_chunk_ids or chunk_b["chunk_id"] in used_chunk_ids:
            continue
        record, reason = _draft_comparative_item(client, chunk_a, chunk_b)
        if record is not None:
            accepted.append(record)
            used_chunk_ids.add(chunk_a["chunk_id"])
            used_chunk_ids.add(chunk_b["chunk_id"])
        else:
            rejections.append((chunk_a["chunk_id"], chunk_b["chunk_id"], reason))
    shortfall = COMPARATIVE_QUOTA - len(accepted)
    print(f"  comparative: drafted {len(accepted)}/{COMPARATIVE_QUOTA}"
          + (f"  -- SHORTFALL {shortfall}" if shortfall else ""))
    return accepted, shortfall, rejections


def main():
    import random
    import chromadb

    load_dotenv()
    client = OpenAI()
    rng = random.Random(SEED)

    chunks = _load_section_aware_chunks()
    chunks_by_paper = defaultdict(list)
    for c in chunks:
        chunks_by_paper[_paper_slug(c["chunk_id"])].append(c)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    quota_by_paper = _allocate_per_paper_quota(chunks_by_paper, TYPE_QUOTA)
    print("Per-paper, per-type quota (largest-remainder allocation by chunk share):")
    for paper in sorted(chunks_by_paper):
        print(f"  {paper} ({len(chunks_by_paper[paper])} chunks): {quota_by_paper[paper]}")
    print()

    all_accepted, used_chunk_ids, shortfalls = draft_single_paper_quota(
        client, chunks_by_paper, quota_by_paper, rng,
    )

    chroma_path = os.environ.get("EVIDENCERAG_STORE") or str(REPO_ROOT / "data" / "chroma")
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    comp_accepted, comp_shortfall, comp_rejections = draft_comparative_quota(
        client, chroma_client, chunks_by_id, used_chunk_ids,
    )
    all_accepted.extend(comp_accepted)

    for i, rec in enumerate(all_accepted):
        rec["qid"] = f"q{i:04d}"

    EVAL_DIR.mkdir(exist_ok=True)
    out_path = EVAL_DIR / "qa_draft.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_accepted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_accepted)} drafted items to {out_path}")
    print("UNREVIEWED machine output. Day 3 Step 2 (manual review) decides what")
    print("survives into eval/qa_gold.jsonl -- nothing here is the benchmark yet.")

    if shortfalls or comp_shortfall:
        print("\nSHORTFALLS (quota not fully met):")
        for paper, qtype, shortfall, rejections in shortfalls:
            print(f"  {paper}/{qtype}: short by {shortfall} "
                  f"(sample reasons: {[r for _, r in rejections[:3]]})")
        if comp_shortfall:
            print(f"  comparative: short by {comp_shortfall} "
                  f"(sample reasons: {[r for _, _, r in comp_rejections[:3]]})")


if __name__ == "__main__":
    main()
