"""
eval/run_retrieval.py
----------------------
Day 4: runs ONE retrieval config -- (chunking strategy, embedding model,
retriever type, rerank on/off) -- against the 73 ANSWERABLE questions in
eval/qa_gold.jsonl (the 12 unanswerable ones are Day 5's abstention
measurement, not this file's job: there is no gold_span for them, so
Recall@K/MRR are undefined), and writes:

    eval/results/{run_id}/results.csv    -- one row per question
    eval/results/{run_id}/config.json    -- config + aggregate metrics

PLAN.md's staged plan (A-F) is 19 total configs across six stages, several
of which reuse the exact same config as an earlier stage's winner (e.g.
Stage B's "winner of A, dense" run for whichever strategy wins Stage A is
literally the same run as one of Stage A's three, just read again under a
different stage's table) -- so this script is invoked once per DISTINCT
config, not once per table row.

DEPENDENCY STRUCTURE, ON PURPOSE: everything above `main()` -- run_id
construction, per-question metric computation (via ir_metrics), aggregate
statistics, CSV/JSON writing -- imports nothing beyond the standard
library plus ir_metrics/span_utils. `chromadb`, `torch`, and
src/retrievers.py / src/rerank.py are imported ONLY inside `main()`,
lazily, right before they're needed to build a real retriever. This is
what let this file be integration-tested end-to-end with a fake retriever
standing in for the real Chroma-backed one, in an environment that has
none of those heavy packages installed or network access to fetch model
weights -- the exact same "fake model standing in for the real
network-gated weights" testing philosophy already used for
embed.py/store.py/retrievers.py/rerank.py (see DECISIONS.md). It is not
merely a workaround for that environment: keeping "what do we do with a
list of Hit-like objects" strictly separate from "how do we obtain one"
is a real design improvement on its own -- run_config() below does not
care, and never needs to know, whether its `retriever` argument is a
DenseRetriever, a BM25Retriever, a HybridRRFRetriever, or a test double.
"""

import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from ir_metrics import (
    RANK_CUTOFFS,
    find_rank,
    find_rank_comparative,
    mrr_at_k,
    page_overlap_hit,
    page_overlap_hit_comparative,
    recall_at_k,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "qa_gold.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

STRATEGIES = [
    "fixed_size", "recursive", "section_aware",
    # Day 4 Stage F chunk-size sweep (see DECISIONS.md / build_corpus_sweep.py):
    # section_aware at max_chunk_size=600/1500, alongside the existing
    # size=1000 "section_aware" -- each is its own opaque strategy name for
    # file/collection-naming purposes, nothing else in this file changes.
    "section_aware_600", "section_aware_1500",
]
MODEL_SLUGS = ["minilm", "bge"]
RETRIEVER_TYPES = ["bm25", "dense", "hybrid"]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_gold_records():
    """All 85 qa_gold.jsonl records, answerable and not."""
    records = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_answerable_questions():
    """The 73 records this script actually evaluates. Unanswerable items
    have no gold_span -- there is nothing for Recall@K/MRR to check."""
    return [r for r in load_gold_records() if r.get("answerable")]


def make_run_id(strategy: str, model_slug: str, retriever_type: str, rerank: bool) -> str:
    suffix = "_rerank" if rerank else ""
    if retriever_type == "bm25":
        # BM25 never touches an embedding model -- baking model_slug into
        # the run_id would imply it affected the result when it didn't.
        return f"{strategy}__bm25{suffix}"
    return f"{strategy}__{model_slug}__{retriever_type}{suffix}"


def evaluate_question(rec: dict, retrieved_hits: list) -> dict:
    """
    rec: one qa_gold.jsonl record with answerable == True.
    retrieved_hits: ordered list of objects exposing .chunk_id, .text,
    .metadata -- already reranked if rerank was requested. This function
    does not know or care which stage produced them (matching
    retrievers.py's own "never needs to know which retriever produced a
    result" philosophy), so a test double with the same three attributes
    exercises the exact same code path a real Hit/RerankedHit does.
    """
    texts = [h.text for h in retrieved_hits]
    metas = [h.metadata for h in retrieved_hits]

    if rec["qtype"] == "comparative":
        rank = find_rank_comparative(texts, rec["gold_span"])
        page_overlap = page_overlap_hit_comparative(rec["paper"], rec["gold_pages"], metas)
        paper_field = "+".join(rec["paper"])
    else:
        rank = find_rank(texts, rec["gold_span"])
        page_overlap = page_overlap_hit(rec["paper"], rec["gold_pages"], metas)
        paper_field = rec["paper"]

    return {
        "qid": rec["qid"],
        "qtype": rec["qtype"],
        "paper": paper_field,
        "rank_found": rank if rank is not None else "",
        "page_overlap_top10": page_overlap,
        "top5_context_chars": sum(len(t) for t in texts[:5]),
        "retrieved_chunk_ids": "|".join(h.chunk_id for h in retrieved_hits),
    }


def aggregate_metrics(rows: list, latencies_ms: list) -> dict:
    ranks = [r["rank_found"] if r["rank_found"] != "" else None for r in rows]
    metrics = {f"recall_at_{k}": recall_at_k(ranks, k) for k in RANK_CUTOFFS}
    metrics["mrr_at_10"] = mrr_at_k(ranks, 10)
    metrics["mean_context_chars_at_5"] = (
        statistics.mean(r["top5_context_chars"] for r in rows) if rows else 0.0
    )
    metrics["n_questions"] = len(rows)
    metrics["n_hits_at_10"] = sum(1 for r in ranks if r is not None)
    metrics["n_page_overlap_misses"] = sum(
        1 for r, rk in zip(rows, ranks) if rk is None and r["page_overlap_top10"]
    )
    if latencies_ms:
        sorted_lat = sorted(latencies_ms)
        p95_idx = min(len(sorted_lat) - 1, int(round(0.95 * (len(sorted_lat) - 1))))
        metrics["median_latency_ms"] = statistics.median(latencies_ms)
        metrics["p95_latency_ms"] = sorted_lat[p95_idx]
    else:
        metrics["median_latency_ms"] = None
        metrics["p95_latency_ms"] = None
    return metrics


def write_run(run_id: str, rows: list, config: dict) -> tuple:
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "qid", "qtype", "paper", "rank_found", "page_overlap_top10",
        "top5_context_chars", "retrieved_chunk_ids",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return csv_path, config_path


def run_config(
    strategy: str,
    model_slug: str,
    retriever_type: str,
    rerank: bool,
    retriever,
    reranker=None,
    candidate_k: int = 20,
    max_k: int = 10,
) -> dict:
    """
    Pure orchestration over an ALREADY-CONSTRUCTED retriever (and,
    if rerank, an already-constructed reranker exposing
    .rerank(query, candidates, k)). Times every question, computes
    per-question and aggregate metrics, writes the run folder, and
    returns everything the caller might want to print or inspect.
    """
    all_gold = load_gold_records()
    questions = [r for r in all_gold if r.get("answerable")]

    rows = []
    latencies_ms = []
    for rec in questions:
        t0 = time.perf_counter()
        if rerank:
            candidates = retriever.retrieve(rec["question"], k=candidate_k)
            hits = reranker.rerank(rec["question"], candidates, k=max_k)
        else:
            hits = retriever.retrieve(rec["question"], k=max_k)
        latencies_ms.append(1000 * (time.perf_counter() - t0))
        rows.append(evaluate_question(rec, hits))

    metrics = aggregate_metrics(rows, latencies_ms)
    run_id = make_run_id(strategy, model_slug, retriever_type, rerank)

    config = {
        "run_id": run_id,
        "strategy": strategy,
        "model_slug": model_slug if retriever_type != "bm25" else None,
        "retriever_type": retriever_type,
        "rerank": rerank,
        "candidate_k": candidate_k if rerank else None,
        "max_k": max_k,
        "corpus_content_hash": _sha256_file(PROCESSED_DIR / f"chunks_{strategy}.jsonl"),
        "benchmark_hash": _sha256_file(GOLD_PATH),
        "n_questions_in_gold_total": len(all_gold),
        "n_questions_answerable_evaluated": len(questions),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }

    csv_path, config_path = write_run(run_id, rows, config)
    return {
        "run_id": run_id,
        "rows": rows,
        "metrics": metrics,
        "config": config,
        "csv_path": csv_path,
        "config_path": config_path,
    }


def print_summary(result: dict) -> None:
    m = result["metrics"]
    print(f"\n=== {result['run_id']} ===")
    for k in RANK_CUTOFFS:
        print(f"  Recall@{k}:  {m[f'recall_at_{k}']:.3f}")
    print(f"  MRR@10:               {m['mrr_at_10']:.4f}")
    print(f"  Mean context chars@5: {m['mean_context_chars_at_5']:.1f}")
    if m["median_latency_ms"] is not None:
        print(f"  Latency median/p95:   {m['median_latency_ms']:.1f}ms / {m['p95_latency_ms']:.1f}ms")
    print(f"  Questions evaluated:  {m['n_questions']}  (found within top-10: {m['n_hits_at_10']})")
    if m["n_page_overlap_misses"]:
        print(
            f"  Diagnostic: {m['n_page_overlap_misses']} of the misses retrieved the "
            f"right paper/page but not the exact span (chunk-boundary split candidates "
            f"-- see PLAN.md Trap 2)."
        )
    print(f"\nWrote {result['csv_path']}")
    print(f"Wrote {result['config_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Day 4: run one retrieval config against qa_gold.jsonl's 73 answerable questions."
    )
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument(
        "--model", default="minilm", choices=MODEL_SLUGS,
        help="Ignored when --retriever bm25 (BM25 never touches an embedding model).",
    )
    parser.add_argument("--retriever", required=True, choices=RETRIEVER_TYPES)
    parser.add_argument("--rerank", action="store_true", help="Apply the cross-encoder second stage.")
    parser.add_argument("--candidate-k", type=int, default=20, help="First-stage pool size when --rerank.")
    parser.add_argument("--max-k", type=int, default=10, help="Final list length (must cover Recall@10).")
    parser.add_argument("--chroma-path", default=None, help="Override EVIDENCERAG_STORE / the repo-local fallback.")
    args = parser.parse_args()

    # Heavy, environment-specific imports deferred to here on purpose --
    # see the module docstring. Only actually building a real retriever
    # needs chromadb/torch/sentence-transformers.
    import os
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import chromadb
    from retrievers import build_retrievers
    from rerank import CrossEncoderReranker

    chroma_path = (
        args.chroma_path
        or os.environ.get("EVIDENCERAG_STORE")
        or str(REPO_ROOT / "data" / "chroma")
    )
    client = chromadb.PersistentClient(path=chroma_path)
    dense, bm25, hybrid = build_retrievers(client, args.strategy, args.model)
    retriever = {"bm25": bm25, "dense": dense, "hybrid": hybrid}[args.retriever]
    reranker = CrossEncoderReranker() if args.rerank else None

    result = run_config(
        args.strategy, args.model, args.retriever, args.rerank,
        retriever, reranker, candidate_k=args.candidate_k, max_k=args.max_k,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
