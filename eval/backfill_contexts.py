"""
eval/backfill_contexts.py
--------------------------
Day 5's real run (`section_aware__bge__hybrid_rerank__generation`) is
already complete and independently, deeply verified (see DECISIONS.md,
"Day 5 results") -- its results.csv/tau_sweep.csv/config.json must never
be recomputed, only read. But it predates run_generation.py's contexts.jsonl
(added for Day 6's RAGAS metrics), so it has no record of the actual
retrieved context TEXT the generator saw for each of its 85 questions --
only citation metadata.

This script backfills exactly that, WITHOUT calling the generator again
(zero new OpenAI cost, zero risk of the answers drifting from the frozen,
already-verified results.csv): it re-runs ONLY retrieval + rerank -- the
same deterministic, local, no-API-cost operations run_generation.py itself
calls -- for the same 85 questions, at the same config (section_aware,
BGE, hybrid, candidate_k=20, max_k=5), and writes contexts.jsonl into the
EXISTING run directory, touching nothing else there.

THE DETERMINISM CHECK THAT MAKES THIS TRUSTWORTHY: dense/BM25/hybrid
retrieval and cross-encoder reranking have no randomness of their own
(no sampling, no temperature -- unlike generation), so the reconstructed
top-1 rerank_score for each question SHOULD exactly match the
top_rerank_score already recorded in Day 5's results.csv. This script
checks that for every one of the 85 rows -- but real hardware caught a
real exception to "should" on the very first run (see DECISIONS.md,
"backfill_contexts.py's determinism check catches a real ANN
non-determinism"): ChromaDB's approximate-nearest-neighbor search is NOT
guaranteed to return bit-identical results across separate query
invocations for a weak/borderline query where many candidates are nearly
tied -- confirmed empirically, not assumed. So a mismatch on a question
that was ALREADY model-abstained in the original run (its response was
INSUFFICIENT_EVIDENCE, which run_ragas.py's own filter already excludes
from every metric it scores) is logged as a WARNING and does not block
the write, since it cannot corrupt anything Day 6 actually uses. A
mismatch on a NON-abstained question -- one whose real, cited answer
DOES depend on knowing the true retrieved text -- is still a hard
failure that refuses to write contexts.jsonl at all, exactly as before.
"""

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
RUN_DIR = EVAL_DIR / "results" / "section_aware__bge__hybrid_rerank__generation"
GOLD_PATH = EVAL_DIR / "qa_gold.jsonl"

STRATEGY = "section_aware"
MODEL_SLUG = "bge"
CANDIDATE_K = 20
MAX_K = 5

sys.path.insert(0, str(REPO_ROOT / "src"))
from generate import ContextBlock  # noqa: E402


def _hit_paper(chunk_id: str) -> str:
    return chunk_id.split("::")[0] if "::" in chunk_id else chunk_id


def build_context_blocks(reranked_hits: list) -> list:
    """Identical logic to run_generation.py's own build_context_blocks --
    duplicated rather than imported so this stays a standalone, obviously
    read-only script that cannot accidentally pull in run_generation.py's
    module-level side effects (its own sys.path insert, its own argparse
    defaults) just to reuse five lines."""
    blocks = []
    for h in reranked_hits:
        meta = h.metadata
        blocks.append(ContextBlock(
            number=h.rank, chunk_id=h.chunk_id, paper=_hit_paper(h.chunk_id),
            start_page=meta.get("start_page"), end_page=meta.get("end_page"),
            text=h.text, rerank_score=h.score,
        ))
    return blocks


def load_gold_records() -> list:
    records = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_recorded_rows(results_csv_path: Path) -> dict:
    """qid -> {top_rerank_score, model_abstained} from Day 5's real,
    frozen results.csv, so this script can verify its OWN reconstruction
    reproduces the score AND knows whether a mismatch is consequential
    (see verify_against_recorded_scores's docstring)."""
    rows = {}
    with open(results_csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["qid"]] = {
                "top_rerank_score": float(row["top_rerank_score"]) if row["top_rerank_score"] != "" else None,
                "model_abstained": row["model_abstained"] == "True",
            }
    return rows


def build_contexts(gold_records: list, retriever, reranker, candidate_k: int = CANDIDATE_K, max_k: int = MAX_K) -> list:
    """Pure(ish) orchestration -- takes an already-built retriever/reranker
    (real ones in main(), fakes in the unit test below), mirrors
    run_generation.py's own retrieve-then-rerank-then-build-blocks
    sequence exactly (same candidate_k/max_k), and returns one context
    record per gold record, in the SAME format run_generation.py's
    build_context_record produces (so run_ragas.py can read either run's
    contexts.jsonl identically)."""
    records = []
    for rec in gold_records:
        candidates = retriever.retrieve(rec["question"], k=candidate_k)
        reranked = reranker.rerank(rec["question"], candidates, k=max_k)
        blocks = build_context_blocks(reranked)
        records.append({
            "qid": rec["qid"],
            "context_chunk_ids": [b.chunk_id for b in blocks],
            "context_pages": [[b.start_page, b.end_page] for b in blocks],
            "context_scores": [b.rerank_score for b in blocks],
            "context_texts": [b.text for b in blocks],
        })
    return records


def verify_against_recorded_scores(context_records: list, recorded_rows: dict) -> dict:
    """Compares every reconstructed top-1 score against Day 5's real,
    frozen results.csv. Returns {"hard": [...], "soft": [...]} -- each a
    list of (qid, reconstructed, recorded) mismatches. A HARD mismatch is
    on a question the original run did NOT abstain on: its real, cited
    answer depends on knowing the true retrieved text, so a mismatch there
    means this reconstruction is not trustworthy and must not be written.
    A SOFT mismatch is on a question the original run DID abstain on
    (response was INSUFFICIENT_EVIDENCE) -- run_ragas.py's own filter
    already excludes that qid from every metric it scores, so a different
    top-1 chunk there cannot corrupt any number Day 6 actually reports;
    it is logged, not treated as disqualifying."""
    hard, soft = [], []
    for rec in context_records:
        qid = rec["qid"]
        recorded_row = recorded_rows.get(qid, {})
        recorded = recorded_row.get("top_rerank_score")
        reconstructed = rec["context_scores"][0] if rec["context_scores"] else None
        if recorded is None and reconstructed is None:
            continue
        if recorded is None or reconstructed is None or abs(recorded - reconstructed) > 1e-6:
            entry = (qid, reconstructed, recorded)
            if recorded_row.get("model_abstained"):
                soft.append(entry)
            else:
                hard.append(entry)
    return {"hard": hard, "soft": soft}


def write_contexts(contexts_path: Path, context_records: list) -> None:
    with open(contexts_path, "w", encoding="utf-8") as f:
        for row in context_records:
            f.write(json.dumps(row) + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Backfill contexts.jsonl for Day 5's frozen winning-config run "
                    "(retrieval+rerank only -- no generator call, no answer changes)."
    )
    parser.add_argument("--chroma-path", default=None, help="Override EVIDENCERAG_STORE / the repo-local fallback.")
    args = parser.parse_args()

    # Deferred until after argparse so --help works without chromadb/torch installed,
    # matching every other eval script's ordering.
    import os

    import chromadb
    from retrievers import build_retrievers
    from rerank import CrossEncoderReranker

    chroma_path = args.chroma_path or os.environ.get("EVIDENCERAG_STORE") or str(REPO_ROOT / "data" / "chroma")
    client = chromadb.PersistentClient(path=chroma_path)
    _, _, hybrid = build_retrievers(client, STRATEGY, MODEL_SLUG)
    reranker = CrossEncoderReranker()

    gold_records = load_gold_records()
    recorded_rows = load_recorded_rows(RUN_DIR / "results.csv")
    context_records = build_contexts(gold_records, hybrid, reranker)

    mismatches = verify_against_recorded_scores(context_records, recorded_rows)
    hard, soft = mismatches["hard"], mismatches["soft"]

    if hard:
        print(f"REFUSING TO WRITE contexts.jsonl: {len(hard)} of {len(context_records)} "
              f"NON-abstained questions' reconstructed top_rerank_score does not match Day 5's "
              f"frozen results.csv -- these rows' cited answers depend on the true retrieved text:")
        for qid, recon, rec in hard[:10]:
            print(f"  {qid}: reconstructed={recon}  recorded={rec}")
        raise SystemExit(1)

    if soft:
        print(f"NOTE: {len(soft)} question(s) reconstructed a different top-1 chunk than Day 5's "
              f"frozen run recorded, but each was ALREADY model-abstained there (response was "
              f"INSUFFICIENT_EVIDENCE) -- run_ragas.py excludes these qids from every metric it "
              f"scores, so this cannot corrupt any Day 6 number. Likely cause: ChromaDB's "
              f"approximate-nearest-neighbor search is not guaranteed bit-identical across query "
              f"invocations for a weak/borderline match (these were already flagged as "
              f"already-known-defective questions -- see DECISIONS.md's Day 5 results). Proceeding:")
        for qid, recon, rec in soft:
            print(f"  {qid}: reconstructed={recon}  recorded={rec}  (model_abstained=True originally)")

    contexts_path = RUN_DIR / "contexts.jsonl"
    write_contexts(contexts_path, context_records)
    n_hard_free = len(context_records) - len(soft)
    print(f"\n{n_hard_free} of {len(context_records)} questions' reconstructed top_rerank_score matched "
          f"Day 5's frozen results.csv exactly; the other {len(soft)} were already-abstained rows noted "
          f"above -- contexts.jsonl is a trustworthy replay for everything Day 6 actually uses.")
    print(f"Wrote {contexts_path}")
    print("results.csv, tau_sweep.csv, and config.json in this directory were NOT touched.")


if __name__ == "__main__":
    main()
