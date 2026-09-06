"""
eval/run_generation.py
------------------------
Day 5 built this against ONE config -- section_aware/BGE/hybrid/rerank,
Day 4's overall winner. Day 6 needs generation (and, for RAGAS, the raw
retrieved context text) over TWO MORE configs for comparison: the naive
baseline (fixed_size chunking, MiniLM, dense-only, no rerank) and the best
retrieval config WITHOUT the reranker (section_aware/BGE/hybrid, rerank
OFF) -- see PLAN.md Day 6: "RAGAS on three configs only: naive baseline,
best retrieval config, best + reranker." So strategy/model_slug/
retriever_type/rerank are now CLI-selectable (mirroring run_retrieval.py's
own --strategy/--model/--retriever/--rerank exactly, including its
make_run_id naming convention) instead of hardcoded module constants.
Calling this with no flags at all reproduces Day 5's EXACT original
invocation and run_id (`section_aware__bge__hybrid_rerank__generation`) --
that run is already complete and independently verified (see DECISIONS.md,
"Day 5 results"); nothing here should ever cause it to be silently
recomputed or overwritten by a default-flags run.

CONTEXTS.JSONL, new this Day: Day 5's results.csv recorded citation
metadata (cited_numbers, invalid_citations) but never the raw retrieved
context TEXT itself -- there was no need to, since Day 5's own gate never
looks at raw context. RAGAS's faithfulness/context-precision/context-recall
metrics need exactly that (the actual passages the model was shown), so
this file now also writes `contexts.jsonl` (one line per qid: chunk_ids,
page ranges, and raw text of every context block shown to the generator)
alongside results.csv. This is purely ADDITIVE -- results.csv's schema,
and every row Day 5 already wrote, are unchanged.

DEPENDENCY STRUCTURE, same philosophy as run_retrieval.py: everything
above main() -- context-block construction, per-question evaluation,
the tau sweep, CSV/JSON writing -- imports nothing beyond stdlib plus
generate/span_utils. chromadb, src/retrievers.py, src/rerank.py, and
generate.py's OpenAIGenerator (which itself lazily imports `openai`) are
all imported inside main() only. Integration-tested end-to-end with a
FakeRetriever/FakeReranker/FakeGenerator triple standing in for the real
network-gated pipeline, exactly like run_retrieval.py was.

WHY TAU IS SWEPT HERE, NOT DECIDED AT GENERATION TIME: score-driven
abstention needs a threshold on top_rerank_score, but PLAN.md's own
instruction is to choose tau BY sweeping it and plotting abstention rate
vs error rate -- so it cannot be a constant baked into the generation
call itself (and doesn't need to be: the model is called exactly once per
question regardless of tau; sweeping tau afterward costs zero additional
API calls, since it only ever changes which already-computed rows get
COUNTED as abstained, never what the model was actually asked or said).
"""

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "qa_gold.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

# generate.py lives in src/, a SIBLING directory of eval/ (this file's own
# directory) -- unlike ir_metrics.py/span_utils.py, which live right here
# in eval/ and are always importable via Python's implicit "script's own
# directory" sys.path entry. This has to happen BEFORE the `from generate
# import ...` below, at module level, not deferred into main() the way
# chromadb/retrievers/rerank are -- generate.py's pure parts (ContextBlock,
# build_prompt, Answer, is_abstained, substitute_citations) import nothing
# heavier than stdlib themselves (OpenAIGenerator's `import openai` is
# lazy, inside its own __init__), so importing them at module level here
# costs nothing and keeps this file's own top-level logic testable exactly
# like run_retrieval.py's. (A real cross-directory import bug was caught
# here on Day 5 by testing against the actual device repo layout, not
# just a flat local test dir -- see DECISIONS.md.)
sys.path.insert(0, str(REPO_ROOT / "src"))

from generate import ContextBlock, build_prompt  # noqa: E402,F401  (build_prompt re-exported for callers/tests)
from span_utils import span_in_text  # noqa: E402

# Day 5's fixed, already-justified DEFAULT config (Day 4's overall winner).
# Kept as the argparse defaults below so a no-flags invocation reproduces
# Day 5's exact original run untouched -- these are no longer read directly
# by run_config(), which now takes strategy/model_slug/retriever_type/
# rerank as explicit parameters (see run_retrieval.py's identical pattern).
STRATEGY = "section_aware"
MODEL_SLUG = "bge"
RETRIEVER_TYPE = "hybrid"
RERANK = True
CANDIDATE_K = 20   # first-stage hybrid candidates handed to the reranker
MAX_K = 5          # context blocks kept after reranking -- PLAN.md Day 5: "keep 5"

# Mirrors run_retrieval.py's own lists exactly -- redefined locally rather
# than imported cross-file, keeping eval/'s scripts independent of each
# other (each already only shares ir_metrics/span_utils/generate).
STRATEGIES = ["fixed_size", "recursive", "section_aware", "section_aware_600", "section_aware_1500"]
MODEL_SLUGS = ["minilm", "bge"]
RETRIEVER_TYPES = ["dense", "hybrid"]  # bm25 excluded: Day 6's 3 configs are both embedding-backed


class NoOpReranker:
    """Stands in for CrossEncoderReranker when a config's rerank stage is
    OFF (Day 6's "best retrieval config, no reranker" -- PLAN.md Day 6).
    Exposes the identical `.rerank(query, candidates, k)` interface
    build_context_blocks/run_config already call, so run_config never
    needs an `if rerank:` branch of its own -- candidates are already
    Hit objects (chunk_id/text/metadata/score/rank), which is exactly the
    shape build_context_blocks expects, so truncating to k is the entire
    job. Mirrors run_retrieval.py's own `reranker=None` handling, but as
    a real object rather than a null check, so run_config's call site
    stays uniform regardless of which config is running."""

    def rerank(self, query: str, candidates: list, k: int = 5) -> list:
        return candidates[:k]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_gold_records() -> list:
    """ALL 85 records -- unlike run_retrieval.py, this file evaluates the
    12 unanswerable ones too (they're the true-abstention half of Day 5's
    gate)."""
    records = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _hit_paper(chunk_id: str) -> str:
    return chunk_id.split("::")[0] if "::" in chunk_id else chunk_id


def make_run_id(strategy: str, model_slug: str, retriever_type: str, rerank: bool) -> str:
    """Identical convention to run_retrieval.py's make_run_id, with a
    `__generation` suffix -- calling this with strategy=section_aware,
    model_slug=bge, retriever_type=hybrid, rerank=True reproduces Day 5's
    exact original run_id byte-for-byte."""
    suffix = "_rerank" if rerank else ""
    return f"{strategy}__{model_slug}__{retriever_type}{suffix}__generation"


def build_context_blocks(reranked_hits: list) -> list:
    """reranked_hits: rerank.py's RerankedHit objects, a plain retrievers.Hit
    (when rerank is off -- NoOpReranker passes Hit objects through
    unchanged, and Hit already carries the same chunk_id/text/metadata/
    score/rank fields ContextBlock needs), or any test double exposing
    those four fields. Numbers context blocks by each hit's OWN rank
    (1-indexed) -- not by re-deriving an order here."""
    blocks = []
    for h in reranked_hits:
        meta = h.metadata
        blocks.append(ContextBlock(
            number=h.rank,
            chunk_id=h.chunk_id,
            paper=_hit_paper(h.chunk_id),
            start_page=meta.get("start_page"),
            end_page=meta.get("end_page"),
            text=h.text,
            rerank_score=h.score,
        ))
    return blocks


def build_context_record(qid: str, context_blocks: list) -> dict:
    """The new, Day-6-motivated companion to evaluate_question: everything
    RAGAS needs about what the generator actually SAW for this question,
    which results.csv deliberately never stored (Day 5 only needed
    citation metadata, not raw text). One dict per question; run_config
    collects these into contexts.jsonl, keyed by qid so run_ragas.py can
    join it back against results.csv without re-deriving anything."""
    return {
        "qid": qid,
        "context_chunk_ids": [b.chunk_id for b in context_blocks],
        "context_pages": [[b.start_page, b.end_page] for b in context_blocks],
        "context_scores": [b.rerank_score for b in context_blocks],
        "context_texts": [b.text for b in context_blocks],
    }


def evaluate_question(rec: dict, answer, context_blocks: list) -> dict:
    """
    rec: one qa_gold.jsonl record (answerable OR unanswerable -- this file
    evaluates all 85). answer: a generate.Answer. context_blocks: what was
    actually shown to the generator for this question (used only to
    record how many chunks were retrieved, and whether any exist at all).

    `gold_answer_found` is tau-INDEPENDENT: whether qa_gold.jsonl's known
    gold_answer phrase is a normalized substring of the model's own
    final_text, using the EXACT SAME span_in_text normalization
    run_retrieval.py's Recall@K hit rule uses (see span_utils.py's own
    docstring on why one shared normalization function matters) --
    computed only when there's a real, non-abstained answer to check
    against a real gold_answer; left as "" otherwise (unanswerable
    questions have no gold_answer BY CONSTRUCTION, and a model-abstained
    answer has no answer text worth checking). This is deliberately the
    simpler, mechanically-verifiable check Day 4's own eval philosophy
    uses throughout (a known short phrase either appears or it doesn't) --
    NOT an LLM-judged semantic grade, which is Day 6's RAGAS job instead.
    """
    gold_answer_found = ""
    if rec.get("answerable") and not answer.model_abstained and rec.get("gold_answer"):
        gold_answer_found = span_in_text(rec["gold_answer"], answer.final_text)

    return {
        "qid": rec["qid"],
        "qtype": rec["qtype"],
        "paper": rec["paper"] if isinstance(rec["paper"], str) else "+".join(rec["paper"]),
        "answerable": bool(rec.get("answerable")),
        "n_context_blocks": len(context_blocks),
        "model_abstained": answer.model_abstained,
        "top_rerank_score": answer.top_rerank_score if answer.top_rerank_score is not None else "",
        "cited_numbers": "|".join(str(n) for n in sorted(answer.cited_numbers)),
        "invalid_citations": "|".join(str(n) for n in sorted(answer.invalid_citations)),
        "gold_answer_found": gold_answer_found,
        "raw_text": answer.raw_text,
        "final_text": answer.final_text,
        "latency_ms": answer.latency_ms,
    }


def _to_bool(v):
    return v if isinstance(v, bool) else (v == "True" if v in ("True", "False") else None)


def sweep_tau(rows: list, taus: list) -> list:
    """Pure post-hoc analysis, zero new generation calls: for each
    candidate tau, apply score-driven abstention (top_rerank_score < tau)
    ON TOP OF the already-fixed model-driven abstention, and report the
    three numbers PLAN.md's gate asks for. `rows` is exactly what
    evaluate_question produced (or an equivalent list of dicts, e.g. a
    test double's), never re-queries anything."""
    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable_rows = [r for r in rows if not r["answerable"]]
    n_answerable = len(answerable_rows)
    n_unanswerable = len(unanswerable_rows)

    def combined_abstained(row, tau):
        if row["model_abstained"]:
            return True
        score = row["top_rerank_score"]
        if score == "" or score is None:
            return False  # no reranked context at all -- can't apply a score threshold
        return score < tau

    sweep = []
    for tau in taus:
        n_false_abstain = sum(1 for r in answerable_rows if combined_abstained(r, tau)) if n_answerable else 0
        n_true_abstain = sum(1 for r in unanswerable_rows if combined_abstained(r, tau)) if n_unanswerable else 0

        non_abstained_answerable = [r for r in answerable_rows if not combined_abstained(r, tau)]
        judged = [r for r in non_abstained_answerable if r["gold_answer_found"] != ""]
        n_correct = sum(1 for r in judged if r["gold_answer_found"] in (True, "True"))
        n_judged = len(judged)

        sweep.append({
            "tau": tau,
            "false_abstention_rate": (n_false_abstain / n_answerable) if n_answerable else 0.0,
            "true_abstention_rate": (n_true_abstain / n_unanswerable) if n_unanswerable else 0.0,
            "n_true_abstained": n_true_abstain,
            "n_unanswerable": n_unanswerable,
            "error_rate_among_non_abstentions": ((n_judged - n_correct) / n_judged) if n_judged else None,
            "n_non_abstained_answerable_judged": n_judged,
            "meets_gate_10_of_12": n_true_abstain >= 10,
        })
    return sweep


def build_tau_grid(rows: list) -> list:
    """Candidate taus: every distinct observed top_rerank_score (a
    standard, complete sweep for a step-function threshold metric -- the
    only tau values where the decision can actually change), plus one
    point below the minimum (score-driven abstention never fires) and one
    above the maximum (score-driven abstention always fires), so the
    sweep table's first and last rows are the two extremes PLAN.md's
    "plot abstention rate vs error rate" wants to see anchored against."""
    scores = sorted({r["top_rerank_score"] for r in rows if r["top_rerank_score"] != ""})
    if not scores:
        return [0.0]
    taus = [scores[0] - 1.0] + scores + [scores[-1] + 1.0]
    return taus


def aggregate_diagnostics(rows: list) -> dict:
    """Tau-independent diagnostics about what the model actually DID,
    not about the abstention decision (which tau governs) -- computed
    once regardless of tau."""
    non_abstained = [r for r in rows if not r["model_abstained"]]
    n_no_citations = sum(1 for r in non_abstained if r["cited_numbers"] == "")
    n_with_invalid = sum(1 for r in non_abstained if r["invalid_citations"] != "")
    return {
        "n_questions_total": len(rows),
        "n_model_abstained": sum(1 for r in rows if r["model_abstained"]),
        "n_non_abstained_answers": len(non_abstained),
        "n_non_abstained_with_no_citations": n_no_citations,
        "n_non_abstained_with_invalid_citation": n_with_invalid,
        "median_latency_ms": statistics.median(r["latency_ms"] for r in rows) if rows else None,
    }


def write_run(run_id: str, rows: list, tau_sweep_rows: list, config: dict, context_rows: list) -> tuple:
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "qid", "qtype", "paper", "answerable", "n_context_blocks", "model_abstained",
        "top_rerank_score", "cited_numbers", "invalid_citations", "gold_answer_found",
        "raw_text", "final_text", "latency_ms",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    tau_path = out_dir / "tau_sweep.csv"
    tau_fieldnames = list(tau_sweep_rows[0].keys()) if tau_sweep_rows else []
    with open(tau_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=tau_fieldnames)
        writer.writeheader()
        for row in tau_sweep_rows:
            writer.writerow(row)

    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    contexts_path = out_dir / "contexts.jsonl"
    with open(contexts_path, "w", encoding="utf-8") as f:
        for row in context_rows:
            f.write(json.dumps(row) + "\n")

    return csv_path, tau_path, config_path, contexts_path


def run_config(
    retriever, reranker, generator,
    strategy: str = STRATEGY, model_slug: str = MODEL_SLUG,
    retriever_type: str = RETRIEVER_TYPE, rerank: bool = RERANK,
    candidate_k: int = CANDIDATE_K, max_k: int = MAX_K,
) -> dict:
    """Pure orchestration over an already-constructed retriever, reranker,
    and generator -- identical philosophy to run_retrieval.py's
    run_config: this function does not care whether `generator` is an
    OpenAIGenerator or a test double, only that it exposes .generate(),
    and does not care whether `reranker` is a real CrossEncoderReranker or
    a NoOpReranker, only that it exposes .rerank(). strategy/model_slug/
    retriever_type/rerank are recorded metadata (run_id, config.json,
    which corpus file to hash) -- they do not themselves select retriever
    behavior; the caller already built the right retriever/reranker for
    them (main() does this from the CLI flags, exactly like
    run_retrieval.py's own main())."""
    all_gold = load_gold_records()

    rows = []
    context_rows = []
    for rec in all_gold:
        candidates = retriever.retrieve(rec["question"], k=candidate_k)
        reranked = reranker.rerank(rec["question"], candidates, k=max_k)
        context_blocks = build_context_blocks(reranked)
        answer = generator.generate(rec["question"], context_blocks)
        rows.append(evaluate_question(rec, answer, context_blocks))
        context_rows.append(build_context_record(rec["qid"], context_blocks))

    tau_grid = build_tau_grid(rows)
    tau_sweep_rows = sweep_tau(rows, tau_grid)
    diagnostics = aggregate_diagnostics(rows)

    run_id = make_run_id(strategy, model_slug, retriever_type, rerank)
    config = {
        "run_id": run_id,
        "strategy": strategy,
        "model_slug": model_slug,
        "retriever_type": retriever_type,
        "rerank": rerank,
        "candidate_k": candidate_k,
        "max_k": max_k,
        "corpus_content_hash": _sha256_file(PROCESSED_DIR / f"chunks_{strategy}.jsonl"),
        "benchmark_hash": _sha256_file(GOLD_PATH),
        "n_questions_total": len(all_gold),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diagnostics": diagnostics,
    }

    csv_path, tau_path, config_path, contexts_path = write_run(run_id, rows, tau_sweep_rows, config, context_rows)
    return {
        "run_id": run_id, "rows": rows, "tau_sweep": tau_sweep_rows,
        "diagnostics": diagnostics, "config": config,
        "csv_path": csv_path, "tau_path": tau_path, "config_path": config_path,
        "contexts_path": contexts_path,
    }


def print_summary(result: dict) -> None:
    d = result["diagnostics"]
    print(f"\n=== {result['run_id']} ===")
    print(f"  Questions: {d['n_questions_total']}  (model-abstained: {d['n_model_abstained']}, "
          f"non-abstained answers: {d['n_non_abstained_answers']})")
    print(f"  Non-abstained answers with NO citation at all: {d['n_non_abstained_with_no_citations']}")
    print(f"  Non-abstained answers with an invalid (hallucinated) citation: {d['n_non_abstained_with_invalid_citation']}")
    if d["median_latency_ms"] is not None:
        print(f"  Median generation latency: {d['median_latency_ms']:.1f}ms")
    print("\n  tau sweep (false_abstention_rate / true_abstention_rate / error_rate_among_non_abstentions):")
    for row in result["tau_sweep"]:
        err = f"{row['error_rate_among_non_abstentions']:.3f}" if row["error_rate_among_non_abstentions"] is not None else "n/a"
        gate = " <-- meets >=10/12 gate" if row["meets_gate_10_of_12"] else ""
        print(f"    tau={row['tau']:.3f}: FAR={row['false_abstention_rate']:.3f}  "
              f"TAR={row['true_abstention_rate']:.3f} ({row['n_true_abstained']}/{row['n_unanswerable']})  "
              f"err={err}{gate}")
    print(f"\nWrote {result['csv_path']}")
    print(f"Wrote {result['tau_path']}")
    print(f"Wrote {result['config_path']}")
    print(f"Wrote {result['contexts_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Grounded generation + abstention over qa_gold.jsonl's 85 questions, "
                    "for one (strategy, model, retriever, rerank) config. Defaults reproduce "
                    "Day 5's original winning-config run exactly."
    )
    parser.add_argument("--strategy", default=STRATEGY, choices=STRATEGIES)
    parser.add_argument("--model-slug", default=MODEL_SLUG, choices=MODEL_SLUGS,
                        help="Embedding model slug for retrieval (NOT the generation model -- see --model).")
    parser.add_argument("--retriever", default=RETRIEVER_TYPE, choices=RETRIEVER_TYPES)
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip the cross-encoder second stage (Day 6's 'best retrieval config, no reranker').")
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--max-k", type=int, default=MAX_K)
    parser.add_argument("--model", default="gpt-5.6-terra", help="OpenAI chat model id (the GENERATOR, not the retriever).")
    parser.add_argument("--chroma-path", default=None, help="Override EVIDENCERAG_STORE / the repo-local fallback.")
    args = parser.parse_args()
    rerank = not args.no_rerank

    # Heavy, environment-specific imports deferred to here on purpose --
    # see the module docstring. Only actually generating for real needs
    # chromadb/torch/sentence-transformers/openai. (src/ is already on
    # sys.path from the module-level generate.py import above.)
    import os

    import chromadb
    from retrievers import build_retrievers
    from rerank import CrossEncoderReranker
    from generate import OpenAIGenerator

    chroma_path = (
        args.chroma_path
        or os.environ.get("EVIDENCERAG_STORE")
        or str(REPO_ROOT / "data" / "chroma")
    )
    client = chromadb.PersistentClient(path=chroma_path)
    dense, _, hybrid = build_retrievers(client, args.strategy, args.model_slug)
    retriever = {"dense": dense, "hybrid": hybrid}[args.retriever]
    reranker = CrossEncoderReranker() if rerank else NoOpReranker()
    generator = OpenAIGenerator(model_id=args.model)

    result = run_config(
        retriever, reranker, generator,
        strategy=args.strategy, model_slug=args.model_slug,
        retriever_type=args.retriever, rerank=rerank,
        candidate_k=args.candidate_k, max_k=args.max_k,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
