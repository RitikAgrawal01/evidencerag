"""
eval/regression.py
-------------------
Day 6's "regression guard, NOT drift detection" (PLAN.md): the corpus is
six static PDFs, it does not drift on its own, so this is not a
background job watching for change -- it is a deliberate gate you run
before trusting a change (a new prompt, a new chunking tweak, a
dependency bump) not to have quietly made the system worse. Loads
eval/results/baseline.json, re-computes Recall@5 and faithfulness for
"the config under test" RIGHT NOW, and exits non-zero if Recall@5 dropped
more than 2 percentage points or faithfulness dropped more than 0.05
versus the baseline -- exactly PLAN.md's two numbers.

WHY THIS NEVER WRITES INTO eval/results/{run_id}/: run_generation.py's own
run_id for "the config under test" -- by construction -- is very likely
IDENTICAL to one of Day 5/6's already-complete, independently-verified,
FROZEN run directories (e.g. the default config's run_id is exactly
"section_aware__bge__hybrid_rerank__generation", Day 5's real run). If
this script called run_generation.run_config() directly, every regression
check would silently overwrite that frozen results.csv/tau_sweep.csv/
config.json/contexts.jsonl with an unverified fresh run -- exactly the
kind of quiet data-clobbering this whole project has been careful to
avoid. So this file does its OWN minimal, in-memory-only orchestration
(retrieval + generation, scored immediately, nothing written to disk
except baseline.json itself, and only when --update-baseline is passed)
-- it reuses ir_metrics (Recall@K) and ragas's Faithfulness metric (via
run_ragas.py's constants, so the judge model can never silently drift
between Day 6's report and this gate) but never touches the results/
directory's run catalog.

COST NOTE: unlike Recall@5 (free, local, no API calls), the faithfulness
half of this gate calls the real generator AND a real LLM judge for every
answerable question checked -- it is not free, and should not be run on
every save. Use --skip-faithfulness for a fast, free Recall@5-only check
when that is all you need; run the full check deliberately, the way
PLAN.md frames it ("regression guard", not continuous monitoring).

DEPENDENCY STRUCTURE, same philosophy as every other eval script: loading/
comparing/writing baseline.json and computing Recall@5 from a list of
Hit-like objects import nothing beyond stdlib + ir_metrics/span_utils.
chromadb, src/retrievers.py, src/rerank.py, generate.py's OpenAIGenerator,
and ragas are all imported inside main() only.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "qa_gold.jsonl"
DECISIONS_PATH = REPO_ROOT / "DECISIONS.md"
BASELINE_PATH = EVAL_DIR / "results" / "baseline.json"

sys.path.insert(0, str(REPO_ROOT / "src"))
from generate import ContextBlock, build_prompt  # noqa: E402,F401
from span_utils import span_in_text  # noqa: E402
from ir_metrics import find_rank, find_rank_comparative, recall_at_k  # noqa: E402

STRATEGY = "section_aware"
MODEL_SLUG = "bge"
RETRIEVER_TYPE = "hybrid"
RERANK = True
CANDIDATE_K = 20
MAX_K = 5

RECALL_DROP_THRESHOLD = 0.02       # PLAN.md: ">2 percentage points"
FAITHFULNESS_DROP_THRESHOLD = 0.05  # PLAN.md: ">0.05"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_gold_records() -> list:
    records = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _hit_paper(chunk_id: str) -> str:
    return chunk_id.split("::")[0] if "::" in chunk_id else chunk_id


def build_context_blocks(reranked_hits: list) -> list:
    blocks = []
    for h in reranked_hits:
        meta = h.metadata
        blocks.append(ContextBlock(
            number=h.rank, chunk_id=h.chunk_id, paper=_hit_paper(h.chunk_id),
            start_page=meta.get("start_page"), end_page=meta.get("end_page"),
            text=h.text, rerank_score=h.score,
        ))
    return blocks


def compute_recall_at_5(retriever, reranker, rerank: bool, candidate_k: int, max_k: int,
                         gold_records: list = None) -> float:
    """Mirrors run_retrieval.py's own per-question hit rule exactly (same
    find_rank/find_rank_comparative calls, same ir_metrics.recall_at_k),
    but purely in-memory -- no CSV/config.json written. Free: retrieval
    and reranking cost no API calls."""
    questions = [r for r in (gold_records or load_gold_records()) if r.get("answerable")]
    ranks = []
    for rec in questions:
        if rerank:
            candidates = retriever.retrieve(rec["question"], k=candidate_k)
            hits = reranker.rerank(rec["question"], candidates, k=max_k)
        else:
            hits = retriever.retrieve(rec["question"], k=max_k)
        texts = [h.text for h in hits]
        if rec["qtype"] == "comparative":
            rank = find_rank_comparative(texts, rec["gold_span"])
        else:
            rank = find_rank(texts, rec["gold_span"])
        ranks.append(rank)
    return recall_at_k(ranks, 5)


def build_faithfulness_records(retriever, reranker, rerank: bool, generator,
                                candidate_k: int, max_k: int, gold_records: list = None) -> list:
    """Generates a real answer for every answerable question under the
    config being tested (real API cost -- see module docstring), entirely
    in-memory: nothing is written to eval/results/. Returns RAGAS-shaped
    records (same four canonical field names run_ragas.py's
    build_ragas_records produces) for every question where the model did
    NOT abstain -- an abstained answer has no claim to check
    faithfulness against, matching run_ragas.py's own filter."""
    questions = [r for r in (gold_records or load_gold_records()) if r.get("answerable")]
    records = []
    for rec in questions:
        if rerank:
            candidates = retriever.retrieve(rec["question"], k=candidate_k)
            hits = reranker.rerank(rec["question"], candidates, k=max_k)
        else:
            hits = retriever.retrieve(rec["question"], k=max_k)
        context_blocks = build_context_blocks(hits)
        answer = generator.generate(rec["question"], context_blocks)
        if answer.model_abstained or not context_blocks:
            continue
        records.append({
            "qid": rec["qid"],
            "user_input": rec["question"],
            "response": answer.final_text,
            "retrieved_contexts": [b.text for b in context_blocks],
            "reference": rec.get("gold_answer"),
        })
    return records


async def _score_faithfulness(records: list, llm) -> float:
    import asyncio

    from ragas.metrics.collections import Faithfulness

    metric = Faithfulness(llm=llm)
    semaphore = asyncio.Semaphore(8)

    async def score_one(rec):
        async with semaphore:
            try:
                result = await metric.ascore(
                    user_input=rec["user_input"], response=rec["response"],
                    retrieved_contexts=rec["retrieved_contexts"],
                )
                return result.value
            except Exception as e:  # noqa: BLE001
                print(f"  WARNING: faithfulness judge call failed for qid={rec['qid']}: {e}")
                return None

    scores = await asyncio.gather(*(score_one(r) for r in records))
    valid = [s for s in scores if s is not None]
    return (sum(valid) / len(valid)) if valid else None


def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def write_baseline(strategy: str, model_slug: str, retriever_type: str, rerank: bool,
                    candidate_k: int, max_k: int, recall_at_5: float, faithfulness, reason: str) -> dict:
    baseline = {
        "strategy": strategy, "model_slug": model_slug, "retriever_type": retriever_type,
        "rerank": rerank, "candidate_k": candidate_k, "max_k": max_k,
        "corpus_content_hash": _sha256_file(PROCESSED_DIR / f"chunks_{strategy}.jsonl"),
        "benchmark_hash": _sha256_file(GOLD_PATH),
        "recall_at_5": recall_at_5,
        "faithfulness": faithfulness,
        "reason": reason,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    decisions_entry = (
        f"\n## Regression baseline updated (auto-logged by regression.py, {datetime.now(timezone.utc).date()})\n\n"
        f"Config: {strategy}/{model_slug}/{retriever_type}{'/rerank' if rerank else ''}, "
        f"candidate_k={candidate_k}, max_k={max_k}. New baseline: Recall@5={recall_at_5:.4f}, "
        f"faithfulness={faithfulness if faithfulness is not None else 'not measured'}. "
        f"Reason given: {reason}\n"
    )
    if DECISIONS_PATH.exists():
        with open(DECISIONS_PATH, "a", encoding="utf-8") as f:
            f.write(decisions_entry)

    return baseline


def check_regression(baseline: dict, current_recall: float, current_faithfulness) -> dict:
    """Pure comparison logic -- no I/O, fully unit-testable. Returns a
    dict with each check's pass/fail and the overall verdict. Faithfulness
    is only checked when BOTH the baseline and the current run actually
    measured it (a --skip-faithfulness run, or a baseline saved before
    Day 6's RAGAS numbers existed, simply doesn't gate on it)."""
    recall_drop = baseline["recall_at_5"] - current_recall
    recall_ok = recall_drop <= RECALL_DROP_THRESHOLD

    faithfulness_checked = baseline.get("faithfulness") is not None and current_faithfulness is not None
    faithfulness_ok = True
    faithfulness_drop = None
    if faithfulness_checked:
        faithfulness_drop = baseline["faithfulness"] - current_faithfulness
        faithfulness_ok = faithfulness_drop <= FAITHFULNESS_DROP_THRESHOLD

    return {
        "recall_ok": recall_ok,
        "recall_drop": recall_drop,
        "faithfulness_checked": faithfulness_checked,
        "faithfulness_ok": faithfulness_ok,
        "faithfulness_drop": faithfulness_drop,
        "passed": recall_ok and faithfulness_ok,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Day 6 regression guard: compares the current config's Recall@5 "
                    "(and, unless --skip-faithfulness, faithfulness) against eval/results/baseline.json."
    )
    parser.add_argument("--strategy", default=STRATEGY)
    parser.add_argument("--model-slug", default=MODEL_SLUG)
    parser.add_argument("--retriever", default=RETRIEVER_TYPE, choices=["dense", "hybrid"])
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--max-k", type=int, default=MAX_K)
    parser.add_argument("--model", default="gpt-5.6-terra", help="Generator model (only used unless --skip-faithfulness).")
    parser.add_argument("--chroma-path", default=None)
    parser.add_argument("--skip-faithfulness", action="store_true",
                         help="Fast, free Recall@5-only check -- skips the real generation+judge calls.")
    parser.add_argument("--update-baseline", metavar="REASON", default=None,
                         help="Overwrite eval/results/baseline.json with THIS run's numbers, logging REASON to DECISIONS.md.")
    args = parser.parse_args()
    rerank = not args.no_rerank

    import asyncio
    import os

    import chromadb
    from retrievers import build_retrievers
    from rerank import CrossEncoderReranker
    from generate import OpenAIGenerator

    chroma_path = args.chroma_path or os.environ.get("EVIDENCERAG_STORE") or str(REPO_ROOT / "data" / "chroma")
    client = chromadb.PersistentClient(path=chroma_path)
    dense, _, hybrid = build_retrievers(client, args.strategy, args.model_slug)
    retriever = {"dense": dense, "hybrid": hybrid}[args.retriever]

    class _NoOpReranker:
        def rerank(self, query, candidates, k=5):
            return candidates[:k]

    reranker = CrossEncoderReranker() if rerank else _NoOpReranker()

    gold_records = load_gold_records()
    print("Computing Recall@5 for the config under test...")
    current_recall = compute_recall_at_5(retriever, reranker, rerank, args.candidate_k, args.max_k, gold_records)
    print(f"  Recall@5 = {current_recall:.4f}")

    current_faithfulness = None
    if not args.skip_faithfulness:
        print("Generating answers + scoring faithfulness (real API cost)...")
        generator = OpenAIGenerator(model_id=args.model)
        records = build_faithfulness_records(retriever, reranker, rerank, generator, args.candidate_k, args.max_k, gold_records)
        from run_ragas import JUDGE_MODEL
        from ragas.llms.base import llm_factory
        from ragas_openai_reasoning_fix import patch_reasoning_model_args
        import openai
        llm = llm_factory(JUDGE_MODEL, provider="openai", client=openai.AsyncOpenAI())
        # See ragas_openai_reasoning_fix.py / DECISIONS.md: ragas==0.4.3's own
        # reasoning-model detection can't parse a decimal minor version like
        # "gpt-5.6-luna" and silently sends max_tokens (which OpenAI rejects for this
        # model) instead of max_completion_tokens. Same real bug run_ragas.py hit.
        patch_reasoning_model_args(llm)
        current_faithfulness = asyncio.run(_score_faithfulness(records, llm))
        print(f"  Faithfulness = {current_faithfulness if current_faithfulness is not None else 'n/a'} (n={len(records)})")

    if args.update_baseline is not None:
        baseline = write_baseline(
            args.strategy, args.model_slug, args.retriever, rerank, args.candidate_k, args.max_k,
            current_recall, current_faithfulness, args.update_baseline,
        )
        print(f"\nBaseline updated: {BASELINE_PATH}")
        print(f"Logged to DECISIONS.md: {args.update_baseline!r}")
        return

    baseline = load_baseline()
    if baseline is None:
        print("\nNo eval/results/baseline.json found. Run again with --update-baseline \"<reason>\" "
              "to create the first one -- there is nothing to compare against yet.")
        raise SystemExit(1)

    result = check_regression(baseline, current_recall, current_faithfulness)
    print(f"\nBaseline: Recall@5={baseline['recall_at_5']:.4f}, "
          f"faithfulness={baseline.get('faithfulness') if baseline.get('faithfulness') is not None else 'not measured'}")
    print(f"Current:  Recall@5={current_recall:.4f}, "
          f"faithfulness={current_faithfulness if current_faithfulness is not None else 'not measured (--skip-faithfulness)'}")
    print(f"\nRecall@5 drop: {result['recall_drop']:.4f}  (threshold {RECALL_DROP_THRESHOLD}) -- "
          f"{'OK' if result['recall_ok'] else 'FAIL'}")
    if result["faithfulness_checked"]:
        print(f"Faithfulness drop: {result['faithfulness_drop']:.4f}  (threshold {FAITHFULNESS_DROP_THRESHOLD}) -- "
              f"{'OK' if result['faithfulness_ok'] else 'FAIL'}")
    else:
        print("Faithfulness: not checked (baseline or current run has no faithfulness measurement)")

    if result["passed"]:
        print("\nPASS")
    else:
        print("\nFAIL -- regression detected")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
