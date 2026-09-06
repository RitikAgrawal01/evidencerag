"""
eval/run_ragas.py
------------------
Day 6: RAGAS's four LLM-judged metrics (faithfulness, answer relevancy,
context precision, context recall) over exactly THREE configs, per
PLAN.md Day 6: "naive baseline (fixed_size chunking, MiniLM, dense), best
retrieval config, best + reranker" -- not all 19+ configs Day 4 swept.

WHAT THIS FILE DOES NOT DO: it does not call the generator. Every one of
the three configs' answers already exists on disk (two from a fresh
run_generation.py invocation, one -- the winning config -- from Day 5's
already-verified real run, its contexts backfilled by
backfill_contexts.py without touching its answers). This file only reads
results.csv + contexts.jsonl for each of the three run directories, builds
the RAGAS-shaped records, and calls the four judge metrics.

WHY ONLY ANSWERABLE, NON-ABSTAINED ROWS: an unanswerable question's
"correct" response is INSUFFICIENT_EVIDENCE, which is not a claim RAGAS's
faithfulness/context-recall metrics are meaningfully asking about (there
is no `reference` answer to compare context recall against either --
qa_gold.jsonl's gold_answer is null by construction for those 12). RAGAS
here measures the QUALITY of answers the system actually gave, not
whether it abstained correctly -- Day 5's tau sweep already measured
that half.

RAGAS API TARGETED: ragas==0.4.3 (current as of Sept 2026), using its
recommended non-deprecated path -- `ragas.metrics.collections` classes
(NOT the older `ragas.metrics.Faithfulness` etc, which still work in
0.4.3 but emit DeprecationWarning) and `ragas.llms.base.llm_factory` /
`ragas.embeddings.base.embedding_factory` (a native OpenAI path that does
NOT require LangChain, unlike the older LangchainLLMWrapper). This
couldn't be tested against a real judge API call in this project's own
device VM (no ragas, no network, no API key there) -- but ragas==0.4.3
itself WAS pip-installed and directly introspected in a separate
network-enabled sandbox after Ritik's real first run hit two real bugs:
(1) every metric except faithfulness raising "got an unexpected keyword
argument" -- an earlier version of this file passed one uniform field
list to all four metrics' .ascore(), but each has a different real
signature (see score_config()'s comment); (2) faithfulness itself hitting
a real OpenAI 400 ("Unsupported parameter: 'max_tokens'... Use
'max_completion_tokens' instead") -- a confirmed ragas==0.4.3 bug where
its own reasoning-model auto-detection fails on a decimal minor version
like "gpt-5.6-luna" (see ragas_openai_reasoning_fix.py). Both are fixed
here, verified against the real installed library's source and behavior,
not assumed.

Canonical RAGAS field names (ragas.dataset_schema.SingleTurnSample,
confirmed unchanged since the v0.1->v0.2 rename): user_input, response,
retrieved_contexts (list[str]), reference (str). Built here from
qa_gold.jsonl's question/gold_answer and a run's results.csv/contexts.jsonl
-- never invented.

SAY CLEARLY WHAT RAGAS DOES NOT MEASURE (PLAN.md Day 6's own instruction,
repeated here so it travels with every report this file writes, not just
DECISIONS.md): RAGAS's context precision/recall are LLM-JUDGED RELEVANCE
-- a model's opinion about whether the retrieved context looks relevant
to/sufficient for the reference answer. That is NOT the same claim as
Day 4's Recall@K, which is an IR metric checking whether the actual
answer-bearing span was retrieved, against ground truth verified by hand.
The two are complementary, not interchangeable, and conflating them is
one of the most common mistakes in a RAG evaluation writeup.

DEPENDENCY STRUCTURE: everything above main() -- loading results.csv/
contexts.jsonl, joining them, filtering to answerable+non-abstained,
building RAGAS-shaped records, computing means, writing the report --
imports nothing beyond stdlib. `ragas`, `openai`, and `asyncio`'s actual
event loop usage are confined to main(), matching every other eval script
this project has built.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "qa_gold.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
REPORTS_DIR = REPO_ROOT / "reports"

# The three configs PLAN.md Day 6 names, in the order they should appear
# in the report. (label, run_id) -- run_id must match run_generation.py's
# make_run_id output for that config (or Day 5's original, for the winner).
RUN_CONFIGS = [
    ("naive baseline (fixed_size / MiniLM / dense)", "fixed_size__minilm__dense__generation"),
    ("best retrieval, no reranker (section_aware / BGE / hybrid)", "section_aware__bge__hybrid__generation"),
    ("best + reranker (section_aware / BGE / hybrid / rerank)", "section_aware__bge__hybrid_rerank__generation"),
]

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

JUDGE_MODEL = "gpt-5.6-luna"
JUDGE_EMBEDDING_MODEL = "text-embedding-3-small"


def load_gold_by_qid() -> dict:
    records = {}
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["qid"]] = rec
    return records


def load_results_rows(run_dir: Path) -> list:
    with open(run_dir / "results.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_contexts_by_qid(run_dir: Path) -> dict:
    contexts_path = run_dir / "contexts.jsonl"
    by_qid = {}
    with open(contexts_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by_qid[rec["qid"]] = rec
    return by_qid


def build_ragas_records(gold_by_qid: dict, results_rows: list, contexts_by_qid: dict) -> list:
    """The pure heart of this file, fully unit-testable without ragas
    installed: joins one run's results.csv (the model's real answers) and
    contexts.jsonl (the real retrieved text) against qa_gold.jsonl (the
    reference), restricted to rows where there is an actual answer to
    judge against an actual reference -- answerable AND not
    model_abstained. Returns a list of dicts with exactly RAGAS's four
    canonical field names, plus qid carried along for traceability in the
    written report (RAGAS itself never sees the qid field)."""
    records = []
    for row in results_rows:
        qid = row["qid"]
        if row["answerable"] != "True" or row["model_abstained"] == "True":
            continue
        gold = gold_by_qid.get(qid)
        if gold is None or not gold.get("gold_answer"):
            continue
        ctx = contexts_by_qid.get(qid)
        if ctx is None or not ctx["context_texts"]:
            continue
        records.append({
            "qid": qid,
            "user_input": gold["question"],
            "response": row["final_text"],
            "retrieved_contexts": ctx["context_texts"],
            "reference": gold["gold_answer"],
        })
    return records


def aggregate_scores(scores: list) -> dict:
    """scores: list of (possibly None -- a judge call that itself failed
    should not silently vanish from the denominator, so this counts and
    reports it) per-question metric values. Returns mean over the
    successfully-scored subset plus how many questions that mean actually
    covers, so a report can never present "the mean" without also saying
    of how many."""
    valid = [s for s in scores if s is not None]
    return {
        "mean": statistics.mean(valid) if valid else None,
        "n_scored": len(valid),
        "n_total": len(scores),
    }


def write_markdown_report(report_path: Path, results_by_config: dict) -> None:
    lines = [
        "# RAGAS metrics: three-config comparison (Day 6)",
        "",
        "Judge model: " + JUDGE_MODEL + " (chat) / " + JUDGE_EMBEDDING_MODEL + " (embeddings, answer relevancy only).",
        "Scored over each config's answerable, non-abstained questions only -- an abstained "
        "answer has no claim to check faithfulness against, and an unanswerable question has "
        "no gold_answer to use as RAGAS's `reference`.",
        "",
        "**What context precision/recall do NOT measure:** these are LLM-JUDGED RELEVANCE -- "
        "a model's opinion about whether the retrieved context looks relevant to the reference "
        "answer. This is a different claim from Day 4's Recall@K, which checks whether the "
        "actual answer-bearing span was retrieved, against hand-verified ground truth. The two "
        "are complementary, not interchangeable -- see DECISIONS.md.",
        "",
        "| Config | n (answerable, non-abstained) | Faithfulness | Answer Relevancy | Context Precision | Context Recall |",
        "|---|---|---|---|---|---|",
    ]
    for label, run_id in RUN_CONFIGS:
        row = results_by_config[run_id]
        n = row["n_records"]
        cells = []
        for metric_name in METRIC_NAMES:
            agg = row["metrics"][metric_name]
            if agg["mean"] is None:
                cells.append("n/a")
            else:
                cells.append(f"{agg['mean']:.3f} (n={agg['n_scored']}/{agg['n_total']})")
        lines.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


async def _score_one_metric(metric, records: list, field_names: list, concurrency: int = 8) -> list:
    import asyncio

    semaphore = asyncio.Semaphore(concurrency)

    async def score_record(rec):
        async with semaphore:
            kwargs = {name: rec[name] for name in field_names}
            try:
                result = await metric.ascore(**kwargs)
                return result.value
            except Exception as e:  # noqa: BLE001 -- a single judge-call failure must not sink the whole run
                print(f"  WARNING: judge call failed for qid={rec['qid']}: {e}")
                return None

    return await asyncio.gather(*(score_record(r) for r in records))


async def score_config(run_id: str, records: list, llm, embeddings) -> dict:
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    # Field lists below are each metric's REAL .ascore() signature in the installed
    # ragas==0.4.3 (measured directly via inspect.signature against the actual installed
    # package, not assumed from docs -- see DECISIONS.md): they differ per metric, and an
    # earlier version of this file passed one uniform field list to all four, which fails
    # immediately with "got an unexpected keyword argument" for three of the four (only
    # Faithfulness's signature happened to match). Faithfulness has no `reference` param;
    # AnswerRelevancy takes only user_input/response (no context at all -- it measures
    # answer-to-question relevance via generated-question embeddings, not context
    # grounding); ContextPrecision and ContextRecall both take `reference`, not `response`
    # (they judge the retrieved context against the KNOWN-CORRECT answer, not the model's
    # actual answer).
    metrics = {
        "faithfulness": (Faithfulness(llm=llm), ["user_input", "response", "retrieved_contexts"]),
        "answer_relevancy": (AnswerRelevancy(llm=llm, embeddings=embeddings), ["user_input", "response"]),
        "context_precision": (ContextPrecision(llm=llm), ["user_input", "reference", "retrieved_contexts"]),
        "context_recall": (ContextRecall(llm=llm), ["user_input", "retrieved_contexts", "reference"]),
    }

    scored = {}
    for metric_name, (metric, field_names) in metrics.items():
        print(f"  scoring {metric_name} over {len(records)} questions...")
        scores = await _score_one_metric(metric, records, field_names)
        scored[metric_name] = aggregate_scores(scores)
    return scored


async def async_main():
    import openai
    from ragas.llms.base import llm_factory
    from ragas.embeddings.base import embedding_factory
    from ragas_openai_reasoning_fix import patch_reasoning_model_args

    client = openai.AsyncOpenAI()
    llm = llm_factory(JUDGE_MODEL, provider="openai", client=client)
    # See ragas_openai_reasoning_fix.py: ragas==0.4.3's own reasoning-model detection
    # (which would otherwise map max_tokens->max_completion_tokens for GPT-5+ judges)
    # fails on a decimal minor version like "gpt-5.6-luna" (int("5.6") raises internally,
    # silently caught) -- confirmed via real 400 errors from Ritik's actual run. This
    # applies the same fix that detection would have, directly.
    patch_reasoning_model_args(llm)
    embeddings = embedding_factory("openai", model=JUDGE_EMBEDDING_MODEL, client=client)

    gold_by_qid = load_gold_by_qid()
    results_by_config = {}
    for label, run_id in RUN_CONFIGS:
        run_dir = RESULTS_DIR / run_id
        print(f"\n=== {label} ({run_id}) ===")
        results_rows = load_results_rows(run_dir)
        contexts_by_qid = load_contexts_by_qid(run_dir)
        records = build_ragas_records(gold_by_qid, results_rows, contexts_by_qid)
        print(f"  {len(records)} answerable, non-abstained questions to judge")
        metrics = await score_config(run_id, records, llm, embeddings)
        for metric_name, agg in metrics.items():
            mean_str = f"{agg['mean']:.3f}" if agg["mean"] is not None else "n/a"
            print(f"    {metric_name}: {mean_str}  (n={agg['n_scored']}/{agg['n_total']})")
        results_by_config[run_id] = {"n_records": len(records), "metrics": metrics}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "ragas_metrics.md"
    write_markdown_report(report_path, results_by_config)
    json_path = REPORTS_DIR / "ragas_metrics.json"
    json_path.write_text(json.dumps(results_by_config, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")
    print(f"Wrote {json_path}")


def main():
    import asyncio
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
