"""
build_corpus.py
----------------
Runs extraction + all three chunking strategies over every paper in
data/papers/, exactly once, and writes the result to disk:

    data/processed/chunks_fixed_size.jsonl
    data/processed/chunks_recursive.jsonl
    data/processed/chunks_section_aware.jsonl
    data/processed/manifest.json

From here on, nothing downstream (embedding, indexing, retrieval,
evaluation) re-parses a PDF or re-runs a chunker -- it reads these
JSONL files. That matters for two reasons: (1) PDF extraction + chunking
is deterministic but not instant, and (2) it makes every later stage
reproducible against a frozen, inspectable corpus instead of a black
box regenerated on every run.

Re-running this script always re-extracts and overwrites everything --
it does not try to detect "did the source PDFs change" and skip work.
What it DOES do is stamp the output with a `code_version` (a hash of
extract.py + chunking.py's own source) and the exact parameters used,
so a later stage -- or a human -- can tell whether a given .jsonl is
stale relative to the code that produced it, without guessing from
file mtimes.
"""

import glob
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from extract import extract_paper
from chunking import fixed_size_chunks, recursive_chunks, section_aware_chunks

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = REPO_ROOT / "data" / "papers"
OUT_DIR = REPO_ROOT / "data" / "processed"

STRATEGIES = {
    "fixed_size": fixed_size_chunks,
    "recursive": recursive_chunks,
    "section_aware": section_aware_chunks,
}

# Every strategy's default chunk_size / overlap (see chunking.py) is
# recorded explicitly here rather than left implicit, so the manifest
# is the single source of truth for "what parameters produced this
# .jsonl" -- not something you have to go read the code to find out.
PARAMS = {
    "fixed_size": {"chunk_size": 1000, "overlap": 100},
    "recursive": {"chunk_size": 1000, "overlap": 100},
    "section_aware": {"max_chunk_size": 1000, "overlap": 100},
}


def _code_version() -> str:
    """
    SHA-256 (first 12 hex chars) of extract.py + chunking.py's own
    source, concatenated. Deliberately NOT a git commit hash -- the
    working tree has uncommitted changes throughout this build, so a
    commit hash would silently describe a stale prior version. This
    hash changes the moment either file's logic changes, committed or
    not, which is the thing a later stage actually needs to detect.
    """
    src_dir = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ("extract.py", "chunking.py"):
        h.update((src_dir / name).read_bytes())
    return h.hexdigest()[:12]


def _git_commit() -> str:
    """Best-effort provenance only -- never gates anything on this."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        commit = out.stdout.strip() if out.returncode == 0 else "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "-dirty"
        return commit
    except Exception:
        return "unknown"


def _chunk_to_record(c) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "content_hash": c.content_hash,
        "text": c.text,
        "source_file": c.source_file,
        "strategy": c.strategy,
        "chunk_index": c.chunk_index,
        "start_page": c.start_page,
        "end_page": c.end_page,
        "char_start": c.char_start,
        "char_end": c.char_end,
        "section": c.section,
        "single_token_line_ratio": round(c.single_token_line_ratio, 4),
    }


def main():
    pdf_paths = sorted(glob.glob(str(PAPERS_DIR / "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {PAPERS_DIR}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    papers = [extract_paper(p) for p in pdf_paths]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_version": _code_version(),
        "git_commit": _git_commit(),
        "papers": [Path(p).name for p in pdf_paths],
        "strategies": {},
    }

    all_ids = set()
    id_collisions = []

    for strategy_name, fn in STRATEGIES.items():
        out_path = OUT_DIR / f"chunks_{strategy_name}.jsonl"
        records = []
        by_paper = {}
        for paper, pdf_path in zip(papers, pdf_paths):
            params = PARAMS[strategy_name]
            chunks = fn(paper, **params)
            paper_slug = Path(pdf_path).stem
            by_paper[paper_slug] = len(chunks)
            for c in chunks:
                if c.chunk_id in all_ids:
                    id_collisions.append(c.chunk_id)
                all_ids.add(c.chunk_id)
                records.append(_chunk_to_record(c))

        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        lengths = [len(r["text"]) for r in records]
        with_section = sum(1 for r in records if r["section"])
        manifest["strategies"][strategy_name] = {
            "params": PARAMS[strategy_name],
            "num_chunks": len(records),
            "avg_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "min_chars": min(lengths) if lengths else 0,
            "max_chars": max(lengths) if lengths else 0,
            "pct_with_section": round(100 * with_section / len(records), 1) if records else 0.0,
            "by_paper": by_paper,
            "output_file": out_path.name,
        }
        print(f"{strategy_name}: {len(records)} chunks -> {out_path}")

    if id_collisions:
        # Should be impossible given chunk_id's construction
        # (paper_slug::strategy::index), but checked explicitly rather
        # than assumed -- this is exactly the kind of silent failure a
        # later stage (dedup, eval-answer lookup) would not notice
        # until results looked subtly wrong.
        manifest["chunk_id_collisions"] = id_collisions
        print(f"WARNING: {len(id_collisions)} chunk_id collisions found: "
              f"{id_collisions[:5]}...", file=sys.stderr)
    else:
        manifest["chunk_id_collisions"] = []

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
