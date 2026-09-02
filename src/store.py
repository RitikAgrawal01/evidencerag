"""
store.py
--------
Builds the six Chroma collections the plan calls for: 3 chunking
strategies x 2 embedding models, named `{strategy}__{model_slug}`.

Reads chunks from data/processed/chunks_{strategy}.jsonl (build_corpus.py,
Day 1.5) -- never re-parses a PDF or re-runs a chunker. Embeds every
chunk's text with embed.py's Embedder for that model, and writes each
chunk's vector into Chroma alongside its citation metadata (chunk_id,
paper, section, start_page, end_page) plus its per-model token_count and
truncated flag from embed.py's token_stats() -- so Day 4 can directly
answer "what fraction of this model's misses were on truncated chunks"
instead of treating a score gap as unexplained.

One Embedder is built per model and reused across all three strategies
(not rebuilt per collection) -- reloading model weights three times over
would just be wasted time for the exact same object.
"""

import json
import os
from pathlib import Path

import chromadb

from embed import Embedder, MODELS

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
STRATEGIES = ["fixed_size", "recursive", "section_aware"]


def _chroma_path() -> str:
    """
    Chroma's persistent store must live OUTSIDE this OneDrive-synced repo
    folder -- see PLAN.md Day 0's OneDrive/SQLite lock trap. OneDrive
    actively syncs while Chroma's backend holds file locks on the same
    files; the two fight each other. EVIDENCERAG_STORE is the env var Day 0
    has you set for exactly this reason:

        setx EVIDENCERAG_STORE "C:\\ml-cache\\evidencerag-chroma"

    (new terminal needed after setx for it to take effect). If it's not
    set, this falls back to a folder INSIDE the repo purely so the script
    still runs -- but that fallback IS the trap Day 0 already flagged, not
    a safe default, so it prints a loud warning rather than silently using
    it.
    """
    path = os.environ.get("EVIDENCERAG_STORE")
    if path:
        return path
    fallback = str(REPO_ROOT / "data" / "chroma")
    print(
        "WARNING: EVIDENCERAG_STORE is not set. Falling back to "
        f"{fallback}, which sits inside the OneDrive-synced repo folder -- "
        "exactly the SQLite/OneDrive lock trap flagged in PLAN.md Day 0. "
        "Run the setx command from Day 0 (new terminal after) to fix this "
        "properly before trusting a long indexing run.",
    )
    return fallback


def _load_chunks(strategy: str):
    path = PROCESSED_DIR / f"chunks_{strategy}.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _clean_metadata(record: dict, token_count: int, truncated: bool) -> dict:
    """
    Chroma's metadata store rejects an explicit None value outright
    (verified directly against chromadb==1.5.9, the exact version pinned
    in requirements.lock.txt: `TypeError: Cannot convert Python object to
    MetadataValue`). `section` is None for fixed_size/recursive chunks by
    design (see chunking.py) -- it must be OMITTED from the dict entirely
    for those, never passed through as None. See DECISIONS.md.
    """
    meta = {
        "chunk_id": record["chunk_id"],
        "content_hash": record["content_hash"],
        "source_file": record["source_file"],
        "strategy": record["strategy"],
        "chunk_index": record["chunk_index"],
        "start_page": record["start_page"],
        "end_page": record["end_page"],
        "single_token_line_ratio": record["single_token_line_ratio"],
        "token_count": token_count,
        "truncated": truncated,
    }
    if record.get("section") is not None:
        meta["section"] = record["section"]
    return meta


def build_collection(client, strategy: str, model_slug: str, embedder: Embedder) -> int:
    name = f"{strategy}__{model_slug}"
    records = _load_chunks(strategy)
    texts = [r["text"] for r in records]
    ids = [r["chunk_id"] for r in records]

    embeddings = embedder.encode_passages(texts)
    stats = embedder.token_stats(texts)
    metadatas = [
        _clean_metadata(r, s.token_count, s.truncated)
        for r, s in zip(records, stats)
    ]

    # Always rebuild from scratch, same philosophy as build_corpus.py:
    # deterministic regeneration beats incremental drift between runs.
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        client.delete_collection(name=name)
    collection = client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}, embedding_function=None,
    )

    batch_size = client.get_max_batch_size()
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            metadatas=metadatas[i:i + batch_size],
            documents=texts[i:i + batch_size],
        )

    n_truncated = sum(1 for s in stats if s.truncated)
    print(
        f"{name}: {collection.count()} chunks indexed "
        f"({n_truncated}/{len(stats)} = {100 * n_truncated / len(stats):.2f}% "
        f"truncated at {embedder.max_seq_length} tokens)"
    )
    return collection.count()


def main():
    chroma_path = _chroma_path()
    print(f"Chroma persistent store: {chroma_path}\n")
    client = chromadb.PersistentClient(path=chroma_path)

    summary = {}
    for model_slug, model_id in MODELS.items():
        embedder = Embedder(model_id)
        for strategy in STRATEGIES:
            summary[f"{strategy}__{model_slug}"] = build_collection(
                client, strategy, model_slug, embedder,
            )

    print("\n=== Summary ===")
    for name, count in summary.items():
        print(f"  {name}: {count}")

    total = sum(summary.values())
    expected = sum(len(_load_chunks(s)) for s in STRATEGIES) * len(MODELS)
    print(f"\nTotal vectors indexed: {total} (expected {expected})")
    if total != expected:
        raise RuntimeError(
            f"Indexed count ({total}) != expected ({expected}) -- "
            f"something silently dropped or duplicated records."
        )


if __name__ == "__main__":
    main()
