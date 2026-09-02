"""
retrievers.py
-------------
One Retriever interface, three implementations, all operating over a
single chunking strategy (dense and BM25 must share the SAME strategy's
chunk_id space for fusion to mean anything):

    DenseRetriever      -- embed the query, Chroma similarity search
    BM25Retriever       -- rank_bm25 over that strategy's chunk texts,
                           in-process, independent of any embedding model
    HybridRRFRetriever  -- reciprocal rank fusion of the two above

All three return the same `Hit` shape (chunk_id, text, metadata, score,
rank), so Day 3+'s eval harness never needs to know which retriever
produced a result.

WHY RANK FUSION, NOT SCORE FUSION: BM25 returns an unbounded positive
score; cosine similarity returns roughly [-1, 1]. Adding them directly is
meaningless without calibration, and calibration needs held-out data
better spent on evaluation. RRF consumes only each list's ORDERING, which
both retrievers produce honestly regardless of their score scales. This
is exactly the weakness in the older RAG project's union-and-truncate
fusion -- see DECISIONS.md.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from embed import Embedder, MODELS

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


@dataclass
class Hit:
    chunk_id: str
    text: str
    metadata: dict
    score: float
    rank: int  # 1-indexed, best result first


def _tokenize(text: str) -> List[str]:
    """
    Lowercase, alphanumeric-only word tokens. rank_bm25 doesn't ship its
    own tokenizer -- whatever splitting rule is used, it MUST be applied
    identically to the indexed corpus and to every incoming query, or
    BM25's term-overlap scoring silently breaks (e.g. "SA-Co" tokenized
    one way at index time and a different way at query time would never
    match, and nothing would error to say so).
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class DenseRetriever:
    """
    Embeds the query with `embedder` -- applying that model's query
    prefix if it has one (see embed.py's QUERY_PREFIXES; BGE needs one,
    MiniLM doesn't) -- then does a Chroma similarity search.
    """

    def __init__(self, collection, embedder: Embedder):
        self.collection = collection
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 10) -> List[Hit]:
        query_vec = self.embedder.encode_queries([query])[0]
        res = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )
        hits = []
        rows = zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        for rank, (cid, doc, meta, dist) in enumerate(rows, start=1):
            # Collections are created with hnsw:space="cosine" (store.py),
            # so Chroma's "distance" here is cosine distance
            # (1 - cosine similarity). Reported as similarity, not
            # distance, so a HIGHER score always means "more relevant"
            # across all three retrievers -- consistent with BM25's own
            # score direction, and with what HybridRRFRetriever assumes
            # when it reads `.rank` (not `.score`) to fuse.
            hits.append(Hit(chunk_id=cid, text=doc, metadata=meta, score=1.0 - dist, rank=rank))
        return hits


class BM25Retriever:
    """
    In-process BM25 over one chunking strategy's chunk texts --
    independent of any embedding model (BM25 never touches a vector).
    Built once per strategy directly from
    data/processed/chunks_{strategy}.jsonl, not from Chroma: this is the
    exact same chunk set dense retrieval for that strategy already uses,
    and Chroma's own `.get()` isn't the natural place to get whole-corpus
    term statistics from anyway.
    """

    def __init__(self, strategy: str):
        self.strategy = strategy
        path = PROCESSED_DIR / f"chunks_{strategy}.jsonl"
        with open(path, encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]
        tokenized_corpus = [_tokenize(r["text"]) for r in self.records]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 10) -> List[Hit]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        hits = []
        for rank, idx in enumerate(ranked_idx, start=1):
            r = self.records[idx]
            meta = {key: val for key, val in r.items() if key != "text"}
            hits.append(Hit(
                chunk_id=r["chunk_id"], text=r["text"], metadata=meta,
                score=float(scores[idx]), rank=rank,
            ))
        return hits


class HybridRRFRetriever:
    """
    Reciprocal Rank Fusion of a DenseRetriever and a BM25Retriever built
    over the SAME chunking strategy -- fusing across two different
    strategies would be combining two different chunk_id spaces, which
    is meaningless. Pulls `candidate_k` results from each list (wider
    than the requested k, so fusion has real candidates from both sides
    to combine, not just whichever one already agreed) and re-ranks by
    RRF score: for a chunk at rank r in a list, add 1/(k_rrf + r); sum
    across both lists; sort descending. k_rrf=60 is the Cormack et al.
    (2009) standard default.
    """

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, k_rrf: int = 60):
        self.dense = dense
        self.bm25 = bm25
        self.k_rrf = k_rrf

    def retrieve(self, query: str, k: int = 10, candidate_k: int = 50) -> List[Hit]:
        dense_hits = self.dense.retrieve(query, candidate_k)
        bm25_hits = self.bm25.retrieve(query, candidate_k)

        rrf_scores: dict = {}
        first_seen: dict = {}
        for hit_list in (dense_hits, bm25_hits):
            for h in hit_list:
                rrf_scores[h.chunk_id] = rrf_scores.get(h.chunk_id, 0.0) + 1.0 / (self.k_rrf + h.rank)
                first_seen.setdefault(h.chunk_id, h)  # keep text/metadata from whichever list saw it first

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]
        hits = []
        for rank, cid in enumerate(ranked_ids, start=1):
            base = first_seen[cid]
            hits.append(Hit(
                chunk_id=cid, text=base.text, metadata=base.metadata,
                score=rrf_scores[cid], rank=rank,
            ))
        return hits


def build_retrievers(client, strategy: str, model_slug: str):
    """
    Convenience factory: one (dense, bm25, hybrid) triple for a given
    (strategy, model) pair. Each call builds its own BM25Retriever
    (loads + tokenizes that strategy's whole chunk set) -- cheap enough
    at this corpus size (under a thousand chunks per strategy) not to
    bother sharing across calls.
    """
    embedder = Embedder(MODELS[model_slug])
    collection = client.get_collection(name=f"{strategy}__{model_slug}")
    dense = DenseRetriever(collection, embedder)
    bm25 = BM25Retriever(strategy)
    hybrid = HybridRRFRetriever(dense, bm25)
    return dense, bm25, hybrid


if __name__ == "__main__":
    import os
    import chromadb

    # The first two probes are answerable as written and were confirmed sensible
    # on a real run (see DECISIONS.md). The last three were originally written
    # as bare category fillers ("...the model", "...the approach") -- but this
    # corpus holds 6 different papers/models, so an unscoped query like "the
    # model" or "the approach" has no single correct answer, and a retriever
    # can't be faulted for not guessing which paper you meant. Rewritten to
    # name a specific paper/topic so each has one intended answer, which is
    # what makes a probe query actually diagnostic rather than just plausible.
    PROBES = [
        ("SA-Co benchmark", "acronym-heavy"),
        ("how is depth predicted from a single camera", "purely semantic"),
        ("how many H100 GPUs were used to train Depth Anything 3", "numerical"),
        ("what limitation of conformal prediction does the paper discuss", "about a limitation"),
        ("which papers in this corpus address surgical scene understanding", "cross-paper"),
    ]

    chroma_path = os.environ.get("EVIDENCERAG_STORE") or str(REPO_ROOT / "data" / "chroma")
    client = chromadb.PersistentClient(path=chroma_path)

    # recursive + minilm: minilm because it is ~8.5x faster to embed queries
    # with (see DECISIONS.md), and this is a sanity smoke test of the
    # retrieval mechanics, not the MiniLM vs BGE comparison itself.
    strategy, model_slug = "recursive", "minilm"
    dense, bm25, hybrid = build_retrievers(client, strategy, model_slug)

    for query, label in PROBES:
        print(f"\n=== [{label}] {query!r} ===")
        for name, retriever in [("dense", dense), ("bm25", bm25), ("hybrid", hybrid)]:
            hits = retriever.retrieve(query, k=5)
            print(f"-- {name} --")
            for h in hits:
                snippet = h.text[:100].replace("\n", " ")
                print(f"  #{h.rank} score={h.score:.4f} {h.chunk_id}  {snippet!r}")
