"""
rerank.py
---------
Second-stage cross-encoder reranker: takes first-stage retrieval's top
`candidate_k` chunk candidates for a query and re-scores each
(query, chunk_text) pair JOINTLY through cross-encoder/ms-marco-MiniLM-L-6-v2,
then returns the top `k` by that score.

WHY A SECOND STAGE: DenseRetriever and BM25Retriever (retrievers.py) both
score a query against a chunk INDEPENDENTLY -- a bi-encoder embeds the
query once, embeds the chunk once (or, for BM25, just counts term
overlap), and compares. That independence is exactly what makes them fast
enough to search 800+ chunks in milliseconds, but it also means neither
one can notice that a chunk merely REUSES the query's vocabulary without
actually answering it -- e.g. the murali_latent_graph::recursive::0059
bibliography-citation line that ranked highly for the cross-paper probe
in Day 2's retrievers.py run (see DECISIONS.md): it repeats topically
relevant words because it's a reference-list entry citing related work,
not because it discusses the topic itself.

A cross-encoder runs the query and the chunk through ONE transformer
TOGETHER, so every query token can attend to every chunk token (and vice
versa) before the model produces a single relevance score. That's much
more accurate at judging one (query, chunk) pair, but the cost scales
with the number of pairs scored -- it cannot be precomputed or cached
per-chunk the way an embedding can, so it does not scale to scoring an
entire corpus. Right tool for re-scoring ~20 first-stage candidates,
wrong tool for retrieval itself. This is exactly the "reranker" PLAN.md
Day 4 Stage D compares (off vs cross-encoder) and Day 5 measures the
latency cost of.

STATUS: this module is prep work. It is written and integration-tested
(against a fake model standing in for the real network-gated weights,
same approach used for embed.py/store.py/retrievers.py -- see
DECISIONS.md) BEFORE Day 3's benchmark (qa_gold.jsonl) exists. That
matters: this file can be verified for CORRECTNESS now -- does it
actually reorder candidates the way the model's own scores say it
should, does it preserve prior rank for inspection, does latency
measurement work -- but NOT for whether reranking IMPROVES retrieval
quality. That second question is Day 4 Stage D's, and it needs Recall@k
against a locked benchmark to answer; it cannot be eyeballed from a
handful of anecdotal probe queries any more than Day 2's retriever
comparison could (see the Day 2 gate in PLAN.md: "Retrieval-quality
judgment... deliberately NOT made from these 5 anecdotal queries").
"""

import time
from dataclasses import dataclass
from typing import List

from sentence_transformers import CrossEncoder

from retrievers import Hit

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankedHit:
    chunk_id: str
    text: str
    metadata: dict
    score: float      # cross-encoder relevance score -- NOT on the same
                       # scale as Hit.score (cosine / BM25 / RRF), never
                       # compare the two directly
    rank: int          # 1-indexed, AFTER reranking
    prior_rank: int    # this candidate's rank BEFORE reranking, as
                       # produced by whichever first-stage retriever
                       # supplied it -- kept so a human (or eval harness)
                       # can see how much reranking moved each chunk


class CrossEncoderReranker:
    """
    Wraps one CrossEncoder model, loaded once and reused across queries.
    Reloading transformer weights per query would dominate any latency
    number and make Day 5's "measure latency honestly" comparison
    meaningless.
    """

    def __init__(self, model_id: str = RERANK_MODEL):
        self.model_id = model_id
        self.model = CrossEncoder(model_id)

    def rerank(self, query: str, candidates: List[Hit], k: int = 5) -> List[RerankedHit]:
        """
        Scores every (query, candidate.text) pair, keeps the top k by
        that score. candidates is expected to already be a first-stage
        retriever's output (Hit objects, in their own prior order) --
        this function does not itself decide how many candidates to
        pull; that's rerank_pipeline's job.
        """
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:k]
        reranked = []
        for rank, i in enumerate(order, start=1):
            c = candidates[i]
            reranked.append(RerankedHit(
                chunk_id=c.chunk_id, text=c.text, metadata=c.metadata,
                score=float(scores[i]), rank=rank, prior_rank=c.rank,
            ))
        return reranked


def rerank_pipeline(retriever, reranker: CrossEncoderReranker, query: str,
                     candidate_k: int = 20, k: int = 5) -> List[RerankedHit]:
    """
    Convenience: pull `candidate_k` results from any retriever that
    exposes retrievers.py's shared `.retrieve(query, k)` interface
    (DenseRetriever, BM25Retriever, or HybridRRFRetriever -- reranking
    doesn't care which produced the candidates), then rerank down to k.

    candidate_k=20 / k=5 matches PLAN.md Day 5 ("top-20 hybrid
    candidates, keep 5"): wide enough that a good chunk buried around
    first-stage rank #14 still gets a chance to surface, cheap enough
    that scoring all 20 pairs with a cross-encoder is fast.
    """
    candidates = retriever.retrieve(query, k=candidate_k)
    return reranker.rerank(query, candidates, k=k)


if __name__ == "__main__":
    import os
    import chromadb

    from retrievers import build_retrievers, REPO_ROOT

    # Same corrected, paper-specific probes as retrievers.py's smoke test
    # (see DECISIONS.md for why the originals were rewritten). Purpose
    # here is narrower than Day 4 Stage D: does the cross-encoder run
    # end-to-end and produce a sane reordering with sane latency -- NOT
    # whether reranking improves recall. That question needs Day 3's
    # locked benchmark, which does not exist yet.
    PROBES = [
        ("SA-Co benchmark", "acronym-heavy"),
        ("how is depth predicted from a single camera", "purely semantic"),
        ("how many H100 GPUs were used to train Depth Anything 3", "numerical"),
        ("what limitation of conformal prediction does the paper discuss", "about a limitation"),
        ("which papers in this corpus address surgical scene understanding", "cross-paper"),
    ]

    chroma_path = os.environ.get("EVIDENCERAG_STORE") or str(REPO_ROOT / "data" / "chroma")
    client = chromadb.PersistentClient(path=chroma_path)

    # recursive + minilm: same reason as retrievers.py's smoke test --
    # this checks reranking mechanics, not the strategy/model comparison
    # itself, and MiniLM is far cheaper to embed queries with.
    strategy, model_slug = "recursive", "minilm"
    _, _, hybrid = build_retrievers(client, strategy, model_slug)

    reranker = CrossEncoderReranker()

    for query, label in PROBES:
        print(f"\n=== [{label}] {query!r} ===")
        t0 = time.perf_counter()
        candidates = hybrid.retrieve(query, k=20)
        t1 = time.perf_counter()
        reranked = reranker.rerank(query, candidates, k=5)
        t2 = time.perf_counter()
        print(f"retrieval: {1000 * (t1 - t0):.1f}ms   rerank: {1000 * (t2 - t1):.1f}ms")
        for h in reranked:
            snippet = h.text[:100].replace("\n", " ")
            moved = h.prior_rank - h.rank
            print(f"  #{h.rank} (was #{h.prior_rank}, moved {moved:+d}) "
                  f"score={h.score:.4f} {h.chunk_id}  {snippet!r}")
