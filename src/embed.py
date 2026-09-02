"""
embed.py
--------
Thin wrapper around sentence-transformers for the two embedding models
under comparison (MiniLM vs BGE), plus per-chunk token accounting.

Deliberately does NOT talk to Chroma -- that's store.py's job (next
piece). This module answers exactly two questions for a batch of texts:
"what are their embeddings" and "were any of them truncated getting
there, and by how much". Keeping those together matters: the truncation
question is meaningless unless it's checked against the SAME model
object that produced the embeddings, using that model's own real
`max_seq_length` -- not a number copied from a doc or measured against
a different library's tokenizer (see DECISIONS.md's tokenization gate
entry for exactly how that distinction was missed once already here).
"""

from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-base-en-v1.5",
}

# BGE was trained asymmetrically: queries get a fixed instruction prefix at
# encode time, passages/documents do not. This is documented on the model's
# own card, not something either embedding model needs at all generically --
# MiniLM was trained symmetrically (no prefix for either side). Applying the
# wrong one (or skipping it for BGE) doesn't error, it just quietly changes
# what the embedding represents -- exactly the kind of silent mismatch this
# project keeps finding, so it's centralised here rather than left for
# retrievers.py to maybe remember later.
QUERY_PREFIXES = {
    MODELS["minilm"]: "",
    MODELS["bge"]: "Represent this sentence for searching relevant passages: ",
}


@dataclass
class TokenStats:
    text_index: int
    token_count: int
    truncated: bool


class Embedder:
    """
    One instance wraps one SentenceTransformer model. Construct one per
    model_id (from MODELS), not one shared instance -- max_seq_length and
    the query prefix are both model-specific, and mixing them up silently
    is exactly the failure mode this class exists to prevent.
    """

    def __init__(self, model_id: str, batch_size: int = 32):
        if model_id not in QUERY_PREFIXES:
            raise ValueError(
                f"Unknown model_id {model_id!r} -- add it to MODELS and "
                f"QUERY_PREFIXES first so its query-prefix behaviour is a "
                f"deliberate decision, not an accidental default of ''."
            )
        self.model_id = model_id
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_id)
        # Read at construction time from the model itself -- never hardcode
        # this number. It's exactly the value that was wrong (512 instead of
        # 256 for MiniLM) when taken from the raw AutoTokenizer instead.
        self.max_seq_length = self.model.max_seq_length

    def encode_passages(self, texts: List[str]):
        """Embed chunk text. No query prefix -- passages are encoded as-is."""
        return self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )

    def encode_queries(self, texts: List[str]):
        """Embed queries. Applies this model's query prefix, if it has one."""
        prefix = QUERY_PREFIXES[self.model_id]
        prefixed = [prefix + t for t in texts] if prefix else list(texts)
        return self.model.encode(
            prefixed, batch_size=self.batch_size, show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )

    def token_stats(self, texts: List[str]) -> List[TokenStats]:
        """
        True (non-truncated) token count per text, via this model's own
        tokenizer, compared against this model's own real max_seq_length --
        NOT against AutoTokenizer.model_max_length, which is a different
        number for at least one of these two models. `truncated` is exactly
        the thing that silently happens inside encode_passages/encode_queries
        above; this makes it visible and countable instead of invisible.

        add_special_tokens=True to match what SentenceTransformer's own
        internal tokenize() counts towards max_seq_length (it includes
        [CLS]/[SEP] in that budget for BERT-family models, not just content
        tokens).
        """
        stats = []
        for i, text in enumerate(texts):
            ids = self.model.tokenizer.encode(
                text, add_special_tokens=True, truncation=False
            )
            n = len(ids)
            stats.append(TokenStats(
                text_index=i, token_count=n, truncated=n > self.max_seq_length,
            ))
        return stats


if __name__ == "__main__":
    import glob
    import json

    chunk_texts = []
    for path in sorted(glob.glob("data/processed/chunks_*.jsonl")):
        for line in open(path, encoding="utf-8"):
            chunk_texts.append(json.loads(line)["text"])

    print(f"Loaded {len(chunk_texts)} chunks from data/processed/\n")

    for slug, model_id in MODELS.items():
        print(f"--- {slug} ({model_id}) ---")
        embedder = Embedder(model_id)
        print(f"real max_seq_length: {embedder.max_seq_length}")

        stats = embedder.token_stats(chunk_texts)
        counts = [s.token_count for s in stats]
        n_truncated = sum(1 for s in stats if s.truncated)
        counts_sorted = sorted(counts)
        n = len(counts_sorted)

        def pct(p):
            return counts_sorted[min(n - 1, int(p * n))]

        print(f"token count -- min {min(counts)}, median {pct(0.5)}, "
              f"p90 {pct(0.9)}, p95 {pct(0.95)}, max {max(counts)}")
        print(f"truncated at {embedder.max_seq_length} tokens: "
              f"{n_truncated}/{n} ({100 * n_truncated / n:.2f}%)\n")
