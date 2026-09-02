"""
check_tokenization.py
---------------------
Checks whether our RAG chunks are longer than the token limits
of the embedding models we plan to compare.

Why this matters:
A chunk may be ~1000 characters, but embedding models work in
TOKENS, not characters. If a chunk exceeds the model's maximum
input length, the model may truncate part of it.

We measure the actual truncation rate rather than assuming that
1000 characters is safe.
"""

from pathlib import Path
import sys

from transformers import AutoTokenizer

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from extract import extract_paper
from chunking import (
    fixed_size_chunks,
    recursive_chunks,
    section_aware_chunks,
)


PAPERS_DIR = PROJECT_ROOT / "data" / "papers"


MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}


def load_all_chunks():
    """Extract papers and generate all three chunking strategies."""

    all_chunks = []

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {PAPERS_DIR}"
        )

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")

        paper = extract_paper(str(pdf_path))

        all_chunks.extend(fixed_size_chunks(paper))
        all_chunks.extend(recursive_chunks(paper))
        all_chunks.extend(section_aware_chunks(paper))

    return all_chunks


def analyze_model(model_name, model_id, chunks):
    """Measure token lengths and truncation rate for one model."""

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    print(f"Loading tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Ask the tokenizer/model what its configured maximum is.
    max_length = tokenizer.model_max_length

    print(f"Tokenizer max length: {max_length} tokens")

    token_lengths = []

    for chunk in chunks:
        # IMPORTANT:
        # truncation=False means we measure the TRUE token length.
        token_ids = tokenizer(
            chunk.text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        token_lengths.append(len(token_ids))

    total = len(token_lengths)

    over_limit = sum(
        length > max_length
        for length in token_lengths
    )

    truncation_rate = over_limit / total * 100

    sorted_lengths = sorted(token_lengths)

    def percentile(values, p):
        index = int(len(values) * p / 100)
        index = min(index, len(values) - 1)
        return values[index]

    print()
    print(f"Chunks checked:       {total:,}")
    print(f"Minimum tokens:       {min(token_lengths):,}")
    print(f"Median tokens:        {percentile(sorted_lengths, 50):,}")
    print(f"90th percentile:      {percentile(sorted_lengths, 90):,}")
    print(f"95th percentile:      {percentile(sorted_lengths, 95):,}")
    print(f"Maximum tokens:       {max(token_lengths):,}")
    print(f"Over model limit:     {over_limit:,}")
    print(f"Truncation rate:      {truncation_rate:.2f}%")

    return {
        "model": model_name,
        "max_length": max_length,
        "total": total,
        "over_limit": over_limit,
        "truncation_rate": truncation_rate,
    }


def main():
    print("=" * 70)
    print("RAG CHUNK TOKENIZATION CHECK")
    print("=" * 70)

    chunks = load_all_chunks()

    print(f"\nTotal chunks: {len(chunks):,}")

    results = []

    for model_name, model_id in MODELS.items():
        result = analyze_model(
            model_name,
            model_id,
            chunks,
        )
        results.append(result)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"{'Model':<25}"
        f"{'Limit':>10}"
        f"{'Over limit':>15}"
        f"{'Rate':>12}"
    )

    print("-" * 62)

    for result in results:
        print(
            f"{result['model']:<25}"
            f"{result['max_length']:>10}"
            f"{result['over_limit']:>15,}"
            f"{result['truncation_rate']:>11.2f}%"
        )


if __name__ == "__main__":
    main()