# Decisions log

One entry per architectural choice: what was chosen, what was rejected, and the
evidence. This file is the interview script for the project — every entry
should be defensible from a number or a measurement, not a preference.

---

## Per-page layout detection instead of a fixed two-column assumption
**Date:** 1 Sep 2026

**Problem found:** `extract.py`'s original `_sort_blocks_by_reading_order`
assumed every paper is two-column and bucketed text blocks left/right of the
page midpoint. Measured directly against the corpus: 3 of 6 papers (sam3,
conformal_prediction, depth_anything_3) are single-column. On a single-column
page every block spans nearly the full width, so its horizontal centre sits
right on the midpoint, and sub-pixel width differences flip it between the
"left" and "right" bucket at random — scrambling paragraph order. Measured on
sam3 page 5: 68% of body blocks on the pages checked were reordered.

**Decision:** detect layout per PAGE (not per paper, since front matter or a
wide table can differ from the body): a block wider than 65% of the page width
is "full-width"; if most qualifying blocks (>15% of page width, to exclude
noise like stray page numbers) on a page are full-width, the page is
single-column and sorted by y0 alone. Otherwise, full-width blocks (headers,
wide figure captions, wide tables) are pulled out of the two-column sort and
re-inserted as page-width "bands" that split the column flow at their vertical
position, rather than being forced into an arbitrary left/right bucket.

**Rejected alternative:** keep the per-paper flag and just special-case the
three single-column papers by name. Rejected because it doesn't generalize —
a 7th paper (Mascagni, still missing) could be either layout, and a per-page
detector costs nothing extra to compute per page.

**Verification:** `src/audit_extraction.py` dumps per-page layout calls and
readable text samples to `reports/extraction_audit.txt` for manual review
against the source PDFs. Full-corpus layout tally after the fix: sam3
59 single / 18 double, conformal_prediction 34/17, depth_anything_3 23/9,
murali_latent_graph 0/12, sages_cvs_challenge 1/20, surgicalsam 0/18 — matches
the expected split (three single-column papers, three two-column) with no
manual per-paper flag.

---

## Deferred: multi-page ToC continuations and figure/table text are not fixed in extract.py
**Date:** 1 Sep 2026

**Problem found (manual review of `reports/extraction_audit.txt`):** two separate-looking
issues surfaced. (1) `_is_likely_toc_page` only catches a ToC's first page (looks for the
literal word "Contents"/"Table of Contents" at the top); conformal_prediction has a
multi-page ToC, and its continuation page slips through as an ordinary page. (2) Complex
figures (e.g. depth_anything_3 p.3) extract as a soup of individually-positioned short
text fragments — axis labels, legend entries, single characters — because PyMuPDF's text
layer has no concept of "these belong to one chart's y-axis". This is not a reading-order
bug; it is PyMuPDF exposing exactly what it sees (isolated positioned text runs) for
content that isn't linear prose to begin with.

**Decision: do not chase either as a new extract.py special case.** Both are instances of
the same underlying thing — linearized non-prose content — so one measurable signal
covers both instead of two page-type-specific heuristics (which is also how the existing
ToC heuristic's own docstring describes rejecting an earlier, broader version for
over-catching results tables). Added `single_token_line_ratio` to every `Chunk` in
`chunking.py`: the fraction of non-empty lines that are a single whitespace-separated
token. High for ToC entries (number + short title per line), figure-label soup (one
fragment per line) and dense results tables alike; low for real prose.

**Measured before committing to it:** across 210 non-ToC-flagged pages, the ratio is a
CONTINUUM (min 0.00, median 0.25, p75 0.48, p90 0.71, max 0.92) — not a clean bimodal
split. By hand inspection the top decile (>0.7) is almost entirely results tables, dense
figures, and (for conformal_prediction specifically) the exact ToC continuation page that
prompted this. There is no single threshold that cleanly separates "prose" from "not
prose" here.

**Because of that, the ratio is stored as metadata, not used to drop or filter anything
at chunking time.** Automatically dropping high-ratio chunks now would risk losing real
content that just happens to be table-adjacent or caption prose near a figure. The
decision of what to do with it is deferred, with a plan: exclude high-ratio chunks from
eval-question sampling (Day 3, since a question drafted from figure-label soup makes a
bad eval item) and consult it during retrieval failure analysis (Day 4) if a miss
correlates with high-ratio source chunks.

**Rejected alternative:** extend `_is_likely_toc_page` to also catch continuation pages
(e.g. by tracking "previous page was ToC-like"), and separately hand-write a
figure-detection heuristic. Rejected because it is two more special cases layered onto a
heuristic the file's own comments already flag as fragile, for a problem one general
per-chunk measurement already covers.

---

## Chunk identity: chunk_id, content_hash, section, char_start/char_end
**Date:** 1 Sep 2026

Every `Chunk` now carries: `chunk_id` (`{paper_slug}::{strategy}::{index:04d}`, stable and
human-readable in logs), `content_hash` (first 12 hex chars of SHA-256 of whitespace-
normalised text — lets a later run tell "this chunk's content changed" from "only its
list position shifted"), `section` (the detected header for section_aware chunks; was
computed and then discarded before this change — now populated for 100% of section_aware
chunks across all 6 papers, comfortably above the 80% target), and `char_start`/`char_end`
(offsets into the paper's joined text — makes page attribution exact rather than
approximated by a `find()` probe, and is what evidence-span ground truth in Day 3 will
anchor against). Verified globally unique across all 2346 chunks in the corpus (3
strategies x 6 papers).

---

## Fixed: recursive chunker's overlap was only applied in the hard-cut fallback
**Date:** 1 Sep 2026

**Problem:** `overlap=100` was accepted as a parameter by `fixed_size_chunks`,
`recursive_chunks`, and (via `_recursive_split`) `section_aware_chunks`'s
oversized-section path, but only the last-resort hard-character-cut branch
(no separator left that fits) actually implemented it. The normal
paragraph/sentence-boundary packing path — which handles nearly every chunk
in this corpus — produced adjacent chunks sharing zero text, despite the
parameter's name and default claiming otherwise.

**Fix:** overlap moved out of `_recursive_split` entirely (which now only
produces boundary-respecting, non-overlapping pieces) and into one new
function, `_stitch_overlap`, applied once by each caller to the final flat
piece list — so it's the same mechanism regardless of which boundary rule
produced a given piece. It borrows a word-boundary-trimmed tail of the
*original* (pre-stitch) previous piece, so overlap never compounds across a
long run of chunks. `section_aware_chunks` deliberately only stitches within
one section's own sub-split pieces, never across a section boundary —
bridging sections would undo the entire point of section-aware chunking
(chunks that don't straddle two different arguments).

**Verified, not just asserted:** measured longest shared substring at every
chunk boundary, corpus-wide.
- `recursive_chunks`: 0/288 zero-overlap boundaries on sam3, 0/152 on
  conformal_prediction; mean ~94 shared characters (target was 100, small
  loss expected from word-boundary trimming).
- `section_aware_chunks` oversized-section sub-splits: overlap present and
  in the same 85-99 char range; confirmed 0 leaks across 10 sampled
  cross-section boundaries on sam3 (checked separately, since overlap must
  NOT bridge sections).
- Chunk size distributions moved as expected: recursive max grew from 1000
  to ~1100 chars, section_aware max from ~1240 to ~1299 — exactly
  `chunk_size + overlap`, confirming overlap is applied and capped, not
  unbounded.

**Side finding while verifying (not fixed, just noted):** a handful of
"zero overlap" boundaries inside surgicalsam's Method-heavy results section
turned out not to be a stitching bug — `_looks_like_header` matched the bare
word "Method" as a new section header 6 separate times, because it's a
repeated column label inside a comparison table, not a real section
heading. Each is treated as its own independent section, so overlap
correctly does not bridge between them; they just happen to share a
section name, which could look like leakage from `.section` alone. This is
the same known header-regex fragility already noted in the file's own
comments, showing up concretely — not a new problem, and not fixed here for
the same reason the ToC-continuation and figure-soup cases weren't: one
more special case bolted onto a heuristic already flagged as approximate,
for a pattern `single_token_line_ratio` already flags as non-prose-like.

---

## chunk_size=1000: not validated, and two concrete issues found while checking
**Date:** 1 Sep 2026
**Prompted by:** Ritik asking why 1000 specifically, and checking the reasoning with GPT.

GPT's answer was directionally right (1000 is an unvalidated starting baseline, chunk
size trades off context-per-chunk against retrieval noise, and the right move is to
measure alternatives once the eval pipeline exists rather than sweep now) but missed two
things that are worth acting on immediately, since they were cheap to fix before Day 2:

**1. The three strategies weren't even using the same size cap.** `fixed_size_chunks`
and `recursive_chunks` both defaulted to `chunk_size=1000`, but `section_aware_chunks`
defaulted to `max_chunk_size=1200` -- a pre-existing mismatch from Day 0, not something
introduced later. This confounds Day 4 Stage A (comparing chunking strategies): any
apparent win for section-aware could partly be "it was allowed bigger chunks," not "its
boundary logic is better" -- exactly the "longer chunks win for free" trap already
flagged for eval-set ground truth in Part III of the plan, showing up one level earlier,
in the chunking parameters themselves. **Fixed:** `section_aware_chunks`'s cap lowered to
1000 to match the other two. Re-verified section coverage still 100% across all 6 papers
after the change; average chunk size across all three strategies now sits in the same
~1000-1050 char band (was ~1140-1240 for section_aware before).

**2. Possible embedding-model truncation at this chunk size -- flagged, not yet confirmed.**
`sentence-transformers/all-MiniLM-L6-v2` truncates input at 256 tokens; `BAAI/bge-base-en-v1.5`
at 512. Using the standard ~4 characters/token rule of thumb, our chunks (1000-1100 chars,
1100 with overlap) land at roughly 250-275 estimated tokens -- right at or just past
MiniLM's limit -- and academic/technical text typically tokenizes to MORE tokens per
character than generic English (rare technical terms fragment into more subword pieces),
so the real count is likely higher than this estimate, not lower.

**Not yet verified with the real tokenizer** -- attempted to check with the actual
MiniLM/BGE tokenizers from both this project's device shell and a second, separate
sandboxed environment; both had network access blocked to huggingface.co, so this remains
an estimate, not a measurement. Added as a hard gate for Day 2: tokenize every chunk with
both real tokenizers before trusting any MiniLM-vs-BGE comparison, and report the
truncation rate for each model. If MiniLM is truncating a meaningful fraction of chunks,
that is itself a legitimate, concrete, mechanistic reason it could underperform BGE --
separate from and more specific than "BGE is just a better embedding model" -- and worth
reporting as its own finding rather than folding silently into the aggregate Recall@K
comparison.

**Also added to Day 4 (not done now):** once Stage A picks a winning chunking strategy,
run that strategy at 2-3 additional chunk sizes (e.g. 600, 1000, 1500) against the SAME
locked benchmark, as a cheap, scoped answer to "is 1000 actually a good choice" --
GPT's suggestion, but scoped to the winning strategy only rather than a full
strategy x size grid, to avoid exploding Stage A's already-staged design.

## Tokenization gate (Day 2 pre-check): measured, one precise number still open
**Date:** 1 Sep 2026
**Prompted by:** the previous entry's open item, checked for real by Ritik on his machine.

Ritik ran the real tokenizers (`check_tokenization.py`, `transformers.AutoTokenizer`,
`truncation=False`) against all 2,492 chunks. Result: both `all-MiniLM-L6-v2` and
`bge-base-en-v1.5` report `model_max_length=512` via `AutoTokenizer` -- not 256 for
MiniLM, contradicting the estimate in the previous entry. Token length distribution:
min 26, median 265, p90 372, p95 411, max 625; 28/2492 chunks (1.12%) exceed 512, for
both models.

That measurement is real and correctly done for what it measures: the raw HF
tokenizer's own configured limit. It does not fully close the gate yet, because of what
Day 2's actual embedding code will call.

**The remaining nuance.** `sentence-transformers` wraps a raw tokenizer with its OWN
`max_seq_length`, read from a `sentence_bert_config.json` file bundled per model --
separate from, and enforced instead of, the tokenizer's own `model_max_length` at encode
time (`SentenceTransformer.encode()` truncates at `self.max_seq_length`). For
`all-MiniLM-L6-v2` this value is widely documented as 256, not 512. If that holds, the
1.12% figure is the truncation rate against the WRONG limit for this model -- given a
measured median of 265 tokens, roughly half the chunks would sit at or past 256, not 28
of them.

This is a specific, falsifiable claim about one config value, not a hedge, and the fix is
one more precise check, not re-litigating the finding above. Could not confirm it
directly this round either: re-tried huggingface.co from this project's device shell
today, still a 403 -- the network restriction noted last entry isn't a one-off.

**Exact check still needed** (run where check_tokenization.py already ran):
```python
from sentence_transformers import SentenceTransformer
for m in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5"]:
    print(m, SentenceTransformer(m).max_seq_length)
```
Then recompute the truncation rate against THAT number for MiniLM specifically (BGE's
raw-tokenizer figure likely stands unchanged, but confirm rather than assume -- print
both). Gate stays open -- not logged as PASS -- until this number is measured and the
truncation rate is recomputed against it. Whichever way it lands (real problem or the
256 figure turns out to not apply to this checkpoint), that result gets logged here
before Day 2 embedding runs, not folded silently into "gate: PASS."

**RESOLVED -- 1 Sep 2026.** Ritik ran the exact check on his machine:

    sentence-transformers/all-MiniLM-L6-v2 -> max_seq_length: 256
    BAAI/bge-base-en-v1.5 -> max_seq_length: 512

Confirmed directly, not estimated: MiniLM's real, encode-time limit is 256 tokens --
half of BGE's 512, and half of what `AutoTokenizer.model_max_length` reported. The
distinction predicted above holds. Given the already-measured distribution (median 265
tokens across all 2,492 chunks), roughly half of MiniLM's chunks sit at or past its real
limit -- materially different from the 1.12% figure the raw-tokenizer check found, which
was answering a question about the wrong object (`AutoTokenizer`, not the
`SentenceTransformer` wrapper Day 2 actually calls). Exact count against 256 (rather than
the ~50% read off the percentiles) is a nice-to-have, not a blocker -- logged here once
available.

**Design implication -- gate CLOSED, chunk_size stays at 1000.** The fix here is NOT to
shrink chunk_size to fit MiniLM. Both models embed the exact same chunks (Day 2's design
is 3 chunkings x 2 embedding models over identical chunk boundaries) -- shrinking chunks
to flatter MiniLM would hide the very effect worth measuring: that MiniLM's own 256-token
ceiling is a real, mechanistic disadvantage on chunks sized for retrieval quality in
general, independent of any chunking-strategy choice. That is exactly the kind of
concrete, reportable finding flagged as worth surfacing rather than folding silently into
Recall@K.

The actual action item is in `embed.py` (Day 2): tokenize every chunk with each model's
OWN tokenizer before encoding, compare against that model's OWN real `max_seq_length`
(256 for MiniLM, 512 for BGE -- read from `SentenceTransformer.max_seq_length` at
runtime, not hardcoded), and store `token_count` + `truncated` as chunk metadata
alongside the embedding. That turns "MiniLM probably truncates some chunks" from a
one-off side-investigation into a per-chunk, per-model measured fact that Day 4's
results can directly cite and correlate with retrieval misses -- e.g. "of MiniLM's
Recall@K misses, X% were on chunks flagged truncated" is a stronger, more specific claim
than "BGE scored higher."

## Day 1.5: corpus persisted (`build_corpus.py`)
**Date:** 1 Sep 2026

Extraction + all three chunking strategies now run exactly once and are cached to
`data/processed/chunks_{strategy}.jsonl` + `manifest.json`. Nothing downstream re-parses
a PDF from here on.

`manifest.json` records a `code_version` -- SHA-256 (12 hex chars) of `extract.py` +
`chunking.py`'s own source, concatenated -- deliberately NOT a git commit hash, since the
working tree has had uncommitted changes through all of Day 1 and a commit hash would
silently point at a stale prior version. `git_commit` is recorded too, but only as
best-effort provenance (with a `-dirty` suffix when the tree has uncommitted changes),
never as the thing anything gates on.

Each strategy's `chunk_size`/`overlap` (or `max_chunk_size`/`overlap` for section_aware)
is written into the manifest explicitly rather than left to be inferred from
`chunking.py`'s current defaults -- so the manifest alone answers "what parameters
produced this file" even if the code's defaults change later.

**Verified:**
- 851 fixed_size + 798 recursive + 843 section_aware = 2,492 chunks total -- exactly
  matches the total Ritik's independently-written `check_tokenization.py` measured on his
  machine in the previous entry. Two different scripts, run in two different
  environments, agree to the chunk.
- 0 `chunk_id` collisions across all 2,492 records checked globally (chunk_id is
  `paper_slug::strategy::index`, so global uniqueness -- not just per-file -- is the
  correct check and was run explicitly rather than assumed).
- section_aware: 100.0% of chunks carry a non-null `section` (exceeds the >=80% Day 1
  gate). fixed_size/recursive: 0.0%, as expected -- they don't track sections.
- Per-paper section_aware chunk counts match the per-paper figures already verified
  earlier in Day 1 exactly (conformal_prediction 153, depth_anything_3 109,
  murali_latent_graph 66, sages_cvs_challenge 146, sam3 295, surgicalsam 74) --
  build_corpus.py calling the same functions with the same params reproduces them, as it
  should.

`data/processed/` is (and was already) in `.gitignore` -- intentional, not an oversight.
The directory is fully reproducible by re-running `build_corpus.py` against the source
PDFs + current code, and the `manifest.json` records exactly which code version produced
it; committing multi-MB jsonl files with full chunk text alongside a hash that can
regenerate them byte-for-byte would be redundant.

## `embed.py`: model-specific query prefix for BGE, and self-tested (not yet run for real)
**Date:** 1 Sep 2026

`BAAI/bge-base-en-v1.5` was trained asymmetrically: at encode time, queries are meant to
get a fixed instruction prefix (`"Represent this sentence for searching relevant
passages: "`), while passages/documents are encoded bare -- documented on the model's own
card, not a generic requirement of embedding models in general. `all-MiniLM-L6-v2` was
trained symmetrically and takes no such prefix on either side. Skipping this for BGE
doesn't error -- it just quietly changes what the resulting embedding represents, which
is exactly the kind of silent mismatch this project keeps finding elsewhere (the
tokenization gate above, the two-column extraction bug in Day 1). `embed.py` centralises
this in one `QUERY_PREFIXES` dict keyed by model_id, applied only in `encode_queries`,
never in `encode_passages` -- so retrievers.py (Day 2, next) can't forget it later.

`embed.py` also reads `max_seq_length` from each model object itself at construction
time (never hardcoded), and exposes `token_stats()` -- true (non-truncated) token count
per text via that model's own tokenizer, compared against that model's own real
`max_seq_length` -- the exact per-chunk instrumentation the tokenization gate entry
above called for.

**Verified so far:** all internal logic (unknown-model-id rejected, max_seq_length read
per-model, truncation flagged against the correct per-model threshold, query prefix
applied for BGE only and never for passages) tested against a fake `SentenceTransformer`
standing in for the real package -- confirms the CODE's logic is correct, independent of
network access.

**RUN FOR REAL -- 1 Sep 2026.** Ritik ran `python src/embed.py` against all 2,492 chunks
with the real package on his machine:

    minilm (max_seq_length=256): token count -- min 26, median 265, p90 372, p95 411, max 625
      truncated at 256 tokens: 1373/2492 (55.10%)
    bge (max_seq_length=512): token count -- min 26, median 265, p90 372, p95 411, max 625
      truncated at 512 tokens: 28/2492 (1.12%)

Tokenization gate CLOSED, final numbers. The "roughly half" estimate from the previous
entry lands at 55.10% exactly -- MiniLM truncates the back end of more than half its
chunks at its real 256-token limit; BGE truncates only 1.12% at its 512-token limit, off
the SAME chunk set (identical token-count distribution for both rows above, as it must
be -- same chunks, same tokenizer family). This is now stored per-chunk, per-model via
`token_stats()` and will be written into Chroma metadata in `store.py` (next), so Day 4
can report, concretely, what fraction of MiniLM's retrieval misses land on chunks
flagged truncated -- rather than treating a possible MiniLM-vs-BGE gap as unexplained.

## `store.py`: two Chroma-specific gotchas found and fixed before they could bite
**Date:** 1 Sep 2026

Writing `store.py` (the six Chroma collections: 3 chunkings x 2 embedding models) surfaced
two version-specific behaviours of `chromadb==1.5.9` (the exact version pinned in
`requirements.lock.txt`), checked directly against the real library rather than assumed
from docs -- same standard as everything else logged here.

**1. Chroma's metadata store rejects an explicit `None` value.** Tried
`collection.add(metadatas=[{"section": None, ...}])` directly against chromadb 1.5.9:

    TypeError: argument 'metadatas': Cannot convert Python object to MetadataValue

`section` is `None` for `fixed_size`/`recursive` chunks by design (`chunking.py` only
tracks it for `section_aware`). Confirmed separately that OMITTING the key entirely
(rather than setting it to `None`) is accepted fine, and that a mixed batch -- some
records with the key, some without -- works too. `store.py`'s `_clean_metadata` drops the
key when `record["section"]` is `None` rather than passing it through. Would otherwise
have crashed `store.py` on its very first `.add()` call for two of the three strategies.

**2. `get_or_create_collection`'s default `embedding_function` is Chroma's own
`DefaultEmbeddingFunction`.** Never actually invoked here, since every `.add()`/`.query()`
call in `store.py` always supplies its own `embeddings=` from `embed.py`'s `Embedder` --
but passed `embedding_function=None` explicitly anyway when creating each collection.
Cheap insurance against exactly the kind of silent substitution this project keeps
finding elsewhere (the wrong tokenizer limit, the missing BGE query prefix): if some
future code path ever called `.add()` without `embeddings=`, an unset embedding_function
would silently compute one with Chroma's own default model instead of erroring loudly.

**Verification approach.** Neither of my own environments can reach huggingface.co (same
restriction as the tokenization gate), so I couldn't run `store.py` against the real
MiniLM/BGE models myself. What I could do, and did: installed `chromadb==1.5.9` (matching
`requirements.lock.txt` exactly) in my own sandbox -- PyPI is reachable even though
huggingface.co isn't -- built synthetic chunk records matching the exact real JSONL
schema (including a realistic mix of `section=None` and real section strings), swapped in
a fake `SentenceTransformer` (same technique as `embed.py`'s self-test), and ran the ACTUAL
`store.py` code end-to-end against the REAL Chroma library: all six collections built with
correct names and counts, `section` correctly omitted/present, `token_count`/`truncated`
present in metadata, re-running a collection build does not duplicate records, and a
forced small batch size correctly exercises the batching loop. This confirms the code's
logic against the real dependency, short of the real embedding models themselves, which
still needs Ritik's machine.

`python src/store.py` is the next thing to run there; paste back the per-collection
truncation rates and the final total-vs-expected count.

## `store.py` run for real: six collections built, cross-checked against embed.py's numbers
**Date:** 1 Sep 2026

`python src/store.py` (after setting `EVIDENCERAG_STORE`, see below) built all six Chroma
collections against the real corpus:

    fixed_size__minilm:     851 chunks (432/851 = 50.76% truncated at 256 tokens)
    recursive__minilm:      798 chunks (484/798 = 60.65% truncated at 256 tokens)
    section_aware__minilm:  843 chunks (457/843 = 54.21% truncated at 256 tokens)
    fixed_size__bge:        851 chunks (  7/851 =  0.82% truncated at 512 tokens)
    recursive__bge:         798 chunks ( 10/798 =  1.25% truncated at 512 tokens)
    section_aware__bge:     843 chunks ( 11/843 =  1.30% truncated at 512 tokens)
    Total vectors indexed: 4984 (expected 4984)

**Cross-check against embed.py's earlier aggregate run:** 432+484+457 = 1373 for MiniLM,
matching the 1373/2492 (55.10%) `embed.py` measured across the whole corpus in one shot,
exactly. 7+10+11 = 28 for BGE, matching 28/2492 (1.12%) exactly. Two different scripts,
two different runs, same underlying per-chunk numbers -- the kind of consistency check
this project keeps running before trusting a result, and it held.

**New, more granular finding:** truncation is NOT uniform across chunking strategies for
MiniLM -- recursive runs hottest (60.65%), section_aware in the middle (54.21%),
fixed_size lowest (50.76%). Plausible mechanism, not yet confirmed: recursive's
boundary-respecting splitter tends to fill each chunk closer to the 1000-char cap
(it only cuts short at a paragraph/sentence boundary, otherwise keeps packing), while
fixed_size's raw sliding window and section_aware's header-triggered splits both produce
more chunks that end early for structural reasons unrelated to hitting the size cap.
Not chased further now -- flagged as a real, secondary data point for Day 4's writeup
rather than a Day 2 blocker.

**Compute-cost data point (the plan's Day 2 gate explicitly asked for this):** MiniLM
embedded all 2,492 chunks in ~104s total (36+33+35s across the three collections); BGE
took ~880s (~14.7 min: 366+259+255s) -- roughly **8.5x slower** for the same corpus, on
top of already truncating far less content. That's the concrete other half of the
MiniLM-vs-BGE tradeoff: BGE is far more complete per chunk, at a real and measurable
compute cost, on CPU. Both numbers are from Ritik's actual machine, not estimated.

**`EVIDENCERAG_STORE` was NOT set** when we checked (`echo $env:EVIDENCERAG_STORE` printed
nothing) -- confirmed and fixed before running store.py for real: `$env:EVIDENCERAG_STORE`
set for the current session plus `setx` for future terminals, both pointing at
`C:\ml-cache\evidencerag-chroma`, keeping Chroma's SQLite files out of the OneDrive-synced
repo folder per Day 0's original warning.

Day 2 gate (six collections, counts match JSONL line counts) -- PASSED. Retrieval
smoke-testing (the three retrievers) still pending `retrievers.py`.

## `retrievers.py`: one interface, three implementations, integration-tested with hand-verifiable math
**Date:** 1 Sep 2026

Implements `DenseRetriever`, `BM25Retriever`, and `HybridRRFRetriever` behind one
`Retriever`-shaped interface (`retrieve(query, k) -> List[Hit]`, `Hit` = chunk_id, text,
metadata, score, rank), exactly as scoped in PLAN.md. `BM25Retriever` is built directly
from `data/processed/chunks_{strategy}.jsonl` (the same chunk set dense retrieval for
that strategy uses), independent of any embedding model. `HybridRRFRetriever` fuses a
dense + BM25 pair built over the SAME strategy -- fusing across two different strategies
would mean combining two different `chunk_id` spaces, which is meaningless -- pulling
`candidate_k` (wider than the requested `k`) from each side before re-ranking by RRF
score: `1/(k_rrf + rank)` per list, summed, `k_rrf=60` (Cormack et al. 2009 default),
matching PLAN.md exactly (rank fusion, not score fusion -- see the existing writeup on
why, and the old `RAG` project's union-and-truncate mistake this avoids repeating).

Chroma's "distance" under `hnsw:space="cosine"` (how `store.py` creates every collection)
is cosine distance, `1 - cosine_similarity` -- confirmed directly against chromadb 1.5.9
(identical vectors -> distance 0.0; orthogonal vectors -> distance 1.0). `DenseRetriever`
reports `1 - distance` as its score, so a HIGHER score means "more relevant" consistently
across all three retrievers, matching BM25's own score direction.

**Verified** with a full integration test using the REAL `chromadb==1.5.9` and REAL
`rank_bm25==0.2.2` (both exactly matching `requirements.lock.txt`), a small synthetic
corpus, and a fake embedder whose output vectors were deliberately hand-picked
(keyword -> fixed embedding) so every score is independently computable by hand rather
than just "looks plausible":
- Dense: query "dogs" against a doc with only "dogs" (score exactly 1.0000) and a doc
  with "cats"+"dogs" both (score 0.7071, matching `cos([0.7,0.7,0],[0,1,0])` computed by
  hand) -- ranked correctly, scores match to 1e-3.
- BM25: same query, correct document ranked first, second document (single "dogs"
  mention) ranked second, both with strictly positive scores.
- Hybrid: RRF scores for the top two results matched the formula `2/(60+rank)` to
  1e-9 -- not just "reasonable-looking," exactly correct.
- `build_retrievers` factory (the (dense, bm25, hybrid) convenience constructor for a
  given strategy+model) verified end-to-end.

**Worth knowing, not a bug:** my first attempt at this synthetic corpus (4 documents,
"dogs" appearing in exactly 2 of them) gave BM25 a score of exactly 0.0 for every
document. Not a bug in `retrievers.py` or `rank_bm25` -- classic Okapi IDF is
`log((N - n + 0.5)/(n + 0.5))`, which is exactly 0 when a term appears in precisely half
the corpus (here N=4, n=2 -> log(2.5/2.5) = 0). `rank_bm25.BM25Okapi` floors NEGATIVE idf
values at `epsilon * average_idf` but does not floor an exact zero -- confirmed by reading
its source directly. Fixed by widening the synthetic corpus to 6 documents so "dogs"
sits at 2/6, not 2/4. This is a real property of classic BM25 worth knowing (a term common
enough to appear in >=half the corpus can contribute nothing, or even get a small floored
positive score if slightly over half), but essentially never triggered on the real
800-1000-chunk-per-strategy corpus, where no real query term will appear in anywhere near
half of all chunks.

Not yet run against the real corpus/collections -- needs Ritik's machine (real MiniLM
weights for query embedding). Running `python src/retrievers.py` prints the top-5 from
all three retrievers for the five hand-written probe queries from PLAN.md's Day 2 smoke
test, side by side, ready to eyeball.

## `retrievers.py` run for real: mechanics confirmed correct, 3 of 5 probe queries were badly worded (mine)
**Date:** 1 Sep 2026

`python src/retrievers.py` (recursive strategy, MiniLM) ran the five Day 2 smoke-test
probes against the real corpus. Two came back clean: "SA-Co benchmark" surfaced SAM 3
chunks that literally discuss the SA-Co/VEval benchmark for all three retrievers, and
"how is depth predicted from a single camera" surfaced Depth Anything 3 passages about
predicting depth maps vs. disparity and single-view geometry -- exactly on topic.

The other three ("how many images were used to train the model", "what is a known
limitation of the approach", "how do surgical phase recognition and depth estimation
relate") came back looking noisy -- dense, BM25 and hybrid each surfacing chunks from
different papers with no clear agreement. **This is my own mistake, not a retrieval bug.**
PLAN.md only specified categories for these three ("one numerical," "one about a
limitation," "one cross-paper") and left the exact wording to whoever wrote
`retrievers.py`'s smoke test -- I filled them in with bare, unscoped phrasing ("the
model", "the approach") without naming which of the corpus's 6 papers each one meant.
With 6 different papers/models in the corpus, an unscoped "the model" has no single
correct answer, so no retriever -- dense, BM25, or hybrid -- can be faulted for not
guessing which one was intended. Fixed by rewriting the three probes to name a specific
paper or topic ("...to train Depth Anything 3", "...conformal prediction...", "...surgical
scene understanding") so each has one intended answer and is actually diagnostic.

**Verified the RRF math is correct on the real run, not just on synthetic data.** For
query "SA-Co benchmark", chunk `sam3::recursive::0024` was dense rank 3 (score 0.5883)
and BM25 rank 1 (score 10.6910); by hand: 1/(60+3) + 1/(60+1) = 0.015873 + 0.016393 =
0.032266, matching the printed hybrid score of 0.0323 for that chunk (which the fusion
correctly promoted to hybrid #1). For the (badly-worded, now-fixed) "how many images..."
query, hybrid's #1 (`sam3::recursive::0174`) was NOT in either individual retriever's
printed top-5 -- not a bug: with `candidate_k=50` under the hood, a chunk ranked
moderately in BOTH lists (e.g. ~rank 10-15 in each) can out-score a chunk that's #1 in
only one list, because RRF sums contributions across lists. That is exactly RRF's
intended behavior -- reward broad agreement over single-signal dominance -- not evidence
of malfunction.

**One concrete, worth-tracking finding, sharper than "the probe was ambiguous":** for
the (now-fixed) cross-paper probe, the top dense/BM25/hybrid hit
(`murali_latent_graph::recursive::0059`) was a **bibliography citation entry** ("[32] T.
Czempiel, M. ... phase recognition in cataracts videos," in MICCAI. Springer, 2018, pp.
265-272."), not body prose discussing the relationship. A reference-list line shares
surface vocabulary with the query (it names the exact topic) without being useful
evidence to cite as an answer. This is a real, specific, plausible failure mode for BOTH
dense and lexical retrieval (a citation line is topically on-target by construction) --
and it is exactly the class of problem a cross-encoder reranker (Day 3) is well-suited
to catch, since it scores the full (query, chunk) pair jointly rather than by vocabulary
or embedding overlap alone. Flagged here to check specifically once `rerank.py` exists:
does reranking demote reference-list chunks relative to dense/BM25's raw ranking.

**Verdict, and why Day 2 closes here rather than tuning further:** implementation
mechanics -- PASS (both clean probes look right, the RRF math checks out by hand on the
real run, hybrid's "surprising" result on the noisy query is explained, not concerning).
Retrieval QUALITY -- not judged now, on purpose. Five hand-picked queries eyeballed by
a human are not a benchmark, and adjusting anything based on how these five happen to
look would risk quietly overfitting to them before the actual eval set (Day 3-4) exists
to measure against. Day 2 is done; Day 3 (reranker) is next.

---

## `rerank.py`: written as prep work, ahead of the benchmark it needs to be judged against

**What it is.** `CrossEncoderReranker`, wrapping `cross-encoder/ms-marco-MiniLM-L-6-v2`
via `sentence_transformers.CrossEncoder`. `.rerank(query, candidates, k)` scores every
`(query, candidate.text)` pair through the cross-encoder (query and chunk attend to each
other jointly, unlike the bi-encoder retrievers in `retrievers.py`, which embed each side
independently and never let one see the other's actual tokens), sorts by that score, and
returns the top `k` as `RerankedHit` objects -- each carrying `prior_rank` (its rank
before reranking, from whichever first-stage retriever produced it) alongside its new
`rank`, so a human or eval harness can see exactly how far reranking moved each chunk.
`rerank_pipeline(retriever, reranker, query, candidate_k=20, k=5)` is a convenience that
pulls `candidate_k` from any of `retrievers.py`'s three retrievers (dense, BM25, or
hybrid -- reranking doesn't care which produced the candidates) and reranks down to `k`.
20/5 matches PLAN.md Day 5's own numbers ("top-20 hybrid candidates, keep 5").

**Why this file exists before Day 3 is done.** PLAN.md's actual next step is the QA
benchmark (previous entry's correction), not this. But Ritik asked to start on the
reranker directly, and unlike Day 4 Stage D's actual QUESTION -- does reranking improve
Recall@k, which genuinely cannot be answered without `qa_gold.jsonl` to measure
against -- the MODULE ITSELF is self-contained, testable code with no dependency on the
benchmark existing yet. Same category of work as writing `embed.py`/`store.py` before Day
2's own gate had real numbers: build the piece, verify it does what it claims
mechanically, and be explicit that "mechanically correct" is not the same claim as
"helps."

**Integration-tested** (huggingface.co is still blocked from both my sandbox and the
device VM, so the real network-gated model weights can't load either place): installed
the exact pinned `sentence-transformers==6.0.1` from PyPI first and confirmed the REAL
`CrossEncoder.__init__(model_name_or_path, ...)` / `.predict(inputs, ...) -> np.ndarray`
signatures via `inspect.signature` -- rerank.py's calls match exactly. Then substituted a
fake `sentence_transformers` module (same technique as embed.py's self-test) with a
deterministic keyword-overlap fake `CrossEncoder.predict` and a bag-of-keyword-counts
fake `SentenceTransformer.encode`, and ran the real `chromadb==1.5.9` + real
`rank_bm25==0.2.2` + `retrievers.py`'s actual `build_retrievers()` against a synthetic
6-chunk corpus. Checks: top-k truncation and hand-verified score ties resolve correctly;
`prior_rank` is copied from each candidate's OWN rank from its source retriever, never
recomputed; empty candidate list returns `[]` without calling the model; `rerank_pipeline`
produces byte-identical output to a manual `retrieve()` + `rerank()` call; the full
`build_retrievers()` -> `hybrid.retrieve()` -> `CrossEncoderReranker.rerank()` chain runs
end to end with real Chroma/BM25 underneath and sane per-stage latency numbers. All
checks passed. Not yet run against the real model weights or the real corpus -- that's
Ritik's machine, same pattern as every other module this project.

**One thing worth flagging about the fake scorer, not the real code**: my synthetic
cross-encoder scored a chunk that name-checked "Depth Anything 3" (the model name in the
query) above a chunk that actually described the GPU training details being asked about
but never used the model's name. That's an artifact of a crude keyword-overlap fake, not
a claim about how the real cross-encoder will rank anything -- flagged here only so it
doesn't get mistaken later for evidence about real reranking behavior.

---

## `draft_qa.py` / `eval/span_utils.py`: Day 3 Step 1, five design decisions PLAN.md leaves open

**What it does.** Drafts `eval/qa_draft.jsonl` -- unreviewed GPT-4o-mini-proposed QA
candidates sampled from the section-aware chunk set, per PLAN.md Day 3 Step 1. It never
writes `qa_gold.jsonl`: Step 2 (entirely manual review) is what turns a trusted subset of
this file into the actual locked benchmark. Only the 90 answerable questions across five
types are drafted here (factual, numerical, method, comparative, limitation) -- the 15
unanswerable questions are Step 3, hand-written by Ritik, since an LLM asked to write "a
question this corpus can't answer" reliably produces obviously-absurd ones rather than
the plausible-but-absent questions that make an abstention measurement meaningful.

PLAN.md specifies WHAT to draft (the counts, the type quota, the verbatim-span
requirement) but leaves HOW underspecified in five places. Each is a real judgment call,
made explicitly rather than silently:

1. **Per-paper quota.** 30/20/20/10/10 totals, but not how those split across six papers
   ranging from 295 section-aware chunks (sam3) to 66 (murali_latent_graph). Split
   proportional to each paper's SHARE OF SECTION-AWARE CHUNKS -- the exact population
   being sampled from, not a proxy like page count -- via the largest-remainder
   (Hare-Niemeyer) apportionment method: floor each paper's exact proportional share,
   then hand the few leftover slots to whichever papers had the largest fractional
   remainder, tie-broken by paper name for a deterministic re-run.

2. **Which chunk gets which type.** A chunk with no digits cannot honestly support a
   numerical question. `_looks_numerical` (digit run of 2+, or a percent sign, with
   citation brackets like `[12]` stripped first so a reference number is never mistaken
   for real content) and `_looks_limitation` (section header matching
   limitation/discussion/conclusion/future work, or the word "limitation" in the body)
   bias SAMPLING ORDER toward promising chunks first, falling back to the rest of that
   paper's pool if the promising ones run out before quota is met. They never gate
   acceptance -- GPT can still decline, or the span check can still fail, on a
   heuristically-promising chunk, in which case the next candidate is tried. Step 2's
   human review is the real filter; this only stops Step 1 from wasting most of its calls
   on chunks that can never honestly support the requested type.

3. **Comparative questions need two chunks from two papers that are ACTUALLY related.**
   Rather than hand-pick topic pairs (bakes in my own assumptions, not a measured fact)
   or compare raw vocabulary overlap (exactly the "reuses the query's words without
   answering it" trap rerank.py's docstring describes), `_build_comparative_candidates`
   reuses Day 2's ALREADY-BUILT `section_aware__minilm` Chroma collection: pulls every
   chunk's own embedding via one `collection.get(include=["embeddings"])` call, computes
   cosine similarity between every cross-paper pair with plain numpy (843 chunks -> ~355k
   pairs, trivial), and ranks candidates by that similarity -- the exact same
   dense-similarity signal `DenseRetriever` already uses for retrieval, repurposed here to
   find which chunks from different papers are topically close enough to support a real
   comparative question, capped at 3 pairs per paper-pair so all 10 can't come from a
   single pair of papers that happen to be very similar.

4. **Schema extension for comparative.** PLAN.md's Part III schema assumes one paper / one
   gold_span per question. A comparative question needs evidence from two different
   papers, so for `qtype=="comparative"` only: `paper` becomes a 2-element list,
   `gold_pages` a 2-element list of per-paper page lists, `gold_span` a 2-element list of
   per-paper spans -- EACH verified as a literal substring of its OWN paper's chunk,
   independently (a response with one fabricated span and one real one is rejected as a
   whole, not half-accepted). Every other qtype keeps PLAN.md's exact single-value schema.
   Flagging this now so Day 4's `ir_metrics.py` doesn't discover it as a surprise: its
   Recall@K hit rule will need to handle comparative's list-shaped fields.

5. **Span verification normalization** lives in a new small shared file,
   `eval/span_utils.py` (`normalize_for_span_match`, `span_in_text`), imported by both
   `draft_qa.py` (today's draft-time check) and Day 4's future `ir_metrics.py` (the actual
   Recall@K hit rule). Both are the exact same operation -- "is this span really in this
   text" -- performed at two different times against two different texts; if they used
   two independently-written normalization routines, they could quietly drift apart in a
   way that corrupts the eval metrics for a reason invisible to anyone reading them.
   Implements PLAN.md's own spec exactly ("collapse whitespace, strip soft hyphens and
   ligatures, lowercase"): `unicodedata.normalize("NFKC", ...)` folds ligatures -- and this
   is not a hypothetical concern, `chunks_section_aware.jsonl` genuinely contains the
   literal single-codepoint "fi" ligature glyph (U+FB01) in "Quantification" from
   conformal_prediction's own extracted PDF text -- plus an explicit U+00AD (soft hyphen)
   strip, since NFKC does not fold that on its own.

**Integration-tested** (no OpenAI API key or spend of Ritik's money involved in testing
logic that doesn't need the real model): installed the real, exact pinned `openai==1.109.1`
first and confirmed `chat.completions.create`'s real signature via `inspect.signature`
before faking it -- same discipline as confirming `CrossEncoder`'s signature before
rerank.py's test. Built a synthetic 3-paper, 20-chunk corpus (deliberately uneven sizes:
10/6/4) with markers embedded in specific chunks' text to force a fabricated-span
rejection, a model-skip rejection, and a from-scratch shortfall (quota impossible to meet
even after trying every candidate). Checks: `_allocate_per_paper_quota` sums to exactly
each type's total quota, every time; `_looks_numerical`/`_looks_limitation` correctly
order candidates and correctly ignore citation-bracket digits; every accepted record's
gold_span (or, for comparative, BOTH gold_spans independently) verified verbatim against
its own source chunk; a chunk carrying a fabricated span or a model-skip marker is never
accepted, only backfilled past; a genuinely impossible quota reports a shortfall after
trying the whole candidate pool rather than hanging or silently under-reporting; the
comparative candidate ranking, run against a real `chromadb` collection populated with
hand-crafted embeddings, correctly surfaced the one genuinely topically-related
cross-paper pair (two chunks both about "surgical video segmentation... foundation
models," one per paper) as the top-ranked candidate, with no chunk or paper-pair reused
past its cap; a partially-bad comparative response (one real span, one fabricated) was
rejected as a whole. All checks passed. Not yet run for real -- that needs Ritik's own
`OPENAI_API_KEY` in `.env` (currently an empty placeholder) and will make ~100 real,
cheap gpt-4o-mini calls.

## `draft_qa.py`: real smoke-test surfaced a bibliography-contamination bug -- fixed before the full run

Ritik ran a shrunk smoke test for real (`TYPE_QUOTA` set to 2 factual / 0 everywhere
else, `COMPARATIVE_QUOTA` 0). It worked -- no crash, 2 items drafted -- but I read the
actual output content (`eval/qa_draft.jsonl`) rather than treating "it ran" as "it's
correct," and one of the two items was bad: a "factual" question drafted from
`conformal_prediction::section_aware::0113` asking "who are the authors of the paper
titled 'Cautious deep learning'?" That question is span-verified correctly -- the
citation text really is a verbatim substring of the chunk -- but the chunk's own
`section` metadata is `"References"`. It's a bibliography entry (a citation list), not
a claim the conformal_prediction paper itself makes. Span verification cannot catch
this class of error, because the span genuinely IS in the chunk; the problem is that
the chunk isn't "content" at all, and no amount of "is this text really there" checking
asks that question.

Checked how common this is across the real corpus, not just guessed: a
`collections.Counter` over all 843 section-aware chunks' `section` labels found
**171 "References" + 12 "REFERENCES" + 3 "Acknowledgment" = 186 chunks, ~22% of the
whole corpus** (roughly 1 in 5). `_looks_numerical` / `_looks_limitation` only ever biased sampling order for
two of the five question types -- factual and method had no protection at all, meaning
roughly a fifth of their candidate pool could silently be bibliography noise.

**Fix:** `_load_section_aware_chunks()` now filters non-content sections
(references/reference/bibliography/acknowledgments/acknowledgements/acknowledgment/
acknowledgement, case-insensitive, exact match on the stripped `section` field -- not a
substring match, so a hypothetical section actually titled "Cross-References in Related
Work" is not swept up by accident) out of its returned list entirely, upstream of every
other function. Both `chunks_by_paper` and `chunks_by_id` in `main()` are built from this
filtered list, so per-type sampling (`_candidate_pool`), the per-paper quota allocator,
and the comparative-candidate builder all inherit the exclusion automatically -- one fix
point, not five.

That fix surfaced a second, real bug it would otherwise have caused silently:
`_build_comparative_candidates` pulls embeddings straight from Day 2's `section_aware__
minilm` Chroma collection, which was built over the full, unfiltered 843-chunk set
*before* this exclusion existed. Once `chunks_by_id` no longer contains the ~186
non-content chunks, the first scored pair touching one of their embeddings would have
hit a bare `KeyError` deep in the ranking loop -- at 22% of the corpus, this was near-
certain to happen on the very first real comparative run, not a rare edge case. Fixed by
filtering the embeddings pulled from Chroma down to only ids present in `chunks_by_id`,
immediately after fetching them and before any similarity computation, with a print
statement reporting how many were dropped.

**Integration-tested** with a synthetic 2-paper corpus deliberately salted with
References/REFERENCES/Bibliography/Acknowledgments chunks AND one deliberate substring
trap (a chunk titled "Cross-References in Related Work", to confirm the match stays
exact rather than swallowing legitimate sections that merely contain the word). Checks:
`_is_non_content_section` exact-match behavior including the substring trap;
`_load_section_aware_chunks` excludes precisely the intended chunks and prints the
correct count; `_candidate_pool` never surfaces an excluded chunk for any of the four
question types; `_build_comparative_candidates`, run against a fake Chroma collection
embedding the FULL unfiltered set (reproducing the real stale-embeddings mismatch),
completes with no KeyError and returns only content-chunk pairs; and both pipeline
halves end-to-end (fake OpenAI client, deterministic span-verifiable responses) produce
zero references-derived items. All checks passed. Pushed to
`eval/draft_qa.py`, confirmed byte-identical to the local copy via matching sha256
(`9bf857f0...cab7ad8`) before and after transfer.

Not yet re-run for real by Ritik. Suggested next step: re-run the same shrunk smoke test
to see the new `_load_section_aware_chunks: excluded 186/843...` print line and confirm
no bibliography-derived items appear, before spending the full ~100-call budget on the
real 90-item run.

## `draft_qa.py`: real full run complete (90 items) — independently re-verified before handing to Step 2

Ritik ran the real, full `draft_qa.py` for real money against gpt-4o-mini: 90/90 items
drafted with zero shortfalls anywhere (every quota slot, across all six papers and all
five question types plus comparative, filled on the first attempt -- no rejections had
to be backfilled). Confirmed the exclusion fix holds at full scale: the run printed
`excluded 186/843` and `dropped 186 embedded chunk(s)`, identical to the smoke test.

Did not stop at "it ran and hit its quota" -- independently re-verified the actual
output file from scratch, without trusting the script's own internal checks:
- Re-ran `span_in_text` (the exact shared `span_utils` logic) against every one of the
  100 spans in the file (80 single-paper items + 20 spans across the 10 comparative
  items' two-span schema) directly against each item's own source chunk's real text,
  pulled fresh from `chunks_section_aware.jsonl`. Zero failures.
- Checked every `source_chunk_id` against the non-content-section set directly. Zero
  leaks -- the fix holds at full scale, not just in the synthetic test or the shrunk
  smoke test.
- Checked for chunk reuse across items (a chunk drafted into two different questions
  would silently overweight that one chunk in the eventual benchmark). Zero reused.
- Checked the qtype/paper distribution against PLAN.md's spec exactly: 30 factual / 20
  numerical / 20 method / 10 limitation / 10 comparative, matching the plan's stated
  totals precisely (the per-paper split differs run to run only because it's
  proportional to the *post-exclusion* per-paper chunk share, which is now smaller and
  differently distributed across papers than before the fix).
- Read every single one of the 90 questions and both spans of all 10 comparative items.
  Substantively, this is real content: architectural and training-objective comparisons
  for the comparative set, genuine paper-reported numbers for the numerical set, and one
  question that looked citation-like on first read ("Who were the key figures in the
  development of conformal prediction?") was checked against its source chunk directly
  and confirmed to come from real narrative prose in the paper's own Discussion section
  (a historical aside naming Vovk/Gammerman/Saunders/Vapnik), not a bibliography entry --
  the exclusion filter correctly let it through because it IS content.

**New finding, not a code bug -- flagged for Step 2 review:** 13 of the 90 questions
(~14%) literally reference "the excerpt" in their own phrasing (e.g. "...as described in
the excerpt", "...mentioned in the excerpts"). This leaks the drafting mechanism into the
question text itself -- no real person querying a RAG system phrases a question this
way, since they don't know they're asking about "an excerpt." This is a distinct problem
from the vocabulary-leakage pattern PLAN.md already warns reviewers about (copying a
chunk's rare terms verbatim); it's about grammatical framing, not word overlap. Not
fixed in code -- Step 2's manual rewrite is the right place for this, the same as
vocabulary leakage. If `draft_qa.py` is ever run again from scratch, tightening
`SYSTEM_PROMPT_SINGLE`/`SYSTEM_PROMPT_COMPARATIVE` to explicitly forbid referencing "the
excerpt"/"the excerpts" in the question text would likely eliminate most of these
up front -- worth doing then, not worth a mid-benchmark patch now.

Also noted, also not a bug: two of the comparative items (both involving
depth_anything_3 vs. surgicalsam/sam3) pulled evidence spans that are mostly raw
number rows from results tables rather than prose. Legitimate content and correctly
span-verified, just a harder kind of "evidence" for a human (or a retriever) to
recognize as relevant -- worth having in mind during review, not something to exclude
outright.

Day 3 Step 1 is now genuinely DONE. Step 2 (manual review, entirely Ritik's) is next.

## Day 3 Steps 2-4: manual review complete, 12 unanswerable questions Claude-drafted (deviating from plan), qa_gold.jsonl locked

**Step 2 (manual review of the 90 drafted items) is done.** Ritik cross-checked every
item against its source chunk himself, corrected several in place in `qa_draft.jsonl`,
and identified 17 qids to drop entirely: q0009, q0016, q0034, q0039, q0043, q0055,
q0056, q0058, q0059, q0071, q0074, q0081, q0083, q0084, q0086, q0087, q0089. This list
is fully consistent with this session's independent review of ChatGPT's own critique of
the same 90 items -- it includes every item both Claude and GPT flagged as REJECT-worthy
(circular/weak questions, ambiguous multi-column tables where the header-to-data-row
correspondence can't be trusted without the source PDF page, and one confirmed factual
error: q0074's `gold_answer` of "70.95" for SurgicalSAM's Mean IoU was actually reading
the table's BF column -- the real Mean IoU is 56.93).

`qa_draft.jsonl` is scratch by design (its own module docstring says so explicitly), and
`qid` is assigned purely by a sequential `enumerate()` in `draft_qa.py`'s `main()` with no
downstream code depending on contiguity -- the real identifier is `source_chunk_id`. So
rather than delete-in-place and leave gaps, `qa_gold.jsonl` was built fresh: the 73
surviving records (90 - 17) were copied over with `reviewed_by_human` flipped from
`false` to `true` (accurately, since Ritik had just done exactly that) and given new
contiguous qids q0000-q0072. Verified directly against the output file, not just the
build script's own claims: 73 records, qids contiguous, every one of the 17 dropped
`source_chunk_id` values confirmed absent from the survivors, and the qtype/paper
distribution (28 factual / 19 method / 14 numerical / 8 limitation / 4 comparative)
sums correctly against what was dropped from each bucket.

**Step 3 (the unanswerable questions) deviates from the plan.** PLAN.md and
`draft_qa.py`'s own docstring both say this must be hand-written by Ritik, specifically
because an LLM asked cold to write "a question this corpus can't answer" reliably
produces obviously-absurd ones instead of the plausible-but-absent questions that make
an abstention measurement meaningful. Given real time pressure, Ritik asked Claude to
write these instead. To avoid the exact failure mode the plan warns about, every
candidate was built and checked the same way: ground it in a real, specific,
topically-adjacent entity that plausibly belongs near this paper's subject, then
grep that paper's ENTIRE chunk set (not just chunks already read) for the specific term
to independently confirm no answer to it exists anywhere in the corpus -- proving
absence rather than assuming it. Two candidates were discarded after this check found
they were NOT actually safe (SAM 3's "thermal imagery" and "aircraft types" niche-domain
examples both turned out to have real numeric results elsewhere in the paper's bundled
2022-Challenge report), which is itself evidence the verification step was doing real
work rather than rubber-stamping.

The 12, each with the specific evidence that makes it plausible-but-unanswerable:

1. conformal_prediction -- robotic-planning collision-avoidance success rate (paper
   cites that such work exists [128,129], reports no numbers for it)
2. conformal_prediction -- MAPIE library's reported experimental results (named as an
   existing tool, no results attached in this paper)
3. depth_anything_3 -- performance on dynamic/moving-scene video (explicitly named as
   future work in the Conclusion; verified no other "dynamic" hit in the paper reports
   dynamic-scene results)
4. depth_anything_3 -- benchmark results for integrating language cues into depth
   predictions (same future-work sentence; verified "language" appears nowhere else)
5. murali_latent_graph -- LG-CVS's mAP on the Cholec80 dataset (Cholec80 never appears
   in this paper's content sections at all, only "cholecystectomy" and bibliography
   entries citing other authors' work on it)
6. sages_cvs_challenge -- performance gain from separate per-criterion confidence heads
   (named as a "complementary next step" in the Limitations section, not implemented)
7. sages_cvs_challenge -- which alternative probabilistic target performed best (same
   Limitations passage, also named as future work, no comparison run)
8. sages_cvs_challenge -- performance difference on LMIC vs. non-LMIC procedures
   (Limitations section explicitly names this stratified analysis as unpursued: "we
   prioritised a single coherent synthesis" instead)
9. sam3 -- accuracy gain from automatic domain expansion on out-of-domain concepts
   (Conclusion names this as a possible mitigation "but requires extra training",
   i.e. not done)
10. sam3 -- performance on the CholecSeg8k surgical segmentation benchmark (verified
    "surgical" and "cholec" both appear zero times anywhere in SAM 3's 295 chunks)
11. surgicalsam -- Dice score on EndoVis2019 (paper only ever reports EndoVis2017/2018;
    EndoVis2019 is a real later edition of the same challenge series, verified zero hits)
12. surgicalsam -- Dice score for segmenting the gallbladder (SurgicalSAM segments
    surgical instruments only, never anatomical structures; verified "gallbladder" and
    "cystic" both appear zero times)

Ritik reviewed all 12 himself before they were accepted -- Step 2's "entirely manual,
non-negotiable" review standard was applied to these exactly as it was to the 73
answerable survivors. `reviewed_by_human` is `true` on all 85 records; `gold_span`,
`gold_answer`, `gold_pages` and `source_chunk_id` are `null` on the 12 unanswerable ones
since by construction no evidence for them exists in the corpus. `qtype` is
`"unanswerable"` for these -- a new value, safe to introduce since nothing outside
`draft_qa.py` (which never touches unanswerable items) reads `qtype`, and the Day 4/5
eval scripts (`ir_metrics.py`, `run_retrieval.py`, `run_ragas.py`, `regression.py`) are
still empty stubs.

**Step 4: locked.** Final `qa_gold.jsonl`: 85 records (73 answerable + 12 unanswerable),
qids q0000-q0084, contiguous. This is 20 short of PLAN.md's original ~105 target
(90 answerable + 15 unanswerable) -- 17 answerable items were cut for cause during
review rather than replaced, and 3 unanswerable candidates were cut for cause during
verification (12 written, 2 of those discarded per above, so effectively 14 attempted
against a target of 15). Both shortfalls are quality cuts, not shortcuts: PLAN.md itself
says "90 you trust beat 150 you don't," and the same principle applies to the negatives.
PLAN.md is updated accordingly (105 -> 85, and the Day 5 gate's abstention threshold
scaled from "12 of 15" to "10 of 12" to preserve the same ~80% bar).

sha256sum eval/qa_gold.jsonl: 2b35cdd12b4716caa438c3124b13fbc26bf3688d5ea4a4d209207dc3753130ed

Per PLAN.md Step 4: do not touch this file again. If a broken question is found later,
fix it, re-record the hash here, and RE-RUN EVERY EXPERIMENT -- editing a benchmark
after seeing results is how honest projects quietly become dishonest ones.

Day 3 is now genuinely DONE. Day 4 (staged retrieval experiments) is next.
