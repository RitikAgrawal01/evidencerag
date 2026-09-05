# EvidenceRAG — Build Plan

An evaluated, citation-backed RAG system over the six GeoCVS thesis papers.
Drafted 31 Aug 2026. Generation: GPT-4o-mini. Runtime: Windows local venv.

Corpus: 6 papers, 212 pages, 770,177 chars.

---

## PART I — What already exists

| Component | Status | Notes |
|---|---|---|
| 6 arXiv PDFs (`data/papers/`) | DONE | Mascagni 2022 still missing (paywalled) |
| PyMuPDF extraction (`src/extract.py`) | **BROKEN** | Reading-order sorter assumes all papers are two-column. 3 of 6 are not. See Part II. |
| ToC filter (`_is_likely_toc`) | DONE | Correctly narrowed; flags exactly 2 pages corpus-wide. Leave alone. |
| Fixed-size chunking | DONE | 851 chunks, 1000 chars / 100 overlap |
| Recursive chunking | GAP | 797 chunks. Overlap applied only in the hard-cut fallback — the paragraph path has ZERO overlap. |
| Section-aware chunking | GAP | 697 chunks. Headers detected, prepended to text, then discarded — no `section` field. That field is the whole point of strategy 3. |
| Stable chunk IDs | MISSING | Only `chunk_index`. No global ID, no content hash. Blocks reproducible eval. |
| Embeddings / store / BM25 / RRF / reranker / generation / eval set / metrics | NOT STARTED | `eval/` and `notebooks/` are empty |
| Repo hygiene | NOT STARTED | No git, no requirements.txt, no venv, no .env, no README. Six loose `*_check.txt` in root. |

---

## PART II — The bug: two-column assumption is wrong for 68% of the corpus

`_sort_blocks_by_reading_order` buckets blocks left/right by whether their horizontal
centre falls left/right of the page midpoint. For a SINGLE-column paper every block is
full-width, so its centre sits on the midpoint and tiny width differences flip blocks
between buckets at random -> paragraph-level scrambling.

Measured fraction of text blocks straddling the midline:

| Paper | Pages | Straddling | True layout | Sorter |
|---|---|---|---|---|
| sam3 | 78 | 68% | **single column** | SCRAMBLED |
| conformal_prediction | 51 | 44% | **single column** | SCRAMBLED |
| depth_anything_3 | 32 | 38% | **single column** | SCRAMBLED |
| sages_cvs_challenge | 21 | 11% | two column | correct |
| surgicalsam | 18 | 8% | two column | correct |
| murali_latent_graph | 12 | 7% | two column | correct |

The three affected papers are the three largest: **161 of 212 pages, 525,622 of
770,177 chars = 68% of the corpus is extracted out of order.**

Proof — SAM 3 page 5 (all blocks x0~71 -> x1~541 on a 612pt page, unambiguously single column):

    CURRENT SORTER emits y = 191, 448, 533, 580, 608, 217, 271, 322, 390
    TRUE ORDER          y = 191, 217, 271, 322, 390, 448, 479, 533, 580

Fix: add `_detect_layout(page)`. A block wider than ~65% of page width is full-width;
if most qualifying blocks on a page are full-width, treat the page as single-column and
sort by y0 alone. Otherwise use the existing two-column path (it is correct where it
applies). Mixed pages: pull full-width blocks out and re-insert at their vertical position.

---

## PART III — Ground truth design (read before writing any eval question)

Labelling each question with a CHUNK ID silently breaks the chunking comparison:
fixed-size chunk #47 and section-aware chunk #47 are different text. Ground truth
labelled against one chunking cannot score another.

**Fix: anchor ground truth to evidence SPANS, not chunk IDs.**
Label each question with a short verbatim quotation from the paper plus its page.
A retrieved chunk is a hit if it contains that span. The span is a property of the
paper, so it scores every strategy / embedding / retriever identically.

Eval record schema:

    {
      "qid": "q0042",
      "question": "What shot-boundary detection library does the SAM 3 data engine use?",
      "qtype": "factual",            // factual | numerical | method | comparative | limitation
      "paper": "sam3",
      "gold_pages": [29],
      "gold_span": "we also use Shot Boundary Detection from the PySceneDetect",
      "gold_answer": "PySceneDetect",
      "answerable": true,
      "reviewed_by_human": true
    }

Hit rule: normalise both sides (collapse whitespace, strip soft hyphens/ligatures,
lowercase), then substring-test gold_span in chunk.text. Secondary diagnostic rule:
same paper AND page overlap with gold_pages.

TWO TRAPS:
1. **Longer chunks win for free.** Never report Recall@5 alone — always beside total
   context characters at K=5. 88% at 6,000 chars beats 90% at 9,500 chars.
2. **Spans straddling a chunk boundary score zero.** Keep every gold_span to one
   sentence, under ~200 chars, so a miss is a real finding not an artefact.

**You also need ~15 unanswerable questions.** Abstention is unmeasurable without
negatives. Write these by hand — the model generates obviously-absurd ones.

---

## DAY 0 — Environment and repo hygiene (~2h)

The project lives in OneDrive. ChromaDB writes SQLite and HuggingFace caches hundreds
of MB of weights — both inside a syncing folder cause file-lock errors. Move both out.

    cd C:\Users\agraw\OneDrive\Desktop\DS_Projects\geocvs-paper-rag
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # if activation blocked
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cpu   # avoids ~2.5GB CUDA
    pip install -r requirements.txt
    setx HF_HOME "C:\ml-cache\hf"
    setx EVIDENCERAG_STORE "C:\ml-cache\evidencerag-chroma"

requirements.txt:

    pymupdf>=1.24
    sentence-transformers>=3.3
    chromadb>=0.5.20
    rank-bm25>=0.2.2
    openai>=1.59
    ragas==0.4.3
    langchain-community==0.3.31   # PIN REQUIRED — see note below
    datasets>=4.0
    pandas
    numpy
    python-dotenv
    tqdm
    tabulate

Then `pip freeze > requirements.lock.txt` and commit it.

Target layout (NEW = to create):

    data/processed/          NEW   cached extractions + chunk sets (.jsonl)
    src/extract.py                 FIX per-page layout detection
    src/chunking.py                FIX chunk_id, section field, real overlap
    src/build_corpus.py      NEW   extract + chunk all 3 strategies, persist
    src/embed.py             DONE  MiniLM / BGE wrapper, batch encode, per-chunk token stats
    src/store.py             DONE  Chroma collection per (strategy, model)
    src/retrievers.py        DONE  Dense, BM25, HybridRRF — one interface
    src/rerank.py            PREP  cross-encoder wrapper written + integration-tested;
                                  NOT yet evaluated (needs Day 3's qa_gold.jsonl -- see
                                  the Day 2 gate note below)
    src/generate.py          NEW   Generator interface + OpenAI + abstention
    src/pipeline.py          NEW   config -> end-to-end answer
    eval/draft_qa.py         DONE  written + integration-tested; not yet run for real
                                  (needs OPENAI_API_KEY -- see DECISIONS.md)
    eval/span_utils.py       NEW   shared span-normalization, draft_qa.py + ir_metrics.py
    eval/qa_draft.jsonl      NEW   machine output, never used directly
    eval/qa_gold.jsonl       NEW   YOUR reviewed set — the locked benchmark
    eval/ir_metrics.py       NEW   span-match hit rule, Recall@K, MRR
    eval/run_retrieval.py    NEW   the experiment grid
    eval/run_ragas.py        NEW   generation-side metrics
    eval/regression.py       NEW   compare run vs baseline, fail on drop
    eval/results/            NEW   one CSV + config JSON per run
    reports/                 NEW   tables and figures
    DECISIONS.md             NEW   every choice + the evidence for it

Move the six loose `*_check.txt` into `reports/scratch/` or delete. Gitignore `.env`,
`.venv/`, `data/papers/*.pdf`, `data/processed/`, `__pycache__/`. Write `fetch_papers.py`
to download the six arXiv PDFs by ID so the repo clones clean.

GATE: `git log` shows an initial commit; `python -c "import fitz, chromadb,
sentence_transformers, rank_bm25, ragas, openai"` exits silently; `python src/extract.py`
still prints six page/char lines.


### Known dependency trap: ragas 0.4.3 + langchain-community

`ragas/llms/base.py` line 12 does an unconditional top-level
`from langchain_community.chat_models.vertexai import ChatVertexAI`.
That module (`langchain_community/chat_models/vertexai.py`) exists in
langchain-community 0.3.31 and was REMOVED in 0.4.x. ragas declares
`langchain-community` with no upper bound, so pip resolves 0.4.2 and
`import ragas` dies with ModuleNotFoundError. Classic missing-upper-bound break,
not a broken library.

Fix (verified in a clean venv):

    pip install "langchain-community==0.3.31"
    python -c "import ragas; print('RAGAS OK', ragas.__version__)"

Verified working resolution: ragas 0.4.3, langchain 1.3.18, langchain-core 1.6.1,
langchain-community 0.3.31, langchain-openai 1.1.9, openai 1.109.1, datasets 5.0.1.
No cascade; nothing else in the stack depends on langchain.

You never use Vertex AI at runtime. ragas 0.4.3 has a native OpenAI path via
`instructor` that does not touch langchain:

    from openai import OpenAI
    from ragas.llms import llm_factory
    llm = llm_factory("gpt-4o-mini", provider="openai", client=OpenAI())

langchain is only needed to satisfy the import at module load.

Use the non-deprecated metric path (note: the class is `AnswerRelevancy` here,
not `ResponseRelevancy`):

    from ragas.metrics.collections import (
        Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
    )

Fallback if ragas ever conflicts with chromadb/sentence-transformers later:
isolate it. `eval/run_ragas.py` only needs a JSONL of {question, answer, contexts,
reference} plus an API key — no torch, no chroma. Put it in a separate `.venv-eval`
and the two dependency trees can never collide. Not needed today.

RAGAS STAYS IN THE PLAN. Do not swap it for a hand-rolled judge because of this
error — a 30-second version pin is not a reason to redesign the evaluation layer.
But DO also write two metrics RAGAS does not have, because they are specific to
this system: **citation correctness** (does `[paper, p.N]` actually point at the
chunk the claim came from) and **abstention accuracy** (Day 5). And validate the
RAGAS faithfulness judge against ~20 hand-labelled answers of your own — agreement
rate between your labels and the judge is a number worth reporting.


---

## DAY 1 — Fix the foundation

1. **Per-page layout detection** — add `_detect_layout(page) -> "single"|"double"`,
   branch the sort. Keep the two-column path untouched.

2. **Verification harness** — write `reports/extraction_audit.txt`: per paper, detected
   layout per page, plus first 400 chars of three sampled body pages in final reading
   order. Then READ IT against the actual PDF. Extraction correctness cannot be
   unit-tested into existence; eyeball it once carefully, then freeze it.

3. **Stable chunk identity** — add to the Chunk dataclass:
   - `chunk_id` = f"{paper_slug}::{strategy}::{index:04d}"
   - `content_hash` = first 12 hex of SHA-256 of normalised text
   - `section` = detected header for section-aware, None otherwise (currently discarded)
   - `char_start`, `char_end` = offsets into the paper's joined text (makes page
     attribution exact instead of the current find()-probe approximation)

4. **Fix recursive overlap** — either implement real overlap by carrying the previous
   chunk's tail forward, or rename the parameter and document that this strategy is
   boundary-preserving without overlap. Both defensible; claiming overlap you don't
   have is not.

5. **DONE — Persist the corpus** — `build_corpus.py` writes
   `data/processed/chunks_{strategy}.jsonl` + `manifest.json` (code_version hash of
   extract.py+chunking.py, git commit, params, counts, per-paper breakdown). 2,492 chunks
   total (851 fixed_size + 798 recursive + 843 section_aware) -- matches Ritik's
   independent `check_tokenization.py` count exactly. 0 chunk_id collisions globally.
   Nothing re-parses a PDF from here on. See DECISIONS.md.

6. **(Added after 1.4, prompted by "why chunk_size=1000?") Equalize size caps across
   strategies** — `section_aware_chunks` defaulted to `max_chunk_size=1200` while
   `fixed_size`/`recursive` used `chunk_size=1000`; a pre-existing mismatch that would
   have confounded Day 4 Stage A's strategy comparison (a size advantage looking like a
   boundary-logic advantage). Lowered to 1000 to match. See DECISIONS.md for the fuller
   writeup, and Day 4 Stage F for the follow-up: is 1000 actually a good number, checked
   properly rather than assumed.

GATE -- PASSED: SAM 3 page 5 reads top-to-bottom in the audit file. Three JSONLs exist
(851/798/843 chunks), every one of 2,492 chunk_ids is globally unique, section_aware
records carry non-null `section` for 100% (>= the 80% target), average chunk size sits
at 989-1051 chars across all three strategies (fixed_size 996.7, recursive 1050.8,
section_aware 989.2).

---

## DAY 2 — Index and retrieve

**1. `embed.py` — written, self-tested, and run for real. DONE.** One `Embedder` class
per model_id (`MODELS = {"minilm": ..., "bge": ...}`), reading `max_seq_length` from the
model object itself (never hardcoded), exposing `encode_passages`, `encode_queries`, and
`token_stats` (true token count + truncated flag per text, against that model's own real
limit). Also centralises BGE's query-only instruction prefix (`"Represent this sentence
for searching relevant passages: "`, applied in `encode_queries` only, never
`encode_passages`) — BGE was trained asymmetrically for this, MiniLM wasn't; see
DECISIONS.md.

Real run against all 2,492 chunks (`python src/embed.py`), on Ritik's machine:

    minilm (max_seq_length=256): median 265 tokens, truncated at 256: 1373/2492 (55.10%)
    bge    (max_seq_length=512): median 265 tokens, truncated at 512:    28/2492 ( 1.12%)

**Tokenization gate CLOSED.** MiniLM truncates the tail of 55.10% of chunks at its real
256-token limit; BGE truncates only 1.12% at its 512-token limit, on the identical chunk
set. `chunk_size` stays 1000 (see rationale above — shrinking it to fit MiniLM would
erase the effect worth measuring). `token_stats()` output gets written into Chroma
metadata in `store.py` below, so Day 4 can report exactly what fraction of MiniLM's
retrieval misses land on truncated chunks.

**2. `store.py` — written, integration-tested, and run for real. DONE.** Builds the
six Chroma collections: 3 chunkings x 2 embedding models, named `{strategy}__{model_slug}`.
Stores `chunk_id`, `paper`, `section`, `start_page`, `end_page` in metadata — the concrete
payoff over FAISS, metadata rides with the vector, so a result knows how to cite itself.
Also stores `token_count` and `truncated` per (chunk, model) pair from `embed.py`'s
`token_stats()` — tokenized with that model's OWN tokenizer, compared against that
model's OWN real `max_seq_length` (see the closed tokenization gate below). Turns
"MiniLM probably truncates some chunks" into a per-chunk, per-model fact Day 4 can cite
directly: what fraction of Recall@K misses were on chunks flagged truncated.

Reuses one `Embedder` per model across all three strategies rather than rebuilding it
three times, and rebuilds each collection from scratch on every run (delete-then-recreate)
— same "deterministic regeneration over incremental drift" philosophy as `build_corpus.py`.
Batches `.add()` calls at `client.get_max_batch_size()`, read at runtime rather than
guessed — in practice a single batch covers this whole corpus, but the loop doesn't
assume that stays true.

Two concrete things found while writing this, checked directly against
chromadb==1.5.9 (the exact version pinned in `requirements.lock.txt`), not assumed from
docs:
- Chroma's metadata store rejects an explicit `None` value outright
  (`TypeError: Cannot convert Python object to MetadataValue`). `section` is `None` for
  fixed_size/recursive chunks by design — it has to be OMITTED from the metadata dict
  entirely for those, never passed through as `None`. `store.py`'s `_clean_metadata`
  does this.
- `get_or_create_collection`'s default `embedding_function` is Chroma's own
  `DefaultEmbeddingFunction` — harmless here since every `.add()` call always supplies
  its own `embeddings=`, but passed explicitly as `embedding_function=None` anyway, so
  nothing could ever silently fall back to Chroma computing its own embedding instead of
  the one MiniLM/BGE comparison this project is actually about.

Verified via a full integration test against the real `chromadb==1.5.9` library (not a
mock), then run for real on Ritik's machine against the actual corpus:

    fixed_size__minilm:     851 chunks (432/851 = 50.76% truncated at 256 tokens)
    recursive__minilm:      798 chunks (484/798 = 60.65% truncated at 256 tokens)
    section_aware__minilm:  843 chunks (457/843 = 54.21% truncated at 256 tokens)
    fixed_size__bge:        851 chunks (  7/851 =  0.82% truncated at 512 tokens)
    recursive__bge:         798 chunks ( 10/798 =  1.25% truncated at 512 tokens)
    section_aware__bge:     843 chunks ( 11/843 =  1.30% truncated at 512 tokens)
    Total vectors indexed: 4984 (expected 4984)

Cross-checked against embed.py's earlier aggregate run: 432+484+457=1373 and 7+10+11=28,
matching the 1373/2492 (55.10%) and 28/2492 (1.12%) figures from the tokenization gate
exactly — two different scripts, same per-chunk numbers.

New finding: truncation isn't uniform across strategies for MiniLM (recursive 60.65% >
section_aware 54.21% > fixed_size 50.76%) — flagged for Day 4's writeup, not chased now.
See DECISIONS.md for the likely mechanism.

Compute-cost data point (the gate below asks for this): MiniLM embedded all 2,492 chunks
in ~104s total; BGE took ~880s (~14.7 min) — roughly 8.5x slower, on top of already
truncating far less. Concrete numbers for the compute half of the MiniLM vs BGE argument.

`EVIDENCERAG_STORE` was confirmed UNSET on Ritik's machine before this run (Day 0's setup
step had been skipped) — fixed by setting it for the current session and via `setx` for
future ones, before running store.py for real.

**3. `retrievers.py` — written and integration-tested, not yet run for real.** One
interface, three implementations:

    class Retriever:
        def retrieve(self, query: str, k: int) -> list[Hit]: ...
    # Hit = (chunk_id, text, metadata, score, rank)

    DenseRetriever     # embed query, Chroma similarity search
    BM25Retriever      # rank_bm25 over the same chunk list, in-process
    HybridRRFRetriever(dense, bm25, k_rrf=60)

RRF: for a doc at rank r in a list, add 1/(k_rrf + r); sum across lists; sort desc.
k_rrf=60 is Cormack et al. (2009), the standard default. Sanity-check k_rrf in {10, 60}
and note whether it moves the metric.

WHY RANK FUSION NOT SCORE FUSION: BM25 is unbounded positive, cosine is ~[-1,1]. Adding
them is meaningless without calibration, and calibration needs held-out data better
spent on evaluation. RRF consumes only ordering. This is exactly the weakness in the
old `RAG` project's union-and-truncate fusion — record that in DECISIONS.md.

`BM25Retriever` is built straight from `data/processed/chunks_{strategy}.jsonl` (the
same chunk set dense retrieval for that strategy uses), independent of any embedding
model. `HybridRRFRetriever` only ever fuses a dense+BM25 pair over the SAME strategy —
fusing across strategies would mean combining two different chunk_id spaces. Chroma's
"distance" (cosine space) is reported by `DenseRetriever` as `1 - distance`, so a higher
score always means "more relevant," consistent with BM25's own direction.

Verified with a full integration test against the real `chromadb==1.5.9` and real
`rank_bm25==0.2.2` (matching `requirements.lock.txt` exactly), a small synthetic corpus,
and a fake embedder with hand-picked embeddings — every score independently
hand-computable, not just plausible-looking: dense cosine similarity matched to 1e-3,
RRF scores matched the `1/(k_rrf+rank)` formula to 1e-9. Found (not a bug) that classic
BM25's IDF is exactly 0 when a term sits in precisely half the corpus — `rank_bm25`
floors negative IDF but not exact zero; essentially never triggered on the real
800-1000-chunk-per-strategy corpus. See DECISIONS.md for the full writeup. Not yet run
against the real corpus — needs Ritik's machine for real MiniLM query embeddings:

    python src/retrievers.py
    # prints top-5 from all three retrievers for the five hand-written probe queries below

GATE (MiniLM vs BGE tokenization) -- RESOLVED. Confirmed directly on Ritik's machine:
`SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").max_seq_length == 256`,
`SentenceTransformer("BAAI/bge-base-en-v1.5").max_seq_length == 512`. The raw-tokenizer
check (both report `model_max_length=512`, 1.12% over) was answering a question about
the wrong object -- `AutoTokenizer`, not the `SentenceTransformer` wrapper `.encode()`
actually calls. Given the already-measured token distribution (median 265 across 2,492
chunks), 55.10% of MiniLM's chunks (1373/2492) sit at or past its real 256-token
limit -- confirmed by the real run in Day 2 item 1 above, not just estimated from
percentiles.

This does NOT mean shrink chunk_size. Both models embed the identical chunk set; making
chunks smaller to flatter MiniLM would erase the exact effect worth measuring -- that
MiniLM's own ceiling, not the chunking strategy, is what limits it. chunk_size stays
1000. Action item is the `token_count`/`truncated` metadata above, so this becomes a
measured, citable fact in Day 4's results instead of an invisible side effect. See
DECISIONS.md for the full writeup.

Smoke test before any eval: five hand-written queries you know the answers to — one
acronym-heavy ("SA-Co benchmark"), one purely semantic ("how is depth predicted from a
single camera"), one numerical, one about a limitation, one cross-paper. Print top-5 for
each retriever side by side and read them.

GATE — PASSED, DAY 2 COMPLETE: six collections built (4984/4984 vectors, matches JSONL
line counts exactly), cold-start build time recorded per embedding model (MiniLM ~104s,
BGE ~880s, ~8.5x slower), all three retrievers run for real against the actual corpus.
Two of five probes ("SA-Co benchmark", the depth-prediction query) came back clean and
on-topic across dense/BM25/hybrid. The other three were badly worded BY ME (bare "the
model"/"the approach" phrasing with 6 papers in the corpus to choose from) rather than a
retrieval defect — fixed in `retrievers.py` to name a specific paper each. RRF math
hand-verified correct on the real run (see DECISIONS.md). One concrete finding carried
into Day 3: the top hit for the cross-paper probe was a bibliography citation line, not
body prose — worth checking whether the reranker demotes reference-list chunks.
Retrieval-quality judgment (not just mechanics) deliberately NOT made from these 5
anecdotal queries — that's what the eval set (Day 3-4) is for.

NOTE (2 Sep 2026): src/rerank.py (the CrossEncoderReranker wrapper) has been written and integration-tested ahead of Day 3 -- see DECISIONS.md. This is prep work only: the module is verified to work mechanically (correct sort order, prior_rank threading, real chromadb/rank_bm25 underneath), but whether reranking actually helps is Day 4 Stage D's question and needs qa_gold.jsonl to answer. Next actual step per this plan is still Day 3.

---

## DAY 3 — Build the benchmark (the long pole — budget the whole day)

Target ~90 answerable + ~15 unanswerable = ~105 questions, ~15 answerable per paper.
ACTUAL (Day 3 complete, 5 Sep 2026): 73 answerable + 12 unanswerable = 85 total.
17 answerable items were cut for cause during Step 2 review (not replaced), and 3 of
15 attempted unanswerable candidates were cut for cause during Step 3 verification.
Both shortfalls are quality cuts, consistent with "90 you trust beat 150 you don't"
below. Full accounting in DECISIONS.md.

**Step 1 — Draft (~1h, mostly unattended).** `draft_qa.py` samples chunks from the
SECTION-AWARE set (the section label lets you enforce coverage) and asks GPT-4o-mini for
a question, the verbatim answer-bearing sentence, and a short reference answer. Force
verbatim quotation from the given chunk, then VERIFY PROGRAMMATICALLY that the span is
really a substring of that chunk; drop the item if not. That check removes most
hallucinated ground truth.
Type quota in the prompt: ~30 factual, 20 numerical, 20 method, 10 comparative
(spanning two papers), 10 limitation.

**Step 2 — Review (4-6h, entirely manual, non-negotiable).** For each item, open the
paper beside it: is the question well-posed, is the span really the evidence, is the
reference answer right, is it answerable ONLY with retrieval rather than general
knowledge. Delete freely — 90 you trust beat 150 you don't.

WATCH FOR QUESTION LEAKAGE: an LLM drafting from a chunk reuses that chunk's rare
vocabulary in the question, making BM25 look artificially brilliant. Rewrite questions
that copy distinctive phrasing verbatim — EXCEPT where the acronym or metric name is
genuinely how a person would ask. You want a realistic mix, deliberately chosen.

**Step 3 — Write the 15 unanswerable questions by hand.** Plausible questions about
things the papers nearly discuss but don't.

ACTUAL: written by Claude (not by hand), under real time pressure, after Ritik asked
for this deviation explicitly. To avoid the exact obviously-absurd failure mode this
section warns about, each candidate was grounded in a real, specific, topically-
adjacent entity and verified absent via a full grep of that paper's entire chunk set
-- not just assumed absent. 12 of 15 attempted candidates survived verification;
Ritik manually reviewed and approved all 12 before they were accepted into
qa_gold.jsonl. Full list and per-item evidence in DECISIONS.md.

**Step 4 — Lock it.** Commit qa_gold.jsonl, record its SHA-256 in DECISIONS.md, never
touch it again. If you later find a broken question: fix it, re-record the hash, and
RE-RUN EVERY EXPERIMENT. Editing a benchmark after seeing results is how honest projects
quietly become dishonest ones.

NOTE (2 Sep 2026): src/rerank.py's prep note above still applies -- draft_qa.py is now ALSO written and integration-tested ahead of actually running Step 1 for real. It drafts eval/qa_draft.jsonl (the 90 answerable candidates); Steps 2-4 below (manual review, the 15 unanswerable questions, locking qa_gold.jsonl) are still entirely undone and are the real remaining work of Day 3. Five underspecified design calls this script had to make (per-paper quota split, which chunk gets which type, how comparative candidates are found, the comparative schema extension, and shared span-normalization logic) are written up in DECISIONS.md -- worth reading before Step 2's manual review, since they explain why a given item was sampled the way it was.

NOTE (2 Sep 2026, later): Ritik's real smoke test surfaced a genuine bug -- a bibliography/citation-list chunk (section="References") produced a technically span-verified but substantively worthless "factual" question. Fixed: _load_section_aware_chunks() now excludes non-content sections (references/bibliography/acknowledgments, 186 of 843 chunks, ~22% of the corpus) from the sampling pool entirely, upstream of every question type and the comparative-candidate builder. This also fixed a latent second bug the exclusion would otherwise have caused: _build_comparative_candidates now drops embeddings pulled from Day 2's Chroma collection that reference now-excluded chunk ids, instead of crashing with KeyError. Full writeup in DECISIONS.md. Integration-tested against a synthetic corpus salted with excluded-section chunks (including a substring-trap case); pushed to eval/draft_qa.py, sha256 confirmed identical on both sides. RE-RUN FOR REAL, SUCCESSFULLY (2 Sep 2026): 90/90 items drafted, zero shortfalls. Independently re-verified from scratch (not just trusting the script): all 100 spans re-checked against the real corpus text, zero non-content-section leaks, zero duplicate evidence chunks, qtype/paper distribution matches PLAN.md's 30/20/20/10/10 spec exactly. One new finding for Step 2 reviewers: 13/90 (~14%) questions literally reference "the excerpt" in their own wording -- a real person querying a RAG system never phrases a question this way. Not a code bug; rewrite during manual review like any other leaked phrasing. Full detail in DECISIONS.md. Day 3 Step 1 is DONE; Step 2 (manual review) is next.

NOTE (5 Sep 2026): Day 3 Steps 2-4 complete -- qa_gold.jsonl is LOCKED. Ritik manually reviewed all 90 drafted items and cut 17 for cause (listed in DECISIONS.md); the survivors were copied into a fresh qa_gold.jsonl with reviewed_by_human flipped true and renumbered contiguously (qid is scratch/non-semantic in qa_draft.jsonl, so no gap-filling was needed). Step 3's 12 unanswerable questions were Claude-drafted under time pressure (a deviation from "write these by hand", done with Ritik's explicit sign-off) using a grounded-entity-plus-full-corpus-grep verification method instead, then manually reviewed and approved by Ritik same as everything else. Final: 85 records (73 answerable + 12 unanswerable), sha256 2b35cdd12b4716caa438c3124b13fbc26bf3688d5ea4a4d209207dc3753130ed, recorded in DECISIONS.md. Per Step 4: do not touch this file again. Day 3 is DONE; Day 4 (staged retrieval experiments) is next.

GATE: every gold_span verified as a substring of the source paper's extracted text.
Type quota met. Hash recorded. You can describe five of your own questions from memory
and say why each is hard.

---

## DAY 4 — The retrieval experiments

Full grid = 3 chunkings x 2 embeddings x 3 retrievers x 2 rerank = 36 configs. Waste of
a day. Staged search instead, then validate:

| Stage | Varying | Held fixed | Runs |
|---|---|---|---|
| A Chunking | fixed / recursive / section-aware | MiniLM, dense | 3 |
| B Embedding | MiniLM / BGE-base | winner of A, dense | 2 |
| C Retriever | BM25 / dense / hybrid RRF | winners of A+B | 3 |
| D Reranking | off / cross-encoder | winners of A+B+C | 2 |
| E Validation | full 3x2 chunk x embed grid | winning retriever + rerank | 6 |

WHY STAGE E EXISTS (and why it's the best answer you'll give in the interview):
A-D are a greedy search — each choice fixed based on a comparison made under one setting
of the others. Only valid if the factors don't interact, and they plausibly do
(section-aware chunking might help BM25 more than dense, since headers add exact terms).
Stage E re-runs the chunking x embedding grid under the winning retriever to check the
greedy path didn't miss an interaction. If it agrees, say so. If it disagrees, you've
found something genuinely interesting. Either beats "I tried things until the number
went up."

Each run records, into `eval/results/{run_id}/`: one CSV of per-question results + one
JSON of config. Per run: Recall@1/@3/@5/@10, MRR@10, mean context chars at K=5, median
and p95 query latency, corpus content_hash, benchmark hash. Per question: the rank at
which the gold span was found (or null) — that column is what lets you do failure
analysis, which is the most valuable half of the exercise.

PLAN FOR THE BORING OUTCOME NOW: hybrid RRF may not beat dense; BGE may tie MiniLM.
Decide TODAY, before seeing numbers, that you'll report it. Then investigate — pull the
15 questions where hybrid lost and read them. A finding like "BM25 hurt on paraphrased
method questions because the corpus shares vocabulary across all six papers, so lexical
matching retrieved the wrong paper's Method section" is more impressive than any
improvement you could manufacture, and it's unfakeable.

**Stage F (added after Day 1) — is 1000 chars actually a good chunk size?** Ritik asked
why chunk_size is 1000 and not some other number -- it wasn't experimentally chosen, it
was a reasonable-sounding default (see DECISIONS.md). Rather than sweep size across all
3 strategies x 2 embeddings (that grid explosion is exactly what NOT to do), run ONLY the
winning strategy from Stage A at 2-3 more sizes (e.g. 600, 1000, 1500) against the same
locked benchmark. Cheap -- 2-3 more retrieval runs, no new eval questions -- and it turns
an arbitrary parameter into either "we checked, 1000 was fine" or a genuine improvement.

GATE: every stage's table in `reports/`, including Stage F. A short note per stage: which
config won, by how much, one sentence on why. Failure cases from the worst stage in
`reports/failure_analysis.md`.

---

## DAY 5 — Reranking and grounded generation

**Cross-encoder** `cross-encoder/ms-marco-MiniLM-L-6-v2` over top-20 hybrid candidates,
keep 5. The first-stage bi-encoder embeds query and chunk independently — fast,
cacheable, but they never interact. The cross-encoder runs both through one transformer
together so attention relates query terms to chunk terms directly. Much more accurate,
far too slow for 851 chunks, exactly right for 20.

MEASURE LATENCY HONESTLY: median and p95 for retrieval-only vs retrieval+rerank on CPU.
If reranking adds 7 points of Recall@5 for 400ms, that's a real trade-off — say which
side you'd pick and under what SLA. This was on `rag_agent_project`'s own "Future
Improvements" list; building it and quantifying its cost closes that loop.

**Generation.** A `Generator` protocol with `generate(query, contexts) -> Answer`,
implemented by `OpenAIGenerator`. Keep the interface even with one implementation — costs
nothing now, lets you add an Ollama comparison later.
Prompt does three things: answer only from the supplied context; cite every claim as
`[paper, p.N]`; emit exactly `INSUFFICIENT_EVIDENCE` if the context doesn't support an
answer. Number the context blocks and require citation by number — the model hallucinates
page numbers far less when copying a label than recalling one.

**Abstention — two independent triggers, measured separately:**
- Model-driven: returns INSUFFICIENT_EVIDENCE
- Score-driven: top reranker score below threshold tau, chosen by sweeping tau and
  plotting abstention rate vs error rate among non-abstentions

Report three numbers: false-abstention rate on answerable questions (low), true
abstention rate on unanswerable questions (high), answer error rate among
non-abstentions. A system that abstains on everything aces the second and is useless on
the first — which is why you report all three.

OPTIONAL STRETCH (only if Days 0-6 done): split-conformal calibration of tau — hold out
part of the eval set, choose tau so non-abstained answers are correct with >=90%
probability, validate coverage on the remainder. Same machinery as the uncertainty
sub-challenge in your thesis, applied to LLM output. FIRST THING TO CUT. Do not let it
block Day 7.

GATE: end-to-end question in, cited answer out, sources verifiable by opening the PDF at
that page. Rerank latency measured. Abstention triggering on >=10 of 12 unanswerables
(scaled down from the original >=12 of 15 to preserve the same ~80% bar against the
actual 12-item unanswerable set -- see DECISIONS.md).

---

## DAY 6 — Generation metrics and the regression guard

**RAGAS on three configs only:** naive baseline (fixed chunking, MiniLM, dense), best
retrieval config, best + reranker. Four metrics: faithfulness, answer relevancy, context
precision, context recall. Running all 36 through an LLM judge is slow and adds nothing.

Cost: ~85 questions x 3 configs x 4 LLM-judged metrics on gpt-4o-mini lands under $2
including drafting and generation. Set a hard spend limit in the OpenAI dashboard anyway.

SAY CLEARLY WHAT RAGAS DOES NOT MEASURE: RAGAS context precision/recall are LLM-JUDGED
RELEVANCE, not IR metrics against known-correct chunks. They ask a model whether the
context looks relevant. Your Recall@K and MRR ask whether the actual answer-bearing
sentence was retrieved, against ground truth you verified by hand. Complementary — and
knowing exactly how they differ is the sharpest thing you can say about evaluation.
Most candidates conflate them.

**Regression guard, NOT "drift detection".** Your corpus is six static PDFs. It does not
drift, and claiming it does invites a question you'd lose. Build the honest version:
`regression.py` loads `eval/results/baseline.json`, runs the current config over the
locked benchmark, exits non-zero if Recall@5 drops >2 percentage points or faithfulness
drops >0.05. Add `--update-baseline` requiring an explicit reason string logged to
DECISIONS.md.

That's a CI quality gate for a RAG system — same instinct as the ">2% F1 gain" promotion
rule in the Drowsiness project, applied where it actually fits. Small, honest, defensible
beats an elaborate dashboard monitoring a distribution that cannot shift.

GATE: RAGAS table for three configs in `reports/`. `regression.py` demonstrably fails
when you deliberately degrade a config (drop K to 1) and passes on the baseline.

---

## DAY 7 — Package it and prepare to defend it

**README** — lead with results tables, not the tech stack. Structure: what it does in two
sentences -> the benchmark (how many questions, how built, how validated) -> results
tables -> what did NOT work and why -> architecture diagram -> how to reproduce. An
honest "what did not work" section is the most credibility-generating thing a student
repo can contain.

**DECISIONS.md** — one entry per architectural choice: what you chose, what you rejected,
the evidence. Chroma over FAISS (metadata co-location; the bookkeeping the prior project
hand-rolled). RRF over score fusion (incomparable scales). Cross-encoder as second stage
only (cost). Span-anchored ground truth over chunk-ID labelling (cross-strategy
comparability). Layout detection over the fixed two-column assumption (68% of the corpus,
with the measurement). This file is your interview script.

**Resume bullets — template, filled only from real numbers:**

    EvidenceRAG: Evaluated Retrieval over Surgical-AI Literature (Self-directed)

    - Built a citation-backed RAG system over 6 surgical-vision papers with hybrid
      BM25 + dense retrieval fused by reciprocal rank fusion and a cross-encoder
      reranker, served through a layout-aware PyMuPDF ingestion pipeline that fixed
      misordered extraction on 68% of the corpus.

    - Constructed and manually validated an 85-question benchmark with span-anchored
      ground truth, enabling like-for-like comparison of 3 chunking strategies and 2
      embedding models: Recall@5 improved from __% to __% and MRR from ___ to ___,
      at +__ ms median latency.

    - Added evidence-gated abstention (__% abstention on 12 unanswerable questions,
      __% false-abstention rate) and a CI regression gate that blocks configs
      dropping Recall@5 by more than 2 points.

Note what these do NOT say: no "leveraged", no "state-of-the-art", no tech-stack list.
Every clause names a thing you built or a number you measured — consistent with the rest
of your resume.

**Interview drill — write these answers out, out loud, once:**

- *Why RRF instead of adding the scores?* BM25 is unbounded, cosine is [-1,1]; combining
  by value needs calibration data better spent on evaluation. RRF needs only rank order.
- *How do you know section-aware chunking is better?* Name the exact Stage A deltas, then
  immediately name the context-length caveat — longer chunks contain more spans for free,
  so cite Recall@5 next to characters-at-K=5.
- *Your ground truth came from an LLM. Why trust it?* Two different constructions, both
  human-gated. The 73 answerable items: gpt-4o-mini drafted, every span programmatically
  verified as a literal substring of the paper, then you hand-reviewed all of them and
  cut 17 for cause. The 12 unanswerable items: Claude drafted under time pressure, each
  one grounded in a real adjacent entity and verified ABSENT via a full corpus grep
  (not assumed absent), then you reviewed and approved all 12 yourself. Either way: a
  model proposed, a programmatic or corpus-level check constrained what could be
  proposed, and you were the last gate before anything entered the locked 85-item file
  by hash.
- *Difference between your Recall@5 and RAGAS context recall?* Recall@5 is IR recall
  against verified gold spans. RAGAS context recall is a second model's opinion about
  whether the context looks sufficient. One has ground truth; the other does not.
- *Would this scale to 10M documents?* No — and say why. In-process BM25 and a flat
  Chroma index are right-sized for 851 chunks. At that scale you need an ANN index with
  sharding, a served sparse index like OpenSearch, and the reranker behind a batching
  service. Naming the break points beats claiming it scales.
- *What went wrong?* Lead with the extraction bug: the sorter assumed two columns, three
  of six papers were single-column, 68% of characters were out of order, and you found it
  by measuring the fraction of blocks straddling the midline. Strongest answer you have.

FINAL GATE: a stranger can clone the repo, run fetch_papers.py, run one command, and
reproduce your headline table. You can answer all six questions without looking anything up.

---

## Five rules for the whole build

1. **No number on the resume that you did not produce.** Every bullet traces to a CSV in
   `eval/results/`.
2. **Lock the benchmark before the first experiment.** If you must change it, re-run
   everything and record why.
3. **Every run records its config.** Corpus hash, benchmark hash, model names, K values.
   A result you cannot reproduce is not a result.
4. **Report the boring outcomes.** "Hybrid did not help, and here is why" is a stronger
   interview answer than a manufactured improvement.
5. **Fix the extraction bug before anything else.** Embeddings, chunking comparisons and
   ground-truth spans built on scrambled text are all wasted work.

**Cut list, in order:** conformal calibration of tau (Day 5 stretch) -> Stage E
validation grid (Day 4) -> RAGAS on the third config (Day 6) -> the regression guard
(Day 6).
**Never cut:** the extraction fix, the manual benchmark review, the failure analysis.
Those three are the project.

---

NEXT STEP: fix `_sort_blocks_by_reading_order`, regenerate `reports/extraction_audit.txt`,
and read SAM 3 page 5 to confirm it flows top-to-bottom. Everything else waits on that.
