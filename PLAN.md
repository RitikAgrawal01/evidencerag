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
    src/embed.py             NEW   MiniLM / BGE wrapper, batch encode
    src/store.py             NEW   Chroma collection per (strategy, model)
    src/retrievers.py        NEW   Dense, BM25, HybridRRF — one interface
    src/rerank.py            NEW   cross-encoder second stage
    src/generate.py          NEW   Generator interface + OpenAI + abstention
    src/pipeline.py          NEW   config -> end-to-end answer
    eval/draft_qa.py         NEW   GPT-4o-mini drafts QA from chunks
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

5. **Persist the corpus** — `build_corpus.py` writes `data/processed/chunks_{strategy}.jsonl`
   + `manifest.json` (code version, params, counts). Nothing re-parses a PDF after this.

GATE: SAM 3 page 5 reads top-to-bottom in the audit file. Three JSONLs exist, every
record has a unique chunk_id, section-aware records carry non-null `section` for >=80%.

---

## DAY 2 — Index and retrieve

Six Chroma collections: 3 chunkings x 2 embedding models, named `{strategy}__{model_slug}`.
Store chunk_id, paper, section, start_page, end_page in metadata. This is the concrete
payoff over FAISS — metadata rides with the vector, so a result knows how to cite itself.

One interface, three implementations:

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

Smoke test before any eval: five hand-written queries you know the answers to — one
acronym-heavy ("SA-Co benchmark"), one purely semantic ("how is depth predicted from a
single camera"), one numerical, one about a limitation, one cross-paper. Print top-5 for
each retriever side by side and read them.

GATE: six collections built, counts match JSONL line counts, all three retrievers give
sensible top-5 on all five probes. Record cold-start build time per embedding model
(needed for the compute half of the MiniLM vs BGE argument).

---

## DAY 3 — Build the benchmark (the long pole — budget the whole day)

Target ~90 answerable + ~15 unanswerable = ~105 questions, ~15 answerable per paper.

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

**Step 4 — Lock it.** Commit qa_gold.jsonl, record its SHA-256 in DECISIONS.md, never
touch it again. If you later find a broken question: fix it, re-record the hash, and
RE-RUN EVERY EXPERIMENT. Editing a benchmark after seeing results is how honest projects
quietly become dishonest ones.

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

GATE: every stage's table in `reports/`. A short note per stage: which config won, by how
much, one sentence on why. Failure cases from the worst stage in
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
that page. Rerank latency measured. Abstention triggering on >=12 of 15 unanswerables.

---

## DAY 6 — Generation metrics and the regression guard

**RAGAS on three configs only:** naive baseline (fixed chunking, MiniLM, dense), best
retrieval config, best + reranker. Four metrics: faithfulness, answer relevancy, context
precision, context recall. Running all 36 through an LLM judge is slow and adds nothing.

Cost: ~105 questions x 3 configs x 4 LLM-judged metrics on gpt-4o-mini lands under $2
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

    - Constructed and manually validated a 105-question benchmark with span-anchored
      ground truth, enabling like-for-like comparison of 3 chunking strategies and 2
      embedding models: Recall@5 improved from __% to __% and MRR from ___ to ___,
      at +__ ms median latency.

    - Added evidence-gated abstention (__% abstention on 15 unanswerable questions,
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
- *Your ground truth came from an LLM. Why trust it?* The model drafted; you verified
  every span as a literal substring of the paper and hand-reviewed all 105. Then locked
  the file by hash and never edited it after seeing results.
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
