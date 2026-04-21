# Implementation

## 1. System Overview

The Loughborough Student Assistant is a semantic question-answering system built to help university students find answers to questions about fees, accommodation, attendance, assessments, and university regulations. It is composed of two independent Streamlit web applications — a public-facing student assistant (`app.py`) and a password-protected admin dashboard (`admin.py`) — backed by a shared pipeline of data ingestion, vector indexing, and retrieval modules.

The system is written entirely in Python 3.10 and uses no database server; all persistent state is stored as CSV files and FAISS binary indexes on disk under the `data/processed/` directory. The public app and admin app run on separate ports (8501 and 8502 respectively) and do not share Streamlit session state.

---

## 2. Architecture and Directory Structure

```
dissertation/
├── app.py                        # Student-facing UI
├── admin.py                      # Admin dashboard UI
├── src/
│   ├── ingestion/
│   │   ├── extract_documents.py  # PDF → text chunks
│   │   ├── generate_draft_qas.py # Chunks → draft Q&A pairs via OpenAI
│   │   └── load_dataset.py       # CSV Q&As → cleaned dataset
│   ├── retrieval/
│   │   ├── model_cache.py        # Singleton SentenceTransformer loader
│   │   ├── build_index.py        # Q&A dataset → FAISS index
│   │   ├── build_chunk_index.py  # Document chunks → FAISS index
│   │   ├── query.py              # QARetriever (semantic Q&A search)
│   │   ├── chunk_query.py        # ChunkRetriever (semantic passage search)
│   │   ├── keyword_retriever.py  # KeywordRetriever (BM25 baseline)
│   │   ├── hybrid_query.py       # HybridRetriever (orchestrator)
│   │   └── query_logger.py       # Query and unanswered-query logging
│   ├── llm/
│   │   └── llm_fallback.py       # OpenAI GPT-4o-mini generation layer
│   └── utils/
│       ├── config.py             # Settings loaded from .env
│       └── text_utils.py         # Word-boundary truncation utility
├── data/
│   ├── raw/                      # Hand-curated and approved Q&A CSVs
│   ├── raw_docs/                 # Uploaded university PDF documents
│   └── processed/                # All generated artifacts (indexes, logs)
├── evaluation/                   # Evaluation query sets
└── tests/                        # Pytest unit tests
```

---

## 3. Configuration (`src/utils/config.py`)

All system-wide settings are defined in a frozen `Settings` dataclass and loaded once from the environment (`.env` file). A proxy object (`settings`) wraps the internal instance so that when `reload_settings()` is called (e.g. after the admin changes thresholds), all importers automatically see the new values without re-importing.

Key settings and their defaults:

| Setting | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model used for encoding |
| `TOP_K` | `5` | Number of Q&A candidates returned by semantic search |
| `SIMILARITY_THRESHOLD` | `0.55` | Minimum cosine similarity to accept a Q&A match |
| `CHUNK_TOP_K` | `3` | Number of document passages returned by chunk search |
| `CHUNK_SIMILARITY_THRESHOLD` | `0.45` | Minimum cosine similarity to accept a passage match |
| `LLM_FALLBACK_ENABLED` | `true` | Whether to invoke the OpenAI model automatically |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model used for generation |
| `ADMIN_PASSWORD` | `admin` | Password for the admin dashboard |

---

## 4. Embedding Model (`src/retrieval/model_cache.py`)

The system uses `sentence-transformers/all-MiniLM-L6-v2` to produce 384-dimensional dense vector embeddings for both documents and queries. The model is loaded lazily on first use via a module-level singleton (`get_model()`). This ensures the model is loaded only once per process, regardless of how many retrieval components request it. The singleton pattern prevents repeated costly disk reads and model initialisation.

All embeddings are L2-normalised (`normalize_embeddings=True`), which means inner-product similarity (used by FAISS `IndexFlatIP`) is mathematically equivalent to cosine similarity. All scoring throughout the pipeline is therefore in the range [−1, 1], though in practice results fall in [0, 1] for semantically related text.

---

## 5. Data Ingestion Pipeline

The ingestion pipeline converts raw source material (PDF documents and curated CSVs) into the indexed knowledge base. It runs in the admin dashboard and has five distinct stages.

### 5.1 PDF Extraction (`src/ingestion/extract_documents.py`)

PDF files placed in `data/raw_docs/` are processed page by page using `pypdf`. Each page's text is extracted, whitespace is normalised (all runs of whitespace collapsed to single spaces), and the text is split into non-overlapping character-level chunks of 800 characters each.

Each chunk is assigned a deterministic ID of the form `{doc_id}_p{page_number}_c{chunk_index}`, where `doc_id` is the lowercase filename stem with spaces replaced by underscores. This ID scheme ensures stability — the same document always produces the same IDs — which is important for incremental processing (chunks already sent to OpenAI are skipped on re-run).

The output is written to `data/processed/document_chunks.csv` with columns: `doc_id`, `chunk_id`, `text`, `source`, `source_file`, `page`.

### 5.2 Draft Q&A Generation (`src/ingestion/generate_draft_qas.py`)

Each extracted text chunk is sent to the OpenAI API (GPT-4o-mini by default) with a carefully structured prompt requesting up to three Q&A pairs per chunk. The prompt instructs the model to generate only pairs that represent reusable student knowledge — definitions, policies, procedures, deadlines, fees, and rights — and to avoid metadata, staff contacts, and content that only makes sense in context.

Pairs are validated before saving. A pair is rejected if:
- The question is shorter than 12 characters
- The answer is shorter than 40 characters or fewer than 6 words
- The question ends with a colon (indicates an incomplete stem)
- Either the question or the answer contains self-referential phrases such as "this document", "the chunk", "the above"

Duplicate question–answer pairs are also deduplicated in memory using a set of `(question.lower(), answer.lower())` tuples.

Processing uses a `ThreadPoolExecutor` with 5 workers to parallelise OpenAI API calls. Results are buffered in memory and flushed to `data/processed/draft_qas.csv` every 10 completed chunks to protect against interruption. Crucially, the function tracks which `chunk_id` values already appear in the drafts file, so re-running after an interruption skips already-processed chunks without duplicating work.

Each draft row carries a `review_status` field initialised to `"pending"`.

### 5.3 Human Review (Admin Dashboard)

Before any AI-generated Q&A pair enters the knowledge base, it passes through a mandatory human review stage in the admin dashboard. Pending pairs are shown one at a time in the Documents → PDFs tab, with the full question, answer, source document, and page number visible. Reviewers can approve, reject, or skip each pair individually, or select multiple pairs via checkboxes and bulk-approve or bulk-reject them.

Approved pairs are appended to `data/raw/approved_qas.csv`. Rejected pairs remain in `draft_qas.csv` with `review_status = "rejected"` and can be re-approved later. Skipped pairs are hidden from the pending view but not permanently discarded.

### 5.4 Dataset Loading and Cleaning (`src/ingestion/load_dataset.py`)

All CSV files in `data/raw/` are loaded and merged. This includes both hand-curated Q&A CSVs (manually written by the administrator) and the `approved_qas.csv` generated via the PDF pipeline. Files missing `question` or `answer` columns are silently skipped. Missing metadata columns (`source`, `source_file`, `page`) are auto-populated from the filename.

After merging, the combined dataset is cleaned:
1. Whitespace is stripped from all string columns.
2. Rows with empty questions or answers are removed.
3. Questions are deduplicated on their normalised (lowercased, whitespace-collapsed) form — the first occurrence is kept.
4. An exclusion filter is applied: any question whose normalised form appears in `data/processed/excluded_qas.csv` is removed. This supports the "soft delete" feature in the admin dashboard without permanently altering the source CSVs.
5. Sequential IDs are regenerated from scratch (`qa_1`, `qa_2`, …) to ensure IDs are always contiguous and never positional in the source files.

The cleaned dataset is saved to `data/processed/qa_dataset_clean.csv`.

### 5.5 Index Building

#### Q&A Index (`src/retrieval/build_index.py`)

The `question` column of `qa_dataset_clean.csv` is encoded in batches of 32 using the SentenceTransformer model. The resulting normalised float32 embeddings are added to a FAISS `IndexFlatIP` (flat inner-product index). This is an exact nearest-neighbour index — no approximation — suitable for the dataset sizes involved (typically hundreds to low thousands of entries).

The index is saved to `data/processed/index.faiss`. A parallel metadata file `data/processed/meta.json` stores the full record for each vector (id, question, answer, source, source_file, page), keyed by the vector's positional index in FAISS. This pairing is essential: FAISS returns only indices and scores, and the metadata file maps those indices back to meaningful content.

#### Chunk Index (`src/retrieval/build_chunk_index.py`)

The same procedure is applied to the `text` column of `document_chunks.csv`. The chunk index is stored as `data/processed/chunk_index.faiss` and `data/processed/chunk_meta.json`.

---

## 6. Retrieval Pipeline

The retrieval pipeline is the core of the system. It is implemented as a three-layer cascade, orchestrated by `HybridRetriever`. Each layer is tried in sequence; as soon as a layer succeeds, the result is returned immediately without invoking later layers.

### 6.1 Layer 1 — Semantic Q&A Search (`src/retrieval/query.py`)

`QARetriever` encodes the incoming query into a normalised 384-dimensional embedding and searches the Q&A FAISS index for the top-K nearest neighbours (default K=5). Each result is scored by inner-product similarity (equivalent to cosine similarity given normalised embeddings).

The results are sorted by score descending. If the top result's score is below `SIMILARITY_THRESHOLD` (default 0.55), the retriever returns `ok=False` along with `best_candidate` (the closest non-matching entry), which the orchestrator uses for logging. If the top result meets the threshold, the retriever returns `ok=True`, `best` (the top result), and `alternatives` (all K results including the best).

The `best` dict contains: `score`, `matched_question`, `answer`, `source`, `source_file`, `page`, `id`.

### 6.2 Layer 2 — Semantic Passage Search (`src/retrieval/chunk_query.py`)

`ChunkRetriever` operates identically to `QARetriever` but searches the chunk FAISS index. It encodes the query, finds the top-K nearest document chunks (default K=3), and applies a lower threshold (default 0.45) since passage text is less semantically concentrated than a curated question.

When this layer succeeds, the result includes `best` (the highest-scoring chunk) and `results` (all K chunks), each carrying: `score`, `chunk_id`, `text`, `source`, `source_file`, `page`.

### 6.3 Layer 3 — LLM Fallback (`src/llm/llm_fallback.py`)

`LLMFallback` calls the OpenAI chat completions API with `gpt-4o-mini`. It supports two operating modes:

- **General mode** (no context): The query is sent directly to the model with a system prompt defining the assistant as a Loughborough student helper. Used when neither the Q&A index nor the chunk index found anything relevant.
- **Grounded mode** (with context): The top-K chunks returned by `ChunkRetriever` are concatenated into a context string (up to 3,000 characters), which is prepended to the user message. A stricter system prompt instructs the model to answer using only the provided excerpts and to explicitly acknowledge when information is insufficient.

The LLM client is lazily initialised on first use. All errors are caught and returned as `{"ok": False, "reason": "..."}` — the LLM layer never raises exceptions to the caller.

In the user-facing app (`app.py`), LLM fallback is disabled by default (`llm_enabled=False` on the `HybridRetriever.search()` call). The user must explicitly click a button to trigger generation. This is a deliberate design choice to give users transparency and control over when AI-generated content is shown.

### 6.4 Orchestration (`src/retrieval/hybrid_query.py`)

`HybridRetriever.search()` coordinates the three layers as follows:

```
Query
  │
  ▼
QARetriever.answer()
  ├─ ok=True  ──► return mode="qa_answer"
  └─ ok=False
       │
       ▼
     ChunkRetriever.search()
       ├─ ok=True  ──► return mode="chunk_fallback"
       └─ ok=False
            │
            ▼
          (llm_enabled?) LLMFallback.generate()
            ├─ ok=True  ──► return mode="llm_fallback"
            └─ (any path) ──► return mode="no_answer"
```

Every search call, regardless of outcome, is logged by `query_logger.log_query()`. Queries that reach `no_answer` are additionally written to a separate unanswered log for gap analysis.

---

## 7. Keyword Retrieval Baseline (`src/retrieval/keyword_retriever.py`)

A BM25 keyword retriever is implemented as a separate, independent component used exclusively in the admin testing tools. It is not part of the live retrieval pipeline shown to students.

`KeywordRetriever` tokenises all questions in `meta.json` (lowercased, punctuation stripped) and builds a `BM25Okapi` index using the `rank_bm25` library. At query time, the query is tokenised the same way and the BM25 scores are computed. The top result and its raw BM25 score are returned.

This component exists to support the BM25 vs Semantic Comparison tool in the admin dashboard, where the accuracy of keyword search and semantic search are measured head-to-head on a set of labelled queries.

---

## 8. Query Logging (`src/retrieval/query_logger.py`)

Every query processed by `HybridRetriever` is appended to `data/processed/query_log.csv`. Each log row captures: timestamp (UTC ISO format), query text, retrieval mode, Q&A score, chunk score, whether the LLM was used, the Q&A and chunk thresholds in effect at the time, the source document, and the source page.

When a query fails all layers and returns `no_answer`, it is additionally written to `data/processed/unanswered_log.csv` with the closest Q&A match details. This log surfaces gaps in the knowledge base — questions real users asked that the system could not answer — allowing the administrator to identify and fill missing content.

---

## 9. User-Facing Application (`app.py`)

The student-facing app presents a single-page search interface. It uses `@st.cache_resource` to load the `HybridRetriever` once and cache it across reruns. The cache key includes the FAISS index's last-modified timestamp (`_index_mtime()`), so the cache is automatically invalidated when the admin rebuilds the index.

Input validation is applied before any retrieval: the query must be at least 3 characters, at most 500 characters, and contain at least one alphanumeric character.

Results are rendered in a four-tier hierarchy based on the retrieval outcome:

**Tier 1 — High confidence Q&A match** (score ≥ 0.75): The answer is shown in a card with a green confidence bar indicating the match score, the matched question shown as a subtitle, and a collapsible expander listing alternative matches (the other top-K results). A "Not satisfied?" button allows the user to invoke the LLM for a general AI answer.

**Tier 2 — Low confidence Q&A match** (score ≥ threshold but < 0.75): All K candidates are shown as ranked cards with score pills. A button prompts the user to request an AI-generated answer.

**Tier 3 — Passage found** (chunk fallback): A card informs the user that relevant document passages were found but no direct answer exists. A button generates a grounded LLM response using those passages as context.

**Tier 4 — No match**: A card explains that neither the Q&A knowledge base nor the uploaded documents contained relevant information. If the user has hit no-match results two or more times in a row (tracked via `no_match_streak` in session state), a hint is shown suggesting rephrasing. A button generates a general AI answer (not grounded in any document).

A retrieval trail is rendered above every result showing which layers ran, their scores, and whether they passed or failed, giving users transparency about how their answer was found.

---

## 10. Admin Dashboard (`admin.py`)

The admin app is protected by a password check on every page load. On successful login, `admin_authenticated` is set in session state and persists for the browser session.

The dashboard is structured into four navigation pages: Documents, Knowledge Base, Evaluation, and Testing.

### 10.1 Documents Page

The Documents page has two tabs: PDFs and Q&A Files.

**PDFs tab** manages the university document corpus. Administrators upload one or more PDFs via a file uploader. On upload, the system:
1. Computes SHA-256 hashes to detect duplicate files before saving.
2. Runs `extract_documents()` to extract and chunk all PDFs in `data/raw_docs/`.
3. Runs `build_chunk_index()` to rebuild the FAISS chunk index.
4. Clears the Streamlit resource cache so the running retriever picks up the new index.

If a re-uploaded PDF has the same filename but different content, the old chunks for that filename are purged from `document_chunks.csv` before extraction runs. PDFs that produce zero chunks (e.g. image-only or password-protected files) trigger a warning without blocking other uploads.

Deleting a PDF removes it from disk, purges its chunks from the CSV, and rebuilds the Q&A index by re-running `load_dataset()` and `build_index()`. Orphaned Q&A pairs (whose `source_file` no longer exists) are also removed from `draft_qas.csv` and `approved_qas.csv`.

The **Generate Q&A Suggestions** section shows the number of unprocessed chunks and calls `generate_draft_qas.generate_drafts()` with a live progress callback that drives a Streamlit progress bar. The user must tick a confirmation checkbox before the button activates, making the OpenAI API cost explicit.

The inline **Review Q&A Suggestions** expander shows a badge count of pending items. Pairs can be approved, rejected, or skipped individually or in bulk. Approved pairs are immediately appended to `data/raw/approved_qas.csv`.

**Q&A Files tab** manages hand-curated Q&A CSVs. Administrators can upload CSVs directly (without going through the PDF → generation → review pipeline), view and edit existing Q&As in an inline data editor, and delete files. Uploading or editing triggers an automatic rebuild of `qa_dataset_clean.csv` and the FAISS Q&A index.

### 10.2 Knowledge Base Page

The Knowledge Base page shows an overview of the dataset: total Q&A count, number of sources, index size, and last-built timestamps. It provides a manual **Rebuild Index** button for cases where the index needs to be refreshed without modifying source files.

A **Deleted Q&As** section shows the current exclusion list. Administrators can restore excluded Q&As (removing them from `excluded_qas.csv`) or permanently delete them.

A knowledge base coverage curve is computed and displayed using `comparison_queries.csv`: the dataset is randomly shuffled, subsets of 10%–100% are indexed on the fly, and the percentage of evaluation queries covered at each size is plotted. This provides visual evidence of how coverage grows with dataset size.

### 10.3 Evaluation Page

The Evaluation page visualises the query log. A pie chart shows the proportion of queries answered by each retrieval layer (Q&A match, passage search, AI fallback, unanswered). Filters allow slicing by retrieval mode, date range, and minimum Q&A score. Filtered results can be exported as CSV.

An Unanswered tab shows all queries that failed all retrieval layers, with the closest Q&A match details included to guide knowledge base improvement.

### 10.4 Testing Page

The Testing page provides four tools:

**Single Query Test**: Runs one query through the live retriever and displays the full result dict, including both Q&A and chunk match details and scores.

**Evaluation Feed**: Accepts a CSV with a `query` column, fires all queries through the live pipeline, and logs results. This populates the query log without requiring real user traffic — useful during development and evaluation.

**BM25 vs Semantic Comparison**: Accepts a CSV with `query` and `expected_question` columns. Each query is run through both `KeywordRetriever` (BM25) and `QARetriever` (semantic). The matched question from each method is compared against the expected question (case-insensitive, whitespace-normalised), and per-query correctness (✓/✗) is shown alongside scores. Aggregate accuracy percentages for BM25 and semantic search are shown as summary metrics. Results can be exported as CSV.

**Full System Capture**: Accepts a CSV with a `query` column, runs all queries through the complete pipeline (with optional LLM fallback), and exports a spreadsheet with the query, the retrieval layer used, and the answer returned. A blank `Correct?` column is included for manual annotation after export.

---

## 11. Data Flow Summary

```
data/raw_docs/*.pdf
        │
        ▼  extract_documents.py
data/processed/document_chunks.csv
        │
        ├──► build_chunk_index.py ──► chunk_index.faiss + chunk_meta.json
        │
        └──► generate_draft_qas.py (OpenAI API)
                    │
                    ▼
        data/processed/draft_qas.csv   (review_status=pending)
                    │
                    ▼  Human review in admin dashboard
        data/raw/approved_qas.csv

data/raw/*.csv  (manual + approved_qas.csv)
        │
        ▼  load_dataset.py  (clean + deduplicate + apply exclusions)
data/processed/qa_dataset_clean.csv
        │
        ▼  build_index.py
data/processed/index.faiss + meta.json

                    ┌────────────────────┐
User query ──────► │   HybridRetriever  │
                    │                    │
                    │ 1. QARetriever     │──► qa_answer
                    │ 2. ChunkRetriever  │──► chunk_fallback
                    │ 3. LLMFallback     │──► llm_fallback / no_answer
                    └────────────────────┘
                              │
                              ▼
                    query_log.csv / unanswered_log.csv
```

---

## 12. Key Design Decisions

**Separation of user and admin apps.** Running `app.py` and `admin.py` as entirely separate Streamlit processes means the admin pipeline (which is CPU- and API-intensive) cannot affect the user-facing app's responsiveness. Index rebuilds in the admin app are detected by the user app via the FAISS file's modification timestamp.

**Cache invalidation via mtime.** The user app's `@st.cache_resource` key includes `_index_mtime()`, so it automatically reloads the retriever the next time a user submits a query after an admin rebuild — without requiring a server restart.

**Human-in-the-loop Q&A generation.** No AI-generated Q&A pair enters the knowledge base without human approval. This prevents hallucinations and factual errors from the generation step from becoming retrievable answers.

**Incremental chunk processing.** The draft generation step tracks which `chunk_id` values have already been processed. If the process is interrupted mid-way (e.g. due to API timeout), re-running resumes from the last unprocessed chunk rather than starting over.

**Soft deletion.** Q&A pairs are never permanently deleted from the source CSVs via the exclusion mechanism. The exclusion list is applied at dataset-cleaning time. This means a mistakenly excluded question can be restored without data loss.

**Tiered confidence display.** The 0.75 high-confidence threshold in the user app is a separate constant from the retrieval threshold (0.55). The retrieval threshold determines whether a result is returned at all; the display threshold determines whether to present it as a confident single answer (Tier 1) or as a list of candidates (Tier 2). This two-threshold design prevents borderline matches from being presented with false confidence.

**No logging below HybridRetriever.** Individual retrievers (`QARetriever`, `ChunkRetriever`) do not log queries. Only `HybridRetriever` logs, once per search call with the final mode. This prevents double-counting and ensures the log accurately reflects the user's experience.
