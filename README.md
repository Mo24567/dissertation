# Loughborough Student Assistant

A semantic Q&A assistant for Loughborough University students. Ask questions about fees, accommodation, attendance, assessments, and university regulations. Answers are drawn directly from official university documents.

## Overview

The system uses a three-layer retrieval pipeline:

1. **Semantic Q&A matching** - queries are matched against a curated knowledge base of question and answer pairs using sentence embeddings and FAISS vector search.
2. **Passage search** - if no Q&A match is found, relevant passages are retrieved from uploaded PDFs using a separate chunk index.
3. **Generated answer fallback** - if no passage is found, an answer is generated using the OpenAI API, grounded in any retrieved context.

## Requirements

- Python 3.10.11
- A `.env` file with the following variables:

```
OPENAI_API_KEY=your_key_here
ADMIN_PASSWORD=your_password_here
```

Optional `.env` settings (with defaults):

```
SIMILARITY_THRESHOLD=0.75
CHUNK_SIMILARITY_THRESHOLD=0.5
LLM_FALLBACK_ENABLED=true
OPENAI_MODEL=gpt-4o-mini
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the apps

**Student assistant** (port 8501):
```bash
streamlit run app.py
```

**Admin dashboard** (port 8502):
```bash
streamlit run admin.py --server.port 8502
```

## Admin workflow

1. **Upload PDFs** - go to Documents and upload university PDF documents.
2. **Generate Q&A suggestions** - the system extracts text passages and generates question and answer pairs via the OpenAI API.
3. **Review suggestions** - approve or reject generated pairs in the review panel.
4. **Rebuild the index** - approved pairs are embedded and indexed, making them searchable.
5. **Upload Q&A files** - alternatively, upload a CSV with `question` and `answer` columns directly.

## Project structure

```
app.py                        # Student-facing assistant
admin.py                      # Admin dashboard

src/
  ingestion/
    extract_documents.py      # PDF text extraction and chunking
    generate_draft_qas.py     # Q&A pair generation from chunks
    load_dataset.py           # Dataset loading and deduplication
  retrieval/
    hybrid_query.py           # Three-layer retrieval orchestrator
    query.py                  # Semantic Q&A retrieval (FAISS)
    chunk_query.py            # Passage chunk retrieval (FAISS)
    keyword_retriever.py      # BM25 keyword search
    build_index.py            # Q&A index builder
    build_chunk_index.py      # Chunk index builder
    model_cache.py            # Sentence transformer model cache
    query_logger.py           # Query and unanswered log writers
  llm/
    llm_fallback.py           # OpenAI answer generation
  utils/
    config.py                 # Settings loaded from .env
    text_utils.py             # Text processing utilities

data/
  raw/                        # Uploaded Q&A CSV files
  raw_docs/                   # Uploaded PDF documents
  processed/                  # Indexes, cleaned dataset, logs

evaluation/                   # Evaluation query CSVs
tests/                        # Unit and integration tests
```

## Evaluation

The admin dashboard includes an Evaluation page with:
- Knowledge base pipeline metrics (PDFs, passages, Q&As, approved, indexed)
- Query activity over time
- Retrieval performance breakdown by layer
- Knowledge base coverage curve (query coverage vs KB size)
- Match quality scores and confidence bands
- Gap analysis from the unanswered query log

Use the Testing page to run individual queries or batch feeds against the live system.

## Stack

| Component | Library |
|---|---|
| Interface | Streamlit |
| Vector search | FAISS |
| Sentence embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Answer generation | OpenAI API |
| PDF extraction | pypdf |
| Data handling | pandas |
| Keyword search | rank-bm25 |
