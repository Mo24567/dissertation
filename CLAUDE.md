# Loughborough Student Assistant

## Project type
Final year dissertation — semantic Q&A assistant

## Python version
3.10.11

## Stack
Python, Streamlit, FAISS, sentence-transformers, OpenAI API, pypdf, pandas

## Key rules
- Never load SentenceTransformer directly, always use get_model() from model_cache.py
- Never log queries from individual retrievers, only from HybridRetriever
- Doc IDs always derived from filename stem, never positional
- ChunkRetriever always uses key "best" in both success and failure states
- LLM fallback always wrapped in try/except, never crashes app
- User app (app.py) and admin app (admin.py) are completely separate
- Python 3.10 compatibility required — no union type hints with |, use Optional and Union from typing instead

## Structure
src/ingestion/, src/retrieval/, src/llm/, src/utils/
data/raw/, data/raw_docs/, data/processed/
evaluation/, tests/
app.py, admin.py

## Environment
Requires .env file with OPENAI_API_KEY and ADMIN_PASSWORD set
