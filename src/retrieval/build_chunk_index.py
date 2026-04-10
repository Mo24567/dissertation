import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from src.retrieval.model_cache import get_model

CHUNKS_PATH = Path("data/processed/document_chunks.csv")
INDEX_PATH = Path("data/processed/chunk_index.faiss")
META_PATH = Path("data/processed/chunk_meta.json")

REQUIRED_COLUMNS = ["doc_id", "chunk_id", "text", "source", "source_file", "page", "section"]


def build_chunk_index():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    df = pd.read_csv(CHUNKS_PATH)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    model = get_model()
    print(f"Model: {model}")

    texts = df["text"].tolist()
    print(f"Encoding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    meta = df[
        ["doc_id", "chunk_id", "text", "source", "source_file", "page", "section"]
    ].to_dict(orient="records")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Chunks indexed: {index.ntotal}, Dimension: {dimension}")
    print(f"Index saved: {INDEX_PATH}")
    print(f"Meta saved: {META_PATH}")


if __name__ == "__main__":
    build_chunk_index()
