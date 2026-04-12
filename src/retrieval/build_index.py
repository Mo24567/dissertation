import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from src.retrieval.model_cache import get_model

QA_DATASET_PATH = Path("data/processed/qa_dataset_clean.csv")
INDEX_PATH = Path("data/processed/index.faiss")
META_PATH = Path("data/processed/meta.json")

REQUIRED_COLUMNS = ["id", "question", "answer", "source", "source_file", "page"]


def build_index():
    if not QA_DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {QA_DATASET_PATH}")

    df = pd.read_csv(QA_DATASET_PATH)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    model = get_model()
    print(f"Model: {model}")

    questions = df["question"].tolist()
    print(f"Encoding {len(questions)} questions...")

    embeddings = model.encode(
        questions,
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

    meta = df[["id", "question", "answer", "source", "source_file", "page"]].to_dict(
        orient="records"
    )
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Vectors: {index.ntotal}, Dimension: {dimension}")
    print(f"Index saved: {INDEX_PATH}")
    print(f"Meta saved: {META_PATH}")


if __name__ == "__main__":
    build_index()
