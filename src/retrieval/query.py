import json
import numpy as np
import faiss
from pathlib import Path
from src.retrieval.model_cache import get_model
from src.utils.config import settings

INDEX_PATH = Path("data/processed/index.faiss")
META_PATH = Path("data/processed/meta.json")


class QARetriever:
    def __init__(self):
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Q&A index not found: {INDEX_PATH}. Run build_index.py first."
            )
        if not META_PATH.exists():
            raise FileNotFoundError(
                f"Q&A metadata not found: {META_PATH}. Run build_index.py first."
            )
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.model = get_model()

    def answer(self, query: str, top_k: int = None, threshold: float = None) -> dict:
        if top_k is None:
            top_k = settings.top_k
        if threshold is None:
            threshold = settings.similarity_threshold

        query = query.strip()
        if not query:
            return {"ok": False, "reason": "Empty query", "best_candidate": None}

        embedding = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = self.meta[idx]
            results.append(
                {
                    "score": float(score),
                    "matched_question": entry.get("question", ""),
                    "answer": entry.get("answer", ""),
                    "source": entry.get("source", ""),
                    "source_file": entry.get("source_file", ""),
                    "page": entry.get("page", ""),
                    "id": entry.get("id", ""),
                }
            )

        if not results:
            return {"ok": False, "reason": "No results", "best_candidate": None}

        best = results[0]

        if best["score"] < threshold:
            return {"ok": False, "reason": "Below threshold", "best_candidate": best}

        return {"ok": True, "best": best, "alternatives": results}
