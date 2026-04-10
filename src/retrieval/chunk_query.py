import json
import numpy as np
import faiss
from pathlib import Path
from src.retrieval.model_cache import get_model
from src.utils.config import settings

INDEX_PATH = Path("data/processed/chunk_index.faiss")
META_PATH = Path("data/processed/chunk_meta.json")


class ChunkRetriever:
    def __init__(self):
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Chunk index not found: {INDEX_PATH}. Run build_chunk_index.py first."
            )
        if not META_PATH.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found: {META_PATH}. Run build_chunk_index.py first."
            )
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.model = get_model()

    def search(self, query: str, top_k: int = None, threshold: float = None) -> dict:
        if top_k is None:
            top_k = settings.chunk_top_k
        if threshold is None:
            threshold = settings.chunk_similarity_threshold

        query = query.strip()
        if not query:
            return {"ok": False, "reason": "Empty query", "best": None}

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
                    "chunk_id": entry.get("chunk_id", ""),
                    "text": entry.get("text", ""),
                    "source": entry.get("source", ""),
                    "source_file": entry.get("source_file", ""),
                    "page": entry.get("page", ""),
                    "section": entry.get("section", ""),
                }
            )

        if not results:
            return {"ok": False, "reason": "No results", "best": None}

        best = results[0]

        if best["score"] < threshold:
            return {"ok": False, "reason": "Below threshold", "best": best}

        return {"ok": True, "best": best, "results": results}
