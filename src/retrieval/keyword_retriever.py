import json
import re
from pathlib import Path
from typing import Dict, List

META_PATH = Path("data/processed/meta.json")


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^\w\s]", "", str(text).lower()).split()


class KeywordRetriever:
    """BM25-based keyword retriever over the Q&A dataset.

    Used as a baseline to compare against semantic search.
    Searches question text only — same corpus as the semantic Q&A layer.
    """

    def __init__(self) -> None:
        if not META_PATH.exists():
            raise FileNotFoundError(
                "meta.json not found — build the Q&A index first."
            )
        with open(META_PATH, encoding="utf-8") as f:
            self.meta = json.load(f)
        if not self.meta:
            raise ValueError("meta.json is empty — no Q&A pairs to index.")

        from rank_bm25 import BM25Okapi
        tokenized = [_tokenize(item["question"]) for item in self.meta]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str) -> Dict:
        """Return the top BM25 match for a query.

        Returns a dict with:
            ok              bool   — True if at least one keyword matched
            score           float  — raw BM25 score (not normalised to 0-1)
            matched_question str   — the question text of the top result
            answer          str   — the answer text of the top result
            id              str   — the Q&A pair ID (e.g. "qa_42")
        """
        if not query.strip():
            return {
                "ok": False,
                "score": 0.0,
                "matched_question": "",
                "answer": "",
                "id": "",
            }

        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        best_idx = int(scores.argsort()[::-1][0])
        best_score = float(scores[best_idx])
        best = self.meta[best_idx]

        return {
            "ok": best_score > 0.0,
            "score": best_score,
            "matched_question": best.get("question", ""),
            "answer": best.get("answer", ""),
            "id": best.get("id", ""),
        }
