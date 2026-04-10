import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

QUERY_LOG_PATH = Path("data/processed/query_log.csv")
UNANSWERED_LOG_PATH = Path("data/processed/unanswered_log.csv")

QUERY_LOG_HEADERS = [
    "timestamp",
    "query",
    "mode",
    "qa_score",
    "chunk_score",
    "llm_used",
    "qa_threshold",
    "chunk_threshold",
    "source",
    "page",
]
UNANSWERED_LOG_HEADERS = [
    "timestamp",
    "query",
    "best_qa_score",
    "best_chunk_score",
    "best_qa_question",
    "best_qa_source",
    "best_qa_page",
]


def _ensure_log(path: Path, headers: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()


def log_query(
    query: str,
    mode: str,
    qa_score,
    chunk_score,
    llm_used: bool,
    qa_threshold: float,
    chunk_threshold: float,
    source: str,
    page,
) -> None:
    _ensure_log(QUERY_LOG_PATH, QUERY_LOG_HEADERS)
    with open(QUERY_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUERY_LOG_HEADERS)
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "mode": mode,
                "qa_score": qa_score,
                "chunk_score": chunk_score,
                "llm_used": llm_used,
                "qa_threshold": qa_threshold,
                "chunk_threshold": chunk_threshold,
                "source": source,
                "page": page,
            }
        )


def log_unanswered(
    query: str,
    best_qa_score,
    best_chunk_score,
    best_qa_match: Optional[dict],
) -> None:
    _ensure_log(UNANSWERED_LOG_PATH, UNANSWERED_LOG_HEADERS)
    best_qa_question = ""
    best_qa_source = ""
    best_qa_page = ""
    if best_qa_match:
        best_qa_question = best_qa_match.get("matched_question", "")
        best_qa_source = best_qa_match.get("source", "")
        best_qa_page = best_qa_match.get("page", "")
    with open(UNANSWERED_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNANSWERED_LOG_HEADERS)
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "best_qa_score": best_qa_score,
                "best_chunk_score": best_chunk_score,
                "best_qa_question": best_qa_question,
                "best_qa_source": best_qa_source,
                "best_qa_page": best_qa_page,
            }
        )
