import pandas as pd
import pytest

from src.ingestion.extract_documents import chunk_text
from src.ingestion.load_dataset import clean_dataset


# ── chunk_text tests ──────────────────────────────────────────────────────────

def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("") == []


def test_chunk_text_short_stays_single_chunk():
    result = chunk_text("Short text", 800)
    assert len(result) == 1
    assert result[0] == "Short text"


def test_chunk_text_long_splits_correctly():
    long_text = "a " * 500  # 1000 chars, normalises to "a a a ..." (1000 chars, well over 800)
    result = chunk_text(long_text, 800)
    assert len(result) >= 2
    for chunk in result:
        assert len(chunk) <= 800


def test_chunk_text_no_empty_chunks():
    text = "word " * 200
    result = chunk_text(text, 800)
    for chunk in result:
        assert chunk.strip() != ""


def test_chunk_text_whitespace_normalised():
    text = "hello   world   foo"
    result = chunk_text(text, 800)
    assert result == ["hello world foo"]


# ── clean_dataset tests ───────────────────────────────────────────────────────

def _make_df(**kwargs) -> pd.DataFrame:
    base = {
        "question": ["valid question"],
        "answer": ["valid answer"],
        "source": ["src"],
        "source_file": ["file.csv"],
        "page": ["1"],
        "section": [""],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


def test_clean_dataset_removes_empty_questions():
    df = pd.DataFrame(
        {
            "question": ["valid question", "", "   "],
            "answer": ["valid answer", "valid answer", "valid answer"],
            "source": ["s", "s", "s"],
            "source_file": ["f", "f", "f"],
            "page": ["1", "1", "1"],
            "section": ["", "", ""],
        }
    )
    result = clean_dataset(df)
    assert len(result) == 1
    assert result.iloc[0]["question"] == "valid question"


def test_clean_dataset_removes_duplicates():
    df = pd.DataFrame(
        {
            "question": ["q1", "q1"],
            "answer": ["a1", "a1"],
            "source": ["s", "s"],
            "source_file": ["f", "f"],
            "page": ["1", "1"],
            "section": ["", ""],
        }
    )
    result = clean_dataset(df)
    assert len(result) == 1


def test_clean_dataset_ids_generated_correctly():
    df = pd.DataFrame(
        {
            "question": ["q1", "q2"],
            "answer": ["a1", "a2"],
            "source": ["s", "s"],
            "source_file": ["f", "f"],
            "page": ["1", "2"],
            "section": ["", ""],
        }
    )
    result = clean_dataset(df)
    assert list(result["id"]) == ["qa_1", "qa_2"]


def test_clean_dataset_whitespace_stripped():
    df = pd.DataFrame(
        {
            "question": ["  question with spaces  "],
            "answer": ["  answer with spaces  "],
            "source": ["  source  "],
            "source_file": ["  file  "],
            "page": ["1"],
            "section": [""],
        }
    )
    result = clean_dataset(df)
    assert result.iloc[0]["question"] == "question with spaces"
    assert result.iloc[0]["answer"] == "answer with spaces"
    assert result.iloc[0]["source"] == "source"


def test_clean_dataset_output_columns_correct_and_ordered():
    df = _make_df()
    result = clean_dataset(df)
    expected_cols = ["id", "question", "answer", "source", "source_file", "page", "section"]
    assert list(result.columns) == expected_cols


def test_clean_dataset_id_is_first_column():
    df = _make_df()
    result = clean_dataset(df)
    assert list(result.columns)[0] == "id"
