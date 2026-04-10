from unittest.mock import MagicMock, patch

from src.retrieval.hybrid_query import HybridRetriever

# ── Fixture return values ─────────────────────────────────────────────────────

QA_SUCCESS = {
    "ok": True,
    "best": {
        "score": 0.82,
        "matched_question": "What is the tuition fee?",
        "answer": "The tuition fee is 9250 per year.",
        "source": "fees",
        "source_file": "fees.pdf",
        "page": 1,
        "section": "",
        "id": "qa_1",
    },
    "alternatives": [],
}

QA_FAIL_BELOW = {
    "ok": False,
    "reason": "Below threshold",
    "best_candidate": {
        "score": 0.32,
        "matched_question": "Something else",
        "source": "src",
        "page": 1,
    },
}

QA_FAIL_NONE = {
    "ok": False,
    "reason": "No results",
    "best_candidate": None,
}

CHUNK_SUCCESS = {
    "ok": True,
    "best": {
        "score": 0.61,
        "chunk_id": "fees_p1_c0",
        "text": "Tuition fees for home students are set by the government.",
        "source": "fees",
        "source_file": "fees.pdf",
        "page": 1,
        "section": "",
    },
    "results": [],
}

CHUNK_FAIL = {
    "ok": False,
    "reason": "Below threshold",
    "best": {"score": 0.18},
}

LLM_SUCCESS = {
    "ok": True,
    "answer": "Loughborough University offers a range of student support services.",
    "model": "gpt-4o-mini",
    "latency_ms": 432,
}

LLM_FAIL = {
    "ok": False,
    "reason": "API error: connection refused",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_retriever(MockQA, MockChunk, MockLLM, qa_return, chunk_return, llm_return):
    MockQA.return_value.answer.return_value = qa_return
    MockChunk.return_value.search.return_value = chunk_return
    MockLLM.return_value.generate.return_value = llm_return
    return HybridRetriever()


# ── Tests ─────────────────────────────────────────────────────────────────────

@patch("src.retrieval.hybrid_query.log_unanswered")
@patch("src.retrieval.hybrid_query.log_query")
@patch("src.retrieval.hybrid_query.LLMFallback")
@patch("src.retrieval.hybrid_query.ChunkRetriever")
@patch("src.retrieval.hybrid_query.QARetriever")
def test_qa_success_returns_qa_answer(MockQA, MockChunk, MockLLM, mock_log, mock_log_unans):
    retriever = _make_retriever(MockQA, MockChunk, MockLLM, QA_SUCCESS, CHUNK_FAIL, LLM_FAIL)
    result = retriever.search("test query")

    assert result["mode"] == "qa_answer"
    assert result["ok"] is True
    assert "qa_result" in result
    assert result["qa_result"] is QA_SUCCESS
    mock_log.assert_called_once()
    mock_log_unans.assert_not_called()


@patch("src.retrieval.hybrid_query.log_unanswered")
@patch("src.retrieval.hybrid_query.log_query")
@patch("src.retrieval.hybrid_query.LLMFallback")
@patch("src.retrieval.hybrid_query.ChunkRetriever")
@patch("src.retrieval.hybrid_query.QARetriever")
def test_qa_fail_chunk_success_returns_chunk_fallback(
    MockQA, MockChunk, MockLLM, mock_log, mock_log_unans
):
    retriever = _make_retriever(MockQA, MockChunk, MockLLM, QA_FAIL_BELOW, CHUNK_SUCCESS, LLM_FAIL)
    result = retriever.search("test query")

    assert result["mode"] == "chunk_fallback"
    assert result["ok"] is True
    assert "qa_result" in result
    assert "chunk_result" in result
    mock_log.assert_called_once()
    mock_log_unans.assert_not_called()


@patch("src.retrieval.hybrid_query.log_unanswered")
@patch("src.retrieval.hybrid_query.log_query")
@patch("src.retrieval.hybrid_query.LLMFallback")
@patch("src.retrieval.hybrid_query.ChunkRetriever")
@patch("src.retrieval.hybrid_query.QARetriever")
def test_qa_fail_chunk_fail_llm_enabled_llm_success_returns_llm_fallback(
    MockQA, MockChunk, MockLLM, mock_log, mock_log_unans
):
    retriever = _make_retriever(MockQA, MockChunk, MockLLM, QA_FAIL_NONE, CHUNK_FAIL, LLM_SUCCESS)
    result = retriever.search("test query", llm_enabled=True)

    assert result["mode"] == "llm_fallback"
    assert result["ok"] is True
    assert "qa_result" in result
    assert "chunk_result" in result
    assert "llm_result" in result
    mock_log.assert_called_once()
    mock_log_unans.assert_not_called()


@patch("src.retrieval.hybrid_query.log_unanswered")
@patch("src.retrieval.hybrid_query.log_query")
@patch("src.retrieval.hybrid_query.LLMFallback")
@patch("src.retrieval.hybrid_query.ChunkRetriever")
@patch("src.retrieval.hybrid_query.QARetriever")
def test_all_fail_returns_no_answer(MockQA, MockChunk, MockLLM, mock_log, mock_log_unans):
    retriever = _make_retriever(MockQA, MockChunk, MockLLM, QA_FAIL_NONE, CHUNK_FAIL, LLM_FAIL)
    result = retriever.search("test query", llm_enabled=False)

    assert result["mode"] == "no_answer"
    assert result["ok"] is False
    assert "qa_result" in result
    assert "chunk_result" in result
    mock_log.assert_called_once()
    mock_log_unans.assert_called_once()
