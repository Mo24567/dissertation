from src.utils.config import settings


def test_all_fields_present():
    assert hasattr(settings, "embedding_model")
    assert hasattr(settings, "top_k")
    assert hasattr(settings, "similarity_threshold")
    assert hasattr(settings, "chunk_top_k")
    assert hasattr(settings, "chunk_similarity_threshold")
    assert hasattr(settings, "llm_fallback_enabled")
    assert hasattr(settings, "openai_api_key")
    assert hasattr(settings, "openai_model")
    assert hasattr(settings, "admin_password")


def test_top_k_positive():
    assert settings.top_k > 0


def test_chunk_top_k_positive():
    assert settings.chunk_top_k > 0


def test_thresholds_in_range():
    assert 0.0 <= settings.similarity_threshold <= 1.0
    assert 0.0 <= settings.chunk_similarity_threshold <= 1.0


def test_embedding_model_nonempty_string():
    assert isinstance(settings.embedding_model, str)
    assert len(settings.embedding_model) > 0
