import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from src.utils.config import settings

_model_instance = None


def get_model():
    global _model_instance
    if _model_instance is None:
        from sentence_transformers import SentenceTransformer
        _model_instance = SentenceTransformer(settings.embedding_model)
    return _model_instance
