import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    embedding_model: str
    top_k: int
    similarity_threshold: float
    chunk_top_k: int
    chunk_similarity_threshold: float
    llm_fallback_enabled: bool
    openai_api_key: str
    openai_model: str
    admin_password: str


def _load_settings() -> Settings:
    return Settings(
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        top_k=int(os.getenv("TOP_K", "5")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.55")),
        chunk_top_k=int(os.getenv("CHUNK_TOP_K", "3")),
        chunk_similarity_threshold=float(os.getenv("CHUNK_SIMILARITY_THRESHOLD", "0.45")),
        llm_fallback_enabled=os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true",
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        admin_password=os.getenv("ADMIN_PASSWORD", "admin"),
    )


class _SettingsProxy:
    """Proxy so reload_settings() is visible to all importers without re-importing."""
    def __getattr__(self, name: str):
        return getattr(_current, name)

    def __repr__(self) -> str:
        return repr(_current)


_current: Settings = _load_settings()
settings: Settings = _SettingsProxy()  # type: ignore[assignment]


def reload_settings() -> None:
    """Re-read .env and refresh the live settings object in place."""
    global _current
    load_dotenv(override=True)
    _current = _load_settings()
