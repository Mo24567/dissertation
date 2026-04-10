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


settings = _load_settings()
