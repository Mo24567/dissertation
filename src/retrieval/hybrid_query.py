from src.retrieval.query import QARetriever
from src.retrieval.chunk_query import ChunkRetriever
from src.llm.llm_fallback import LLMFallback
from src.retrieval.query_logger import log_query, log_unanswered
from src.utils.config import settings


class HybridRetriever:
    def __init__(self):
        self.qa_retriever = QARetriever()
        self.chunk_retriever = ChunkRetriever()
        self.llm_fallback = LLMFallback()

    def search(
        self,
        query: str,
        qa_top_k: int = None,
        qa_threshold: float = None,
        chunk_top_k: int = None,
        chunk_threshold: float = None,
        llm_enabled: bool = None,
    ) -> dict:
        # Resolve parameters — use passed value if not None, else fall back to settings
        qa_top_k = qa_top_k if qa_top_k is not None else settings.top_k
        qa_threshold = qa_threshold if qa_threshold is not None else settings.similarity_threshold
        chunk_top_k = chunk_top_k if chunk_top_k is not None else settings.chunk_top_k
        chunk_threshold = (
            chunk_threshold if chunk_threshold is not None else settings.chunk_similarity_threshold
        )
        llm_enabled = llm_enabled if llm_enabled is not None else settings.llm_fallback_enabled

        # Step 1: Q&A retrieval
        qa_result = self.qa_retriever.answer(query, top_k=qa_top_k, threshold=qa_threshold)

        if qa_result["ok"]:
            best = qa_result["best"]
            log_query(
                query=query,
                mode="qa_answer",
                qa_score=best["score"],
                chunk_score=None,
                llm_used=False,
                qa_threshold=qa_threshold,
                chunk_threshold=chunk_threshold,
                source=best.get("source", ""),
                page=best.get("page", ""),
            )
            return {"mode": "qa_answer", "ok": True, "qa_result": qa_result}

        # Step 2: Chunk retrieval
        chunk_result = self.chunk_retriever.search(query, top_k=chunk_top_k, threshold=chunk_threshold)

        best_qa_score = None
        if qa_result.get("best_candidate"):
            best_qa_score = qa_result["best_candidate"]["score"]

        if chunk_result["ok"]:
            best_chunk = chunk_result["best"]
            log_query(
                query=query,
                mode="chunk_fallback",
                qa_score=best_qa_score,
                chunk_score=best_chunk["score"],
                llm_used=False,
                qa_threshold=qa_threshold,
                chunk_threshold=chunk_threshold,
                source=best_chunk.get("source", ""),
                page=best_chunk.get("page", ""),
            )
            return {
                "mode": "chunk_fallback",
                "ok": True,
                "qa_result": qa_result,
                "chunk_result": chunk_result,
            }

        best_chunk_score = None
        if chunk_result.get("best"):
            best_chunk_score = chunk_result["best"]["score"]

        # Step 3: LLM fallback
        if llm_enabled:
            llm_result = self.llm_fallback.generate(query)
            if llm_result["ok"]:
                log_query(
                    query=query,
                    mode="llm_fallback",
                    qa_score=best_qa_score,
                    chunk_score=best_chunk_score,
                    llm_used=True,
                    qa_threshold=qa_threshold,
                    chunk_threshold=chunk_threshold,
                    source="",
                    page="",
                )
                return {
                    "mode": "llm_fallback",
                    "ok": True,
                    "qa_result": qa_result,
                    "chunk_result": chunk_result,
                    "llm_result": llm_result,
                }

        # Step 4: No answer
        log_query(
            query=query,
            mode="no_answer",
            qa_score=best_qa_score,
            chunk_score=best_chunk_score,
            llm_used=False,
            qa_threshold=qa_threshold,
            chunk_threshold=chunk_threshold,
            source="",
            page="",
        )
        log_unanswered(
            query=query,
            best_qa_score=best_qa_score,
            best_chunk_score=best_chunk_score,
            best_qa_match=qa_result.get("best_candidate"),
        )
        return {
            "mode": "no_answer",
            "ok": False,
            "qa_result": qa_result,
            "chunk_result": chunk_result,
        }
