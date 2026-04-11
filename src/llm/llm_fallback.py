import time
from typing import Optional
from src.utils.config import settings

_SYSTEM_PROMPT = (
    "You are a helpful assistant for Loughborough University students. "
    "Answer the student's question clearly and concisely based on your knowledge of "
    "UK university policies and general student life. "
    "If you are not confident in your answer, say so explicitly."
)

_SYSTEM_PROMPT_GROUNDED = (
    "You are a helpful assistant for Loughborough University students. "
    "Answer the student's question using ONLY the provided document excerpts. "
    "Be clear and concise. If the excerpts do not contain enough information "
    "to fully answer the question, say so and summarise what is available. "
    "Do not invent information not present in the excerpts."
)


class LLMFallback:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def generate(self, query: str, context: Optional[str] = None) -> dict:
        try:
            client = self._get_client()
            start = time.time()

            if context:
                system = _SYSTEM_PROMPT_GROUNDED
                user_content = f"Document excerpts:\n\n{context}\n\nStudent question: {query}"
            else:
                system = _SYSTEM_PROMPT
                user_content = query

            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            )
            latency_ms = int((time.time() - start) * 1000)
            answer = response.choices[0].message.content
            return {
                "ok": True,
                "answer": answer,
                "model": response.model,
                "latency_ms": latency_ms,
                "grounded": context is not None,
            }
        except Exception as e:
            return {"ok": False, "reason": str(e)}
