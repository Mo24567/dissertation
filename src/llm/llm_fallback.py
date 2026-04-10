import time
from src.utils.config import settings

_SYSTEM_PROMPT = (
    "You are a helpful assistant for Loughborough University students. "
    "Answer the student's question clearly and concisely based on your knowledge of "
    "UK university policies and general student life. "
    "If you are not confident in your answer, say so explicitly."
)


class LLMFallback:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def generate(self, query: str) -> dict:
        try:
            client = self._get_client()
            start = time.time()
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
            )
            latency_ms = int((time.time() - start) * 1000)
            answer = response.choices[0].message.content
            return {
                "ok": True,
                "answer": answer,
                "model": response.model,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            return {"ok": False, "reason": str(e)}
