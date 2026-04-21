import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI
from src.utils.config import settings

CHUNKS_PATH = Path("data/processed/document_chunks.csv")
DRAFTS_PATH = Path("data/processed/draft_qas.csv")

PROMPT_TEMPLATE = (
    "You are helping build a curated Q&A dataset for a semantic retrieval system for\n"
    "Loughborough University students.\n\n"
    "Generate up to 3 Q&A pairs from the chunk below.\n\n"
    "Only generate pairs if the content is reusable knowledge a student would search for.\n"
    "Good pairs capture: definitions, policies, processes, requirements, deadlines,\n"
    "rules, fees, rights, procedures, or explanations that stand alone without needing\n"
    "the original document.\n\n"
    "Do not generate pairs for: administrative metadata, staff contact details,\n"
    "document headers/footers, highly specific dates unlikely to be searched,\n"
    "content that only makes sense in context of surrounding text.\n\n"
    "Rules:\n"
    "- Use only information from the chunk\n"
    "- Do not invent or assume facts\n"
    '- Do not reference "this chunk", "this document", "the text", "the above"\n'
    "- Questions must be clear and standalone\n"
    "- Answers must be factual and complete enough to be useful on their own\n"
    "- Return empty qas array if chunk has no suitable content\n\n"
    "Metadata:\n"
    "source: {source}\n"
    "source_file: {source_file}\n"
    "page: {page}\n\n"
    "Chunk:\n"
    "{text}\n\n"
    "Respond with a JSON object matching this schema exactly:\n"
    '{{"source": "string", "source_file": "string", "page": "integer or string", '
    '"qas": [{{"question": "string", "answer": "string", "is_useful": true, "notes": "string"}}]}}'
)

BANNED_PHRASES = [
    "this chunk",
    "the chunk",
    "this text",
    "the text",
    "this document",
    "the document",
    "the above",
    "provided chunk",
]

DRAFTS_COLUMNS = [
    "chunk_id",
    "source",
    "source_file",
    "page",
    "question",
    "answer",
    "is_useful",
    "notes",
    "review_status",
]

SAVE_INTERVAL = 10
WORKERS = 5


def _load_existing_chunk_ids() -> set:
    if not DRAFTS_PATH.exists():
        return set()
    existing = set()
    with open(DRAFTS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add(row["chunk_id"])
    return existing


def _load_existing_pairs() -> set:
    if not DRAFTS_PATH.exists():
        return set()
    pairs = set()
    with open(DRAFTS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.add((row["question"].lower(), row["answer"].lower()))
    return pairs


def _validate_pair(question: str, answer: str) -> bool:
    if len(question) < 12:
        return False
    if len(answer) < 40:
        return False
    if len(answer.split()) < 6:
        return False
    if question.rstrip().endswith(":"):
        return False
    q_lower = question.lower()
    a_lower = answer.lower()
    for phrase in BANNED_PHRASES:
        if phrase in q_lower or phrase in a_lower:
            return False
    return True


def _flush_buffer(buffer: list) -> None:
    if not buffer:
        return
    with open(DRAFTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DRAFTS_COLUMNS)
        writer.writerows(buffer)


def _call_openai(client: OpenAI, chunk: dict) -> tuple:
    """Call OpenAI for one chunk. Returns (chunk, raw_qas). Raises on API error."""
    prompt = PROMPT_TEMPLATE.format(
        source=chunk["source"],
        source_file=chunk["source_file"],
        page=chunk["page"],
        text=chunk["text"],
    )
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    return chunk, data.get("qas", [])


def generate_drafts(on_progress=None):
    """Generate draft Q&A pairs from unprocessed chunks.

    Args:
        on_progress: optional callable(completed: int, total: int) called after
                     each chunk finishes, suitable for driving a UI progress bar.
    """
    if not CHUNKS_PATH.exists():
        print(f"Chunks file not found: {CHUNKS_PATH}")
        return

    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append(row)

    existing_ids = _load_existing_chunk_ids()
    unprocessed = [c for c in chunks if c["chunk_id"] not in existing_ids]

    print(
        f"Total chunks: {len(chunks)}, "
        f"Already processed: {len(existing_ids)}, "
        f"Remaining: {len(unprocessed)}"
    )

    if not unprocessed:
        print("All chunks already processed.")
        return

    DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DRAFTS_PATH.exists():
        with open(DRAFTS_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DRAFTS_COLUMNS)
            writer.writeheader()

    seen_pairs = _load_existing_pairs()
    client = OpenAI(api_key=settings.openai_api_key, timeout=60.0)
    total = len(unprocessed)
    buffer = []
    completed = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_call_openai, client, chunk): chunk for chunk in unprocessed}
        for future in as_completed(futures):
            chunk = futures[future]
            completed += 1
            try:
                _, qas = future.result()
            except Exception as e:
                _flush_buffer(buffer)
                raise RuntimeError(f"OpenAI API error on chunk {chunk['chunk_id']}: {e}") from e

            count = 0
            for qa in qas:
                if not qa.get("is_useful", False):
                    continue
                question = str(qa.get("question", "")).strip()
                answer = str(qa.get("answer", "")).strip()
                if not _validate_pair(question, answer):
                    continue
                pair_key = (question.lower(), answer.lower())
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                buffer.append({
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "source_file": chunk["source_file"],
                    "page": chunk["page"],
                    "question": question,
                    "answer": answer,
                    "is_useful": True,
                    "notes": str(qa.get("notes", "")),
                    "review_status": "pending",
                })
                count += 1

            if count == 0:
                buffer.append({
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk.get("source", ""),
                    "source_file": chunk.get("source_file", ""),
                    "page": chunk.get("page", ""),
                    "question": "",
                    "answer": "",
                    "is_useful": False,
                    "notes": "No valid pairs generated",
                    "review_status": "skipped",
                })

            print(f"[{completed}/{total}] {chunk['chunk_id']} — {count} pairs")

            if on_progress is not None:
                on_progress(completed, total)

            if completed % SAVE_INTERVAL == 0:
                _flush_buffer(buffer)
                buffer = []

    _flush_buffer(buffer)
    print("Done.")


if __name__ == "__main__":
    generate_drafts()
