import csv
import re
from pathlib import Path
from pypdf import PdfReader

RAW_DOCS_DIR = Path("data/raw_docs")
OUTPUT_PATH = Path("data/processed/document_chunks.csv")
CHUNK_SIZE = 800
OUTPUT_COLUMNS = ["doc_id", "chunk_id", "text", "source", "source_file", "page", "section"]


def clean_text(text: str) -> str:
    """Collapse all whitespace to single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """Split text into non-overlapping chunks of chunk_size characters.

    Normalises whitespace before splitting. Filters empty chunks.
    """
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end
    return chunks


def extract_documents():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = list(RAW_DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/raw_docs/")
        return

    all_chunks = []

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem.lower().replace(" ", "_")
        print(f"Processing {pdf_path.name}...")

        try:
            reader = PdfReader(str(pdf_path))
            page_count = 0
            chunk_count = 0

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = clean_text(text)
                if not text:
                    continue

                page_chunks = chunk_text(text)
                for chunk_idx, chunk in enumerate(page_chunks):
                    chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"
                    all_chunks.append(
                        {
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "text": chunk,
                            "source": pdf_path.stem,
                            "source_file": pdf_path.name,
                            "page": page_num,
                            "section": "",
                        }
                    )
                    chunk_count += 1
                page_count += 1

            print(f"  -> {page_count} pages, {chunk_count} chunks")
        except Exception as e:
            print(f"  ERROR processing {pdf_path.name}: {e}")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(all_chunks)

    print(f"\nTotal: {len(all_chunks)} chunks from {len(pdf_files)} PDFs -> {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_documents()
