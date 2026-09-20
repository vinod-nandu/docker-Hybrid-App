"""Simple recursive character chunking with overlap, page-aware."""
import uuid
from backend.config import settings


def chunk_pages(pages: list[dict], doc_id: str, doc_name: str) -> list[dict]:
    """pages: [{page, text}] -> list of chunk dicts with metadata."""
    chunks = []
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP

    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": p["page"],
                    "text": piece,
                })
            if end == len(text):
                break
            start = end - overlap
    return chunks
