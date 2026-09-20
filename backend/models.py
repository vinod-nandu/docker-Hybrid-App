from typing import List, Optional
from pydantic import BaseModel


class ChunkMeta(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    text: str


class UploadResponse(BaseModel):
    doc_id: str
    doc_name: str
    num_pages: int
    num_chunks: int
    num_entities: int
    message: str


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class SourceChunk(BaseModel):
    chunk_id: str
    doc_name: str
    page: int
    text: str
    retrieval_type: str  # "vector" | "graph"


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    blocked: bool
    block_reason: Optional[str] = None
