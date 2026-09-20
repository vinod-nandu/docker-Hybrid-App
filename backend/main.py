import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.extraction import extract_pdf_pages
from backend.chunking import chunk_pages
from backend.embeddings import embed_texts, extract_entities_and_relations
from backend.vector_store import store
from backend.knowledge_graph import kg
from backend.retrieval import hybrid_retrieve
from backend.llm import generate_answer
from backend.guardrails.input_guard import check_input
from backend.guardrails.output_guard import check_grounding
from backend.models import UploadResponse, QueryRequest, QueryResponse, SourceChunk

app = FastAPI(title="Hybrid RAG (FAISS + Neo4j) with Guardrails")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp/hybrid_rag_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
def startup():
    kg.ensure_constraints()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    tmp_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pages = extract_pdf_pages(tmp_path)
        if not pages:
            raise HTTPException(400, "No extractable text found in PDF.")

        chunks = chunk_pages(pages, doc_id, file.filename)

        # --- Vector pipeline: embed + store in FAISS ---
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)
        store.add(vectors, chunks)

        # --- Knowledge graph pipeline: entities/relations -> Neo4j AuraDB ---
        kg.add_document(doc_id, file.filename, len(pages))
        total_entities = 0
        for c in chunks:
            extracted = extract_entities_and_relations(c["text"])
            kg.add_chunk_with_entities(doc_id, c, extracted["entities"], extracted["triples"])
            total_entities += len(extracted["entities"])

        return UploadResponse(
            doc_id=doc_id,
            doc_name=file.filename,
            num_pages=len(pages),
            num_chunks=len(chunks),
            num_entities=total_entities,
            message="Document processed and stored in FAISS + Neo4j.",
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # 1. Input guardrail
    guard_in = check_input(req.question)
    if not guard_in.allowed:
        return QueryResponse(answer="", sources=[], blocked=True, block_reason=guard_in.reason)

    clean_question = guard_in.redacted_text or req.question

    # 2. Hybrid retrieval (FAISS + Neo4j)
    chunks = hybrid_retrieve(clean_question, req.top_k)

    # 3. LLM generation grounded in retrieved context
    answer = generate_answer(clean_question, chunks)

    # 4. Output guardrail: grounding + disallowed-content check
    guard_out = check_grounding(answer, [c["text"] for c in chunks])
    if not guard_out.allowed:
        return QueryResponse(answer="", sources=[], blocked=True, block_reason=guard_out.reason)

    sources = [
        SourceChunk(
            chunk_id=c["chunk_id"],
            doc_name=c["doc_name"],
            page=c["page"],
            text=c["text"][:500],
            retrieval_type=c.get("retrieval_type", "vector"),
        )
        for c in chunks
    ]

    return QueryResponse(answer=answer, sources=sources, blocked=False)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
