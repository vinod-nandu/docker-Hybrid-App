"""Hybrid retrieval: FAISS vector search + Neo4j graph traversal, merged & deduped."""
import re
from backend.config import settings
from backend.vector_store import store
from backend.knowledge_graph import kg
from backend.embeddings import embed_query

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "it", "this", "that", "with", "as", "by", "be", "at", "from",
    "what", "who", "when", "where", "why", "how", "does", "do", "did",
}


def _naive_keyword_entities(question: str) -> list[str]:
    """Very lightweight entity-ish extraction from the query: capitalized words + noun-ish tokens."""
    caps = re.findall(r"\b[A-Z][a-zA-Z0-9\-]{2,}\b", question)
    words = [w for w in re.findall(r"[a-zA-Z0-9]+", question) if w.lower() not in _STOPWORDS and len(w) > 3]
    candidates = list(dict.fromkeys(caps + words))  # dedupe, keep order
    return candidates[:8]


def hybrid_retrieve(question: str, top_k: int | None = None) -> list[dict]:
    top_k = top_k or settings.TOP_K_VECTOR

    # 1. Vector retrieval from FAISS
    q_vec = embed_query(question)
    vector_hits = store.search(q_vec, top_k)
    for h in vector_hits:
        h["retrieval_type"] = "vector"

    # 2. Graph retrieval from Neo4j via naive entity matching
    entities = _naive_keyword_entities(question)
    graph_hits = kg.get_chunks_for_entities(entities, settings.TOP_K_GRAPH)
    for h in graph_hits:
        h["retrieval_type"] = "graph"

    # 3. Merge & dedupe by chunk_id, vector hits take priority on conflict
    merged: dict[str, dict] = {}
    for h in graph_hits:
        merged[h["chunk_id"]] = h
    for h in vector_hits:
        merged[h["chunk_id"]] = h  # overwrite so vector score/type wins

    return list(merged.values())
