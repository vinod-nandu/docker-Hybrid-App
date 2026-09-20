"""OpenAI embedding + lightweight LLM-based entity/relation extraction."""
import json
from openai import OpenAI
from backend.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of strings using OpenAI embeddings API."""
    if not texts:
        return []
    resp = client.embeddings.create(model=settings.OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


ENTITY_EXTRACTION_PROMPT = """Extract key entities and relationships from the text below.
Return STRICT JSON only, no prose, in this exact shape:
{{"entities": ["Entity A", "Entity B"], "triples": [["Entity A", "relation", "Entity B"]]}}
Keep entities short (proper nouns, key concepts). Max 8 entities, max 8 triples.

TEXT:
{chunk_text}
"""


def extract_entities_and_relations(chunk_text: str) -> dict:
    """Uses the chat model to pull a small knowledge graph out of one chunk."""
    try:
        resp = client.chat.completions.create(
            model=settings.OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You extract concise knowledge graphs from text and reply only with JSON."},
                {"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(chunk_text=chunk_text[:3000])},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        entities = [e.strip() for e in data.get("entities", []) if isinstance(e, str) and e.strip()]
        triples = [
            t for t in data.get("triples", [])
            if isinstance(t, list) and len(t) == 3
        ]
        return {"entities": entities, "triples": triples}
    except Exception:
        return {"entities": [], "triples": []}
