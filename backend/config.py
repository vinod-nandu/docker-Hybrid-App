import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1536"))

    # Neo4j AuraDB
    NEO4J_URI: str = os.getenv("NEO4J_URI", "")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # FAISS / storage
    FAISS_INDEX_DIR: str = os.getenv("FAISS_INDEX_DIR", "./storage")
    FAISS_INDEX_FILE: str = os.path.join(FAISS_INDEX_DIR, "faiss.index")
    METADATA_FILE: str = os.path.join(FAISS_INDEX_DIR, "metadata.json")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Retrieval
    TOP_K_VECTOR: int = int(os.getenv("TOP_K_VECTOR", "5"))
    TOP_K_GRAPH: int = int(os.getenv("TOP_K_GRAPH", "5"))

    # Guardrails
    MAX_QUERY_CHARS: int = int(os.getenv("MAX_QUERY_CHARS", "2000"))
    MIN_GROUNDING_OVERLAP: float = float(os.getenv("MIN_GROUNDING_OVERLAP", "0.15"))

    # App
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")


settings = Settings()
os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
