"""Local FAISS vector store with a JSON sidecar for chunk metadata."""
import json
import os
import threading
import numpy as np
import faiss
from backend.config import settings

_lock = threading.Lock()


class FaissStore:
    def __init__(self):
        self.dim = settings.EMBEDDING_DIM
        self.index_path = settings.FAISS_INDEX_FILE
        self.meta_path = settings.METADATA_FILE
        self.metadata: list[dict] = []  # row i -> chunk metadata
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)
        else:
            # Inner product on L2-normalized vectors == cosine similarity
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    def add(self, vectors: list[list[float]], chunks: list[dict]):
        with _lock:
            arr = np.array(vectors, dtype="float32")
            arr = self._normalize(arr)
            self.index.add(arr)
            self.metadata.extend(chunks)
            self._save()

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        arr = np.array([query_vector], dtype="float32")
        arr = self._normalize(arr)
        scores, idxs = self.index.search(arr, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            meta = dict(self.metadata[idx])
            meta["score"] = float(score)
            results.append(meta)
        return results


store = FaissStore()
