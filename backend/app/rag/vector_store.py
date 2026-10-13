from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np

from backend.app.config import settings


class FAISSVectorStore:
    """Stores embedding vectors in FAISS and the original chunks on disk.

    FAISS only stores numbers. The matching text and metadata live in
    corpus.pkl — without that file, a loaded index cannot return useful
    retrieval results.
    """

    def __init__(self, index_path: str | None = None):
        self.index_path = Path(index_path or settings.VECTOR_INDEX_PATH)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.documents: List[str] = []
        self.metadata: List[dict] = []

    @property
    def index_file(self) -> Path:
        return self.index_path / "portfolio.index"

    @property
    def corpus_file(self) -> Path:
        return self.index_path / "corpus.pkl"

    def build(self, embeddings: List[List[float]], documents: List[str], metadata: List[dict]):
        if not embeddings:
            return

        vector_array = np.array(embeddings, dtype="float32")
        dimension = vector_array.shape[1]
        # Exact L2 search. Fine for a few hundred CV chunks; IVF/HNSW would
        # only matter at much larger scale.
        index = faiss.IndexFlatL2(dimension)
        index.add(vector_array)

        self.index = index
        self.documents = documents
        self.metadata = metadata

    def save(self):
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_file))
        with self.corpus_file.open("wb") as handle:
            pickle.dump({"documents": self.documents, "metadata": self.metadata}, handle)

    def load(self) -> bool:
        if not self.index_file.exists() or not self.corpus_file.exists():
            return False
        self.index = faiss.read_index(str(self.index_file))
        with self.corpus_file.open("rb") as handle:
            payload = pickle.load(handle)
        self.documents = payload.get("documents", [])
        self.metadata = payload.get("metadata", [])
        if self.index.ntotal != len(self.documents):
            self.index = None
            self.documents = []
            self.metadata = []
            return False
        return True
