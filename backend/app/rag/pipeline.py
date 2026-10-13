from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from backend.app.config import settings
from backend.app.rag.chunker import SimpleChunker
from backend.app.rag.document_loader import LocalDocumentLoader
from backend.app.rag.embeddings import EmbeddingService
from backend.app.rag.llm import build_llm
from backend.app.rag.retriever import SimpleRetriever
from backend.app.rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir or settings.DATA_DIR)
        self.chunker = SimpleChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = FAISSVectorStore(index_path=settings.VECTOR_INDEX_PATH)
        self.llm = build_llm()
        self.retriever = SimpleRetriever(self.vector_store, self.embedding_service)

    def ensure_index(self):
        if self.vector_store.load():
            logger.info(
                "Loaded FAISS index with %s chunks from %s",
                len(self.vector_store.documents),
                settings.VECTOR_INDEX_PATH,
            )
            return
        logger.info("No valid index on disk; building from %s", self.data_dir)
        self.build_index()

    def build_index(self):
        loader = LocalDocumentLoader(self.data_dir)
        raw_documents = loader.load()
        all_chunks: List[str] = []
        all_metadata: List[dict] = []

        for document in raw_documents:
            chunks = self.chunker.chunk(document.page_content)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({**document.metadata, "text_preview": chunk[:200]})

        self._write_inventory(all_chunks, all_metadata)

        if not all_chunks:
            self.vector_store.index = None
            self.vector_store.documents = []
            self.vector_store.metadata = []
            return []

        embeddings = self.embedding_service.embed_documents(all_chunks)
        self.vector_store.build(embeddings, all_chunks, all_metadata)
        self.vector_store.save()
        logger.info("Indexed %s chunks from %s files", len(all_chunks), len(raw_documents))
        return all_chunks

    def _write_inventory(self, chunks: List[str], metadata: List[dict]):
        """Pandas is used here on purpose: a readable table of what went into FAISS."""
        settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "source": meta.get("source", ""),
                "filename": meta.get("filename", ""),
                "n_chars": len(chunk),
                "preview": chunk[:200].replace("\n", " "),
            }
            for chunk, meta in zip(chunks, metadata)
        ]
        frame = pd.DataFrame(rows, columns=["source", "filename", "n_chars", "preview"])
        frame.to_csv(settings.CHUNKS_INVENTORY_PATH, index=False)

    def query(self, question: str, top_k: int | None = None):
        if self.vector_store.index is None:
            self.ensure_index()

        if self.vector_store.index is None:
            return {"answer": "Ainda não há contexto indexado.", "source_documents": []}

        results = self.retriever.retrieve(question, top_k=top_k or settings.TOP_K)
        answer = self.llm.generate(question, results)
        sources = []
        for doc in results:
            source = doc["metadata"].get("source", "unknown")
            if source not in sources:
                sources.append(source)
        return {"answer": answer, "source_documents": sources}
