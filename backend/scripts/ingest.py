"""Rebuild the FAISS index from markdown/csv under data/ and write a Pandas inventory."""

from backend.app.config import settings
from backend.app.rag.pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline()
    chunks = pipeline.build_index()
    print(f"Indexed {len(chunks)} chunks.")
    print(f"FAISS index: {pipeline.vector_store.index_file}")
    print(f"Inventory: {settings.CHUNKS_INVENTORY_PATH}")


if __name__ == "__main__":
    main()
