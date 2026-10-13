from typing import List


class SimpleChunker:
    """Splits long text into overlapping windows.

    Overlap keeps a sentence that sits on a boundary from being cut in half
    without any neighbouring context. Character-based splitting is enough for
    a small markdown knowledge base; later you can swap this for recursive
    splitters from LangChain if the corpus grows.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.chunk_size, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == length:
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks
