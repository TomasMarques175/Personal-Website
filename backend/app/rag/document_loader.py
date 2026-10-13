from pathlib import Path
from typing import List


class Document:
    """Minimal stand-in for a LangChain Document.

    Keeping our own type makes the ingestion path easy to explain:
    text plus metadata, nothing else.
    """

    def __init__(self, text: str, metadata: dict | None = None):
        self.page_content = text
        self.metadata = metadata or {}


class LocalDocumentLoader:
    """Walks a folder and loads .md / .txt / .csv files as documents."""

    SUPPORTED = {".txt", ".md", ".csv"}

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def load(self) -> List[Document]:
        documents: List[Document] = []
        if not self.directory.exists():
            return documents

        for file_path in sorted(self.directory.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in self.SUPPORTED:
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            try:
                source = str(file_path.relative_to(self.directory))
            except ValueError:
                source = file_path.name
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source": source,
                        "filename": file_path.name,
                        "type": file_path.suffix.lower().lstrip("."),
                    },
                )
            )
        return documents
