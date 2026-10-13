from langchain_huggingface import HuggingFaceEmbeddings

from backend.app.config import settings


class EmbeddingService:
    """Turns text into dense vectors with a sentence-transformers model.

    HuggingFaceEmbeddings loads the model with PyTorch. We do not train
    anything here: the model is already trained to put similar sentences
    close together in vector space (cosine / L2).
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def embed_documents(self, texts):
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str):
        return self._embeddings.embed_query(text)
