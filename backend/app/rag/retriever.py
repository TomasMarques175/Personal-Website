from typing import List

import numpy as np


class SimpleRetriever:
    """Embeds the question and asks FAISS for the nearest chunks.

    Smaller L2 distance means closer in embedding space. We return the
    original text so the LLM can cite it instead of inventing facts.
    """

    def __init__(self, vector_store, embeddings_service):
        self.vector_store = vector_store
        self.embeddings_service = embeddings_service

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        if self.vector_store.index is None or not self.vector_store.documents:
            return []

        k = min(top_k, len(self.vector_store.documents))
        query_embedding = np.array(
            self.embeddings_service.embed_query(query), dtype="float32"
        ).reshape(1, -1)
        distances, indices = self.vector_store.index.search(query_embedding, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            results.append(
                {
                    "text": self.vector_store.documents[int(idx)],
                    "metadata": self.vector_store.metadata[int(idx)],
                    "score": float(distance),
                }
            )
        return results
