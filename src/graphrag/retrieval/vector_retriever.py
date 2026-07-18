"""Vector half of retrieval: embed the query, search Qdrant for seed passages."""
from __future__ import annotations

from graphrag.vector.embedder import Embedder
from graphrag.vector.qdrant_store import search


class VectorRetriever:
    def __init__(self, embedder: Embedder, qdrant_client, collection_name: str):
        self.embedder = embedder
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        """Returns [{"passage_id", "title", "score"}, ...] ranked best-first."""
        query_vector = self.embedder.embed_query(question)
        return search(self.qdrant_client, self.collection_name, query_vector, top_k)
