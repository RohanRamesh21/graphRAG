"""Passage-level embedding (v1 embeds passages, not entities — per instructions.md).

Model: BAAI/bge-small-en-v1.5 (384-dim). Chosen for v1 because it's a strong,
widely-used open-source retrieval embedder that's small/fast enough to run on CPU for a
~6.5k-passage corpus with no GPU dependency. BGE models expect an instruction prefix on
the *query* side only (asymmetric encoding) — passages are embedded as-is.
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

EMBED_DIM = 384
QUERY_INSTRUCTION = "Represent this question for retrieving supporting Wikipedia passages: "


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            QUERY_INSTRUCTION + query, normalize_embeddings=True, show_progress_bar=False
        )
        return embedding.tolist()
