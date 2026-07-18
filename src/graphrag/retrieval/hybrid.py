"""Fuse vector-ranked and graph-ranked candidate lists via Reciprocal Rank Fusion (RRF).

RRF is chosen over a weighted sum of raw scores because vector cosine similarity and
graph hop-distance live on incomparable scales — RRF only needs each list's *rank
order*, which sidesteps having to tune a weighting between them.
"""
from __future__ import annotations

from graphrag.schemas import Passage, RetrievedPassage

RRF_K = 60  # standard RRF smoothing constant


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = RRF_K
) -> dict[str, float]:
    """ranked_lists: each a list of passage_ids, best-first. Returns {passage_id: score}."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, vector_retriever, graph_retriever, passage_lookup: dict[str, Passage]):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.passage_lookup = passage_lookup

    def retrieve(
        self, question: str, top_k_vector: int, top_k_final: int
    ) -> list[RetrievedPassage]:
        vector_hits = self.vector_retriever.retrieve(question, top_k_vector)
        vector_ranked_ids = [h["passage_id"] for h in vector_hits]
        vector_scores = {h["passage_id"]: h["score"] for h in vector_hits}

        seed_norms = self.graph_retriever.seed_entity_norms(vector_ranked_ids)
        graph_hits = self.graph_retriever.traverse(seed_norms)
        graph_ranked_ids = [h["passage_id"] for h in graph_hits]
        graph_meta = {h["passage_id"]: h for h in graph_hits}

        fused_scores = reciprocal_rank_fusion([vector_ranked_ids, graph_ranked_ids])
        ordered_ids = sorted(fused_scores, key=lambda pid: fused_scores[pid], reverse=True)

        results = []
        for pid in ordered_ids[:top_k_final]:
            passage = self.passage_lookup.get(pid)
            if passage is None:
                continue  # defensive: shouldn't happen if corpus/graph/vector stay in sync
            graph_hit = graph_meta.get(pid)
            results.append(
                RetrievedPassage(
                    passage_id=pid,
                    title=passage.title,
                    text=passage.text,
                    vector_score=vector_scores.get(pid),
                    graph_score=(1.0 / (1 + graph_hit["hops"])) if graph_hit else None,
                    fused_score=fused_scores[pid],
                    hop_path=graph_hit["hop_path"] if graph_hit else [],
                )
            )
        return results
