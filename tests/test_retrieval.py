from graphrag.retrieval.graph_retriever import NullGraphRetriever
from graphrag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from graphrag.schemas import Passage


def test_rrf_single_list_preserves_order():
    scores = reciprocal_rank_fusion([["a", "b", "c"]])
    ranked = sorted(scores, key=lambda pid: scores[pid], reverse=True)
    assert ranked == ["a", "b", "c"]


def test_rrf_boosts_items_ranked_highly_in_both_lists():
    # "b" is top of the graph list and second in vector list -> should outrank "a",
    # which only appears in the vector list.
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]])
    ranked = sorted(scores, key=lambda pid: scores[pid], reverse=True)
    assert ranked[0] == "b"


def test_rrf_item_in_no_lists_is_absent():
    scores = reciprocal_rank_fusion([["a", "b"]])
    assert "z" not in scores


class _FakeVectorRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, question, top_k):
        return self.hits[:top_k]


def test_hybrid_retriever_degenerates_to_vector_only_with_null_graph_retriever():
    """This is exactly the mechanism the eval baseline relies on: plugging
    NullGraphRetriever into HybridRetriever should reproduce pure vector ranking."""
    passages = {
        "p1": Passage(id="p1", title="T1", text="text one"),
        "p2": Passage(id="p2", title="T2", text="text two"),
    }
    vector_retriever = _FakeVectorRetriever(
        [
            {"passage_id": "p1", "title": "T1", "score": 0.9},
            {"passage_id": "p2", "title": "T2", "score": 0.5},
        ]
    )
    hybrid = HybridRetriever(vector_retriever, NullGraphRetriever(), passages)

    results = hybrid.retrieve("some question", top_k_vector=5, top_k_final=5)

    assert [r.passage_id for r in results] == ["p1", "p2"]
    assert all(r.graph_score is None for r in results)
    assert all(r.hop_path == [] for r in results)
