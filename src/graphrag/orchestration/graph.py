"""Stage 6: LangGraph orchestration — retrieve -> generate -> validate, with at most one
re-retrieval loop if the first pass comes back empty. Kept deliberately linear per
instructions.md: "a linear retrieve-then-generate flow with one optional re-retrieval
loop is enough" — no deeper agent loop before the baseline is validated end to end.
"""
from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from graphrag.retrieval.hybrid import HybridRetriever
from graphrag.schemas import RetrievedPassage

logger = logging.getLogger(__name__)

MAX_RETRIES = 1
EXPAND_TOP_K_STEP = 5


class GraphRAGState(TypedDict):
    question: str
    top_k_vector: int
    top_k_final: int
    retrieved: list[RetrievedPassage]
    answer: str
    supporting_passage_ids: list[str]
    retries: int


def _low_confidence(state: GraphRAGState) -> bool:
    return not state["answer"].strip() or not state["supporting_passage_ids"]


def build_graphrag_app(hybrid_retriever: HybridRetriever, gemini_generator):
    async def retrieve_node(state: GraphRAGState) -> dict:
        retrieved = hybrid_retriever.retrieve(
            state["question"], state["top_k_vector"], state["top_k_final"]
        )
        return {"retrieved": retrieved}

    async def generate_node(state: GraphRAGState) -> dict:
        passages = [
            {"passage_id": r.passage_id, "title": r.title, "text": r.text}
            for r in state["retrieved"]
        ]
        answer, supporting_ids = await gemini_generator.generate(state["question"], passages)
        return {"answer": answer, "supporting_passage_ids": supporting_ids}

    def validate_route(state: GraphRAGState) -> str:
        if _low_confidence(state) and state["retries"] < MAX_RETRIES:
            return "expand"
        return "end"

    async def expand_node(state: GraphRAGState) -> dict:
        logger.info("Low-confidence answer for %r — expanding retrieval and retrying.", state["question"])
        return {
            "top_k_vector": state["top_k_vector"] + EXPAND_TOP_K_STEP,
            "top_k_final": state["top_k_final"] + EXPAND_TOP_K_STEP,
            "retries": state["retries"] + 1,
        }

    graph = StateGraph(GraphRAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("expand", expand_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", validate_route, {"expand": "expand", "end": END})
    graph.add_edge("expand", "retrieve")

    return graph.compile()


async def answer_question(
    app, question: str, top_k_vector: int, top_k_final: int
) -> GraphRAGState:
    initial_state: GraphRAGState = {
        "question": question,
        "top_k_vector": top_k_vector,
        "top_k_final": top_k_final,
        "retrieved": [],
        "answer": "",
        "supporting_passage_ids": [],
        "retries": 0,
    }
    final_state = await app.ainvoke(initial_state)
    return final_state
