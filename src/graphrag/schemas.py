"""Shared pydantic schemas used across ingestion, extraction, graph, retrieval, and eval."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


def normalize_name(name: str) -> str:
    """Casefold + strip + collapse whitespace. The only entity resolution done in v1 —
    no coreference or ontology alignment (see README limitations)."""
    return " ".join(name.casefold().strip().split())


class Passage(BaseModel):
    """One deduplicated context paragraph from the pooled 2WikiMultiHopQA corpus."""

    id: str  # stable hash of (title, text)
    title: str
    text: str
    source_question_ids: list[str] = Field(default_factory=list)


class SampledQuestion(BaseModel):
    """One sampled 2WikiMultiHopQA question with gold answer + supporting facts."""

    question_id: str
    question: str
    answer: str
    answer_aliases: list[str] = Field(default_factory=list)
    # gold supporting facts as (title, sentence_id) pairs, per the official format
    supporting_facts: list[tuple[str, int]] = Field(default_factory=list)
    # passage ids (post-pooling) that contain the gold supporting facts, by title match
    supporting_passage_ids: list[str] = Field(default_factory=list)
    type: str | None = None  # 2Wiki question type (comparison, bridge, etc.)


class Entity(BaseModel):
    name: str
    type: str

    @field_validator("name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("entity name must be non-empty")
        return v


class Triple(BaseModel):
    head: str
    relation: str
    tail: str

    @field_validator("head", "relation", "tail")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("triple fields must be non-empty")
        return v


class ExtractionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0


class Extraction(BaseModel):
    """One passage's validated extraction result, checkpointed as one JSON line."""

    passage_id: str
    entities: list[Entity] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)
    usage: ExtractionUsage = Field(default_factory=ExtractionUsage)
    error: str | None = None  # set if extraction failed after retries; entities/triples empty


class RetrievedPassage(BaseModel):
    passage_id: str
    title: str
    text: str
    vector_score: float | None = None
    graph_score: float | None = None
    fused_score: float = 0.0
    hop_path: list[str] = Field(default_factory=list)  # entity names traversed to reach it


class Prediction(BaseModel):
    """One eval record: what the pipeline produced for one question, checkpointed to disk."""

    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    supporting_passage_ids: list[str] = Field(default_factory=list)
    gold_supporting_passage_ids: list[str] = Field(default_factory=list)
    reasoning_path: list[str] = Field(default_factory=list)
    mode: Literal["graphrag", "baseline"] = "graphrag"
    retries: int = 0
    # Populated only when generation used DeepSeek (pay-per-use) — stays zeroed for the
    # free-tier Gemini/Gemma path. Lets a resumed run reseed its cumulative spend
    # tracker accurately across both prediction files (see eval/runner.py).
    usage: ExtractionUsage = Field(default_factory=ExtractionUsage)
