"""Stage 2: build the Neo4j graph from corpus.jsonl + extractions.jsonl.

Idempotent by construction — every write is a Cypher MERGE keyed on a stable id
(Passage.id, Entity.name_norm), so re-running this over the same (or a superset of the
same) input is always safe and never duplicates nodes/edges. This is what makes
`POST /ingest` safe to call repeatedly.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from neo4j import Driver

from graphrag.graph.neo4j_client import (
    ensure_constraints,
    sanitize_relation_type,
    write_entities_batch,
    write_mentions_batch,
    write_passages_batch,
    write_triples_batch,
)
from graphrag.schemas import Extraction, Passage, normalize_name

logger = logging.getLogger(__name__)

BATCH_SIZE = 1000


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _load_corpus(corpus_path: Path) -> dict[str, Passage]:
    passages = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            p = Passage.model_validate_json(line)
            passages[p.id] = p
    return passages


def _load_successful_extractions(extractions_path: Path) -> dict[str, Extraction]:
    """Last successful (error=None) record wins per passage_id."""
    out: dict[str, Extraction] = {}
    with extractions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = Extraction.model_validate_json(line)
            if rec.error is None:
                out[rec.passage_id] = rec
    return out


def build_graph(
    driver: Driver,
    database: str,
    corpus_path: Path,
    extractions_path: Path,
) -> dict:
    ensure_constraints(driver, database)

    passages = _load_corpus(corpus_path)
    extractions = _load_successful_extractions(extractions_path)

    logger.info(
        "Loaded %d passages, %d successful extractions (%d passages have no usable extraction yet).",
        len(passages),
        len(extractions),
        len(passages) - len(extractions),
    )

    # --- Passages ---
    passage_rows = [{"id": p.id, "title": p.title, "text": p.text} for p in passages.values()]
    for chunk in _chunks(passage_rows, BATCH_SIZE):
        write_passages_batch(driver, database, chunk)

    # --- Entities (deduped by normalized name across the whole corpus) ---
    entity_rows_by_norm: dict[str, dict] = {}
    mention_rows: list[dict] = []
    # relation rows grouped by sanitized Cypher relationship type (can't be parameterized)
    triple_rows_by_type: dict[str, list[dict]] = defaultdict(list)

    for passage_id, extraction in extractions.items():
        for entity in extraction.entities:
            norm = normalize_name(entity.name)
            if not norm:
                continue
            entity_rows_by_norm.setdefault(
                norm, {"name_norm": norm, "name": entity.name, "type": entity.type}
            )
            mention_rows.append({"passage_id": passage_id, "name_norm": norm})

        for triple in extraction.triples:
            head_norm = normalize_name(triple.head)
            tail_norm = normalize_name(triple.tail)
            if not head_norm or not tail_norm:
                continue
            rel_type = sanitize_relation_type(triple.relation)
            triple_rows_by_type[rel_type].append(
                {
                    "head_norm": head_norm,
                    "tail_norm": tail_norm,
                    "relation": triple.relation,
                    "source_passage_id": passage_id,
                }
            )

    entity_rows = list(entity_rows_by_norm.values())
    for chunk in _chunks(entity_rows, BATCH_SIZE):
        write_entities_batch(driver, database, chunk)

    # Entities must exist before MENTIONS/relation edges reference them.
    for chunk in _chunks(mention_rows, BATCH_SIZE):
        write_mentions_batch(driver, database, chunk)

    n_triples = 0
    for rel_type, rows in triple_rows_by_type.items():
        for chunk in _chunks(rows, BATCH_SIZE):
            write_triples_batch(driver, database, rel_type, chunk)
        n_triples += len(rows)

    return {
        "n_passages_written": len(passage_rows),
        "n_entities_written": len(entity_rows),
        "n_mentions_written": len(mention_rows),
        "n_relation_types": len(triple_rows_by_type),
        "n_triples_written": n_triples,
        "n_passages_missing_extraction": len(passages) - len(extractions),
    }
