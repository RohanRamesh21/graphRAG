"""Thin Neo4j Aura driver wrapper: schema setup, batched UNWIND writes, stats."""
from __future__ import annotations

import logging
import re

from neo4j import GraphDatabase, Driver

logger = logging.getLogger(__name__)


def make_driver(uri: str, username: str, password: str) -> Driver:
    return GraphDatabase.driver(uri, auth=(username, password))


def ensure_constraints(driver: Driver, database: str) -> None:
    """Idempotent — CREATE CONSTRAINT IF NOT EXISTS is safe to run on every startup."""
    statements = [
        "CREATE CONSTRAINT entity_name_norm IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.name_norm IS UNIQUE",
        "CREATE CONSTRAINT passage_id IF NOT EXISTS "
        "FOR (p:Passage) REQUIRE p.id IS UNIQUE",
    ]
    with driver.session(database=database) as session:
        for stmt in statements:
            session.run(stmt)


_REL_TYPE_SAFE = re.compile(r"[^A-Z0-9_]")


def sanitize_relation_type(relation: str) -> str:
    """Cypher relationship types can't be parameterized, so extracted relation strings
    (arbitrary LLM output) must be turned into a safe uppercase identifier before being
    interpolated into a query. The original text is kept as a `.relation` property."""
    upper = relation.strip().upper().replace(" ", "_").replace("-", "_")
    safe = _REL_TYPE_SAFE.sub("", upper) or "RELATED_TO"
    # Cypher identifiers can't start with a digit.
    if safe[0].isdigit():
        safe = f"REL_{safe}"
    return safe[:100]  # guard against pathological lengths


def write_passages_batch(driver: Driver, database: str, passages: list[dict]) -> None:
    """passages: [{"id", "title", "text"}, ...]"""
    query = """
    UNWIND $rows AS row
    MERGE (p:Passage {id: row.id})
    SET p.title = row.title, p.text = row.text
    """
    with driver.session(database=database) as session:
        session.run(query, rows=passages)


def write_entities_batch(driver: Driver, database: str, entities: list[dict]) -> None:
    """entities: [{"name_norm", "name", "type"}, ...]. MERGE on name_norm — the only
    entity resolution in v1 (see README limitations); `name` keeps the first-seen
    surface form for display."""
    query = """
    UNWIND $rows AS row
    MERGE (e:Entity {name_norm: row.name_norm})
    ON CREATE SET e.name = row.name, e.type = row.type
    """
    with driver.session(database=database) as session:
        session.run(query, rows=entities)


def write_mentions_batch(driver: Driver, database: str, mentions: list[dict]) -> None:
    """mentions: [{"passage_id", "name_norm"}, ...]"""
    query = """
    UNWIND $rows AS row
    MATCH (p:Passage {id: row.passage_id})
    MATCH (e:Entity {name_norm: row.name_norm})
    MERGE (p)-[:MENTIONS]->(e)
    """
    with driver.session(database=database) as session:
        session.run(query, rows=mentions)


def write_triples_batch(
    driver: Driver, database: str, rel_type: str, triples: list[dict]
) -> None:
    """triples: [{"head_norm", "tail_norm", "relation", "source_passage_id"}, ...], all
    sharing one sanitized `rel_type` (Cypher relationship types can't be parameterized,
    so callers must group triples by rel_type before calling this)."""
    query = f"""
    UNWIND $rows AS row
    MATCH (h:Entity {{name_norm: row.head_norm}})
    MATCH (t:Entity {{name_norm: row.tail_norm}})
    MERGE (h)-[r:{rel_type}]->(t)
    ON CREATE SET r.relation = row.relation, r.source_passage_ids = [row.source_passage_id]
    ON MATCH SET r.source_passage_ids =
        CASE WHEN NOT row.source_passage_id IN r.source_passage_ids
             THEN r.source_passage_ids + row.source_passage_id
             ELSE r.source_passage_ids END
    """
    with driver.session(database=database) as session:
        session.run(query, rows=triples)


def get_stats(driver: Driver, database: str) -> dict:
    """Four independent counts run as separate statements (rather than one query with
    subquery-scoping syntax) so this works across older Aura/Neo4j versions too."""
    queries = {
        "entity_count": "MATCH (e:Entity) RETURN count(e) AS n",
        "passage_count": "MATCH (p:Passage) RETURN count(p) AS n",
        "mentions_count": "MATCH ()-[m:MENTIONS]->() RETURN count(m) AS n",
        "entity_relation_count": "MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS n",
    }
    stats = {}
    with driver.session(database=database) as session:
        for key, query in queries.items():
            record = session.run(query).single()
            stats[key] = record["n"] if record else 0
    return stats


def get_all_passage_ids(driver: Driver, database: str) -> set[str]:
    with driver.session(database=database) as session:
        return {r["id"] for r in session.run("MATCH (p:Passage) RETURN p.id AS id")}


def delete_passages(driver: Driver, database: str, passage_ids: list[str]) -> None:
    """DETACH DELETE removes the Passage node and its MENTIONS edges. Entity nodes and
    Entity-Entity relationship edges are left untouched — they may still be shared with
    passages that remain in the corpus (only their `source_passage_ids` provenance list
    may retain a now-stale id, which doesn't affect traversal correctness)."""
    query = "MATCH (p:Passage) WHERE p.id IN $ids DETACH DELETE p"
    with driver.session(database=database) as session:
        session.run(query, ids=passage_ids)
