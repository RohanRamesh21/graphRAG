"""Graph half of retrieval: from seed entities (mentioned in the vector-retrieved seed
passages), traverse the Neo4j graph outward up to `hop_depth` hops and collect passages
that mention entities found along the way.

Traversal patterns are built with every intermediate node explicitly labeled `:Entity`
(rather than Cypher's `*1..N` variable-length syntax). Since MENTIONS edges always touch
a `:Passage` node on one side, requiring every node in the pattern to be `:Entity` is
what keeps a "hop" strictly meaning "an extracted relation edge between two entities" —
never an incidental co-mention hop through a passage.

Every node in the chain (seed *and* every node reached after it) is degree-filtered —
see `max_seed_entity_degree` on `GraphRetriever`. This was added after a live 429/None:
at full corpus scale (41k+ entities), a couple of generic "hub" entities (a bare year,
a nationality adjective — both correctly typed DATE/OTHER by extraction, just not useful
traversal points) had degree 30-40 against an otherwise single-digit norm, and a 2-hop
match through even one such hub made a single question's retrieval take 10s of seconds
to minutes and return thousands of barely-relevant passages. Filtering only the seed
cut that from 5+ minutes to 10-48s — filtering *every* hop (this file's current design)
is what actually fixes it, since the blowup was in the intermediate node's fan-out, not
just the seed's.
"""
from __future__ import annotations

from neo4j import Driver


def _build_traversal_query(depth: int) -> str:
    """Builds a chain of MATCH/WHERE stages (n0 -> n1 -> ... -> n_depth), applying the
    degree filter to each node as soon as it's introduced so Neo4j's planner prunes
    high-degree hubs before expanding the next hop, rather than after the fact."""
    lines = ["MATCH (n0:Entity)", "WHERE n0.name_norm IN $seed_norms"]
    for i in range(1, depth + 1):
        lines.append(f"MATCH (n{i - 1})-[r{i}]-(n{i}:Entity)")
        condition = f"COUNT {{ (n{i})--() }} <= $max_degree"
        if i == depth:
            condition += f" AND n{i}.name_norm <> n0.name_norm"
        lines.append(f"WHERE {condition}")
    path_expr = "[" + ", ".join(f"n{i}.name" for i in range(depth + 1)) + "]"
    lines.append(f"WITH DISTINCT n{depth} AS reached, {path_expr} AS hop_path")
    lines.append("MATCH (reached)<-[:MENTIONS]-(p:Passage)")
    lines.append("RETURN DISTINCT p.id AS passage_id, p.title AS title, p.text AS text, hop_path")
    return "\n".join(lines)


class GraphRetriever:
    def __init__(self, driver: Driver, database: str, hop_depth: int = 2, max_seed_entity_degree: int = 30):
        self.driver = driver
        self.database = database
        self.hop_depth = hop_depth
        # Applied to every node encountered during traversal (seed and beyond) — see
        # this module's docstring for why filtering only the seed wasn't enough.
        self.max_seed_entity_degree = max_seed_entity_degree

    def seed_entity_norms(self, seed_passage_ids: list[str]) -> list[str]:
        if not seed_passage_ids:
            return []
        query = """
        MATCH (p:Passage)-[:MENTIONS]->(e:Entity)
        WHERE p.id IN $passage_ids
        WITH DISTINCT e
        WHERE COUNT { (e)--() } <= $max_degree
        RETURN e.name_norm AS name_norm
        """

        def _tx(tx):
            return [
                r["name_norm"]
                for r in tx.run(query, passage_ids=seed_passage_ids, max_degree=self.max_seed_entity_degree)
            ]

        # execute_read (a managed transaction function) automatically retries on
        # transient driver errors (ServiceUnavailable, SessionExpired, etc.) with
        # backoff — unlike a bare session.run(), which doesn't retry at all. This
        # matters a lot here: a 1000-question eval run can span hours, and a
        # long-lived driver hitting one transient Aura routing hiccup shouldn't crash
        # the entire batch (this happened live — see git history/PR notes).
        with self.driver.session(database=self.database) as session:
            return session.execute_read(_tx)

    def traverse(self, seed_norms: list[str]) -> list[dict]:
        """Returns [{"passage_id", "title", "text", "hops", "hop_path"}, ...], deduped
        by passage_id keeping the smallest hop count (and its path) seen across depths,
        sorted by hops ascending (closer = more relevant)."""
        if not seed_norms:
            return []

        def _tx(tx) -> dict[str, dict]:
            best: dict[str, dict] = {}
            for depth in range(1, self.hop_depth + 1):
                query = _build_traversal_query(depth)
                for record in tx.run(
                    query, seed_norms=seed_norms, max_degree=self.max_seed_entity_degree
                ):
                    pid = record["passage_id"]
                    if pid not in best or depth < best[pid]["hops"]:
                        best[pid] = {
                            "passage_id": pid,
                            "title": record["title"],
                            "text": record["text"],
                            "hops": depth,
                            "hop_path": record["hop_path"],
                        }
            return best

        with self.driver.session(database=self.database) as session:
            best = session.execute_read(_tx)

        return sorted(best.values(), key=lambda r: r["hops"])


class NullGraphRetriever:
    """Stub matching GraphRetriever's interface that always returns no expansion. Used
    by the eval harness's baseline mode: plugging this into HybridRetriever makes it
    degenerate to pure vector retrieval (RRF over a single ranked list preserves that
    list's order) — so baseline and GraphRAG share the exact same retrieve/generate/
    validate code path, differing only in this one component. That's what makes the
    baseline-vs-GraphRAG delta a measurement of the graph's contribution specifically,
    rather than of two differently-built pipelines.
    """

    def seed_entity_norms(self, seed_passage_ids: list[str]) -> list[str]:
        return []

    def traverse(self, seed_norms: list[str]) -> list[dict]:
        return []
