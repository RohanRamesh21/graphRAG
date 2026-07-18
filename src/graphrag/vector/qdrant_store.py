"""Qdrant Cloud vector store: idempotent collection setup + upsert + search.

Point IDs are deterministic (uuid5 of the passage id) so re-running ingestion re-upserts
the same points instead of creating duplicates — this is what makes vector indexing safe
to call repeatedly from `POST /ingest`.
"""
from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from graphrag.vector.embedder import EMBED_DIM

logger = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("f4a6e2f0-2b8a-4b8b-9b8e-2f8a4b8b9b8e")  # fixed, arbitrary


def passage_point_id(passage_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, passage_id))


def make_client(url: str, api_key: str) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
    )
    logger.info("Created Qdrant collection %r (dim=%d, cosine)", collection_name, EMBED_DIM)


def upsert_passages(
    client: QdrantClient,
    collection_name: str,
    passage_ids: list[str],
    titles: list[str],
    vectors: list[list[float]],
) -> None:
    points = [
        qmodels.PointStruct(
            id=passage_point_id(pid),
            vector=vec,
            payload={"passage_id": pid, "title": title},
        )
        for pid, title, vec in zip(passage_ids, titles, vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)


def search(
    client: QdrantClient, collection_name: str, query_vector: list[float], top_k: int
) -> list[dict]:
    """Returns [{"passage_id", "title", "score"}, ...], best first."""
    results = client.query_points(
        collection_name=collection_name, query=query_vector, limit=top_k, with_payload=True
    ).points
    return [
        {
            "passage_id": r.payload["passage_id"],
            "title": r.payload["title"],
            "score": r.score,
        }
        for r in results
    ]


def count(client: QdrantClient, collection_name: str) -> int:
    try:
        return client.count(collection_name=collection_name, exact=True).count
    except Exception:
        return 0


def get_all_passage_ids(client: QdrantClient, collection_name: str) -> set[str]:
    """Scrolls the full collection to collect every point's `passage_id` payload
    field — used to find points orphaned by a corpus resample (see
    scripts/prune_orphaned_graph_passages.py for the analogous Neo4j case)."""
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        ids.update(p.payload["passage_id"] for p in points)
        if offset is None:
            break
    return ids


def delete_passages(client: QdrantClient, collection_name: str, passage_ids: list[str]) -> None:
    point_ids = [passage_point_id(pid) for pid in passage_ids]
    client.delete(collection_name=collection_name, points_selector=point_ids)
