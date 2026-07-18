#!/usr/bin/env python
"""Liveness ping for Neo4j Aura Free + Qdrant Cloud Free, run on a schedule (see
.github/workflows/keep-alive.yml) so neither free-tier instance gets auto-paused/deleted
from inactivity:
  - Neo4j Aura Free: paused after 72h idle, deleted after 90 days paused.
  - Qdrant Cloud Free: suspended after 1 week idle, deleted after 4 weeks.

Deliberately standalone (no dependency on the graphrag package or its .env loading) so
CI only needs to `pip install neo4j qdrant-client` — not the full project's ML stack —
and so this script never touches unrelated secrets (DeepSeek/Gemini keys) that have
nothing to do with keeping these two instances alive.

Never prints credential values, only OK/FAIL per service — safe to view in CI logs.
Exits non-zero on any failure, which is what makes GitHub email the repo owner on a
failed scheduled run (the built-in notification, not something this script sends itself).
"""
from __future__ import annotations

import os
import sys


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def ping_neo4j() -> tuple[bool, str]:
    uri = _env("NEO4J_URI", "NEO4J_AURA_URI")
    username = _env("NEO4J_USERNAME", "NEO4J_AURA_USER")
    password = _env("NEO4J_PASSWORD", "NEO4J_AURA_PASSWORD")
    if not (uri and username and password):
        return False, "missing NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD (or NEO4J_AURA_* equivalents)"

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session() as session:
            session.run("RETURN 1").consume()
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        driver.close()


def ping_qdrant() -> tuple[bool, str]:
    url = _env("QDRANT_URL", "QDRANT_CLUSTER_ENDPOINT")
    api_key = _env("QDRANT_API_KEY")
    if not (url and api_key):
        return False, "missing QDRANT_URL (or QDRANT_CLUSTER_ENDPOINT) / QDRANT_API_KEY"

    from qdrant_client import QdrantClient

    client = QdrantClient(url=url, api_key=api_key)
    try:
        client.get_collections()
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    checks = {"neo4j": ping_neo4j, "qdrant": ping_qdrant}
    all_ok = True
    for name, fn in checks.items():
        ok, detail = fn()
        all_ok &= ok
        print(f"{'OK  ' if ok else 'FAIL'}  {name:<8} {detail}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
