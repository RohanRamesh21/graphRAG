#!/usr/bin/env python
"""Delete Passage nodes in Neo4j that no longer belong to the current corpus.jsonl.

Needed when the corpus is rebuilt with a different sample composition (e.g. scaling
from a smaller smoke-test sample to the full 1,000-question sample) — MERGE-based graph
writes are idempotent/additive, so they never remove nodes on their own, and a resample
can leave orphaned Passage nodes (and their MENTIONS edges) from a prior corpus version.
Harmless to eval correctness (retrieval already skips passages not in the current
corpus) but inflates /graph/stats and adds unnecessary traversal surface — prune it.

Entity nodes and Entity-Entity edges are never touched, since they may still be shared
with passages that remain in the corpus.

Usage:
    python scripts/prune_orphaned_graph_passages.py [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.config import get_settings
from graphrag.graph.neo4j_client import delete_passages, get_all_passage_ids, get_stats, make_driver
from graphrag.schemas import Passage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prune_orphaned_graph_passages")


def _load_corpus_ids(corpus_path: Path) -> set[str]:
    ids = set()
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            ids.add(Passage.model_validate_json(line).id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Report orphans without deleting")
    args = parser.parse_args()

    settings = get_settings()
    corpus_ids = _load_corpus_ids(settings.corpus_path)

    driver = make_driver(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
    try:
        graph_ids = get_all_passage_ids(driver, settings.neo4j_database)
        orphans = graph_ids - corpus_ids
        logger.info(
            "Corpus has %d passages; graph has %d; %d orphaned Passage nodes found.",
            len(corpus_ids),
            len(graph_ids),
            len(orphans),
        )

        if not orphans:
            logger.info("Nothing to prune.")
            return

        if args.dry_run:
            logger.info("Dry run — not deleting. Re-run without --dry-run to prune.")
            return

        delete_passages(driver, settings.neo4j_database, list(orphans))
        logger.info("Deleted %d orphaned Passage nodes.", len(orphans))
        logger.info("Graph stats after prune: %s", get_stats(driver, settings.neo4j_database))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
