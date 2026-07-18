#!/usr/bin/env python
"""CLI: build the Neo4j graph from corpus.jsonl + extractions.jsonl. Idempotent (MERGE-based).

Usage:
    python scripts/build_graph.py
    python scripts/build_graph.py --stats-only   # just print current graph stats
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.config import get_settings
from graphrag.graph.build import build_graph
from graphrag.graph.neo4j_client import get_stats, make_driver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_graph")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    driver = make_driver(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
    try:
        if not args.stats_only:
            extractions_path = settings.extraction_dir / "extractions.jsonl"
            stats = build_graph(
                driver=driver,
                database=settings.neo4j_database,
                corpus_path=settings.corpus_path,
                extractions_path=extractions_path,
            )
            logger.info("Graph build complete: %s", stats)

        graph_stats = get_stats(driver, settings.neo4j_database)
        logger.info("Current graph stats: %s", graph_stats)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
