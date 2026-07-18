#!/usr/bin/env python
"""Delete Qdrant points whose passage_id no longer belongs to the current corpus.jsonl.

Same rationale as scripts/prune_orphaned_graph_passages.py: upsert is additive, so a
corpus resample (different sample composition) can leave orphaned points from a prior
corpus version. Harmless to eval correctness (retrieval already skips passages not in
the current corpus) but wastes index space and can occasionally displace a real result
out of top-K before being silently discarded.

Usage:
    python scripts/prune_orphaned_vector_points.py [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.config import get_settings
from graphrag.schemas import Passage
from graphrag.vector.qdrant_store import count, delete_passages, get_all_passage_ids, make_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prune_orphaned_vector_points")


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

    client = make_client(settings.qdrant_url, settings.qdrant_api_key)
    collection_ids = get_all_passage_ids(client, settings.qdrant_collection)
    orphans = collection_ids - corpus_ids
    logger.info(
        "Corpus has %d passages; Qdrant has %d; %d orphaned points found.",
        len(corpus_ids),
        len(collection_ids),
        len(orphans),
    )

    if not orphans:
        logger.info("Nothing to prune.")
        return

    if args.dry_run:
        logger.info("Dry run — not deleting. Re-run without --dry-run to prune.")
        return

    delete_passages(client, settings.qdrant_collection, list(orphans))
    logger.info("Deleted %d orphaned points.", len(orphans))
    logger.info(
        "Qdrant collection %r now has %d points.",
        settings.qdrant_collection,
        count(client, settings.qdrant_collection),
    )


if __name__ == "__main__":
    main()
