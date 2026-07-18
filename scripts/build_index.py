#!/usr/bin/env python
"""CLI: embed every passage in corpus.jsonl and upsert into Qdrant. Idempotent
(deterministic point ids) — safe to re-run over the same or a superset corpus.

Usage:
    python scripts/build_index.py [--batch-size 64]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from graphrag.config import get_settings
from graphrag.schemas import Passage
from graphrag.vector.embedder import Embedder
from graphrag.vector.qdrant_store import count, ensure_collection, make_client, upsert_passages

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_index")


def _load_corpus(corpus_path: Path) -> list[Passage]:
    passages = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            passages.append(Passage.model_validate_json(line))
    return passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    settings = get_settings()
    passages = _load_corpus(settings.corpus_path)
    logger.info("Loaded %d passages to embed.", len(passages))

    embedder = Embedder(settings.embed_model)
    client = make_client(settings.qdrant_url, settings.qdrant_api_key)
    ensure_collection(client, settings.qdrant_collection)

    for i in tqdm(range(0, len(passages), args.batch_size), desc="Embedding+upserting"):
        batch = passages[i : i + args.batch_size]
        vectors = embedder.embed_passages([p.text for p in batch])
        upsert_passages(
            client,
            settings.qdrant_collection,
            passage_ids=[p.id for p in batch],
            titles=[p.title for p in batch],
            vectors=vectors,
        )

    total = count(client, settings.qdrant_collection)
    logger.info("Qdrant collection %r now has %d points.", settings.qdrant_collection, total)


if __name__ == "__main__":
    main()
