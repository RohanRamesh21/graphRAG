#!/usr/bin/env python
"""CLI: download 2WikiMultiHopQA dev split, sample 1k questions, pool+dedupe corpus.

Only ever prints small aggregates (counts, one sample record) — never the raw file.
Usage:
    python scripts/prepare_data.py [--n-questions 1000] [--seed 42] [--force] [--peek]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.config import get_settings
from graphrag.data.build_corpus import build
from graphrag.data.download import download_2wiki, peek

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-questions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Re-download/re-build even if outputs exist")
    parser.add_argument("--peek", action="store_true", help="Print one raw record's top-level keys only")
    args = parser.parse_args()

    settings = get_settings()
    raw_dir = settings.data_dir / "raw"

    logger.info("Downloading 2WikiMultiHopQA dev split (idempotent)...")
    raw_path = download_2wiki(raw_dir, force=args.force)

    if args.peek:
        sample = peek(raw_path, n=1)
        if sample:
            logger.info("Sample record top-level keys: %s", list(sample[0].keys()))
            logger.info("Sample question: %r", sample[0].get("question"))
            logger.info("Sample answer: %r", sample[0].get("answer"))
            logger.info("Sample #context passages: %d", len(sample[0]["context"]["title"]))

    logger.info("Sampling %d questions + pooling corpus (seed=%d)...", args.n_questions, args.seed)
    stats = build(
        raw_path=raw_path,
        corpus_path=settings.corpus_path,
        sample_questions_path=settings.sample_questions_path,
        n_questions=args.n_questions,
        seed=args.seed,
        force=args.force,
    )

    logger.info("Corpus build stats: %s", stats)
    if stats.get("skipped"):
        logger.info("Nothing to do — corpus.jsonl and sample_questions.jsonl already exist. Use --force to rebuild.")
    else:
        logger.info(
            "Wrote %d passages to %s, %d questions to %s",
            stats["n_unique_passages"],
            settings.corpus_path,
            stats["n_questions_sampled"],
            settings.sample_questions_path,
        )


if __name__ == "__main__":
    main()
