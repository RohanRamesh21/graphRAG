#!/usr/bin/env python
"""CLI: run Stage 1 (DeepSeek entity/relation extraction) over the pooled corpus.

Usage:
    python scripts/run_extraction.py [--limit 20]   # smoke test on first 20 passages
    python scripts/run_extraction.py                # full corpus, resumable
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.config import get_settings
from graphrag.extraction.run_extraction import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_extraction")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N passages (smoke test)")
    args = parser.parse_args()

    settings = get_settings()
    stats = asyncio.run(run(settings, limit=args.limit))
    logger.info("Extraction run complete: %s", stats)


if __name__ == "__main__":
    main()
