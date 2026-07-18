#!/usr/bin/env python
"""CLI: run the GraphRAG (hybrid) eval over the sampled questions. Resumable — safe to
re-run after hitting the Gemini daily quota; picks up where it left off.

Usage:
    python scripts/run_eval.py --limit 50   # subset smoke test
    python scripts/run_eval.py              # full 1,000-question set
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.config import get_settings
from graphrag.eval.run_eval import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_eval")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    stats = asyncio.run(run(settings, limit=args.limit))
    logger.info("GraphRAG eval run: %s", stats)


if __name__ == "__main__":
    main()
