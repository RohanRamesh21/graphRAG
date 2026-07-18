#!/usr/bin/env python
"""CLI: run the vector-only baseline eval (no graph expansion) over the same sampled
questions and corpus. Resumable, and shares the same Gemini daily-quota counter as
run_eval.py since both draw against the same account's daily cap.

Usage:
    python scripts/run_baseline.py --limit 50
    python scripts/run_baseline.py
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.config import get_settings
from graphrag.eval.run_baseline import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_baseline")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    stats = asyncio.run(run(settings, limit=args.limit))
    logger.info("Baseline eval run: %s", stats)


if __name__ == "__main__":
    main()
