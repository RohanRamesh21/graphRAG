#!/usr/bin/env python
"""CLI: aggregate GraphRAG + baseline predictions into data/results/report.md."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.config import get_settings
from graphrag.eval.report import write_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_report")


def main() -> None:
    settings = get_settings()
    out_path = write_report(settings.results_dir)
    logger.info("Report written to %s", out_path)


if __name__ == "__main__":
    main()
