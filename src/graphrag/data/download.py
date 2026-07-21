"""Stage: download the 2WikiMultiHopQA dev (validation) split.

Uses `framolfese/2WikiMultihopQA` on the Hugging Face Hub — a parquet-native repackaging
of the original 2WikiMultihopQA data (HotpotQA-style field layout, content unaltered).
Chosen over the older `xanhho/2WikiMultihopQA` mirror because that one relies on a legacy
loading script, which `datasets>=3.x` no longer executes without extra flags.

IMPORTANT: this module and its CLI never print full records or the raw file to the
console — only counts and small aggregates. See scripts/prepare_data.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HF_DATASET_ID = "framolfese/2WikiMultihopQA"
HF_SPLIT = "validation"  # 2Wiki's "dev" split (test is blind)


def download_2wiki(raw_dir: Path, force: bool = False) -> Path:
    """Download the dev split and persist it as local JSONL (our copy of record).

    Idempotent: if the output file already exists and `force` is False, skips re-download
    and returns the existing path immediately.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "2wikimultihopqa_dev.jsonl"

    if out_path.exists() and not force:
        logger.info("Raw dataset already present at %s, skipping download.", out_path)
        return out_path

    # Imported lazily so the rest of the package doesn't require `datasets` at import time.
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)

    tmp_path = out_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)

    logger.info("Downloaded %d rows to %s", len(ds), out_path)
    return out_path


def peek(raw_path: Path, n: int = 1) -> list[dict]:
    """Return the first `n` records for structural inspection. Caller must only print
    small aggregates/keys — never dump these records wholesale to a viewer/log."""
    records = []
    with raw_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            records.append(json.loads(line))
    return records
