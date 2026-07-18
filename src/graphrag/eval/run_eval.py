"""Thin entrypoint: run the full GraphRAG pipeline (hybrid retrieval) over the eval set."""
from __future__ import annotations

from graphrag.config import Settings
from graphrag.eval.runner import run as _run


async def run(settings: Settings, limit: int | None = None) -> dict:
    return await _run(settings, mode="graphrag", limit=limit)
