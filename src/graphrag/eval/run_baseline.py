"""Thin entrypoint: run the vector-only baseline (no graph expansion) over the eval set,
so the graph's actual contribution is measurable against this, rather than asserted."""
from __future__ import annotations

from graphrag.config import Settings
from graphrag.eval.runner import run as _run


async def run(settings: Settings, limit: int | None = None) -> dict:
    return await _run(settings, mode="baseline", limit=limit)
