"""Shared DeepSeek infrastructure used by both Stage 1 (extraction) and, per an
explicit user decision to route generation through DeepSeek too instead of Gemini's
free tier, Stages 5/7 (answer generation): base URL, pricing, cost computation, and the
account-wide spend guard.

Both extraction and generation draw against the SAME DeepSeek account balance (real
credit, under $3 total per instructions.md), so callers should size their own ceiling
with that shared balance in mind — see config.py's deepseek_cost_ceiling_usd
(extraction) and deepseek_generation_cost_ceiling_usd (generation).
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Pricing confirmed against https://api-docs.deepseek.com/quick_start/pricing/ (2026-07-18)
# for deepseek-v4-flash, USD per 1M tokens. Re-check before relying on this for budgeting
# again if it's been a while — DeepSeek has changed pricing before.
PRICE_PER_MTOK = {
    "cache_hit_input": 0.0028,
    "cache_miss_input": 0.14,
    "output": 0.28,
}

# deepseek-v4-flash defaults to THINKING mode (hidden chain-of-thought before the
# visible answer) — confirmed against api-docs.deepseek.com/guides/thinking_mode. Left
# on, reasoning tokens can consume the entire max_tokens budget before any real content
# is emitted, producing an empty completion with finish_reason="length" (observed on
# ~27% of extraction calls before this was found and disabled). Every DeepSeek call in
# this project — extraction or generation — needs a single structured JSON object, not
# reasoning, so this is applied everywhere calls are made.
DISABLE_THINKING = {"thinking": {"type": "disabled"}}


class CostCeilingExceeded(RuntimeError):
    """Raised when cumulative DeepSeek spend would exceed the configured ceiling."""


def compute_cost(cache_hit_tokens: int, cache_miss_tokens: int, output_tokens: int) -> float:
    return (
        cache_hit_tokens * PRICE_PER_MTOK["cache_hit_input"]
        + cache_miss_tokens * PRICE_PER_MTOK["cache_miss_input"]
        + output_tokens * PRICE_PER_MTOK["output"]
    ) / 1_000_000


class SpendTracker:
    """Thread/async-safe running total, seeded from any usage already logged on disk so
    a resumed run doesn't reset the ceiling check to zero."""

    def __init__(self, ceiling_usd: float, starting_cost_usd: float = 0.0):
        self.ceiling_usd = ceiling_usd
        self._cost = starting_cost_usd
        self._lock = asyncio.Lock()

    @property
    def cost(self) -> float:
        return self._cost

    async def check_before_call(self) -> None:
        if self._cost >= self.ceiling_usd:
            raise CostCeilingExceeded(
                f"Cumulative DeepSeek cost ${self._cost:.4f} has reached the "
                f"${self.ceiling_usd:.2f} ceiling — halting before making another call."
            )

    async def add(self, cost_usd: float) -> None:
        async with self._lock:
            self._cost += cost_usd
            if self._cost >= self.ceiling_usd:
                logger.error(
                    "DeepSeek cumulative cost $%.4f has reached the $%.2f ceiling.",
                    self._cost,
                    self.ceiling_usd,
                )
