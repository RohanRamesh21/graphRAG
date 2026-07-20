"""Async DeepSeek client wrapper for Stage 1 extraction: JSON-mode calls, usage/cost
accounting, and the hard spend guard described in instructions.md (halt before
exceeding DEEPSEEK_COST_CEILING_USD, default $2, rather than trusting the ~$1 estimate
alone). Shared cost/spend-guard infrastructure lives in graphrag.deepseek_common."""
from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI, APIError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from graphrag.deepseek_common import (
    DEEPSEEK_BASE_URL,
    DISABLE_THINKING,
    CostCeilingExceeded,  # noqa: F401  (re-exported: existing callers import it from here)
    SpendTracker,
    compute_cost,
)
from graphrag.extraction.prompt import build_messages

logger = logging.getLogger(__name__)


class ExtractionFailed(RuntimeError):
    """Raised when DeepSeek returns malformed JSON after all retries. Carries the
    summed usage/cost across every attempt for this passage (all of them cost real
    money even though none produced valid JSON) — the caller should persist this on
    the checkpointed error record. Without it, a process restart would reseed its
    SpendTracker from disk at an undercounted total, since the failed attempts'
    spend would otherwise never be written anywhere."""

    def __init__(self, message: str, usage: dict):
        super().__init__(message)
        self.usage = usage


class DeepSeekExtractor:
    def __init__(self, api_key: str, model: str, spend_tracker: SpendTracker):
        self.client = AsyncOpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = model
        self.spend_tracker = spend_tracker

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def _call(self, messages: list[dict[str, str]]):
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            # Headroom for entity/relation-dense passages (e.g. long biographical
            # articles) now that thinking is disabled and every output token is real
            # JSON content, not hidden reasoning — this cap costs nothing unless a
            # passage actually needs it.
            max_tokens=4096,
            extra_body=DISABLE_THINKING,
        )

    async def extract(self, title: str, text: str) -> tuple[dict, dict]:
        """Returns (parsed_json_dict, usage_dict). Raises on cost ceiling breach or if
        the model still returns malformed JSON after retries — caller (run_extraction)
        is responsible for turning that into a checkpointed error record, not crashing
        the whole batch."""
        await self.spend_tracker.check_before_call()

        messages = build_messages(title, text)

        last_err: Exception | None = None
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0}
        for attempt in range(3):
            response = await self._call(messages)
            usage = response.usage
            cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
            cache_miss = getattr(usage, "prompt_cache_miss_tokens", 0) or (
                usage.prompt_tokens - cache_hit
            )
            completion_tokens = usage.completion_tokens
            cost = compute_cost(cache_hit, cache_miss, completion_tokens)
            await self.spend_tracker.add(cost)

            usage_dict = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cache_hit,
                "cost_usd": cost,
            }
            # Every attempt costs real money regardless of whether it parses — keep a
            # running total across the whole call so a final failure can still report
            # accurate spend (see ExtractionFailed's docstring for why this matters).
            total_usage["prompt_tokens"] += usage_dict["prompt_tokens"]
            total_usage["completion_tokens"] += usage_dict["completion_tokens"]
            total_usage["cached_tokens"] += usage_dict["cached_tokens"]
            total_usage["cost_usd"] += usage_dict["cost_usd"]

            raw = response.choices[0].message.content
            try:
                parsed = json.loads(raw)
                return parsed, total_usage
            except json.JSONDecodeError as e:
                last_err = e
                logger.warning(
                    "Malformed JSON from DeepSeek for %r (attempt %d/3): %s",
                    title,
                    attempt + 1,
                    e,
                )
                # Re-prompt with the same fixed prefix; the malformed output itself isn't
                # fed back in, keeping the cacheable prefix untouched on retry.
                continue

        raise ExtractionFailed(
            f"DeepSeek returned malformed JSON after 3 attempts: {last_err}", total_usage
        )
