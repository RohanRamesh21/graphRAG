"""DeepSeek-backed answer generator for Stages 5 & 7 — used in place of
generation/gemini_client.py per an explicit decision to route generation through
DeepSeek instead of Gemini's free tier: no daily request cap (pure pay-per-use), and
empirically ~$0.45 total for a full 1,000-question x 2-mode eval run (vs. Gemini/Gemma's
multi-day rate-limit wall). Same `generate(question, passages) -> (answer, ids)`
interface as GeminiGenerator, so it's a drop-in replacement in orchestration/graph.py,
eval/runner.py, and api/main.py.

Draws against the SAME DeepSeek account balance as Stage 1 extraction — see
deepseek_common.SpendTracker and config.deepseek_generation_cost_ceiling_usd.
"""
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

from graphrag.deepseek_common import DEEPSEEK_BASE_URL, DISABLE_THINKING, SpendTracker, compute_cost
from graphrag.generation.deepseek_prompt import build_messages

logger = logging.getLogger(__name__)


class DeepSeekGenerator:
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
            max_tokens=512,  # answer + a handful of passage ids — plenty of headroom
            extra_body=DISABLE_THINKING,
        )

    async def generate(self, question: str, passages: list[dict]) -> tuple[str, list[str]]:
        """passages: [{"passage_id", "title", "text"}, ...]. Raises CostCeilingExceeded
        before making a call once the configured ceiling is reached."""
        await self.spend_tracker.check_before_call()

        messages = build_messages(question, passages)

        last_err: Exception | None = None
        for attempt in range(3):
            response = await self._call(messages)
            usage = response.usage
            cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
            cache_miss = getattr(usage, "prompt_cache_miss_tokens", 0) or (
                usage.prompt_tokens - cache_hit
            )
            cost = compute_cost(cache_hit, cache_miss, usage.completion_tokens)
            await self.spend_tracker.add(cost)

            raw = response.choices[0].message.content
            try:
                data = json.loads(raw)
                return data.get("answer", ""), data.get("supporting_passage_ids", [])
            except json.JSONDecodeError as e:
                last_err = e
                logger.warning(
                    "Malformed JSON from DeepSeek for question %r (attempt %d/3): %s",
                    question,
                    attempt + 1,
                    e,
                )
                continue

        logger.warning(
            "DeepSeek returned malformed JSON after 3 attempts for question %r: %s",
            question,
            last_err,
        )
        return "", []
