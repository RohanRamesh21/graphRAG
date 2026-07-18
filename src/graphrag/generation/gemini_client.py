"""Free-tier client for answer generation (Stages 5 & 7), via the google-genai SDK.

Despite the module name, GEMINI_MODEL currently points at a Gemma model
(gemma-4-31b-it) — served through the same API/SDK/key as Gemini models, but on a
separate quota bucket that empirically has far more free-tier headroom than
gemini-2.5-flash did for this project (20 req/day measured live). Nothing else about
this client is Gemini-specific; it'll work unchanged if GEMINI_MODEL is pointed back at
a Gemini model.

Rate limiting has two layers, per instructions.md's expectation that RPD — not RPM or
TPM — is the binding constraint on a multi-day eval run:
  1. `RpmLimiter`: a simple async pacer that spaces calls at least `60/RPM` seconds
     apart, in-memory only (doesn't need to survive a restart — worst case is one
     slightly-too-fast call right after a resume).
  2. `DailyQuotaTracker`: a persistent JSON counter keyed by (model, UTC date), surviving
     process restarts. Keying by model matters: different models are independent quota
     buckets, so without this a model switch would inherit a stale, irrelevant count.
     Once the configured RPD is hit, raises `DailyQuotaExceeded` so the caller (run_eval
     / run_baseline) can checkpoint and stop cleanly instead of hammering a 429 wall —
     the run resumes the next session once the date rolls over. If the configured RPD
     is set too high, the real API's 429 is also caught and handled gracefully by the
     eval runner, just with one wasted retry cycle first.

Free-tier RPD varies by project/region/date and can be far below what docs advertise —
re-confirm the live number in AI Studio and set GEMINI_RPD accordingly before a full run.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import time
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)


class DailyQuotaExceeded(RuntimeError):
    """Raised when the configured Gemini RPD cap has been reached for today (UTC)."""


class GeminiAnswer(BaseModel):
    answer: str
    supporting_passage_ids: list[str] = []


GENERATION_SYSTEM_PROMPT = """You are a careful multi-hop question-answering assistant. \
You are given a question and a numbered list of retrieved passages (which may include \
irrelevant distractors). Using ONLY the passages provided:
1. Give the shortest correct answer to the question (a name, date, yes/no, or short \
phrase — not a full sentence).
2. List the passage_id values of the passages you actually relied on to answer.
If the passages don't contain enough information, give your best-guess answer anyway \
and return an empty supporting_passage_ids list."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ClientError):
        return exc.code == 429
    return isinstance(exc, ServerError)


class DailyQuotaTracker:
    """Persistent per-UTC-day call counter, stored as JSON so it survives restarts."""

    def __init__(self, state_path: Path, rpd: int, model: str):
        # Different models are independent quota buckets (empirically confirmed: a
        # Gemma model succeeded via this same API key on a day gemini-2.5-flash's
        # quota was already exhausted) — keying state by model name stops a model
        # switch from inheriting a stale, irrelevant count from a different bucket.
        self.state_path = state_path
        self.rpd = rpd
        self.model = model
        self._date, self._count = self._load()

    def _load(self) -> tuple[str, int]:
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        if self.state_path.exists():
            all_state = json.loads(self.state_path.read_text(encoding="utf-8"))
            entry = all_state.get(self.model, {})
            if entry.get("date") == today:
                return today, entry.get("count", 0)
        return today, 0

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        all_state = {}
        if self.state_path.exists():
            try:
                all_state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        all_state[self.model] = {"date": self._date, "count": self._count}
        self.state_path.write_text(json.dumps(all_state), encoding="utf-8")

    def check_and_increment(self) -> None:
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        if today != self._date:
            self._date, self._count = today, 0  # new UTC day — quota resets
        if self._count >= self.rpd:
            raise DailyQuotaExceeded(
                f"Gemini daily request cap ({self.rpd}) reached for {self.model} on "
                f"{today} (UTC). Resume this run after the quota resets."
            )
        self._count += 1
        self._save()

    @property
    def count_today(self) -> int:
        return self._count


class RpmLimiter:
    """Spaces calls at least 60/rpm seconds apart. In-memory only — not meant to
    survive restarts, unlike the daily tracker."""

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / rpm
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


class GeminiGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        rpm_limiter: RpmLimiter,
        daily_tracker: DailyQuotaTracker,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.rpm_limiter = rpm_limiter
        self.daily_tracker = daily_tracker

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _call(self, prompt: str) -> types.GenerateContentResponse:
        return await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GENERATION_SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=GeminiAnswer,
            ),
        )

    async def generate(
        self, question: str, passages: list[dict]
    ) -> tuple[str, list[str]]:
        """passages: [{"passage_id", "title", "text"}, ...]. Raises DailyQuotaExceeded
        before making a call once the RPD cap is hit for today."""
        self.daily_tracker.check_and_increment()
        await self.rpm_limiter.wait()

        numbered = "\n\n".join(
            f"[{i}] passage_id={p['passage_id']} title={p['title']}\n{p['text']}"
            for i, p in enumerate(passages, start=1)
        )
        prompt = f"Question: {question}\n\nRetrieved passages:\n{numbered}"

        response = await self._call(prompt)
        parsed = response.parsed
        if isinstance(parsed, GeminiAnswer):
            return parsed.answer, parsed.supporting_passage_ids

        # Fallback if structured parsing didn't populate `.parsed` for any reason.
        try:
            data = json.loads(response.text)
            return data.get("answer", ""), data.get("supporting_passage_ids", [])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse Gemini response for question: %r", question)
            return response.text or "", []
