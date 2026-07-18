"""Stage 1 batch runner: extract triples for every passage in the corpus.

Resumable: a passage counts as "done" only once a line with `error: null` exists for its
id in extractions.jsonl. Passages that only have error records (malformed JSON that
survived all in-call retries, or a run that got interrupted mid-flight) are retried on
the next invocation — nothing is silently skipped, and nothing requires re-running the
whole batch. Halts (does not crash) as soon as the cost ceiling is reached, leaving
already-written checkpoints intact.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from tqdm import tqdm

from graphrag.config import Settings
from graphrag.extraction.deepseek_client import (
    CostCeilingExceeded,
    DeepSeekExtractor,
    ExtractionFailed,
    SpendTracker,
)
from graphrag.schemas import Extraction, ExtractionUsage, Passage

logger = logging.getLogger(__name__)


def _load_corpus(corpus_path: Path) -> list[Passage]:
    passages = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            passages.append(Passage.model_validate_json(line))
    return passages


def _load_done_ids_and_cost(extractions_path: Path) -> tuple[set[str], float]:
    done: set[str] = set()
    total_cost = 0.0
    if not extractions_path.exists():
        return done, total_cost
    with extractions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total_cost += rec.get("usage", {}).get("cost_usd", 0.0)
            if rec.get("error") is None:
                done.add(rec["passage_id"])
    return done, total_cost


async def run(settings: Settings, limit: int | None = None) -> dict:
    corpus_path = settings.corpus_path
    extraction_dir = settings.extraction_dir
    extraction_dir.mkdir(parents=True, exist_ok=True)
    extractions_path = extraction_dir / "extractions.jsonl"
    cost_state_path = extraction_dir / "cost_state.json"

    passages = _load_corpus(corpus_path)
    if limit is not None:
        passages = passages[:limit]

    done_ids, starting_cost = _load_done_ids_and_cost(extractions_path)
    pending = [p for p in passages if p.id not in done_ids]

    logger.info(
        "Extraction: %d total passages, %d already done, %d pending. Starting cost: $%.4f",
        len(passages),
        len(done_ids),
        len(pending),
        starting_cost,
    )

    if not pending:
        return {"total": len(passages), "done": len(done_ids), "pending": 0, "halted": False}

    spend_tracker = SpendTracker(settings.deepseek_cost_ceiling_usd, starting_cost_usd=starting_cost)
    extractor = DeepSeekExtractor(settings.deepseek_api_key, settings.deepseek_model, spend_tracker)

    semaphore = asyncio.Semaphore(settings.extraction_concurrency)
    write_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    processed = 0
    errored = 0

    async def write_checkpoint(record: Extraction) -> None:
        async with write_lock:
            with extractions_path.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            with cost_state_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cumulative_cost_usd": spend_tracker.cost,
                        "ceiling_usd": spend_tracker.ceiling_usd,
                    },
                    f,
                )

    async def worker(passage: Passage, pbar: tqdm) -> None:
        nonlocal processed, errored
        if stop_event.is_set():
            return
        async with semaphore:
            if stop_event.is_set():
                return
            try:
                parsed, usage_dict = await extractor.extract(passage.title, passage.text)
                record = Extraction(
                    passage_id=passage.id,
                    entities=parsed.get("entities", []),
                    triples=parsed.get("triples", []),
                    usage=ExtractionUsage(**usage_dict),
                )
            except CostCeilingExceeded as e:
                logger.error(str(e))
                stop_event.set()
                return
            except ExtractionFailed as e:
                # Every attempt inside extract() cost real money even though none
                # parsed — persist that spend on the error record so a future process
                # restart's SpendTracker seeds from the true cumulative total, not an
                # undercount (see ExtractionFailed's docstring).
                logger.warning("Extraction failed for passage %s: %s", passage.id, e)
                record = Extraction(passage_id=passage.id, error=str(e), usage=ExtractionUsage(**e.usage))
                errored += 1
            except Exception as e:  # network failure after retries, etc. — no response ever received, so no spend to record
                logger.warning("Extraction failed for passage %s: %s", passage.id, e)
                record = Extraction(passage_id=passage.id, error=str(e))
                errored += 1
            await write_checkpoint(record)
            processed += 1
            pbar.update(1)

    with tqdm(total=len(pending), desc="Extracting") as pbar:
        await asyncio.gather(*(worker(p, pbar) for p in pending))

    return {
        "total": len(passages),
        "done_before": len(done_ids),
        "processed_this_run": processed,
        "errored_this_run": errored,
        "halted_on_cost_ceiling": stop_event.is_set(),
        "final_cumulative_cost_usd": spend_tracker.cost,
    }
