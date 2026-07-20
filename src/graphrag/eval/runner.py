"""Shared eval-run core for both GraphRAG and baseline (vector-only) modes.

Resumable by construction: predictions are checkpointed one JSON line per question as
soon as they're produced, and a re-invocation skips any question_id already present in
that mode's results file. Several conditions halt a run cleanly instead of crashing it:
a DeepSeek cost-ceiling breach, the local Gemini daily-quota pre-check, a real 429 from
the Gemini API (if GEMINI_RPD is set too high), and Neo4j becoming unavailable for
longer than the driver's own retry budget — all resumable by just re-running the same
command later (or, for the cost ceiling, after raising it deliberately).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from google.genai.errors import ClientError
from neo4j.exceptions import DriverError

from graphrag.config import Settings
from graphrag.deepseek_common import CostCeilingExceeded, SpendTracker
from graphrag.generation.deepseek_generator import DeepSeekGenerator
from graphrag.generation.gemini_client import DailyQuotaExceeded, DailyQuotaTracker, GeminiGenerator, RpmLimiter
from graphrag.graph.neo4j_client import make_driver
from graphrag.orchestration.graph import answer_question, build_graphrag_app
from graphrag.retrieval.graph_retriever import GraphRetriever, NullGraphRetriever
from graphrag.retrieval.hybrid import HybridRetriever
from graphrag.retrieval.vector_retriever import VectorRetriever
from graphrag.schemas import ExtractionUsage, Passage, Prediction, SampledQuestion
from graphrag.vector.embedder import Embedder
from graphrag.vector.qdrant_store import make_client

logger = logging.getLogger(__name__)

_OTHER_MODE = {"graphrag": "baseline", "baseline": "graphrag"}


def _load_questions(path: Path) -> list[SampledQuestion]:
    with path.open("r", encoding="utf-8") as f:
        return [SampledQuestion.model_validate_json(line) for line in f]


def _load_corpus_lookup(path: Path) -> dict[str, Passage]:
    lookup = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            p = Passage.model_validate_json(line)
            lookup[p.id] = p
    return lookup


def _load_done_ids(predictions_path: Path) -> set[str]:
    if not predictions_path.exists():
        return set()
    done = set()
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["question_id"])
    return done


def _sum_usage_cost(predictions_path: Path) -> float:
    if not predictions_path.exists():
        return 0.0
    total = 0.0
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                total += json.loads(line).get("usage", {}).get("cost_usd", 0.0)
    return total


def _reasoning_path(retrieved) -> list[str]:
    lines = []
    for r in retrieved:
        parts = []
        if r.vector_score is not None:
            parts.append(f"vector_score={r.vector_score:.3f}")
        if r.hop_path:
            parts.append(f"graph_path={' -> '.join(r.hop_path)}")
        lines.append(f"{r.title} ({r.passage_id}): {', '.join(parts) if parts else 'no signal'}")
    return lines


def _build_generator(settings: Settings, results_dir: Path, mode: str):
    """Returns (generator, tracker) — tracker is a SpendTracker for the DeepSeek
    backend or a DailyQuotaTracker for the Gemini backend; `run()` checks which via
    isinstance to report the right stats and to compute per-question cost deltas."""
    if settings.generation_backend == "deepseek":
        # Both eval modes (and Stage 1 extraction) draw against the SAME DeepSeek
        # account balance, so the ceiling check must see cost from both prediction
        # files, not just this mode's — otherwise running graphrag then baseline could
        # together exceed the ceiling while each individually looked fine.
        this_mode_cost = _sum_usage_cost(results_dir / f"{mode}_predictions.jsonl")
        other_mode_cost = _sum_usage_cost(results_dir / f"{_OTHER_MODE[mode]}_predictions.jsonl")
        starting_cost = this_mode_cost + other_mode_cost
        spend_tracker = SpendTracker(settings.deepseek_generation_cost_ceiling_usd, starting_cost)
        generator = DeepSeekGenerator(settings.deepseek_api_key, settings.deepseek_model, spend_tracker)
        logger.info(
            "[%s] Using DeepSeek (%s) for generation. Starting cumulative generation "
            "cost (both modes): $%.4f of $%.2f ceiling.",
            mode,
            settings.deepseek_model,
            starting_cost,
            settings.deepseek_generation_cost_ceiling_usd,
        )
        return generator, spend_tracker

    if settings.generation_backend == "gemini":
        daily_tracker = DailyQuotaTracker(
            results_dir / "gemini_daily_quota.json", settings.gemini_rpd, model=settings.gemini_model
        )
        rpm_limiter = RpmLimiter(settings.gemini_rpm)
        generator = GeminiGenerator(
            settings.gemini_api_key, settings.gemini_model, rpm_limiter, daily_tracker
        )
        return generator, daily_tracker

    raise ValueError(f"Unknown generation_backend: {settings.generation_backend!r}")


async def _generate_and_track_usage(app, question, top_k_vector, top_k_final, spend_tracker):
    """Wraps answer_question, capturing the DeepSeek spend delta (if any) for this one
    question so it can be checkpointed on the Prediction record. Free-tier backends
    (spend_tracker is a DailyQuotaTracker, not a SpendTracker) simply record zero cost."""
    cost_before = spend_tracker.cost if isinstance(spend_tracker, SpendTracker) else 0.0
    final_state = await answer_question(app, question, top_k_vector, top_k_final)
    cost_after = spend_tracker.cost if isinstance(spend_tracker, SpendTracker) else 0.0
    return final_state, ExtractionUsage(cost_usd=cost_after - cost_before)


async def run(settings: Settings, mode: str, limit: int | None = None) -> dict:
    if mode not in ("graphrag", "baseline"):
        raise ValueError(f"mode must be 'graphrag' or 'baseline', got {mode!r}")

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = settings.results_dir / f"{mode}_predictions.jsonl"

    questions = _load_questions(settings.sample_questions_path)
    if limit is not None:
        questions = questions[:limit]
    passage_lookup = _load_corpus_lookup(settings.corpus_path)

    done_ids = _load_done_ids(predictions_path)
    pending = [q for q in questions if q.question_id not in done_ids]
    logger.info(
        "[%s] %d total questions, %d already done, %d pending.",
        mode,
        len(questions),
        len(done_ids),
        len(pending),
    )
    if not pending:
        return {"mode": mode, "total": len(questions), "done": len(done_ids), "pending": 0, "halted": False}

    embedder = Embedder(settings.embed_model)
    qdrant_client = make_client(settings.qdrant_url, settings.qdrant_api_key)
    vector_retriever = VectorRetriever(embedder, qdrant_client, settings.qdrant_collection)

    driver = make_driver(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
    graph_retriever = (
        NullGraphRetriever()
        if mode == "baseline"
        else GraphRetriever(
            driver,
            settings.neo4j_database,
            hop_depth=settings.hop_depth,
            max_seed_entity_degree=settings.max_seed_entity_degree,
        )
    )
    hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever, passage_lookup)

    generator, tracker = _build_generator(settings, settings.results_dir, mode)
    app = build_graphrag_app(hybrid_retriever, generator)

    processed = 0
    halted = False
    try:
        with predictions_path.open("a", encoding="utf-8") as out_f:
            for question in pending:
                try:
                    final_state, usage = await _generate_and_track_usage(
                        app, question.question, settings.top_k_vector, settings.top_k_final, tracker
                    )
                except CostCeilingExceeded as e:
                    logger.warning(
                        "%s — stopping this run. Re-running now will hit the same "
                        "ceiling; raise deepseek_generation_cost_ceiling_usd to continue.",
                        e,
                    )
                    halted = True
                    break
                except DriverError as e:
                    # Retrieval already retries transient Neo4j errors automatically
                    # (see GraphRetriever — execute_read/execute_write retry on
                    # ServiceUnavailable/SessionExpired with backoff). If it still
                    # raises here, the outage outlasted the driver's own retry budget.
                    # A 1000-question eval can span hours; that shouldn't crash the
                    # whole batch — halt cleanly (checkpoints up to this question are
                    # already flushed) and let the next invocation resume once Neo4j is
                    # reachable again.
                    logger.warning(
                        "Neo4j became unavailable (%s: %s) — stopping this run; "
                        "resume later once it's reachable again (check with "
                        "scripts/check_connections.py).",
                        type(e).__name__,
                        e,
                    )
                    halted = True
                    break
                except DailyQuotaExceeded as e:
                    # Only reachable when generation_backend="gemini".
                    logger.warning("%s — stopping this run; resume later to continue.", e)
                    halted = True
                    break
                except ClientError as e:
                    # Only reachable when generation_backend="gemini": our local
                    # DailyQuotaTracker is a pre-emptive estimate seeded by GEMINI_RPD —
                    # if that's set higher than the account's actual live quota, the
                    # real API rejects the call with a 429 after retries are exhausted.
                    if e.code == 429:
                        logger.warning(
                            "Gemini API rejected the call with 429 (real quota reached, "
                            "exceeding the locally configured GEMINI_RPD=%d) — stopping "
                            "this run; resume later to continue. Re-check the live quota "
                            "in AI Studio and lower GEMINI_RPD accordingly.",
                            settings.gemini_rpd,
                        )
                        halted = True
                        break
                    raise

                prediction = Prediction(
                    question_id=question.question_id,
                    question=question.question,
                    gold_answer=question.answer,
                    predicted_answer=final_state["answer"],
                    supporting_passage_ids=final_state["supporting_passage_ids"],
                    gold_supporting_passage_ids=question.supporting_passage_ids,
                    reasoning_path=_reasoning_path(final_state["retrieved"]),
                    mode=mode,
                    retries=final_state["retries"],
                    usage=usage,
                )
                out_f.write(prediction.model_dump_json() + "\n")
                out_f.flush()
                processed += 1
    finally:
        driver.close()

    stats = {
        "mode": mode,
        "total": len(questions),
        "done_before": len(done_ids),
        "processed_this_run": processed,
        "halted": halted,
        "generation_backend": settings.generation_backend,
    }
    if isinstance(tracker, SpendTracker):
        stats["deepseek_generation_cumulative_cost_usd"] = tracker.cost
    else:
        stats["gemini_calls_today"] = tracker.count_today
    return stats
