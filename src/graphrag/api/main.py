"""FastAPI service: POST /query, GET /health, POST /ingest, GET /graph/stats.

Shared clients (Neo4j driver, Qdrant client, embedder, LangGraph app) are built once at
startup via the lifespan context and stashed on `app.state`, rather than reconnecting
per-request.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graphrag.config import get_settings
from graphrag.data.build_corpus import build as build_corpus
from graphrag.data.download import download_2wiki
from graphrag.deepseek_common import SpendTracker
from graphrag.extraction.run_extraction import run as run_extraction
from graphrag.generation.deepseek_generator import DeepSeekGenerator
from graphrag.generation.gemini_client import DailyQuotaTracker, GeminiGenerator, RpmLimiter
from graphrag.graph.build import build_graph
from graphrag.graph.neo4j_client import get_stats, make_driver
from graphrag.orchestration.graph import answer_question, build_graphrag_app
from graphrag.retrieval.graph_retriever import GraphRetriever
from graphrag.retrieval.hybrid import HybridRetriever
from graphrag.retrieval.vector_retriever import VectorRetriever
from graphrag.schemas import Passage
from graphrag.vector.embedder import Embedder
from graphrag.vector.qdrant_store import count as qdrant_count
from graphrag.vector.qdrant_store import ensure_collection, make_client, upsert_passages

logger = logging.getLogger(__name__)


def _load_corpus_lookup(path) -> dict[str, Passage]:
    lookup = {}
    if not path.exists():
        return lookup
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            p = Passage.model_validate_json(line)
            lookup[p.id] = p
    return lookup


def _sum_usage_cost(predictions_path) -> float:
    """Mirrors eval/runner.py's helper of the same name — both the batch eval and this
    live API draw against the same DeepSeek account balance, so the API's spend
    tracker must seed from whatever the eval runs have already spent."""
    if not predictions_path.exists():
        return 0.0
    total = 0.0
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                total += json.loads(line).get("usage", {}).get("cost_usd", 0.0)
    return total


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings

    app.state.embedder = Embedder(settings.embed_model)
    app.state.qdrant_client = make_client(settings.qdrant_url, settings.qdrant_api_key)
    ensure_collection(app.state.qdrant_client, settings.qdrant_collection)
    app.state.neo4j_driver = make_driver(
        settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password
    )

    app.state.passage_lookup = _load_corpus_lookup(settings.corpus_path)

    vector_retriever = VectorRetriever(
        app.state.embedder, app.state.qdrant_client, settings.qdrant_collection
    )
    graph_retriever = GraphRetriever(
        app.state.neo4j_driver,
        settings.neo4j_database,
        hop_depth=settings.hop_depth,
        max_seed_entity_degree=settings.max_seed_entity_degree,
    )
    hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever, app.state.passage_lookup)

    if settings.generation_backend == "deepseek":
        # Seeds from both eval prediction files — this server, the GraphRAG eval, and
        # the baseline eval all draw against the same DeepSeek account balance.
        starting_cost = _sum_usage_cost(
            settings.results_dir / "graphrag_predictions.jsonl"
        ) + _sum_usage_cost(settings.results_dir / "baseline_predictions.jsonl")
        spend_tracker = SpendTracker(settings.deepseek_generation_cost_ceiling_usd, starting_cost)
        generator = DeepSeekGenerator(settings.deepseek_api_key, settings.deepseek_model, spend_tracker)
    else:
        daily_tracker = DailyQuotaTracker(
            settings.results_dir / "gemini_daily_quota.json", settings.gemini_rpd, model=settings.gemini_model
        )
        rpm_limiter = RpmLimiter(settings.gemini_rpm)
        generator = GeminiGenerator(
            settings.gemini_api_key, settings.gemini_model, rpm_limiter, daily_tracker
        )

    app.state.langgraph_app = build_graphrag_app(hybrid_retriever, generator)
    app.state.ingest_running = False

    yield

    app.state.neo4j_driver.close()


app = FastAPI(title="GraphRAG API", lifespan=lifespan)

# CORS is configured at app-construction time (not inside lifespan), so settings are
# loaded once here too — see Settings.allowed_origins in config.py for the rationale.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class QueryRequest(BaseModel):
    question: str
    top_k_vector: int | None = None
    top_k_final: int | None = None


class QueryResponse(BaseModel):
    answer: str
    supporting_passage_ids: list[str]
    reasoning_path: list[str]


class IngestResponse(BaseModel):
    status: str
    detail: str


class StatsResponse(BaseModel):
    neo4j: dict
    qdrant_points: int


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    settings = app.state.settings
    if not app.state.passage_lookup:
        raise HTTPException(
            status_code=503,
            detail="Corpus not loaded — run ingestion (POST /ingest) before querying.",
        )

    final_state = await answer_question(
        app.state.langgraph_app,
        request.question,
        request.top_k_vector or settings.top_k_vector,
        request.top_k_final or settings.top_k_final,
    )
    reasoning_path = [
        f"{r.title} ({r.passage_id}): "
        + ", ".join(
            filter(
                None,
                [
                    f"vector_score={r.vector_score:.3f}" if r.vector_score is not None else None,
                    f"graph_path={' -> '.join(r.hop_path)}" if r.hop_path else None,
                ],
            )
        )
        for r in final_state["retrieved"]
    ]
    return QueryResponse(
        answer=final_state["answer"],
        supporting_passage_ids=final_state["supporting_passage_ids"],
        reasoning_path=reasoning_path,
    )


@app.get("/graph/stats", response_model=StatsResponse)
async def graph_stats():
    settings = app.state.settings
    neo4j_stats = get_stats(app.state.neo4j_driver, settings.neo4j_database)
    points = qdrant_count(app.state.qdrant_client, settings.qdrant_collection)
    return StatsResponse(neo4j=neo4j_stats, qdrant_points=points)


async def _full_ingest(app: FastAPI) -> None:
    settings = app.state.settings
    try:
        logger.info("Ingest: downloading/sampling/pooling corpus (idempotent)...")
        raw_path = await asyncio.to_thread(download_2wiki, settings.data_dir / "raw")
        await asyncio.to_thread(
            build_corpus,
            raw_path,
            settings.corpus_path,
            settings.sample_questions_path,
            1000,
            42,
            False,
        )

        logger.info("Ingest: running extraction (resumable, cost-guarded)...")
        await run_extraction(settings)

        logger.info("Ingest: building graph (idempotent MERGE)...")
        extractions_path = settings.extraction_dir / "extractions.jsonl"

        def _build_graph_sync():
            driver = make_driver(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
            try:
                return build_graph(driver, settings.neo4j_database, settings.corpus_path, extractions_path)
            finally:
                driver.close()

        graph_stats_result = await asyncio.to_thread(_build_graph_sync)
        logger.info("Ingest: graph build result: %s", graph_stats_result)

        logger.info("Ingest: embedding + upserting into Qdrant (idempotent)...")

        def _build_index_sync():
            passages = list(_load_corpus_lookup(settings.corpus_path).values())
            embedder = Embedder(settings.embed_model)
            client = make_client(settings.qdrant_url, settings.qdrant_api_key)
            ensure_collection(client, settings.qdrant_collection)
            batch_size = 64
            for i in range(0, len(passages), batch_size):
                batch = passages[i : i + batch_size]
                vectors = embedder.embed_passages([p.text for p in batch])
                upsert_passages(
                    client,
                    settings.qdrant_collection,
                    [p.id for p in batch],
                    [p.title for p in batch],
                    vectors,
                )

        await asyncio.to_thread(_build_index_sync)

        # Refresh in-memory lookup used by /query now that ingestion has (re)run.
        app.state.passage_lookup = _load_corpus_lookup(settings.corpus_path)
        logger.info("Ingest: complete.")
    except Exception:
        logger.exception("Ingest run failed.")
    finally:
        app.state.ingest_running = False


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    """Idempotent: every stage skips already-processed work on re-run (extraction
    checkpoints per passage, graph writes are MERGE-based, vector upserts use
    deterministic point ids). Runs as a background task since a full corpus build can
    take well beyond a typical HTTP timeout — poll GET /graph/stats for progress."""
    if app.state.ingest_running:
        return IngestResponse(status="already_running", detail="An ingest run is already in progress.")
    app.state.ingest_running = True
    asyncio.create_task(_full_ingest(app))
    return IngestResponse(
        status="started",
        detail="Ingestion running in the background (idempotent, resumable). Poll GET /graph/stats for progress.",
    )
