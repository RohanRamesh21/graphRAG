"""Central configuration, loaded once from `.env` via pydantic-settings.

Field names follow the instructions.md spec (NEO4J_AURA_URI, QDRANT_URL, ...), but each
credential also accepts the alternate name actually present in this project's `.env`
(NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD, QDRANT_CLUSTER_ENDPOINT) so either convention
works without editing the file. See `.env.example` for the full set of variables.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Credentials -------------------------------------------------------
    deepseek_api_key: str = Field(validation_alias=AliasChoices("DEEPSEEK_API_KEY"))
    gemini_api_key: str = Field(validation_alias=AliasChoices("GEMINI_API_KEY"))

    neo4j_uri: str = Field(validation_alias=AliasChoices("NEO4J_URI", "NEO4J_AURA_URI"))
    neo4j_username: str = Field(
        validation_alias=AliasChoices("NEO4J_USERNAME", "NEO4J_AURA_USER")
    )
    neo4j_password: str = Field(
        validation_alias=AliasChoices("NEO4J_PASSWORD", "NEO4J_AURA_PASSWORD")
    )
    neo4j_database: str = Field(default="neo4j", validation_alias=AliasChoices("NEO4J_DATABASE"))

    qdrant_url: str = Field(
        validation_alias=AliasChoices("QDRANT_URL", "QDRANT_CLUSTER_ENDPOINT")
    )
    qdrant_api_key: str = Field(validation_alias=AliasChoices("QDRANT_API_KEY"))

    # --- Models --------------------------------------------------------------
    # Confirmed live against provider docs on 2026-07-18 — re-verify if it has been a
    # while, both providers rotate model strings and free-tier limits over time.
    deepseek_model: str = Field(default="deepseek-v4-flash")
    # gemini-2.5-flash's real free-tier quota for this project measured only 20
    # requests/day (confirmed via a live 429:
    # "GenerateRequestsPerDayPerProjectPerModel-FreeTier ... quotaValue: '20'"), far
    # below the 250-1500/day range docs/third-party sources suggested — that would put
    # a full 1,000-question x 2-mode eval at 100+ days. Switched to a Gemma model
    # served via the same API/SDK: empirically confirmed to sit on a SEPARATE quota
    # bucket (20 rapid calls succeeded on a day gemini-2.5-flash's bucket was already
    # exhausted). "gemma-4-31b-it" was picked over the smaller "gemma-4-26b-a4b-it" for
    # answer quality, since RPD — not latency — is the binding constraint here anyway.
    gemini_model: str = Field(default="gemma-4-31b-it")
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # --- Cost guard (Stage 1 / DeepSeek extraction) ---------------------------
    deepseek_cost_ceiling_usd: float = Field(default=2.0)
    extraction_concurrency: int = Field(default=8)

    # --- Generation backend (Stages 5 & 7) ------------------------------------
    # "deepseek": pay-per-use via the same DeepSeek account as extraction, no daily
    #   request cap — chosen after Gemini/Gemma's free-tier daily quota made a full
    #   1,000-question x 2-mode eval take multiple days; DeepSeek was empirically
    #   estimated at ~$0.45 total for the same workload (see deepseek_common.py).
    # "gemini": the original free-tier path (gemini_model, gemini_rpm/rpd below) —
    #   still fully implemented in generation/gemini_client.py, just not the default.
    # Both draw from the SAME account balance as Stage 1 extraction, so this ceiling is
    # sized with that already-spent cost in mind, not just this stage in isolation.
    generation_backend: str = Field(default="deepseek")
    deepseek_generation_cost_ceiling_usd: float = Field(default=1.5)

    # --- Gemini/Gemma rate limiting (Stages 5 & 7, only used if generation_backend="gemini") ---
    # These are informed estimates, not a confirmed ceiling — 20 rapid calls to
    # gemma-4-31b-it succeeded with no throttling, but that's a lower bound, not the
    # true cap (deliberately didn't burn further free-tier quota just to find the
    # exact edge). Re-confirm live in AI Studio before a full run. Precision doesn't
    # matter much either way: DailyQuotaTracker is keyed per-model (so switching models
    # never inherits a stale count from a different quota bucket) and the eval runner
    # halts gracefully on a real 429 from the API regardless of what's configured here.
    gemini_rpm: int = Field(default=15)
    gemini_rpd: int = Field(default=200)

    # --- Retrieval -------------------------------------------------------------
    hop_depth: int = Field(default=2)
    top_k_vector: int = Field(default=5)
    top_k_final: int = Field(default=8)
    qdrant_collection: str = Field(default="graphrag_passages")
    # Excludes generic "hub" entities (e.g. a bare year, a nationality adjective) from
    # being used as graph-traversal seeds — found live at full corpus scale (41k+
    # entities), where a couple of seed entities with degree ~40 caused a 2-hop
    # traversal to stall for minutes per question. See GraphRetriever's docstring.
    max_seed_entity_degree: int = Field(default=30)

    # --- API / CORS ----------------------------------------------------------
    # Comma-separated list of origins allowed to call this API directly from a browser.
    # The primary frontend integration (web/src/app/api/query/route.ts) is a server-side
    # proxy, which isn't subject to CORS at all — this exists as defense-in-depth and to
    # support hitting the API directly (e.g. /docs) from a browser during development.
    allowed_origins_raw: str = Field(
        default="http://localhost:3000", validation_alias=AliasChoices("ALLOWED_ORIGINS")
    )

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins_raw.split(",") if origin.strip()]

    # --- Paths -------------------------------------------------------------
    # REPO_ROOT/DATA_DIR are computed from this file's location, which only resolves
    # correctly for an editable/source-tree install (the local .venv dev setup). Under
    # a regular `pip install .` — exactly what the Dockerfile does — the package lands
    # in site-packages, so that computation silently points at the wrong place (found
    # live: a container's /query returned "corpus not loaded" even though corpus.jsonl
    # was right there at /app/data, because DATA_DIR had resolved to
    # /usr/local/lib/python3.11/data instead). The Dockerfile sets DATA_DIR=/app/data
    # explicitly so a real deployment never depends on this fragile default at all.
    data_dir: Path = Field(default=DATA_DIR, validation_alias=AliasChoices("DATA_DIR"))

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus.jsonl"

    @property
    def sample_questions_path(self) -> Path:
        return self.data_dir / "sample_questions.jsonl"

    @property
    def extraction_dir(self) -> Path:
        return self.data_dir / "extraction"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"


class ConfigError(RuntimeError):
    """Raised in place of pydantic's ValidationError for missing/invalid settings.

    Deliberately does NOT wrap or chain the original ValidationError: pydantic-settings
    error reprs embed each field's `input_value`, which — for a validation failure on
    one field — still includes the raw values of *other* fields that loaded fine (i.e.
    real secrets from .env). Letting that propagate (via `raise ... from e`, logging,
    or an uncaught traceback) would print credential material. Only field names, never
    values, make it into this exception's message.
    """


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — import and call this, don't instantiate Settings()."""
    try:
        return Settings()
    except ValidationError as e:
        missing = ", ".join(
            ".".join(str(p) for p in err["loc"])
            for err in e.errors(include_url=False, include_input=False, include_context=False)
        )
        raise ConfigError(
            f"Missing/invalid settings: {missing}. Check .env against .env.example "
            "(field names, not values, are shown here on purpose)."
        ) from None
