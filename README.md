# GraphRAG

An agentic GraphRAG system that answers multi-hop questions by combining graph
traversal (Neo4j Aura) with vector retrieval (Qdrant), using DeepSeek V4 Flash for graph
construction and a free-tier model (served via the Gemini API) for answer generation,
orchestrated with LangGraph and served via FastAPI. Evaluated on the
[2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) benchmark against a
vector-only baseline.

## How it works

```
2WikiMultiHopQA (dev split)
  -> sample 1,000 questions, pool + dedupe supporting/distractor passages   [Stage 0/data]
  -> DeepSeek V4 Flash: extract (entity, relation, entity) triples per passage  [Stage 1]
  -> Neo4j Aura: (:Entity)-[REL]->(:Entity), (:Passage)-[:MENTIONS]->(:Entity)  [Stage 2]
  -> Qdrant: passage-level embeddings (BAAI/bge-small-en-v1.5)                  [Stage 3]

Query time (LangGraph: retrieve -> generate -> validate, one re-retrieval max)  [Stage 6]
  -> vector search Qdrant for seed passages                                    [Stage 4]
  -> traverse Neo4j outward from seed entities (default 2 hops)                [Stage 4]
  -> fuse vector-ranked + graph-ranked candidates via Reciprocal Rank Fusion    [Stage 4]
  -> Gemma (via Gemini API): generate answer + cite supporting passage ids     [Stage 5]
```

Eval (Stage 7) runs the same pipeline over all 1,000 questions, plus a vector-only
baseline over the identical corpus, and reports Exact Match / Answer F1 /
Supporting-Fact F1 for both, so the graph's actual contribution is measurable.

## Tech stack

| Purpose | Choice |
|---|---|
| Graph database | Neo4j Aura (free tier) |
| Vector store | Qdrant Cloud (see [swapping vector stores](#swapping-vector-stores)) |
| Extraction LLM | DeepSeek `deepseek-v4-flash` (paid, cost-guarded) |
| Generation LLM | `gemma-4-31b-it` via the Gemini API (free tier, rate-limited — see below) |
| Embeddings | `BAAI/bge-small-en-v1.5` (open-source, passage-level) |
| Orchestration | LangGraph |
| API | FastAPI |

Model strings and free-tier limits drift over time and can vary by project — both were
re-confirmed live on 2026-07-18 (see `src/graphrag/config.py` and
`src/graphrag/generation/gemini_client.py` for specifics). Notably, `gemini-2.5-flash`'s
*actual* free-tier quota for this project measured only 20 requests/day — far below the
250-1,500/day range docs and third-party sources suggested, and low enough that a full
1,000-question x 2-mode eval would take 100+ days. Switching generation to a Gemma model
(`gemma-4-31b-it`) served through the same Gemini API/key resolved this: empirically
confirmed to sit on a separate, much more generous quota bucket (20 rapid calls
succeeded on a day `gemini-2.5-flash`'s bucket was already exhausted). If it's been a
while, re-check `https://api-docs.deepseek.com` and AI Studio's live rate-limit page
before a full run — don't trust either provider's advertised numbers at face value.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"

cp .env.example .env   # then fill in real values — see below
```

`.env` needs 7 credentials: `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, a Neo4j Aura
connection URI + username + password, and a Qdrant Cloud URL + API key. `config.py`
accepts either the instructions' naming (`NEO4J_AURA_URI`, `NEO4J_AURA_USER`, ...) or the
plainer `NEO4J_URI`/`NEO4J_USERNAME`/... names — use whichever your provisioning gave
you. See `.env.example` for the full list and format (e.g. the Neo4j URI looks like
`neo4j+s://<db-id>.databases.neo4j.io`).

Verify all four services are reachable before running anything expensive:

```bash
python scripts/check_connections.py
```

This only ever prints per-service OK/FAIL and an error class name — never credential
values — so its output is safe to paste when debugging.

## Running ingestion

Each stage is idempotent/resumable — safe to re-run after an interruption, a rate limit,
or hitting the DeepSeek cost ceiling; already-completed work is skipped, not repeated.

```bash
# 1. Download 2Wiki dev split, sample 1,000 questions, pool+dedupe the corpus
python scripts/prepare_data.py --peek   # --peek prints one record's structure only

# 2. Extract (entity, relation, entity) triples via DeepSeek — smoke test first
python scripts/run_extraction.py --limit 20
python scripts/run_extraction.py        # full corpus; resumes if interrupted

# 3. Build the Neo4j graph (MERGE-based — safe to re-run)
python scripts/build_graph.py

# 4. Embed passages and upsert into Qdrant (deterministic point ids — safe to re-run)
python scripts/build_index.py
```

Stage 1 halts (doesn't crash) before cumulative DeepSeek spend crosses
`DEEPSEEK_COST_CEILING_USD` (default $2.00; full-corpus estimate is ≈$1 thanks to
DeepSeek's automatic prompt-prefix caching — see `extraction/prompt.py`). Check progress
any time via `data/extraction/cost_state.json`.

## Running eval

Get the loop working on a small subset before committing to the full run:

```bash
python scripts/run_eval.py --limit 50
python scripts/run_baseline.py --limit 50
python scripts/build_report.py   # writes data/results/report.md
```

Then scale up. Gemini's free-tier daily request cap (not RPM/TPM) is the binding
constraint — `run_eval.py` and `run_baseline.py` **share one daily-quota counter**
(`data/results/gemini_daily_quota.json`) since both draw against the same account. A
full 1,000-question run of each (~2,000+ calls total) will very likely span multiple
days; re-running either command simply resumes from the last checkpointed question:

```bash
python scripts/run_eval.py
python scripts/run_baseline.py
python scripts/build_report.py
```

`report.md` includes the EM / Answer F1 / Supporting-Fact F1 table for both modes, the
delta between them, and a few qualitative traces showing multi-hop graph paths that
contributed to a correct answer the vector-only baseline missed.

## API

```bash
uvicorn graphrag.api.main:app --reload
```

- `GET /health` — liveness check.
- `POST /ingest` — runs the full idempotent ingestion pipeline (data prep -> extraction
  -> graph build -> vector index) as a background task; poll `GET /graph/stats` for
  progress. Safe to call repeatedly.
- `GET /graph/stats` — Neo4j node/edge counts + Qdrant point count.
- `POST /query` — `{"question": "..."}` -> answer, supporting passage ids, and a
  human-readable reasoning path (vector scores + graph hop paths per retrieved passage).

## Keeping free-tier instances alive

Since this app is meant to be hosted going forward, `.github/workflows/keep-alive.yml`
pings Neo4j Aura and Qdrant Cloud daily so neither free-tier instance auto-pauses or
gets deleted from inactivity:

- **Neo4j Aura Free**: paused after 72h idle, deleted after 90 days paused.
- **Qdrant Cloud Free**: suspended after 1 week idle, deleted after 4 weeks.

To enable it, add these as **repository secrets** (Settings -> Secrets and variables ->
Actions) — never commit them: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`,
`QDRANT_URL`, `QDRANT_API_KEY` (same alternate names as `.env` are also accepted). The
workflow can also be triggered manually from the Actions tab (`workflow_dispatch`) to
verify it's wired up correctly. GitHub disables scheduled workflows after 60 days with
no repo activity — if the ping stops running, push any commit or re-enable it manually
from the Actions tab.

## Swapping vector stores

Qdrant is the default. To swap to Pinecone: implement the four functions in
`src/graphrag/vector/qdrant_store.py` (`ensure_collection`, `upsert_passages`, `search`,
`count`) against Pinecone's client, keeping the same signatures, and swap the import in
`retrieval/vector_retriever.py` and `api/main.py`. Nothing else in the pipeline is
Qdrant-specific.

## Known limitations (v1) — deliberately out of scope

- **No entity resolution or coreference** — entities are deduplicated only by
  casefolded/whitespace-normalized name (`schemas.normalize_name`). "NYC" and "New York
  City" become different nodes.
- **No ontology alignment** — relation types are whatever DeepSeek extracts, sanitized
  into Cypher-safe identifiers, not mapped to a fixed schema.
- **No community summarization** — no clustering/summarization layer over the graph.
- **Passage-level, not entity-level, embeddings** — retrieval seeds come from
  passage vectors only.
- **Supporting-fact F1 is passage-level, not the official sentence-level** —
  2WikiMultiHopQA's gold supporting facts are `(title, sentence_id)` pairs; this
  pipeline retrieves/cites whole passages, so gold sentences are rolled up to their
  containing passage for scoring. This is a real deviation from the official metric,
  not just a rounding difference — treat cross-paper F1 comparisons with that in mind.

These are reasonable v1 cuts, not oversights — revisit them if scaling past this
benchmark-sized corpus (~6-7k passages).

## Repo structure

```
src/graphrag/
  config.py, schemas.py        # settings + shared pydantic models
  data/                         # download, sample, pool+dedupe
  extraction/                   # DeepSeek prompt, client, checkpointed batch runner
  graph/                        # Neo4j client, batched idempotent graph build
  vector/                       # embedder, Qdrant store
  retrieval/                    # vector + graph retrievers, RRF fusion
  generation/                   # Gemini client, RPM/RPD-limited
  orchestration/                # LangGraph retrieve->generate->validate
  eval/                         # metrics, resumable runners, report
  api/                          # FastAPI app
scripts/                        # thin CLI entrypoints for every stage above
tests/                          # unit tests (metrics, schemas, fusion, graph helpers)
data/                           # gitignored: corpus, checkpoints, results
```
