# GraphRAG System — Build Instructions for Claude Code

## Objective
Build an agentic GraphRAG system that answers multi-hop questions by combining graph
traversal (Neo4j Aura) with vector retrieval (Qdrant), using DeepSeek V4 Flash for both
graph construction (entity/relation extraction) and answer generation, orchestrated via
LangGraph, and exposed through a FastAPI service. Evaluate the system on the
2WikiMultiHopQA benchmark.

## Tech stack
- Graph database: Neo4j Aura (free tier)
- Vector store: Qdrant (default choice; note in README how to swap to Pinecone if needed)
- LLM (graph construction / entity-relation extraction): DeepSeek V4 Flash via API
  (confirm exact model string from current DeepSeek API docs at build time, don't
  hardcode from memory)
- LLM (answer generation, both GraphRAG and baseline eval runs): Gemini free tier
  (confirm current model name and rate limits — RPM/RPD/TPM — from Google's docs at
  build time, these shift over time)
- Orchestration: LangGraph
- API layer: FastAPI
- Language: Python 3.11+

## Data & corpus construction
1. Download 2WikiMultiHopQA (dev split — test is likely blind/hidden, dev is the
   standard choice for this kind of eval).
2. Sample 1,000 questions.
3. Pool the union of all supporting + distractor context passages across the sampled
   questions into a single deduplicated corpus. Expect roughly 6,000-6,500 unique
   passages, but verify the actual count once loaded rather than assuming it.
4. Persist this pooled corpus (id, title, text, source_question_ids) to disk as the
   ingestion source of truth before running any extraction, so extraction is resumable
   without re-downloading or re-pooling.

## Pipeline stages

### Stage 1: Entity & relation extraction (DeepSeek V4 Flash)
- For each passage, prompt DeepSeek to extract (entity, relation, entity) triples plus
  entity types.
- Design the prompt so the instruction/schema portion is a fixed, reusable prefix (to
  benefit from DeepSeek's automatic prompt caching), with only the passage text varying
  per call.
- Implement checkpointing: write extraction results incrementally (e.g. one JSON line
  per passage) so a crash or rate limit doesn't require re-running the whole batch.
- Implement retry/backoff for rate limits and malformed JSON output; validate extraction
  output against a strict schema before accepting it.
- Log token usage per call so actual cost can be reconciled against estimates later.

### Stage 2: Graph construction (Neo4j Aura)
- Schema: `(:Entity {name, type})`, `(:Passage {id, title, text})`, relationship types
  taken directly from extracted triples between Entity nodes, plus
  `(:Passage)-[:MENTIONS]->(:Entity)`.
- Deduplicate entities via normalized name matching (case-fold, strip whitespace) before
  merging nodes. Flag this as a known limitation (no proper entity resolution or
  coreference) rather than trying to solve it in v1.
- Batch writes via Cypher `UNWIND` rather than one write per triple.

### Stage 3: Vector indexing (Qdrant)
- Embed each passage (pick and document the embedding model — an open-source
  sentence-transformers model is fine for v1) and upsert into Qdrant with passage id as
  payload, linking back to the corresponding Neo4j Passage node.
- Embed at the passage level (not entity level) for v1.

### Stage 4: Retrieval (hybrid graph + vector)
- Given a question: vector search in Qdrant to find seed passages/entities.
- From seed entities, traverse the Neo4j graph outward (configurable hop depth, default
  2) to collect candidate supporting passages along relation paths.
- Merge and rank candidates (vector similarity score + graph path relevance) before
  passing to generation.

### Stage 5: Answer generation (Gemini free tier)
- Given the question and retrieved passages/graph context, generate a final answer plus
  the supporting passage IDs used (needed for supporting-fact F1 scoring).
- This stage uses Gemini's free tier, not DeepSeek. Throttle requests to stay under the
  RPM limit (e.g. a simple rate limiter around the client, one call every few seconds)
  and expect the RPD cap to be the binding constraint on how many questions you can
  process per day, not TPM or RPM.

### Stage 6: Orchestration (LangGraph)
- Wrap stages 4-5 in a LangGraph graph: retrieve → (optionally expand/re-retrieve if
  confidence is low) → generate → validate.
- Keep this simple for v1: a linear retrieve-then-generate flow with one optional
  re-retrieval loop is enough. Don't build a more complex agent loop before the baseline
  is working end to end.

### Stage 7: Evaluation harness
- Run the full 1,000-question sample through the pipeline.
- Report: Exact Match, F1 (answer), and supporting-fact F1, matching 2WikiMultiHopQA's
  standard metrics.
- Save per-question predictions and gold answers to a results file for later error
  analysis.
- Include a simple baseline (vector-only retrieval, no graph) run over the same corpus,
  so the graph's actual contribution is measurable rather than asserted.
- Both the GraphRAG eval run and the baseline run use Gemini's free tier for generation
  (1,000 questions each, ~2,000 calls total, plus any extra calls from a re-retrieval
  loop). This will likely span more than one day given the daily request cap, so the
  eval script must checkpoint per-question results and be resumable across sessions
  rather than assuming it completes in one run.

## API layer (FastAPI)
- `POST /query` — accepts a question, returns answer + supporting passages + reasoning
  path.
- `GET /health` — basic liveness check.
- `POST /ingest` — trigger corpus ingestion/graph build; must be idempotent and skip
  already-processed passages on re-run.
- `GET /graph/stats` — node/edge counts, for sanity-checking the build.

## Config & secrets
- All credentials (`DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `NEO4J_AURA_URI`,
  `NEO4J_AURA_USER`, `NEO4J_AURA_PASSWORD`, `QDRANT_URL`, `QDRANT_API_KEY`) via `.env`,
  never hardcoded.
- Include a `.env.example` with placeholder values.

## Deliverables
- Clean repo structure separating ingestion/extraction, graph, retrieval, generation,
  API, and eval code.
- README covering setup, how to run ingestion, how to run eval, and the reported
  metrics.
- Requirements file (or `pyproject.toml`) pinning key dependency versions.
- Eval results (metrics + a few qualitative example traces showing multi-hop retrieval
  actually working) written to a results file/report.

## Constraints
- DeepSeek is used only for graph construction (Stage 1). That budget is small (well
  under $5 for this corpus size) but not unlimited — the available DeepSeek credit is
  under $3 total, so add a hard spend guard (e.g. track cumulative token usage/cost
  locally and halt with a clear error before exceeding a set ceiling, such as $2) rather
  than relying on the estimate alone. Prioritize correctness and resumability so a bug
  can't force an expensive full re-run.
- Gemini free tier is used for all answer generation (Stages 5 and 7, both the GraphRAG
  eval run and the baseline run). This costs no money but is rate-limited, so design for
  a multi-day, resumable batch job rather than a single continuous run.
- Don't build entity resolution, ontology alignment, or community summarization for v1.
  Document these as future work in the README rather than implementing them now.
- Get the retrieve → generate → eval loop working end-to-end on a small subset (e.g. 50
  questions) before scaling to the full 1,000, so failures are caught early and cheaply.
- **Never read the raw 2WikiMultiHopQA dataset file directly into context** (e.g. via a
  file-viewing/read tool on the full JSON). It's large enough to consume most of a usage
  budget in a single read. All dataset access — sampling questions, pooling passages,
  deduplication, counting — must go through a Python script executed via the shell, with
  only small, aggregated outputs (counts, a handful of example records, summary stats)
  ever printed back to the console. If you need to inspect the data's structure, print
  just one or two sample records, not the full file or a large slice of it.