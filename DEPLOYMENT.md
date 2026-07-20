# Deployment Plan

## Architecture

```
Browser
  │
  ▼
Vercel (Next.js — web/)
  │  same-origin, no CORS involved
  ▼
web/src/app/api/query/route.ts  (server-side proxy)
  │  server-to-server HTTP call, GRAPHRAG_API_URL
  ▼
FastAPI backend (src/graphrag/api/main.py)
  │
  ├──▶ Neo4j Aura (graph)
  ├──▶ Qdrant Cloud (vectors)
  └──▶ DeepSeek API (generation — pay-per-use, no daily cap)
```

The frontend never talks to the backend directly from the browser — `POST /api/query`
(a Next.js Route Handler) proxies the request server-side. This keeps the backend's URL
out of the client bundle and sidesteps browser CORS entirely in production. CORS is
still configured on the FastAPI side (`ALLOWED_ORIGINS`) as a fallback for hitting the
backend directly (e.g. its `/docs` page) during development.

Ingestion (corpus build → extraction → graph → vector index) has **already run** against
the live Neo4j/Qdrant instances — the deployed backend only needs to serve `GET /health`
and `POST /query`. It ships `data/corpus.jsonl` (3.4MB) baked into its image for the
passage-text lookup `/query` needs; it does not need to re-run `/ingest`.

## Two paths — pick based on how much cold-start latency is acceptable

Both are fully free. Neither requires code changes beyond what's already in this repo
(`Dockerfile`, `.dockerignore`, CORS middleware, the `web/` app) — whichever is picked,
these steps are ready to execute as-is.

### Path A — Vercel for both frontend and backend

Simplest mental model (one platform, one dashboard). Vercel's Python serverless function
limit was raised to 500MB in Feb 2026, so the backend's heaviest dependency
(sentence-transformers/torch + the ~130MB embedding model) fits — but expect:
- **Cold-start latency**: every request after an idle gap reloads torch, the embedding
  model, and reconnects to Neo4j — likely 10-30s added to that first message.
- **Execution-time risk**: `/query` typically takes 3-10s (retrieval + DeepSeek
  generation); Hobby-tier function timeouts are tight against that, especially with a
  cold start stacked on top.

**Steps:**
1. `vercel.com` → New Project → import this repo → set **Root Directory** to `web/`.
   Vercel auto-detects Next.js; no build config needed.
2. Add env var `GRAPHRAG_API_URL` once the backend project exists (step 4 creates a
   circular dependency — deploy the backend project first, or redeploy the frontend
   after adding this var).
3. For the backend: a **second** Vercel project, root directory `/` (repo root), using
   Vercel's [Python runtime](https://vercel.com/docs/functions/runtimes/python). This
   needs a `vercel.json` routing all requests to `src/graphrag/api/main:app` and,
   likely, `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` set as an env var given the dependency
   size — not included in this repo yet since this path is deferred; ask if you want it
   scaffolded before executing this path.
4. Add the 7 backend secrets from `.env.example` (`DEEPSEEK_API_KEY`, `NEO4J_URI`,
   `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `QDRANT_URL`, `QDRANT_API_KEY`) plus
   `ALLOWED_ORIGINS=https://<your-frontend>.vercel.app` as Vercel project env vars.
5. Redeploy the frontend project with `GRAPHRAG_API_URL` pointed at the backend
   project's URL.

### Path B — Vercel frontend + Railway backend (originally recommended)

Two platforms, but the backend runs as a persistent container: no cold-start reload, no
per-request timeout. Railway's free tier ($5 first month, $1/month credit after)
confirmed to require no credit card.

**Steps:**
1. **Backend on Railway**: railway.app → New Project → Deploy from GitHub repo → Railway
   detects the root `Dockerfile` automatically. Set the 7 backend secrets from
   `.env.example` plus `ALLOWED_ORIGINS` as Railway environment variables (Settings →
   Variables). Railway injects `PORT` automatically — the `Dockerfile`'s `CMD` already
   reads it. Once deployed, note the generated `*.up.railway.app` URL.
2. **Frontend on Vercel**: same as Path A step 1, but simpler — no circular dependency,
   since the backend URL from step 1 is already known.
   Set `GRAPHRAG_API_URL=https://<your-backend>.up.railway.app`.
3. Set `ALLOWED_ORIGINS=https://<your-frontend>.vercel.app` on Railway (redeploy to
   apply) — belt-and-suspenders alongside the proxy pattern.
4. Monitor Railway's $1/month post-trial budget if the backend needs to stay warm
   24/7 — a low-traffic personal chat app should fit comfortably, but it's the one
   number worth watching over time.

## Environment variables reference

| Variable | Where | Notes |
|---|---|---|
| `GRAPHRAG_API_URL` | Vercel (frontend project) | Server-only, no `NEXT_PUBLIC_` prefix |
| `DEEPSEEK_API_KEY` | Backend host | Generation + (already-run) extraction |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Backend host | Aura connection |
| `QDRANT_URL` / `QDRANT_API_KEY` | Backend host | Qdrant Cloud connection |
| `ALLOWED_ORIGINS` | Backend host | Comma-separated; add the frontend's URL |

See `.env.example` for the full set of optional tunables (generation backend/model,
cost ceilings, retrieval params) — defaults are sensible for either path.

## Verification (either path)

1. `GET https://<backend>/health` → `{"status": "ok"}`.
2. `GET https://<backend>/graph/stats` → non-zero `neo4j`/`qdrant_points` counts,
   confirming the deployed backend reached the same live Neo4j/Qdrant instances.
3. Open the deployed frontend, ask a real multi-hop question (e.g. "Who is the mother
   of the director of Polish-Russian War?"), confirm an answer renders with a working
   "Show reasoning" section.

## Out of scope for now

`.github/workflows/keep-alive.yml` already keeps Neo4j Aura and Qdrant Cloud from being
auto-paused — unaffected by either path above, no changes needed. If the backend host
itself has an inactivity-based free-tier sleep (Railway's hobby tier does not, on the
$1/month credit model, as long as budget remains), that would need its own keep-alive
ping — not set up yet since the hosting path itself is still undecided.
