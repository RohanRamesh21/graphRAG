# FastAPI backend image. Not meant for Vercel's serverless Python runtime (see
# DEPLOYMENT.md) — built for a persistent-container host (Railway, Render, Fly, etc.)
# so the embedding model and Neo4j connection load once and stay warm, matching how
# api/main.py's lifespan is designed to work.
FROM python:3.11-slim

WORKDIR /app

# Installed first (before copying source) so this layer only rebuilds when
# dependencies actually change, not on every source edit.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# torch is a transitive dependency (via sentence-transformers) and `pip install torch`
# defaults to the CUDA-enabled build on Linux — several GB, including ~500MB of NVIDIA
# cudnn/cusparselt wheels that do nothing for us: every realistic host for this image
# (Railway, Render, Fly, Vercel) is CPU-only. Installing the CPU wheel first, from
# PyTorch's own CPU index, satisfies that dependency before `pip install .` ever gets a
# chance to pull the GPU build — confirmed live: without this, the build both balloons
# to multiple GB and is prone to timing out mid-download on those CUDA wheels.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir .

# Ingestion has already been run against the live Neo4j/Qdrant instances this image
# talks to — only the small pooled corpus ships (needed by _load_corpus_lookup for
# passage text at query time), not the raw dataset or extraction/eval artifacts.
COPY data/corpus.jsonl ./data/corpus.jsonl

# Real secrets (DEEPSEEK_API_KEY, NEO4J_URI, etc.) are injected at runtime by the
# hosting platform — never baked into the image. See .env.example for the full list.
#
# DATA_DIR is set explicitly rather than relying on config.py's __file__-relative
# default, which only resolves correctly for an editable/source-tree install — under
# this image's regular `pip install .`, the package lands in site-packages, and that
# default would silently point at the wrong place (confirmed live: /query returned
# "corpus not loaded" despite data/corpus.jsonl being right here at /app/data).
ENV DATA_DIR=/app/data
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn graphrag.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
