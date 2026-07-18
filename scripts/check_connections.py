#!/usr/bin/env python
"""Connectivity smoke test for all four external services. Prints only pass/fail per
service and short error classes — never any credential value — so it's safe to run and
share output from. Run this before a full ingestion/eval pass.

Usage:
    python scripts/check_connections.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def check_deepseek(settings) -> tuple[bool, str]:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com")
        client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_gemini(settings) -> tuple[bool, str]:
    try:
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)
        client.models.generate_content(model=settings.gemini_model, contents="ping")
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_neo4j(settings) -> tuple[bool, str]:
    try:
        from graphrag.graph.neo4j_client import make_driver

        driver = make_driver(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
        try:
            driver.verify_connectivity()
        finally:
            driver.close()
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_qdrant(settings) -> tuple[bool, str]:
    try:
        from graphrag.vector.qdrant_store import make_client

        client = make_client(settings.qdrant_url, settings.qdrant_api_key)
        client.get_collections()
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    from graphrag.config import ConfigError, get_settings

    try:
        settings = get_settings()
    except ConfigError as e:
        # get_settings() already sanitizes this (field names only, never values) — see
        # config.ConfigError's docstring for why that sanitization lives there and not
        # here: any uncaught pydantic ValidationError would otherwise print other,
        # perfectly-valid secrets that happened to load alongside the missing one.
        print(f"FAIL  config      {e}")
        sys.exit(1)

    checks = {
        "deepseek": check_deepseek,
        "gemini": check_gemini,
        "neo4j": check_neo4j,
        "qdrant": check_qdrant,
    }

    all_ok = True
    for name, fn in checks.items():
        ok, detail = fn(settings)
        all_ok &= ok
        status = "OK  " if ok else "FAIL"
        print(f"{status}  {name:<10} {detail}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
