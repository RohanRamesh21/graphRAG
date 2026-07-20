import { NextResponse } from "next/server";
import type { QueryRequest, QueryResponse } from "@/lib/types";

// Server-side proxy to the FastAPI backend. The browser only ever calls this
// same-origin route — GRAPHRAG_API_URL (no NEXT_PUBLIC_ prefix) stays out of the
// client bundle, and this also sidesteps CORS entirely in production.
const BACKEND_URL = process.env.GRAPHRAG_API_URL ?? "http://localhost:8000";

export async function POST(request: Request) {
  let body: QueryRequest;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  if (!body.question || !body.question.trim()) {
    return NextResponse.json({ error: "question is required." }, { status: 400 });
  }

  let backendResponse: Response;
  try {
    backendResponse = await fetch(`${BACKEND_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      // Retrieval + generation can take several seconds — don't let Next.js's
      // default fetch behavior cut it short.
      signal: AbortSignal.timeout(60_000),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { error: `Could not reach the GraphRAG backend: ${message}` },
      { status: 502 },
    );
  }

  if (!backendResponse.ok) {
    // Surface the backend's own detail (e.g. 503 "corpus not loaded — run ingestion")
    // rather than a generic failure.
    let detail = `Backend returned ${backendResponse.status}`;
    try {
      const errorBody = await backendResponse.json();
      detail = errorBody.detail ?? detail;
    } catch {
      // response wasn't JSON — keep the generic detail
    }
    return NextResponse.json({ error: detail }, { status: backendResponse.status });
  }

  const data: QueryResponse = await backendResponse.json();
  return NextResponse.json(data);
}
