"""Answer-generation prompt for DeepSeek (Stages 5 & 7).

Same fixed-prefix design as extraction/prompt.py: the schema/instructions live entirely
in the system message and never change, so DeepSeek's automatic prefix caching applies
to that portion on every call. The retrieved passages and question are the only
variable content, in the user message — unlike extraction, this varies per *question*
rather than per fixed passage, so the cache-hit fraction of each call is smaller here
(the system prompt is still cached, but the passages dominate the token count and
differ every time).
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a careful multi-hop question-answering assistant. You are \
given a question and a numbered list of retrieved passages (which may include \
irrelevant distractors). Using ONLY the passages provided:
1. Give the shortest correct answer to the question (a name, date, yes/no, or short \
phrase — not a full sentence).
2. List the passage_id values of the passages you actually relied on to answer.
If the passages don't contain enough information, give your best-guess answer anyway \
and return an empty supporting_passage_ids list.

Respond with ONLY a single JSON object (no markdown fences, no commentary), matching \
exactly this schema:
{"answer": "<string>", "supporting_passage_ids": ["<string>", ...]}"""


def build_messages(question: str, passages: list[dict]) -> list[dict[str, str]]:
    """passages: [{"passage_id", "title", "text"}, ...]"""
    numbered = "\n\n".join(
        f"[{i}] passage_id={p['passage_id']} title={p['title']}\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nRetrieved passages:\n{numbered}"},
    ]
