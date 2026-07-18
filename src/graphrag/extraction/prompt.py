"""Extraction prompt design.

The entire instruction/schema payload lives in the SYSTEM message and is byte-for-byte
identical across every call. Only the passage (title + text) goes in the user message,
with nothing else variable ahead of it. DeepSeek's automatic disk-based prefix caching
keys off the shared prefix across requests, so keeping *all* fixed content in one place
(the system message) and *only* the passage after it maximizes the cache-hit portion of
every call — this is what makes Stage 1 cheap (~$0.0028/MTok cached vs $0.14/MTok
uncached, per DeepSeek's pricing page).
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are an information extraction system. Given a passage from Wikipedia, \
extract every distinct entity and every factual relation between entities as (head, \
relation, tail) triples.

Rules:
- Entities: use the surface form as it appears in the passage (don't invent canonical \
names). Assign a short, open-vocabulary type: one of PERSON, ORGANIZATION, LOCATION, \
WORK (film/book/song/artwork), EVENT, DATE, OTHER.
- Triples: only relations explicitly stated or directly implied by the passage text. \
Use a short, general relation label (e.g. "director", "spouse", "birth_place", \
"publication_date", "based_on") — snake_case, no punctuation.
- Do not invent facts not present in the passage. Do not extract relations to entities \
outside the passage.
- If the passage contains no clear entities/relations, return empty lists.

Respond with ONLY a single JSON object (no markdown fences, no commentary), matching \
exactly this schema:
{
  "entities": [{"name": "<string>", "type": "<PERSON|ORGANIZATION|LOCATION|WORK|EVENT|DATE|OTHER>"}],
  "triples": [{"head": "<string>", "relation": "<string>", "tail": "<string>"}]
}

Example passage:
"Marie Curie was a Polish-French physicist and chemist. She was born in Warsaw in 1867 \
and conducted her research at the University of Paris."

Example response:
{"entities": [{"name": "Marie Curie", "type": "PERSON"}, {"name": "Polish-French", \
"type": "OTHER"}, {"name": "Warsaw", "type": "LOCATION"}, {"name": "1867", "type": \
"DATE"}, {"name": "University of Paris", "type": "ORGANIZATION"}], "triples": \
[{"head": "Marie Curie", "relation": "birth_place", "tail": "Warsaw"}, {"head": "Marie \
Curie", "relation": "birth_year", "tail": "1867"}, {"head": "Marie Curie", "relation": \
"affiliated_with", "tail": "University of Paris"}]}"""


def build_messages(title: str, text: str) -> list[dict[str, str]]:
    """Everything variable is confined to this one user message, after the fixed
    system prompt — this is the entire "cache-miss" surface of a call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Title: {title}\n\nPassage: {text}"},
    ]
