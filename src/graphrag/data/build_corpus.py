"""Stage: sample questions + pool the deduplicated passage corpus.

Combines sampling and pooling in one pass because resolving each sampled question's gold
`supporting_passage_ids` requires the pooled passage ids at the same time the question
records are written — doing it in two passes would mean re-reading the raw file twice for
no benefit.

Idempotent: if both output files exist and `force` is False, does nothing.
All aggregate stats are returned as a plain dict for the caller to print — this module
never prints raw records itself.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path

from graphrag.schemas import Passage, SampledQuestion

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def _passage_id(title: str, text: str) -> str:
    """Stable content hash — same (title, text) pair always yields the same id, so the
    same Wikipedia passage pulled in via different questions dedupes automatically."""
    digest = hashlib.sha1(f"{title}\x1f{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def _iter_raw(raw_path: Path):
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _sample_ids(raw_path: Path, n: int, seed: int) -> set[str]:
    """Shuffle once, then take a prefix — unlike `random.Random(seed).sample(pop, k)`,
    whose internal algorithm varies with k, this guarantees sample(n=50) is always a
    strict subset of sample(n=1000) at the same seed, so scaling up later never
    invalidates extraction/graph work already done for the smaller sample."""
    all_ids = sorted(row["id"] for row in _iter_raw(raw_path))
    if n >= len(all_ids):
        return set(all_ids)
    rng = random.Random(seed)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)
    return set(shuffled[:n])


def build(
    raw_path: Path,
    corpus_path: Path,
    sample_questions_path: Path,
    n_questions: int = 1000,
    seed: int = RANDOM_SEED,
    force: bool = False,
) -> dict:
    if corpus_path.exists() and sample_questions_path.exists() and not force:
        logger.info("Corpus + sample questions already built, skipping.")
        return {"skipped": True}

    wanted_ids = _sample_ids(raw_path, n_questions, seed)

    passages: dict[str, Passage] = {}
    questions: list[SampledQuestion] = []
    type_counts: Counter = Counter()
    passage_refs_total = 0

    for row in _iter_raw(raw_path):
        if row["id"] not in wanted_ids:
            continue

        titles = row["context"]["title"]
        sentences_lists = row["context"]["sentences"]

        # title -> passage id, scoped to this question's own context (titles are unique
        # within one question's context list).
        title_to_pid: dict[str, str] = {}
        for title, sentences in zip(titles, sentences_lists):
            text = " ".join(s.strip() for s in sentences).strip()
            pid = _passage_id(title, text)
            title_to_pid[title] = pid
            passage_refs_total += 1

            if pid in passages:
                if row["id"] not in passages[pid].source_question_ids:
                    passages[pid].source_question_ids.append(row["id"])
            else:
                passages[pid] = Passage(
                    id=pid, title=title, text=text, source_question_ids=[row["id"]]
                )

        sf_titles = row["supporting_facts"]["title"]
        sf_sent_ids = row["supporting_facts"]["sent_id"]
        supporting_facts = list(zip(sf_titles, sf_sent_ids))
        supporting_passage_ids = sorted(
            {title_to_pid[t] for t in sf_titles if t in title_to_pid}
        )

        q_type = row.get("type")
        type_counts[q_type] += 1

        questions.append(
            SampledQuestion(
                question_id=row["id"],
                question=row["question"],
                answer=row["answer"],
                answer_aliases=[],
                supporting_facts=supporting_facts,
                supporting_passage_ids=supporting_passage_ids,
                type=q_type,
            )
        )

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as f:
        for p in passages.values():
            f.write(p.model_dump_json() + "\n")

    sample_questions_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_questions_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(q.model_dump_json() + "\n")

    stats = {
        "skipped": False,
        "n_questions_sampled": len(questions),
        "n_passage_refs_total": passage_refs_total,
        "n_unique_passages": len(passages),
        "avg_passages_per_question": round(passage_refs_total / max(len(questions), 1), 2),
        "question_type_counts": dict(type_counts),
    }
    return stats
