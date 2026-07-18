import json

from graphrag.data.build_corpus import build


def _write_raw(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(qid, extra_title="Extra"):
    return {
        "id": qid,
        "question": f"Question {qid}?",
        "answer": "Answer",
        "type": "bridge",
        "evidences": [],
        "supporting_facts": {"title": ["Shared Title"], "sent_id": [0]},
        "context": {
            "title": ["Shared Title", extra_title],
            "sentences": [["Shared sentence one.", "Shared sentence two."], ["Extra sentence."]],
        },
    }


def test_build_corpus_dedupes_shared_passage_across_questions(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    _write_raw(raw_path, [_row("q1", "Extra A"), _row("q2", "Extra B")])

    corpus_path = tmp_path / "corpus.jsonl"
    questions_path = tmp_path / "sample_questions.jsonl"

    stats = build(raw_path, corpus_path, questions_path, n_questions=2, seed=0)

    assert stats["n_questions_sampled"] == 2
    # "Shared Title" passage is identical across both questions -> deduped to 1;
    # "Extra A" / "Extra B" differ -> 2 more unique passages. Total unique = 3.
    assert stats["n_unique_passages"] == 3

    passages = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    shared = next(p for p in passages if p["title"] == "Shared Title")
    assert set(shared["source_question_ids"]) == {"q1", "q2"}


def test_build_corpus_resolves_supporting_passage_ids(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    _write_raw(raw_path, [_row("q1")])

    corpus_path = tmp_path / "corpus.jsonl"
    questions_path = tmp_path / "sample_questions.jsonl"

    build(raw_path, corpus_path, questions_path, n_questions=1, seed=0)

    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines()]
    assert len(questions) == 1
    q = questions[0]
    assert len(q["supporting_passage_ids"]) == 1  # only "Shared Title" is a supporting fact


def test_build_corpus_is_idempotent_by_default(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    _write_raw(raw_path, [_row("q1")])
    corpus_path = tmp_path / "corpus.jsonl"
    questions_path = tmp_path / "sample_questions.jsonl"

    build(raw_path, corpus_path, questions_path, n_questions=1, seed=0)
    stats_second_run = build(raw_path, corpus_path, questions_path, n_questions=1, seed=0)

    assert stats_second_run == {"skipped": True}
