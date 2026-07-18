"""2WikiMultiHopQA-standard metrics: answer Exact Match, answer F1 (token-level), and
supporting-fact F1.

Answer normalization follows the standard SQuAD/HotpotQA/2Wiki convention: lowercase,
strip punctuation, drop articles, collapse whitespace.

Supporting-fact F1 here is scored at **passage granularity** rather than the official
sentence-level (title, sent_id) granularity — a deliberate, documented deviation (see
README limitations), since this pipeline retrieves and cites whole passages, not
individual sentences.
"""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def supporting_fact_f1(predicted_ids: list[str], gold_ids: list[str]) -> tuple[float, float, float]:
    """Returns (precision, recall, f1) over passage-id sets."""
    pred_set, gold_set = set(predicted_ids), set(gold_ids)
    if not gold_set:
        return (1.0, 1.0, 1.0) if not pred_set else (0.0, 1.0, 0.0)
    if not pred_set:
        return (0.0, 0.0, 0.0)

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return (precision, recall, f1)


def aggregate(predictions: list[dict]) -> dict:
    """predictions: [{"predicted_answer", "gold_answer", "supporting_passage_ids",
    "gold_supporting_passage_ids"}, ...]. Returns mean EM/F1/support-F1 across the set."""
    if not predictions:
        return {"n": 0, "em": 0.0, "f1": 0.0, "support_f1": 0.0}

    ems, f1s, support_f1s = [], [], []
    for p in predictions:
        ems.append(exact_match(p["predicted_answer"], p["gold_answer"]))
        f1s.append(f1_score(p["predicted_answer"], p["gold_answer"]))
        _, _, sf1 = supporting_fact_f1(
            p["supporting_passage_ids"], p["gold_supporting_passage_ids"]
        )
        support_f1s.append(sf1)

    n = len(predictions)
    return {
        "n": n,
        "em": sum(ems) / n,
        "f1": sum(f1s) / n,
        "support_f1": sum(support_f1s) / n,
    }
