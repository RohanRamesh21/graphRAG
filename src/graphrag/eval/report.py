"""Stage 7 reporting: aggregate metrics for both modes + a few qualitative multi-hop
traces, written to a single markdown report."""
from __future__ import annotations

import json
from pathlib import Path

from graphrag.eval.metrics import aggregate, exact_match


def _load_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pick_qualitative_examples(graphrag_preds: list[dict], baseline_preds: list[dict], n: int = 5) -> list[dict]:
    """Prefer questions GraphRAG got right, the baseline got wrong, and the reasoning
    path actually shows a multi-hop graph traversal — the clearest evidence the graph
    contributed something the vector-only baseline could not."""
    baseline_by_id = {p["question_id"]: p for p in baseline_preds}

    candidates = []
    for p in graphrag_preds:
        has_graph_hop = any("graph_path=" in step for step in p.get("reasoning_path", []))
        if not has_graph_hop:
            continue
        graphrag_correct = exact_match(p["predicted_answer"], p["gold_answer"]) == 1.0
        baseline_pred = baseline_by_id.get(p["question_id"])
        baseline_correct = (
            exact_match(baseline_pred["predicted_answer"], baseline_pred["gold_answer"]) == 1.0
            if baseline_pred
            else None
        )
        candidates.append(
            {
                **p,
                "graphrag_correct": graphrag_correct,
                "baseline_correct": baseline_correct,
            }
        )

    # Best evidence first: GraphRAG right + baseline wrong.
    candidates.sort(
        key=lambda c: (c["graphrag_correct"] and c["baseline_correct"] is False, c["graphrag_correct"]),
        reverse=True,
    )
    return candidates[:n]


def build_report(results_dir: Path) -> str:
    graphrag_preds = _load_predictions(results_dir / "graphrag_predictions.jsonl")
    baseline_preds = _load_predictions(results_dir / "baseline_predictions.jsonl")

    graphrag_metrics = aggregate(
        [
            {
                "predicted_answer": p["predicted_answer"],
                "gold_answer": p["gold_answer"],
                "supporting_passage_ids": p["supporting_passage_ids"],
                "gold_supporting_passage_ids": p["gold_supporting_passage_ids"],
            }
            for p in graphrag_preds
        ]
    )
    baseline_metrics = aggregate(
        [
            {
                "predicted_answer": p["predicted_answer"],
                "gold_answer": p["gold_answer"],
                "supporting_passage_ids": p["supporting_passage_ids"],
                "gold_supporting_passage_ids": p["gold_supporting_passage_ids"],
            }
            for p in baseline_preds
        ]
    )

    lines = ["# GraphRAG Evaluation Report", ""]
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Mode | N | EM | Answer F1 | Supporting-Fact F1 (passage-level) |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| GraphRAG (hybrid) | {graphrag_metrics['n']} | {graphrag_metrics['em']:.3f} | "
        f"{graphrag_metrics['f1']:.3f} | {graphrag_metrics['support_f1']:.3f} |"
    )
    lines.append(
        f"| Baseline (vector-only) | {baseline_metrics['n']} | {baseline_metrics['em']:.3f} | "
        f"{baseline_metrics['f1']:.3f} | {baseline_metrics['support_f1']:.3f} |"
    )
    lines.append("")
    if graphrag_metrics["n"] and baseline_metrics["n"]:
        lines.append(
            f"**Delta (GraphRAG − baseline):** EM {graphrag_metrics['em'] - baseline_metrics['em']:+.3f}, "
            f"F1 {graphrag_metrics['f1'] - baseline_metrics['f1']:+.3f}, "
            f"Support-F1 {graphrag_metrics['support_f1'] - baseline_metrics['support_f1']:+.3f}"
        )
    lines.append("")

    lines.append("## Qualitative traces (multi-hop retrieval in action)")
    lines.append("")
    examples = _pick_qualitative_examples(graphrag_preds, baseline_preds)
    if not examples:
        lines.append("_No qualifying multi-hop traces found yet (run more of the eval set)._")
    for i, ex in enumerate(examples, start=1):
        lines.append(f"### Example {i}: {ex['question']}")
        lines.append(f"- Gold answer: `{ex['gold_answer']}`")
        lines.append(f"- GraphRAG answer: `{ex['predicted_answer']}` (correct: {ex['graphrag_correct']})")
        if ex["baseline_correct"] is not None:
            lines.append(f"- Baseline (vector-only) correct: {ex['baseline_correct']}")
        lines.append("- Reasoning path:")
        for step in ex["reasoning_path"]:
            lines.append(f"  - {step}")
        lines.append("")

    return "\n".join(lines)


def write_report(results_dir: Path) -> Path:
    report = build_report(results_dir)
    out_path = results_dir / "report.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path
