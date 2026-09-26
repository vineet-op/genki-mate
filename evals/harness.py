"""Shared utilities used by every eval script.

Layer 1 — Python IR metrics (no LLM, free, run as often as you want):
    hit_at_k, recall_at_k, mrr

Layer 2 — DeepEval result helpers (wraps the evaluate() return value):
    summarize_by_metric, print_summary
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


# ── golden set loader ─────────────────────────────────────────────────────────

def load_goldens(path: str) -> list[dict]:
    """Load the JSON golden set from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── Python IR metrics (Layer 1) ───────────────────────────────────────────────

def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> int:
    """1 if any relevant chunk appears in the top-k results, else 0."""
    return int(any(r in relevant_ids for r in retrieved_ids[:k]))


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Fraction of relevant chunks that appear in the top-k results."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids[:k] if r in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant chunk (1/rank, or 0 if not found)."""
    for rank, r in enumerate(retrieved_ids, start=1):
        if r in relevant_ids:
            return 1.0 / rank
    return 0.0


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_ir_metrics(goldens: list[dict], retriever, k: int = 5) -> dict:
    """Run Hit@k, Recall@k, MRR over the full golden set. Returns averages."""
    hits, recalls, rrs = [], [], []

    for g in goldens:
        retrieved_docs = retriever.invoke(g["query"])
        retrieved_ids  = [doc.metadata["chunk_id"] for doc in retrieved_docs]
        relevant_ids   = g["relevant_chunk_ids"]

        hits.append(hit_at_k(retrieved_ids, relevant_ids, k))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
        rrs.append(mrr(retrieved_ids, relevant_ids))

    return {
        f"Hit@{k}":    _avg(hits),
        f"Recall@{k}": _avg(recalls),
        "MRR":         _avg(rrs),
    }


# ── DeepEval result helpers (Layer 2) ────────────────────────────────────────

def summarize_by_metric(result) -> dict:
    """Extract per-metric average scores from a DeepEval EvaluationResult."""
    scores: dict[str, list[float]] = defaultdict(list)
    for test_result in result.test_results:
        for metric_data in test_result.metrics_data:
            if metric_data.score is not None:
                scores[metric_data.name].append(metric_data.score)
    return {name: _avg(vals) for name, vals in scores.items()}


def print_summary(label: str, summary: dict, threshold: float = 0.7) -> None:
    width = 50
    print(f"\n{'=' * width}")
    print(f"  {label.upper()} EVAL SUMMARY")
    print(f"{'=' * width}")
    for metric, score in summary.items():
        status = "✓" if score >= threshold else "✗"
        print(f"  {status}  {metric:<35} {score:.3f}")
    print(f"{'=' * width}\n")
