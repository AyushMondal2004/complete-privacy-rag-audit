from __future__ import annotations
from dataclasses import dataclass
from .label_matcher import MatchResult


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def compute_metrics(tp: int, fp: int, fn: int) -> Metrics:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def aggregate(results: list[MatchResult]) -> Metrics:
    """Micro-averaged across all policies in the evaluation set — sums raw
    TP/FP/FN rather than averaging per-policy scores, which is the more
    standard choice when class frequency varies a lot between policies.
    State this choice explicitly in your Methodology chapter."""
    tp = sum(len(r.true_positives) for r in results)
    fp = sum(len(r.false_positives) for r in results)
    fn = sum(len(r.false_negatives) for r in results)
    return compute_metrics(tp, fp, fn)


def per_policy_metrics(results: list[MatchResult]) -> list[dict]:
    rows = []
    for r in results:
        m = compute_metrics(len(r.true_positives), len(r.false_positives), len(r.false_negatives))
        rows.append(
            {
                "policy_id": r.policy_id,
                "precision": round(m.precision, 3),
                "recall": round(m.recall, 3),
                "f1": round(m.f1, 3),
                "tp": m.tp,
                "fp": m.fp,
                "fn": m.fn,
            }
        )
    return rows
