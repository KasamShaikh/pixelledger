"""Schema-level metrics: field-wise precision/recall/F1 against ground-truth JSON."""

from __future__ import annotations

from typing import Any


def _flatten(obj: Any, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            out.update(_flatten(v, key))
    elif obj is None:
        return out
    else:
        out[prefix] = str(obj).strip().lower()
    return out


def field_scores(
    predicted: dict | None, ground_truth: dict | None
) -> dict[str, float | int]:
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
    pred = _flatten(predicted or {})
    gt = _flatten(ground_truth or {})
    tp = sum(1 for k, v in gt.items() if k in pred and pred[k] == v)
    fn = len(gt) - tp
    fp = sum(1 for k in pred if k not in gt or pred[k] != gt.get(k))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def section_scores(
    predicted: dict | None, ground_truth: dict | None
) -> dict[str, dict[str, float | int]]:
    """Per top-level section: precision/recall/F1 computed against fields under that key."""
    if not ground_truth:
        return {}
    out: dict[str, dict[str, float | int]] = {}
    for section in ground_truth.keys():
        gt_sub = {section: ground_truth.get(section)}
        pred_sub = {section: (predicted or {}).get(section)}
        out[section] = field_scores(pred_sub, gt_sub)
    return out
