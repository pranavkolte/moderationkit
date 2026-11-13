from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from moderationkit import ModGuard, TextInput
from moderationkit.exceptions import ConfigurationError
from moderationkit.types import ModerationResult

from .dataset import DatasetRecord


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def category_metrics(
    labels: Sequence[bool | None],
    predictions: Sequence[bool | None],
) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    tp = tn = fp = fn = missing = unavailable = 0
    for label, predicted in zip(labels, predictions, strict=True):
        if label is None:
            missing += 1
        elif predicted is None:
            unavailable += 1
        elif predicted:
            if label:
                tp += 1
            else:
                fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    evaluated = tp + tn + fp + fn
    labelled = len(labels) - missing
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "evaluated": evaluated,
        "labelled": labelled,
        "missing_labels": missing,
        "unavailable": unavailable,
        "coverage": _ratio(evaluated, labelled),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_negative_rate": _ratio(fn, fn + tp),
        "denominators": {"precision": tp + fp, "recall": tp + fn, "fpr": fp + tn, "fnr": fn + tp},
    }


def _percentiles(values: list[float]) -> dict[str, float | None]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        index = (len(ordered) - 1) * fraction
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {"p50": percentile(0.5), "p95": percentile(0.95), "p99": percentile(0.99)}


def evaluate_dataset(
    guard: ModGuard,
    records: Sequence[DatasetRecord],
    *,
    split: str = "test",
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ConfigurationError("split must be train, validation, or test")
    selected = [record for record in records if record.split == split]
    if len({record.id for record in records}) != len(records):
        raise ConfigurationError("dataset IDs must be unique across splits")
    if not selected:
        raise ConfigurationError(f"Dataset has no {split} records")
    results: list[ModerationResult] = []
    for record in selected:
        # Batch mode collects errors even if the client was configured to raise.
        from moderationkit.exceptions import BatchModerationError

        try:
            batch = guard.moderate_batch(
                (TextInput(text=record.text, item_id=record.id, language=record.language),)
            )
        except BatchModerationError as exc:
            batch = exc.result
        results.append(batch.items[0].result)
    categories = sorted(
        set(guard.policy.requested_categories)
        | {name for record in selected for name in record.labels}
    )

    def metrics(indices: list[int]) -> dict[str, Any]:
        return {
            name: category_metrics(
                [selected[i].labels.get(name) for i in indices],
                [
                    next(
                        (
                            c.detected
                            for c in results[i].categories
                            if c.name == name and results[i].status == "completed"
                        ),
                        None,
                    )
                    for i in indices
                ],
            )
            for name in categories
        }

    slices: dict[str, Any] = {}
    for field in ("language", "content_type"):
        for value in sorted({getattr(record, field) or "unknown" for record in selected}):
            indices = [
                i
                for i, record in enumerate(selected)
                if (getattr(record, field) or "unknown") == value
            ]
            slices[f"{field}:{value}"] = metrics(indices)
    canonical = json.dumps(
        [record.model_dump(mode="json") for record in selected],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    profiles = {
        json.dumps(
            {
                key: value
                for key, value in result.metadata.model_dump().items()
                if key
                in {
                    "provider",
                    "model",
                    "revision",
                    "runtime_version",
                    "adapter_version",
                    "taxonomy_version",
                    "template_version",
                    "questions_hash",
                }
            },
            sort_keys=True,
        )
        for result in results
    }
    costs = [result.metadata.cost for result in results]
    return {
        "schema_version": "1.0",
        "split": split,
        "count": len(selected),
        "dataset_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "policy": guard.policy.model_dump(mode="json"),
        "profiles": [json.loads(profile) for profile in sorted(profiles)],
        "categories": metrics(list(range(len(selected)))),
        "slices": slices,
        "actions": dict(Counter(result.decision.action.value for result in results)),
        "failures": sum(result.status != "completed" for result in results),
        "latency_ms": _percentiles([r.metadata.latency_ms for r in results]),
        "warm_latency_ms": _percentiles(
            [r.metadata.latency_ms for r in results if r.metadata.load_ms == 0]
        ),
        "cold_requests": sum(r.metadata.load_ms > 0 for r in results),
        "cost": sum(cost for cost in costs if cost is not None)
        if all(cost is not None for cost in costs)
        else None,
        "predictions": [
            {
                "id": record.id,
                "status": result.status,
                "labels": record.labels,
                "categories": [c.model_dump(mode="json") for c in result.categories],
            }
            for record, result in zip(selected, results, strict=True)
        ],
    }


def threshold_sweep(
    guard: ModGuard,
    records: Sequence[DatasetRecord],
    *,
    category: str,
    thresholds: Sequence[float],
    min_recall: float | None = None,
) -> dict[str, Any]:
    if category not in guard.policy.requested_categories:
        raise ConfigurationError("sweep category must be requested by the policy")
    if min_recall is not None and not 0 <= min_recall <= 1:
        raise ConfigurationError("min_recall must lie between zero and one")
    if not thresholds or any(isinstance(t, bool) or not 0 <= t <= 1 for t in thresholds):
        raise ConfigurationError("thresholds must contain finite values between zero and one")
    report = evaluate_dataset(guard, records, split="validation")
    predictions = report["predictions"]
    rows = []
    for threshold in sorted(set(thresholds)):
        labels = [p["labels"].get(category) for p in predictions]
        scores = [
            next(
                (
                    c["confidence"]
                    for c in p["categories"]
                    if c["name"] == category and p["status"] == "completed"
                ),
                None,
            )
            for p in predictions
        ]
        metrics = category_metrics(labels, [None if s is None else s >= threshold for s in scores])
        rows.append({"threshold": threshold, **metrics})
    eligible = [
        row
        for row in rows
        if min_recall is not None
        and row["recall"] is not None
        and row["recall"] >= min_recall
        and row["precision"] is not None
    ]
    best = max(eligible, key=lambda row: (row["precision"], row["threshold"]), default=None)
    return {
        "category": category,
        "split": "validation",
        "rows": rows,
        "recommended_threshold": None if best is None else best["threshold"],
        "target_min_recall": min_recall,
        "dataset_sha256": report["dataset_sha256"],
    }
