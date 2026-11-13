from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from moderationkit.categories import CategoryDefinition
from moderationkit.exceptions import ProviderResponseError
from moderationkit.types import CategorySignal, ClassificationResult, ProviderMetadata


def normalize(
    raw: Any,
    categories: Sequence[CategoryDefinition],
    metadata: ProviderMetadata,
) -> ClassificationResult:
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
        raise ProviderResponseError("Laya response must contain an answers mapping")
    signals = []
    for category in categories:
        answer = raw["answers"].get(category.id)
        value = answer.get("noul") if isinstance(answer, dict) else None
        valid = (
            isinstance(answer, dict)
            and answer.get("type") == "noul"
            and isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        )
        signals.append(
            CategorySignal(
                name=category.id,
                probability=value if valid else None,
                unavailable_reason=None if valid else "missing_or_invalid_answer",
            )
        )
    usage = raw.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        if type(value) is int and value >= 0:
            setattr(metadata, key, value)
    model = raw.get("model")
    routing = raw.get("routing")
    if isinstance(routing, dict) and isinstance(routing.get("repo"), str):
        metadata.model = routing["repo"]
    elif isinstance(model, str):
        metadata.model = model
    # Missing coverage information is deliberately not interpreted as full coverage.
    coverage = usage.get("truncated") is False
    return ClassificationResult(
        categories=tuple(signals), metadata=metadata, coverage_complete=coverage
    )
