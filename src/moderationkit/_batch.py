from collections import Counter
from collections.abc import Sequence

from moderationkit.exceptions import InputError
from moderationkit.types import Action, BatchItem, BatchResult


def validate_batch(items: object) -> None:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise InputError("batch must be a sequence of TextInput records")


def summarize(items: tuple[BatchItem, ...], latency_ms: float) -> BatchResult:
    actions = Counter(item.result.decision.action for item in items)
    totals: Counter[str] = Counter()
    evaluated: Counter[str] = Counter()
    for item in items:
        for category in item.result.categories:
            if category.detected is not None:
                evaluated[category.name] += 1
                totals[category.name] += int(category.detected)
    return BatchResult(
        items=items,
        submitted=len(items),
        processed=len(items),
        allowed=actions[Action.ALLOW],
        warned=actions[Action.WARN],
        reviewed=actions[Action.REVIEW],
        blocked=actions[Action.BLOCK],
        failed=actions[Action.ERROR],
        category_totals=dict(totals),
        category_evaluated=dict(evaluated),
        latency_ms=latency_ms,
    )
