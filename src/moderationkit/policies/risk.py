from typing import Protocol

from moderationkit.types import ClassificationResult

from .models import Policy


class RiskScorer(Protocol):
    """An injected, validated scorer must explicitly accept the full provenance profile."""

    def compatible(self, classification: ClassificationResult, policy: Policy) -> bool: ...

    def score(self, classification: ClassificationResult, policy: Policy) -> float: ...
