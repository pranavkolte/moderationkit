"""Illustrative thresholds only. Evaluate and version your own deployment policy."""

from moderationkit.types import Action, Severity

from .models import Policy, Rule


def community() -> Policy:
    return Policy(
        id="community_example",
        version="1.0",
        rules=(
            Rule(
                id="hate_review",
                category="hate_speech",
                threshold=0.8,
                action=Action.REVIEW,
                severity=Severity.HIGH,
            ),
            Rule(
                id="harassment_review",
                category="harassment",
                threshold=0.8,
                action=Action.REVIEW,
                severity=Severity.MEDIUM,
            ),
            Rule(
                id="explicit_block",
                category="sexual_content",
                threshold=0.9,
                action=Action.BLOCK,
                severity=Severity.HIGH,
            ),
        ),
    )
