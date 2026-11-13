from __future__ import annotations

import math

from moderationkit.types import (
    Action,
    CategoryResult,
    ClassificationResult,
    Decision,
    ErrorDetails,
    ModerationResult,
    PolicyReference,
    Severity,
    Violation,
)

from .models import Policy
from .risk import RiskScorer

PRIORITY = {Action.ALLOW: 0, Action.WARN: 1, Action.REVIEW: 2, Action.BLOCK: 3}
SEVERITY_PRIORITY = {value: index for index, value in enumerate(Severity)}


def evaluate(
    classification: ClassificationResult,
    policy: Policy,
    *,
    request_id: str,
    item_id: str | None = None,
    risk_scorer: RiskScorer | None = None,
) -> ModerationResult:
    signals = {signal.name: signal for signal in classification.categories}
    matched = []
    categories = []
    violations: list[Violation] = []
    errors = []
    for name in policy.requested_categories:
        signal = signals.get(name)
        probability = None if signal is None else signal.probability
        rules = [
            r
            for r in policy.rules
            if r.category == name and probability is not None and probability >= r.threshold
        ]
        matched.extend(rules)
        enforcing = [r for r in rules if r.action != Action.ALLOW]
        action = max((r.action for r in rules), key=PRIORITY.__getitem__, default=Action.ALLOW)
        severity = max(
            (r.severity for r in enforcing if r.severity is not None),
            key=SEVERITY_PRIORITY.__getitem__,
            default=None,
        )
        reason = None if signal is not None else "missing_category"
        if signal is not None:
            reason = signal.unavailable_reason
        if probability is None:
            errors.append(
                ErrorDetails(
                    code="category_unavailable", message=reason or "unavailable", category=name
                )
            )
        categories.append(
            CategoryResult(
                name=name,
                confidence=probability,
                detected=None if probability is None else probability >= policy.threshold_for(name),
                detection_threshold=policy.threshold_for(name),
                action=action if probability is not None else None,
                severity=severity,
                severity_source="policy" if severity is not None else None,
                unavailable_reason=reason,
            )
        )
        violations.extend(
            Violation(
                category=name,
                rule_id=r.id,
                reason=f"Policy rule {r.id} matched {name} at threshold {r.threshold}",
            )
            for r in enforcing
        )
    unavailable_required = any(
        name not in signals or signals[name].probability is None
        for name in policy.required_categories
    )
    no_evaluation = not any(
        name in signals and signals[name].probability is not None
        for name in policy.requested_categories
    )
    incomplete = unavailable_required or not classification.coverage_complete or no_evaluation
    if not classification.coverage_complete:
        errors.append(
            ErrorDetails(
                code="incomplete_input_coverage",
                message="The provider did not evaluate the entire input",
            )
        )
    action = max((r.action for r in matched), key=PRIORITY.__getitem__, default=Action.ALLOW)
    if incomplete:
        action = Action.ERROR
    strongest = max(
        (r for r in matched if r.action != Action.ALLOW),
        key=lambda r: (
            PRIORITY[r.action],
            (-1 if r.severity is None else SEVERITY_PRIORITY[r.severity]),
        ),
        default=None,
    )
    severity = max(
        (c.severity for c in categories if c.severity is not None),
        key=SEVERITY_PRIORITY.__getitem__,
        default=None,
    )
    metadata = classification.metadata.model_copy(deep=True)
    metadata.calibration_status = "unavailable"
    risk = None
    if (
        not incomplete
        and risk_scorer is not None
        and risk_scorer.compatible(classification, policy)
    ):
        risk = risk_scorer.score(classification, policy)
        if isinstance(risk, bool) or not math.isfinite(risk) or not 0 <= risk <= 100:
            raise ValueError("risk scorer returned an invalid score")
        metadata.calibration_status = "validated"
    publish = action in (Action.ALLOW, Action.WARN)
    return ModerationResult(
        request_id=request_id,
        item_id=item_id,
        status="incomplete" if incomplete else "completed",
        decision=Decision(
            action=action,
            safe=publish,
            can_publish=publish,
            requires_review=action in (Action.REVIEW, Action.ERROR),
            severity=severity,
            risk_score=risk,
            highest_priority_violation=None if strongest is None else strongest.category,
        ),
        categories=tuple(categories),
        violations=tuple(violations),
        errors=tuple(errors),
        policy=PolicyReference(
            id=policy.id,
            version=policy.version,
            matched_rules=tuple(r.id for r in matched),
            rule_thresholds={r.id: r.threshold for r in policy.rules},
        ),
        metadata=metadata,
    )
