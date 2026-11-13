from __future__ import annotations

import time
from collections.abc import Sequence
from uuid import uuid4

from moderationkit.categories import CORE_CATEGORIES, CategoryDefinition
from moderationkit.exceptions import ConfigurationError, InputError, ProviderError
from moderationkit.policies import Policy
from moderationkit.types import (
    Action,
    Decision,
    ErrorDetails,
    InferenceOptions,
    ModerationResult,
    PolicyReference,
    ProviderMetadata,
    TextInput,
)


def resolve_categories(
    policy: Policy, custom: Sequence[CategoryDefinition]
) -> tuple[CategoryDefinition, ...]:
    registry = {c.id: c for c in CORE_CATEGORIES}
    for category in custom:
        if category.id in registry or not category.id.startswith("custom."):
            raise ConfigurationError("Custom categories must have unique IDs prefixed with custom.")
        registry[category.id] = category
    unknown = set(policy.requested_categories) - registry.keys()
    if unknown:
        raise ConfigurationError(f"Undefined categories: {sorted(unknown)}")
    return tuple(registry[name] for name in policy.requested_categories)


def request_options(timeout: float | None) -> InferenceOptions:
    return InferenceOptions(
        request_id=f"mod_{uuid4().hex}",
        deadline=None if timeout is None else time.monotonic() + timeout,
    )


def text_input(text: str, item_id: str | None, language: str | None) -> TextInput:
    try:
        return TextInput(text=text, item_id=item_id, language=language)
    except ValueError as exc:
        raise InputError(str(exc)) from exc


def failure(
    policy: Policy,
    error: ProviderError | InputError,
    *,
    options: InferenceOptions,
    item_id: str | None = None,
    provider_name: str = "unknown",
) -> ModerationResult:
    return ModerationResult(
        request_id=options.request_id,
        item_id=item_id,
        status="error",
        decision=Decision(action=Action.ERROR, safe=False, can_publish=False, requires_review=True),
        policy=PolicyReference(
            id=policy.id,
            version=policy.version,
            rule_thresholds={r.id: r.threshold for r in policy.rules},
        ),
        metadata=ProviderMetadata(provider=provider_name),
        errors=(ErrorDetails(code=getattr(error, "code", "invalid_input"), message=str(error)),),
    )
