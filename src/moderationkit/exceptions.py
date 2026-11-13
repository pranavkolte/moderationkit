"""Public SDK exceptions. Operational failures carry stable machine-readable codes."""

from __future__ import annotations

from typing import Any


class ModerationKitError(Exception):
    """Base exception for SDK errors."""


class ConfigurationError(ModerationKitError, ValueError):
    pass


class InputError(ModerationKitError, ValueError):
    pass


class MissingDependencyError(ModerationKitError, ImportError):
    pass


class ProviderError(ModerationKitError):
    code = "provider_error"


class ProviderResponseError(ProviderError):
    code = "invalid_provider_response"


class ModerationTimeout(ProviderError):
    code = "timeout"


class CapacityError(ProviderError):
    code = "capacity_exceeded"


class ClosedError(ProviderError):
    code = "closed"


class ModerationError(ModerationKitError):
    def __init__(self, result: Any) -> None:
        self.result = result
        super().__init__("Moderation could not complete; see result.errors")


class BatchModerationError(ModerationError):
    """Raised after collecting every batch item; result is the full batch report."""
