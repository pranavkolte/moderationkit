"""Policy-aware moderation with interchangeable providers. Model imports stay lazy."""

from importlib.metadata import version

from ._client import AsyncModGuard, ModGuard
from .categories import CORE_CATEGORIES, CategoryDefinition
from .policies import Policy, RiskScorer, Rule
from .types import Action, BatchResult, ModerationResult, Severity, TextInput

__version__ = version("moderationkit")
__all__ = [
    "Action",
    "AsyncModGuard",
    "BatchResult",
    "CORE_CATEGORIES",
    "CategoryDefinition",
    "ModGuard",
    "ModerationResult",
    "Policy",
    "RiskScorer",
    "Rule",
    "Severity",
    "TextInput",
]
