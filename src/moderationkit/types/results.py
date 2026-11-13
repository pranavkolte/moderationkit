from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .providers import ProviderMetadata


class Action(StrEnum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"
    ERROR = "ERROR"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Decision(FrozenModel):
    action: Action
    safe: bool
    can_publish: bool
    requires_review: bool
    risk_score: float | None = Field(default=None, ge=0, le=100)
    severity: Severity | None = None
    highest_priority_violation: str | None = None

    @model_validator(mode="after")
    def publication(self) -> Decision:
        publish = self.action in (Action.ALLOW, Action.WARN)
        review = self.action in (Action.REVIEW, Action.ERROR)
        if (self.safe, self.can_publish, self.requires_review) != (publish, publish, review):
            raise ValueError("publication flags contradict action")
        return self


class CategoryResult(FrozenModel):
    name: str
    detected: bool | None
    confidence: float | None = Field(default=None, ge=0, le=1)
    detection_threshold: float = Field(ge=0, le=1)
    action: Action | None = None
    severity: Severity | None = None
    severity_source: Literal["policy"] | None = None
    unavailable_reason: str | None = None


class TextSpan(FrozenModel):
    type: Literal["text_span"] = "text_span"
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> TextSpan:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        return self


class Violation(FrozenModel):
    category: str
    reason: str
    rule_id: str
    evidence: TextSpan | None = None


class ErrorDetails(FrozenModel):
    code: str
    message: str
    category: str | None = None


class PolicyReference(FrozenModel):
    id: str
    version: str
    matched_rules: tuple[str, ...] = ()
    rule_thresholds: dict[str, float] = Field(default_factory=dict)


class ModerationResult(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    item_id: str | None = None
    status: Literal["completed", "incomplete", "error"]
    input_type: Literal["text"] = "text"
    decision: Decision
    categories: tuple[CategoryResult, ...] = ()
    violations: tuple[Violation, ...] = ()
    policy: PolicyReference
    metadata: ProviderMetadata
    errors: tuple[ErrorDetails, ...] = ()

    @model_validator(mode="after")
    def status_matches(self) -> ModerationResult:
        if (self.status == "completed") == (self.decision.action == Action.ERROR):
            raise ValueError("status contradicts decision action")
        return self


class BatchItem(FrozenModel):
    index: int = Field(ge=0)
    result: ModerationResult


class BatchResult(FrozenModel):
    items: tuple[BatchItem, ...]
    submitted: int
    processed: int
    allowed: int
    warned: int
    reviewed: int
    blocked: int
    failed: int
    category_totals: dict[str, int]
    category_evaluated: dict[str, int]
    latency_ms: float

    @model_validator(mode="after")
    def counts(self) -> BatchResult:
        if self.processed != self.submitted or self.processed != len(self.items):
            raise ValueError("batch counts do not match items")
        if (
            self.processed
            != self.allowed + self.warned + self.reviewed + self.blocked + self.failed
        ):
            raise ValueError("batch outcome counts do not sum to processed")
        return self
