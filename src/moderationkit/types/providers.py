from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

Probability = StrictFloat | StrictInt


class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    modalities: frozenset[str] = frozenset({"text"})
    custom_categories: bool = True
    supported_categories: frozenset[str] | None = None
    evidence: bool = False
    native_batch: bool = False
    max_input_tokens: int | None = None


class CategorySignal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    name: str
    probability: Probability | None = Field(default=None, ge=0, le=1)
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def availability(self) -> CategorySignal:
        if self.probability is None and not self.unavailable_reason:
            raise ValueError("unavailable probability requires a reason")
        if self.probability is not None and self.unavailable_reason:
            raise ValueError("available probability cannot have an unavailable reason")
        return self


class ProviderMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    provider: str
    model: str | None = None
    requested_model: str | None = None
    revision: str | None = None
    runtime_version: str | None = None
    adapter_version: str = "0.1"
    taxonomy_version: str = "1.0"
    template_version: str = "1.0"
    questions_hash: str | None = None
    latency_ms: float = Field(default=0, ge=0)
    queue_ms: float = Field(default=0, ge=0)
    inference_ms: float = Field(default=0, ge=0)
    load_ms: float = Field(default=0, ge=0)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    attempts: int = Field(default=1, ge=0)
    fallback_used: bool = False
    evidence_available: bool = False
    unsupported_capabilities: tuple[str, ...] = ()
    calibration_status: Literal["unavailable", "validated"] = "unavailable"


class ClassificationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    categories: tuple[CategorySignal, ...]
    metadata: ProviderMetadata
    coverage_complete: bool = False

    @model_validator(mode="after")
    def unique(self) -> ClassificationResult:
        names = [item.name for item in self.categories]
        if len(set(names)) != len(names):
            raise ValueError("duplicate category signals")
        return self
