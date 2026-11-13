from __future__ import annotations

import math
import time

from pydantic import BaseModel, ConfigDict, field_validator


class TextInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    text: str
    item_id: str | None = None
    language: str | None = None

    @field_validator("text")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value


class InferenceOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str
    deadline: float | None = None

    @field_validator("deadline")
    @classmethod
    def finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("deadline must be finite")
        return value

    def remaining(self) -> float | None:
        return None if self.deadline is None else max(0.0, self.deadline - time.monotonic())
