from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LayaConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model: Literal["auto", "english", "multilingual", "typed-decisions"] = "auto"
    revision: str | None = None
    device: str = "cpu"
    max_loaded: int = Field(default=2, ge=1, le=3)
    capacity: int = Field(default=8, ge=1)
    max_len: int | None = Field(default=None, gt=0)
