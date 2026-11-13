from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from moderationkit.exceptions import InputError


class DatasetRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    labels: dict[str, StrictBool]
    language: str | None = None
    content_type: str = "text"
    split: Literal["train", "validation", "test"] = "test"

    @field_validator("text")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


def load_jsonl(path: str | Path) -> tuple[DatasetRecord, ...]:
    records = []
    ids = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = DatasetRecord.model_validate(json.loads(line))
            if record.id in ids:
                raise ValueError("duplicate record ID")
        except ValueError as exc:
            raise InputError(f"Invalid dataset record at line {line_number}: {exc}") from exc
        ids.add(record.id)
        records.append(record)
    return tuple(records)
