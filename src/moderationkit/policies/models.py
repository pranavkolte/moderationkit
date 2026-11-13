from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from moderationkit.types import Action, Severity


class Rule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    threshold: float = Field(ge=0, le=1)
    action: Action
    severity: Severity | None = None

    @model_validator(mode="after")
    def enforcement(self) -> Rule:
        if self.action == Action.ERROR:
            raise ValueError("ERROR is an evaluation outcome, not a rule action")
        return self


class Policy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    rules: tuple[Rule, ...]
    observation_categories: tuple[str, ...] = ()
    detection_thresholds: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid(self) -> Policy:
        ids = [rule.id for rule in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate rule IDs")
        if not self.requested_categories:
            raise ValueError("policy must request at least one category")
        if set(self.detection_thresholds) - set(self.requested_categories):
            raise ValueError("detection threshold refers to an unrequested category")
        for name in self.requested_categories:
            self.threshold_for(name)
        return self

    @property
    def required_categories(self) -> frozenset[str]:
        return frozenset(rule.category for rule in self.rules)

    @property
    def requested_categories(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys([r.category for r in self.rules] + list(self.observation_categories))
        )

    def threshold_for(self, category: str) -> float:
        if category in self.detection_thresholds:
            value = self.detection_thresholds[category]
        else:
            values = {r.threshold for r in self.rules if r.category == category}
            if len(values) != 1:
                raise ValueError(f"explicit detection threshold required for {category}")
            value = values.pop()
        if not 0 <= value <= 1:
            raise ValueError("detection thresholds must be finite and between zero and one")
        return value
