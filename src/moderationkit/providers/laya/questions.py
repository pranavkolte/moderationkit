import hashlib
import json
from collections.abc import Sequence
from typing import Any

from moderationkit.categories import CategoryDefinition

TEMPLATE_VERSION = "1.0"


def compile_questions(categories: Sequence[CategoryDefinition]) -> dict[str, Any]:
    if len({c.id for c in categories}) != len(categories):
        raise ValueError("duplicate category IDs")
    return {
        c.id: {
            "type": "noul",
            "instructions": f"Does the supplied content match this category? {c.description}",
            "criteria": {
                "true": c.positive or c.description,
                "false": c.negative
                or f"The content does not match this definition: {c.description}",
            },
        }
        for c in categories
    }


def questions_hash(questions: dict[str, Any]) -> str:
    encoded = json.dumps(questions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((TEMPLATE_VERSION + encoded).encode()).hexdigest()
