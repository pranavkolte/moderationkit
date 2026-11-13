from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CategoryDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")
    description: str = Field(min_length=1)
    positive: str | None = None
    negative: str | None = None


TAXONOMY_VERSION = "1.0"
_DEFINITIONS = {
    "hate_speech": (
        "Targeted hateful abuse or dehumanization based on protected identity; "
        "exclude reporting, quotation for criticism, and counterspeech."
    ),
    "harassment": (
        "Targeted personal abuse, intimidation, or bullying; exclude respectful "
        "disagreement and reporting of abuse."
    ),
    "sexual_content": (
        "Sexually explicit descriptions or solicitations; exclude non-graphic "
        "health education and neutral reporting."
    ),
    "violence": (
        "Threats, encouragement, or graphic descriptions of physical violence; "
        "exclude neutral non-graphic reporting."
    ),
    "politics": (
        "Political discussion, campaigning, or advocacy; detection does not itself "
        "imply a violation."
    ),
    "self_harm": (
        "Encouragement, instructions, or expressed intent for self-harm; "
        "distinguish prevention, support, and neutral reporting."
    ),
    "extremism": (
        "Promotion, recruitment, or praise of violent extremist groups or acts; "
        "exclude criticism and neutral reporting."
    ),
    "spam_scams": (
        "Unsolicited repetitive promotion, deceptive solicitations, phishing, or "
        "fraudulent schemes."
    ),
    "privacy": (
        "Disclosure or solicitation of private sensitive personal information "
        "without consent; exclude synthetic examples."
    ),
}
CORE_CATEGORIES = tuple(
    CategoryDefinition(id=key, description=value) for key, value in _DEFINITIONS.items()
)
