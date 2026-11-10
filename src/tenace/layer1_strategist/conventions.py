from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = [
    "ConventionCard",
    "SAYC_CONVENTION_CARD",
]


class ConventionCard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    summary_rules: str

    @field_validator("name", "summary_rules")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("ConventionCard fields must be non-empty strings")
        return stripped


SAYC_CONVENTION_CARD = ConventionCard(
    name="SAYC (Standard American Yellow Card)",
    summary_rules=(
        "5-card majors (1H/1S promise 5+ cards; 1C/1D may be 3+ cards). "
        "1NT opening = 15-17 HCP balanced; 2C Stayman asks for 4-card major; "
        "2D/2H Jacoby Transfers to Hearts/Spades. "
        "Strong artificial 2C = 22+ HCP or game-forcing playing strength (2D waiting). "
        "Weak two-bids (2D/2H/2S) = 6-card suit, 5-11 HCP. "
        "2NT opening = 20-21 HCP balanced (3C Stayman, 3D/3H Jacoby Transfers). "
        "Defensive carding: standard attitude (high encourages) and standard count (high-low = even)."
    ),
)
