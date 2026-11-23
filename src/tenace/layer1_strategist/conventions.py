from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

VALID_L2_DISCIPLINE_SLOTS: frozenset[str] = frozenset(
    {"partner_active", "competitive", "unopened"}
)

__all__ = [
    "ConventionCard",
    "SAYC_CONVENTION_CARD",
    "VALID_L2_DISCIPLINE_SLOTS",
]


class ConventionCard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = "custom"
    name: str
    summary_rules: str
    detailed_rules: str = ""
    calibration_notes: str = ""
    l1_auction_directive: str = ""
    l2_discipline_overrides: dict[str, str] = Field(default_factory=dict)

    @field_validator("id", "name", "summary_rules")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("ConventionCard fields must be non-empty strings")
        return stripped

    @field_validator("detailed_rules", "calibration_notes", "l1_auction_directive")
    @classmethod
    def _strip_optional_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("l2_discipline_overrides")
    @classmethod
    def _validate_l2_overrides(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for k, v in value.items():
            if k not in VALID_L2_DISCIPLINE_SLOTS:
                allowed = ", ".join(sorted(VALID_L2_DISCIPLINE_SLOTS))
                raise ValueError(
                    f"Unknown l2_discipline_overrides slot {k!r}; allowed: {allowed}"
                )
            stripped = v.strip()
            if stripped:
                cleaned[k] = stripped
        return cleaned


def _load_builtin_sayc_card() -> ConventionCard:
    from tenace.skills.registry import DEFAULT_BIDDING_SYSTEM_ID, SkillRegistry

    return (
        SkillRegistry(extra_dirs=())
        .load(DEFAULT_BIDDING_SYSTEM_ID, allow_file_path=False)
        .to_convention_card()
    )


SAYC_CONVENTION_CARD = _load_builtin_sayc_card()
