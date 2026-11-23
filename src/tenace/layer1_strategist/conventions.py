from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = [
    "ConventionCard",
    "SAYC_CONVENTION_CARD",
]


class ConventionCard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = "custom"
    name: str
    summary_rules: str
    detailed_rules: str = ""

    @field_validator("id", "name", "summary_rules")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("ConventionCard fields must be non-empty strings")
        return stripped

    @field_validator("detailed_rules")
    @classmethod
    def _strip_optional_text(cls, value: str) -> str:
        return value.strip()


def _load_builtin_sayc_card() -> ConventionCard:
    from tenace.skills.registry import DEFAULT_BIDDING_SYSTEM_ID, SkillRegistry

    return (
        SkillRegistry(extra_dirs=())
        .load(DEFAULT_BIDDING_SYSTEM_ID, allow_file_path=False)
        .to_convention_card()
    )


SAYC_CONVENTION_CARD = _load_builtin_sayc_card()
