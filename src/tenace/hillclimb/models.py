from __future__ import annotations

from datetime import datetime, timezone
import difflib
import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tenace.layer1_strategist.conventions import ConventionCard
from tenace.layer1_strategist.strategist import (
    DEFAULT_L1_AUCTION_DIRECTIVE,
    build_strategist_system_prompt,
)
from tenace.layer2_evaluator.evaluator import (
    DEFAULT_L2_AUCTION_SYSTEM_PROMPT,
    DEFAULT_L2_DISCIPLINE_COMPETITIVE,
    DEFAULT_L2_DISCIPLINE_PARTNER_ACTIVE,
    DEFAULT_L2_DISCIPLINE_UNOPENED,
)
from tenace.skills.registry import BiddingSkill, BiddingSkillMeta

MatchMode = Literal["controlled", "duplicate"]
VersionStatus = Literal["baseline", "candidate", "accepted", "rejected"]
RunKind = Literal["eval", "ab_retest", "holdout"]

PROMPT_VARIABLE_SLOTS: tuple[str, ...] = (
    "summary_rules",
    "detailed_rules",
    "calibration_notes",
    "l1_auction_directive",
    "l2_discipline_partner_active",
    "l2_discipline_competitive",
    "l2_discipline_unopened",
)

__all__ = [
    "ABRetestRecord",
    "AuctionDivergence",
    "BoardDealSnapshot",
    "BoardTelemetryRecord",
    "IterationRecord",
    "MatchMode",
    "PROMPT_VARIABLE_SLOTS",
    "PromptVariables",
    "ProposedPromptPatch",
    "RoomSnapshot",
    "RunKind",
    "RunManifest",
    "SkillVersionMeta",
    "VersionStatus",
    "apply_patch_to_variables",
    "extract_prompt_variables",
    "render_prompt_snapshots",
    "sha256_text",
    "unified_diff_variables",
    "utc_now_iso",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class PromptVariables(BaseModel):
    """Explicit decomposition of every mutable prompt component for a bidding-system skill."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    summary_rules: str
    detailed_rules: str = ""
    calibration_notes: str = ""
    l1_auction_directive: str = DEFAULT_L1_AUCTION_DIRECTIVE
    l2_discipline_partner_active: str = DEFAULT_L2_DISCIPLINE_PARTNER_ACTIVE
    l2_discipline_competitive: str = DEFAULT_L2_DISCIPLINE_COMPETITIVE
    l2_discipline_unopened: str = DEFAULT_L2_DISCIPLINE_UNOPENED

    def slot_hashes(self) -> dict[str, str]:
        return {slot: sha256_text(getattr(self, slot)) for slot in PROMPT_VARIABLE_SLOTS}

    def slot_char_counts(self) -> dict[str, int]:
        return {slot: len(getattr(self, slot)) for slot in PROMPT_VARIABLE_SLOTS}

    def combined_hash(self) -> str:
        payload = "\n<SLOT>\n".join(
            f"{slot}={getattr(self, slot)}" for slot in PROMPT_VARIABLE_SLOTS
        )
        return sha256_text(payload)

    def to_convention_card(self, meta: BiddingSkillMeta) -> ConventionCard:
        overrides: dict[str, str] = {}
        if self.l2_discipline_partner_active.strip() != DEFAULT_L2_DISCIPLINE_PARTNER_ACTIVE:
            overrides["partner_active"] = self.l2_discipline_partner_active.strip()
        if self.l2_discipline_competitive.strip() != DEFAULT_L2_DISCIPLINE_COMPETITIVE:
            overrides["competitive"] = self.l2_discipline_competitive.strip()
        if self.l2_discipline_unopened.strip() != DEFAULT_L2_DISCIPLINE_UNOPENED:
            overrides["unopened"] = self.l2_discipline_unopened.strip()

        l1_override = (
            self.l1_auction_directive.strip()
            if self.l1_auction_directive.strip() != DEFAULT_L1_AUCTION_DIRECTIVE
            else ""
        )
        return ConventionCard(
            id=meta.id,
            name=meta.name,
            summary_rules=self.summary_rules,
            detailed_rules=self.detailed_rules,
            calibration_notes=self.calibration_notes,
            l1_auction_directive=l1_override,
            l2_discipline_overrides=overrides,
        )


def extract_prompt_variables(skill: BiddingSkill) -> PromptVariables:
    return PromptVariables(
        summary_rules=skill.summary_rules,
        detailed_rules=skill.detailed_rules,
        calibration_notes=skill.calibration_notes,
    )


def render_prompt_snapshots(
    meta: BiddingSkillMeta,
    variables: PromptVariables,
) -> dict[str, str]:
    """Return the exact rendered L1/L2 prompt texts sent to the LLM for audit/diffing."""
    card = variables.to_convention_card(meta)
    return {
        "l1_system_prompt": build_strategist_system_prompt(card, phase="auction"),
        "l1_play_system_prompt": build_strategist_system_prompt(card, phase="play"),
        "l2_system_prompt": DEFAULT_L2_AUCTION_SYSTEM_PROMPT,
        "l2_partnership_summary": card.summary_rules,
        "l2_calibration_notes": card.calibration_notes,
        "l2_discipline_partner_active": variables.l2_discipline_partner_active,
        "l2_discipline_competitive": variables.l2_discipline_competitive,
        "l2_discipline_unopened": variables.l2_discipline_unopened,
    }


def changed_variable_slots(
    parent: PromptVariables,
    child: PromptVariables,
) -> list[str]:
    return [
        slot
        for slot in PROMPT_VARIABLE_SLOTS
        if getattr(parent, slot) != getattr(child, slot)
    ]


def unified_diff_variables(
    from_label: str,
    from_vars: PromptVariables,
    to_label: str,
    to_vars: PromptVariables,
) -> str:
    """Return a clean readable unified diff across every changed `PromptVariables` slot."""
    blocks: list[str] = []
    for slot in PROMPT_VARIABLE_SLOTS:
        before = getattr(from_vars, slot)
        after = getattr(to_vars, slot)
        if before == after:
            continue
        diff_lines = list(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"{from_label}/{slot}",
                tofile=f"{to_label}/{slot}",
                lineterm="\n",
            )
        )
        if diff_lines:
            text = "".join(
                line if line.endswith("\n") else f"{line}\n" for line in diff_lines
            )
            blocks.append(text.rstrip("\n"))
    return "\n\n".join(blocks)


class ProposedPromptPatch(BaseModel):
    """Structured prompt mutation emitted by an optimizer or human calibrator."""

    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(min_length=1)
    rationale: str = Field(default="")
    target_boards: list[str] = Field(default_factory=list)
    summary_rules_replacement: str | None = None
    detailed_rules_append: str | None = None
    detailed_rules_replacement: str | None = None
    calibration_notes_append: str | None = None
    calibration_notes_replacement: str | None = None
    l1_auction_directive_replacement: str | None = None
    l2_discipline_partner_active_replacement: str | None = None
    l2_discipline_competitive_replacement: str | None = None
    l2_discipline_unopened_replacement: str | None = None


def apply_patch_to_variables(
    base: PromptVariables,
    patch: ProposedPromptPatch,
) -> PromptVariables:
    """Apply a `ProposedPromptPatch` cleanly to `base` and validate at least one slot changes."""
    summary_rules = base.summary_rules
    if patch.summary_rules_replacement is not None:
        cleaned = patch.summary_rules_replacement.strip()
        if not cleaned:
            raise ValueError("summary_rules_replacement cannot be blank")
        summary_rules = cleaned

    detailed_rules = base.detailed_rules
    if patch.detailed_rules_replacement is not None:
        detailed_rules = patch.detailed_rules_replacement.strip()
    elif patch.detailed_rules_append and patch.detailed_rules_append.strip():
        addition = patch.detailed_rules_append.strip()
        detailed_rules = f"{detailed_rules}\n{addition}".strip() if detailed_rules else addition

    calibration_notes = base.calibration_notes
    if patch.calibration_notes_replacement is not None:
        calibration_notes = patch.calibration_notes_replacement.strip()
    elif patch.calibration_notes_append and patch.calibration_notes_append.strip():
        addition = patch.calibration_notes_append.strip()
        calibration_notes = (
            f"{calibration_notes}\n{addition}".strip() if calibration_notes else addition
        )

    l1_dir = (
        patch.l1_auction_directive_replacement.strip()
        if patch.l1_auction_directive_replacement is not None
        and patch.l1_auction_directive_replacement.strip()
        else base.l1_auction_directive
    )
    l2_partner = (
        patch.l2_discipline_partner_active_replacement.strip()
        if patch.l2_discipline_partner_active_replacement is not None
        and patch.l2_discipline_partner_active_replacement.strip()
        else base.l2_discipline_partner_active
    )
    l2_comp = (
        patch.l2_discipline_competitive_replacement.strip()
        if patch.l2_discipline_competitive_replacement is not None
        and patch.l2_discipline_competitive_replacement.strip()
        else base.l2_discipline_competitive
    )
    l2_unopened = (
        patch.l2_discipline_unopened_replacement.strip()
        if patch.l2_discipline_unopened_replacement is not None
        and patch.l2_discipline_unopened_replacement.strip()
        else base.l2_discipline_unopened
    )

    candidate = PromptVariables(
        summary_rules=summary_rules,
        detailed_rules=detailed_rules,
        calibration_notes=calibration_notes,
        l1_auction_directive=l1_dir,
        l2_discipline_partner_active=l2_partner,
        l2_discipline_competitive=l2_comp,
        l2_discipline_unopened=l2_unopened,
    )
    if not changed_variable_slots(base, candidate):
        raise ValueError("ProposedPromptPatch produced zero changes vs base PromptVariables")
    return candidate


class SkillVersionMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    parent_version: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    status: VersionStatus = "candidate"
    hypothesis: str = ""
    rationale: str = ""
    proposer: str = "manual"
    target_boards: list[str] = Field(default_factory=list)
    changed_slots: list[str] = Field(default_factory=list)
    combined_hash: str
    slot_hashes: dict[str, str] = Field(default_factory=dict)
    slot_char_counts: dict[str, int] = Field(default_factory=dict)
    skill_meta: BiddingSkillMeta
    decision_note: str = ""


class AuctionDivergence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_index: int = Field(ge=0)
    seat: str
    reference_call: str
    challenger_call: str


class BoardDealSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hands: dict[str, str]
    cards_by_seat: dict[str, list[str]]
    hand_hcp: dict[str, int]
    shapes: dict[str, str] = Field(default_factory=dict)


class RoomSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: str
    auction: str
    tricks: list[str] = Field(default_factory=list)
    ns_score: float
    strategy_traces: dict[str, list[str]] = Field(default_factory=dict)
    elapsed_s: float = 0.0


class BoardTelemetryRecord(BaseModel):
    """Full raw per-board telemetry stored in `raw/boards.jsonl`."""

    model_config = ConfigDict(extra="forbid")

    recorded_at: str = Field(default_factory=utc_now_iso)
    run_kind: RunKind = "eval"
    batch_id: str
    prompt_version: str
    match_mode: MatchMode
    tournament_seed: int
    board_number: int = Field(ge=1)
    dealer: str
    vul: str
    deal: BoardDealSnapshot
    table_a: RoomSnapshot
    table_b: RoomSnapshot
    score_diff: float
    imps_challenger: int
    auction_exact_match: bool
    contract_match: bool
    first_divergence: AuctionDivergence | None = None
    swing_category: str
    reference_cache_hit: bool = False
    telemetry: dict[str, int] = Field(default_factory=dict)

    @property
    def board_key(self) -> str:
        return f"{self.tournament_seed}:{self.board_number}"


class IterationRecord(BaseModel):
    """Batch-level rollup appended to `raw/iterations.jsonl`."""

    model_config = ConfigDict(extra="forbid")

    recorded_at: str = Field(default_factory=utc_now_iso)
    batch_id: str
    run_kind: RunKind = "eval"
    prompt_version: str
    match_mode: MatchMode
    tournament_seed: int
    start_board: int
    num_boards: int
    net_imps: int
    gross_won_imps: int
    gross_lost_imps: int
    imp_per_board: float
    vp_challenger: float
    vp_reference: float
    wins: int
    losses: int
    pushes: int
    auction_match_rate: float
    contract_match_rate: float
    swing_counts: dict[str, int] = Field(default_factory=dict)
    telemetry: dict[str, int] = Field(default_factory=dict)
    elapsed_s: float = 0.0


class ABRetestRecord(BaseModel):
    """Matched-seed A/B retest verdict appended to `raw/ab_retests.jsonl`."""

    model_config = ConfigDict(extra="forbid")

    recorded_at: str = Field(default_factory=utc_now_iso)
    candidate_version: str
    parent_version: str
    board_keys: list[str]
    before_imps: int
    after_imps: int
    delta_imps: int
    verdict: Literal["accepted", "rejected"]
    min_delta_required: int = 1
    note: str = ""
    per_board: list[dict[str, object]] = Field(default_factory=list)


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    system_id: str
    system_name: str
    match_mode: MatchMode = "controlled"
    reference_label: str = "reference"
    challenger_profile_label: str = ""
    active_version: str = "v000"
    best_version: str = "v000"
    base_skill_meta: BiddingSkillMeta
    notes: str = ""
