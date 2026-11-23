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
        "Over 1C - 1D, bid 4-card majors up the line (rebid 1H/1S rather than bypassing them with 1NT). "
        "Respond to partner's 1-level suit opening with 6+ HCP (new suit at 1-level or 2-level is forcing 1 round; "
        "1NT response is 6-10 non-forcing; single major raise 2M is 6-9 pts — opener invites game with 16-18 pts "
        "or bids 4M with 19+ pts). "
        "1NT opening = 15-17 HCP balanced (1NT overcall = 15-18 HCP balanced w/ stopper, Systems On); "
        "2C Stayman asks for 4-card major; 2D/2H Jacoby Transfers to Hearts/Spades; "
        "invite 2NT with 8-9 HCP (8 HCP w/ 5-card suit or good intermediates!) and bid 3NT with 10+ HCP. "
        "Strong artificial 2C = 22+ HCP or game-forcing playing strength (2D waiting). "
        "Weak two-bids (2D/2H/2S) = 6-card suit, 5-11 HCP; a new suit response by an unpassed hand is FORCING 1 round (2NT asks for feature). "
        "2NT opening = 20-21 HCP balanced (3C Stayman, 3D/3H Jacoby Transfers). "
        "Over opponent's opening: Unusual 2NT shows 5-5+ in the two lowest unbid suits; Michaels cuebid shows 5-5 two-suiter; "
        "Negative Double through 2S/3S shows unbid major(s) (NOT penalty — pull without a 4+ trump stack); "
        "cuebid of opponent's suit raises partner's overcall showing a limit+ raise (10+ pts with fit). "
        "Penalty doubles at 4/5-level: do NOT double distributional two-suiters on A/K side honors likely to get ruffed unless holding natural trump tricks. "
        "Slam control: Blackwood 4NT (0/1/2/3 Aces) or Gerber 4C over NT. "
        "Defensive carding: standard attitude (high encourages) and standard count (high-low = even)."
    ),
)
