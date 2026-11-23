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
        "Opening discipline: open 1-level suit in 1st/2nd seat with 12+ HCP (or Rule of 20: HCP + lengths of two longest suits >= 20) and 2+ quick tricks — do NOT pass a 12+ HCP unbalanced hand merely for a singleton honor or weak 5-card major (in 4th seat use Rule of 15: HCP + spades >= 15). "
        "Over 1C - 1D, bid 4-card majors up the line (rebid 1H/1S rather than bypassing them with 1NT). "
        "Respond to partner's 1-level suit opening with 6+ HCP (new suit at 1-level or 2-level is forcing 1 round; "
        "1NT response is 6-10 non-forcing; single major raise 2M is 6-9 pts — opener invites game with 16-18 pts "
        "or bids 4M with 19+ pts). "
        "When raising partner's 1M response with 4-card support, count dummy/support points (+1 doubleton, +3 singleton, +5 void): "
        "single raise 2M = 12-15, jump raise 3M = 16-18, game raise 4M = 19+. "
        "Over opener's 1NT rebid (1m-1M-1NT, 12-14 balanced), responder's jump to 3M is invitational (6+ suit, 10-12 pts): accept to 4M with 13-14 HCP, 3-card support, or prime A/K controls (2.5+ quick tricks) and Hx support. "
        "1NT opening = 15-17 HCP balanced (1NT overcall = 15-18 HCP balanced w/ stopper, Systems On); "
        "2C Stayman asks for 4-card major; 2D/2H Jacoby Transfers to Hearts/Spades; "
        "invite 2NT with 8-9 HCP (8 HCP w/ 5-card suit or good intermediates!) and bid 3NT with 10+ HCP. "
        "Strong artificial 2C = 22+ HCP or game-forcing playing strength (2D waiting). "
        "Weak two-bids (2D/2H/2S) = 6-card suit, 5-11 HCP; a new suit response by an unpassed hand is FORCING 1 round (2NT asks for feature). "
        "2NT opening = 20-21 HCP balanced (3C Stayman, 3D/3H Jacoby Transfers). "
        "Over opponent's opening: Unusual 2NT shows 5-5+ in the two lowest unbid suits; Michaels cuebid shows 5-5 two-suiter; "
        "Negative Double through 2S/3S shows unbid major(s) (NOT penalty — pull without a 4+ trump stack); "
        "cuebid of opponent's suit raises partner's overcall showing a limit+ raise (10+ pts with fit) — EXCEPT over a minor overcall (2C/2D), show a 5+ card major first rather than bypassing it for a minor cuebid! "
        "With opening strength and shortness (singleton/void) in opponents' 2-level overcall (e.g., 1D-1NT-2S), prefer a balancing/reopening takeout double over selling out cheaply. "
        "Penalty doubles at 4/5-level: do NOT double distributional two-suiters on A/K side honors likely to get ruffed unless holding natural trump tricks; "
        "HOWEVER, when WE own the hand (after a limit+ cuebid raise, strong bid, or game bid) and opponents jump to a sacrifice (e.g., 1M-(2S)-3S-(4S)), NEVER let them play undoubled — either Double to collect the penalty or bid 5M with shortness/extra playing strength! "
        "Conversely, at favorable vulnerability with a 10+ card fit over partner's weak jump overcall, use the Law of Total Tricks to jump straight to 4M to jam the opponents' auction. "
        "Avoid stretching to an unsupported 4-level minor contract (especially vulnerable) opposite a partner who passed an enemy preempt, unless holding extra shape/trick strength. "
        "Slam control: Blackwood 4NT asks for Aces (5C=0/4, 5D=1, 5H=2, 5S=3 Aces; 5NT asks for Kings: 6C=0, 6D=1, 6H=2, 6S=3) or Gerber 4C over NT. "
        "After a 4NT Blackwood answer, NEVER pass out the artificial Ace-response when holding 3 or 4 combined Aces — bid the slam (6-level suit/NT, or 5NT for Kings); sign off at the 5-level only when missing 2+ Aces. "
        "Defensive carding: standard attitude (high encourages) and standard count (high-low = even)."
    ),
)
