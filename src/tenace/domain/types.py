from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field

NUM_CARDS = 52
NUM_SUITS = 4
NUM_RANKS = 13
BIDDING_ACTION_BASE = 52
PASS_ACTION = 52
DBL_ACTION = 53
RDBL_ACTION = 54
FIRST_CONTRACT_BID_ACTION = 55
MAX_ACTION = 89

RANK_CHARS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
SUIT_CHARS = ("C", "D", "H", "S")
DENOMINATION_CHARS = ("C", "D", "H", "S", "NT")
HCP_BY_RANK = {12: 4, 11: 3, 10: 2, 9: 1}


class Seat(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @property
    def partner(self) -> Seat:
        return Seat(int(self) ^ 2)

    @property
    def partnership(self) -> int:
        return int(self) & 1


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


def card_id(suit: Suit, rank: int) -> int:
    if not (0 <= rank < NUM_RANKS):
        raise ValueError(f"Invalid rank: {rank}")
    return rank * NUM_SUITS + int(suit)


def card_suit(card: int) -> Suit:
    if not (0 <= card < NUM_CARDS):
        raise ValueError(f"Invalid card action id: {card}")
    return Suit(card % NUM_SUITS)


def card_rank(card: int) -> int:
    if not (0 <= card < NUM_CARDS):
        raise ValueError(f"Invalid card action id: {card}")
    return card // NUM_SUITS


def action_label(action: int) -> str:
    if 0 <= action < NUM_CARDS:
        return f"{SUIT_CHARS[int(card_suit(action))]}:{RANK_CHARS[card_rank(action)]}"
    if action == PASS_ACTION:
        return "Pass"
    if action == DBL_ACTION:
        return "Dbl"
    if action == RDBL_ACTION:
        return "RDbl"
    if FIRST_CONTRACT_BID_ACTION <= action <= MAX_ACTION:
        offset = action - FIRST_CONTRACT_BID_ACTION
        level = 1 + offset // 5
        denom = DENOMINATION_CHARS[offset % 5]
        return f"{level}{denom}"
    raise ValueError(f"Unknown OpenSpiel bridge action id: {action}")


def hand_hcp(cards: Iterable[int]) -> int:
    return sum(HCP_BY_RANK.get(card_rank(c), 0) for c in cards)


class SeatConstraint(BaseModel):
    seat: Seat
    min_hcp: int = Field(default=0, ge=0, le=37)
    max_hcp: int = Field(default=37, ge=0, le=37)
    min_suit_lengths: dict[Suit, int] = Field(default_factory=dict)
    max_suit_lengths: dict[Suit, int] = Field(default_factory=dict)
    likely_cards: list[int] = Field(default_factory=list)


class Assumption(BaseModel):
    claim: str
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class StrategyState(BaseModel):
    phase: Literal["auction", "play"]
    plan_summary: str
    target_or_threat: str = ""
    assumptions: list[Assumption] = Field(default_factory=list)
    seat_constraints: dict[Seat, SeatConstraint] = Field(default_factory=dict)
    revision: int = 0


class CandidateOption(BaseModel):
    action: int
    label: str
    equivalent_actions: list[int] = Field(default_factory=list)
    baseline_ev: float = 0.0
    make_prob: float | None = None


class TacticalDecision(BaseModel):
    assumption_violated: bool = False
    violation_reason: str = ""
    option_scores: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
