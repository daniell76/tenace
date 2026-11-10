from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

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

CardId = Annotated[int, Field(ge=0, le=NUM_CARDS - 1)]
ActionId = Annotated[int, Field(ge=0, le=MAX_ACTION)]
SuitLength = Annotated[int, Field(ge=0, le=NUM_RANKS)]

__all__ = [
    "ActionId",
    "Assumption",
    "BIDDING_ACTION_BASE",
    "CandidateOption",
    "CardId",
    "DBL_ACTION",
    "DENOMINATION_CHARS",
    "FIRST_CONTRACT_BID_ACTION",
    "HCP_BY_RANK",
    "MAX_ACTION",
    "NUM_CARDS",
    "NUM_RANKS",
    "NUM_SUITS",
    "PASS_ACTION",
    "RANK_CHARS",
    "RDBL_ACTION",
    "SUIT_CHARS",
    "Seat",
    "SeatConstraint",
    "StrategyState",
    "Suit",
    "SuitLength",
    "TacticalDecision",
    "action_label",
    "card_id",
    "card_rank",
    "card_suit",
    "hand_hcp",
]


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
    suit = Suit(suit)
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
    card_list = list(cards)
    if len(set(card_list)) != len(card_list):
        raise ValueError("Duplicate card in hand")
    return sum(HCP_BY_RANK.get(card_rank(c), 0) for c in card_list)


class SeatConstraint(BaseModel):
    seat: Seat
    min_hcp: int = Field(default=0, ge=0, le=37)
    max_hcp: int = Field(default=37, ge=0, le=37)
    min_suit_lengths: dict[Suit, SuitLength] = Field(default_factory=dict)
    max_suit_lengths: dict[Suit, SuitLength] = Field(default_factory=dict)
    likely_cards: list[CardId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_bounds(self) -> SeatConstraint:
        if self.min_hcp > self.max_hcp:
            raise ValueError(
                f"min_hcp ({self.min_hcp}) must be <= max_hcp ({self.max_hcp})"
            )
        for suit, min_len in self.min_suit_lengths.items():
            max_len = self.max_suit_lengths.get(suit)
            if max_len is not None and min_len > max_len:
                raise ValueError(
                    f"min_suit_lengths[{suit.name}] ({min_len}) must be <= "
                    f"max_suit_lengths[{suit.name}] ({max_len})"
                )
        return self


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
    action: ActionId
    label: str
    equivalent_actions: list[ActionId] = Field(default_factory=list)
    baseline_ev: float = 0.0
    make_prob: float | None = Field(default=None, ge=0.0, le=1.0)


class TacticalDecision(BaseModel):
    assumption_violated: bool = False
    violation_reason: str = ""
    option_scores: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
