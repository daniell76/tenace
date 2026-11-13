from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import random
import threading
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.api_core import exceptions as gcp_exceptions
from google.genai import errors as genai_errors
from google.genai._gaos.lib import compat_errors
import pyspiel
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
import uvicorn

_UPSTREAM_MODEL_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    genai_errors.APIError,
    compat_errors.GeminiNextGenAPIClientError,
    gcp_exceptions.GoogleAPIError,
)

from tenace.agent import TenaceAgent
from tenace.cli import (
    DEFAULT_STRATEGIST_MODEL,
    ModelProfile,
    build_clients_for_profile,
    format_match_report,
)
from tenace.domain.types import (
    BIDDING_ACTION_BASE,
    DBL_ACTION,
    DENOMINATION_CHARS,
    FIRST_CONTRACT_BID_ACTION,
    MAX_ACTION,
    NUM_CARDS,
    PASS_ACTION,
    RDBL_ACTION,
    Seat,
    StrategyState,
    Suit,
    action_label,
    card_rank,
    card_suit,
    hand_hcp,
)
from tenace.harness.duplicate import (
    BridgeSeatBot,
    DuplicateMatchRunner,
    FirstLegalPassBaselineBot,
    MatchSummary,
    wbf_board_params,
)
from tenace.layer1_strategist import SAYC_CONVENTION_CARD
from tenace.layer3_rules import (
    RuleDDSEvaluator,
    RuleFactTracker,
    collapse_equivalent_actions,
    trick_winner,
)
from tenace.models.clients import StructuredModelClient

SERVICE_VERSION = "0.1.0"
_MAX_CACHED_PROFILES = 16
_MAX_INT32 = 2_147_483_647
_MAX_BOARD_NUMBER = 1024
WEB_DIR = Path(__file__).resolve().parent / "web"

SeatName = Literal["NORTH", "EAST", "SOUTH", "WEST"]
StrictToken = StrictInt | StrictStr

_CARD_LABEL_TO_ID: dict[str, int] = {
    action_label(card).upper(): card for card in range(NUM_CARDS)
}
_ACTION_LABEL_TO_ID: dict[str, int] = {
    action_label(action).upper(): action for action in range(MAX_ACTION + 1)
}

__all__ = [
    "MatchRequest",
    "MatchResponse",
    "StepRequest",
    "StepResponse",
    "TableStateRequest",
    "TableStateResponse",
    "compute_table_state",
    "create_app",
    "default_dds_worlds_from_env",
    "default_profile_from_env",
    "main",
    "parse_action_token",
    "parse_card_token",
    "reconstruct_step_state",
    "resolve_server_bind",
]


def _opt_env(mapping: Mapping[str, str], key: str) -> str | None:
    raw = mapping.get(key)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


def parse_card_token(token: int | str) -> int:
    """Parse a canonical card ID (`0..51`) or label (`'C:2'`..`'S:A'`, case-insensitive)."""
    if isinstance(token, bool):
        raise ValueError("Card token cannot be a boolean")
    if isinstance(token, int):
        if 0 <= token < NUM_CARDS:
            return int(token)
        raise ValueError(f"Card token ID must be in [0, {NUM_CARDS - 1}], got {token}")
    if isinstance(token, str):
        key = token.strip().upper()
        card = _CARD_LABEL_TO_ID.get(key)
        if card is not None:
            return card
        raise ValueError(f"Unknown card token: {token!r}")
    raise ValueError(f"Card token must be int or str, got {type(token).__name__}")


def parse_action_token(token: int | str) -> int:
    """Parse an OpenSpiel Bridge action ID (`0..89`) or canonical label (`'Pass'`, `'1NT'`, `'S:A'`)."""
    if isinstance(token, bool):
        raise ValueError("Action token cannot be a boolean")
    if isinstance(token, int):
        if 0 <= token <= MAX_ACTION:
            return int(token)
        raise ValueError(f"Action token ID must be in [0, {MAX_ACTION}], got {token}")
    if isinstance(token, str):
        key = token.strip().upper()
        action = _ACTION_LABEL_TO_ID.get(key)
        if action is not None:
            return action
        raise ValueError(f"Unknown action token: {token!r}")
    raise ValueError(f"Action token must be int or str, got {type(token).__name__}")


def default_profile_from_env(env: Mapping[str, str] | None = None) -> ModelProfile:
    """Build the default `ModelProfile` from Cloud Run environment variables."""
    mapping = os.environ if env is None else env

    mode = _opt_env(mapping, "TENACE_MODE") or "vertex"
    project = _opt_env(mapping, "GOOGLE_CLOUD_PROJECT")
    strategist_location = _opt_env(mapping, "TENACE_STRATEGIST_LOCATION") or "global"
    strategist_model = (
        _opt_env(mapping, "TENACE_STRATEGIST_MODEL") or DEFAULT_STRATEGIST_MODEL
    )
    evaluator_backend = _opt_env(mapping, "TENACE_EVALUATOR_BACKEND") or "decisions"
    evaluator_location = _opt_env(mapping, "TENACE_EVALUATOR_LOCATION")
    evaluator_model = _opt_env(mapping, "TENACE_EVALUATOR_MODEL")
    evaluator_endpoint_id = _opt_env(mapping, "TENACE_EVALUATOR_ENDPOINT_ID")
    decisions_auth = _opt_env(mapping, "TENACE_DECISIONS_AUTH") or "vertex"

    raw_threshold = _opt_env(mapping, "TENACE_DECISIONS_VIOLATION_THRESHOLD")
    threshold = 0.70 if raw_threshold is None else float(raw_threshold)

    return ModelProfile(
        mode=mode,  # type: ignore[arg-type]
        project=project,
        strategist_location=strategist_location,
        strategist_model=strategist_model,
        evaluator_backend=evaluator_backend,  # type: ignore[arg-type]
        evaluator_location=evaluator_location,  # type: ignore[arg-type]
        evaluator_model=evaluator_model,  # type: ignore[arg-type]
        evaluator_endpoint_id=evaluator_endpoint_id,
        decisions_auth=decisions_auth,  # type: ignore[arg-type]
        decisions_violation_threshold=threshold,
    )


def default_dds_worlds_from_env(env: Mapping[str, str] | None = None) -> int:
    """Read `TENACE_DDS_WORLDS` from environment variables (default `16`, must be `>= 1`)."""
    mapping = os.environ if env is None else env
    raw = _opt_env(mapping, "TENACE_DDS_WORLDS")
    if raw is None:
        return 16
    worlds = int(raw)
    if worlds < 1:
        raise ValueError(f"TENACE_DDS_WORLDS must be >= 1, got {worlds}")
    return worlds


def resolve_server_bind(env: Mapping[str, str] | None = None) -> tuple[str, int]:
    """Resolve `(host, port)` from `HOST` (default `'0.0.0.0'`) and `PORT` (default `8080`)."""
    mapping = os.environ if env is None else env

    if "HOST" in mapping:
        host = mapping["HOST"].strip()
        if not host:
            raise ValueError("HOST must be a non-blank string")
    else:
        host = "0.0.0.0"

    if "PORT" in mapping:
        raw_port = mapping["PORT"].strip()
        if not raw_port:
            raise ValueError("PORT must be an integer between 1 and 65535")
        port = int(raw_port)
    else:
        port = 8080

    if not (1 <= port <= 65535):
        raise ValueError(f"PORT must be between 1 and 65535, got {port}")
    return host, port


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seat: SeatName
    board_number: int = Field(default=1, ge=1, le=_MAX_BOARD_NUMBER)
    own_hand: list[StrictToken] | None = None
    dummy_hand: list[StrictToken] | None = None
    tournament_seed: int | None = Field(default=None, ge=0, le=_MAX_INT32)
    action_history: list[StrictToken] = Field(default_factory=list)
    prior_strategy: StrategyState | None = None
    profile: ModelProfile | None = None
    dds_worlds: int | None = Field(default=None, ge=1, le=256)
    seed: int = Field(default=42, ge=0, le=_MAX_INT32)
    bot: Literal["tenace", "baseline"] = "tenace"

    @field_validator("seat", mode="before")
    @classmethod
    def _normalize_seat(cls, value: Any) -> Any:
        if type(value) is Seat:
            return value.name
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("board_number", "tournament_seed", "dds_worlds", "seed", mode="before")
    @classmethod
    def _reject_bool_ints(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("Boolean value is not valid for integer field")
        return value

    @model_validator(mode="after")
    def _validate_deal_mode_and_tokens(self) -> StepRequest:
        if (self.own_hand is None) == (self.tournament_seed is None):
            raise ValueError(
                "Provide exactly one of 'own_hand' (imperfect-information mode) "
                "or 'tournament_seed' (benchmark mode)"
            )
        if self.tournament_seed is not None and self.dummy_hand is not None:
            raise ValueError("'dummy_hand' cannot be combined with 'tournament_seed'")

        for tok in self.action_history:
            parse_action_token(tok)

        if self.own_hand is not None:
            own_cards = [parse_card_token(tok) for tok in self.own_hand]
            if len(own_cards) != 13 or len(set(own_cards)) != 13:
                raise ValueError("own_hand must contain exactly 13 unique cards")
            if self.dummy_hand is not None:
                dummy_cards = [parse_card_token(tok) for tok in self.dummy_hand]
                if len(dummy_cards) != 13 or len(set(dummy_cards)) != 13:
                    raise ValueError("dummy_hand must contain exactly 13 unique cards")
                if set(own_cards) & set(dummy_cards):
                    raise ValueError("own_hand and dummy_hand must be disjoint")

        return self


class StepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: int
    action_label: str
    seat: SeatName
    phase: Literal["auction", "play"]
    strategy: StrategyState | None
    telemetry: dict[str, int]


class TableStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tournament_seed: int = Field(default=42, ge=0, le=_MAX_INT32)
    board_number: int = Field(default=1, ge=1, le=_MAX_BOARD_NUMBER)
    action_history: list[StrictToken] = Field(default_factory=list)
    viewer_seat: SeatName = "SOUTH"
    reveal_all: StrictBool = False

    @field_validator("viewer_seat", mode="before")
    @classmethod
    def _normalize_viewer_seat(cls, value: Any) -> Any:
        if type(value) is Seat:
            return value.name
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("tournament_seed", "board_number", mode="before")
    @classmethod
    def _reject_bool_ints(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("Boolean value is not valid for integer field")
        return value

    @model_validator(mode="after")
    def _validate_action_tokens(self) -> TableStateRequest:
        for tok in self.action_history:
            parse_action_token(tok)
        return self


class AuctionCallView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seat: SeatName
    label: str
    action_id: int


class ContractView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: int | None
    trump: Literal["C", "D", "H", "S", "NT"] | None
    doubled: Literal["", "X", "XX"] = ""
    declarer: SeatName | None
    dummy: SeatName | None
    formatted: str


class TrickCardView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seat: SeatName
    card: str
    action_id: int


class CompletedTrickView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trick_number: int
    leader: SeatName
    winner: SeatName
    cards: list[TrickCardView]


class LegalActionView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: int
    label: str


class TerminalResultView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ns_score: int
    ew_score: int
    contract: str
    ns_tricks: int
    ew_tricks: int


class TableStateResponse(BaseModel):
    """Imperfect-information Bridge table snapshot for an interactive viewer seat.

     Visibility & phase invariants:
    - `contract` is `None` while `phase == "auction"`, and populated once the auction
      finishes (`"Passed Out"` or a played contract such as `"3NT by SOUTH"`).
    - `legal_actions` contains public calls during `phase == "auction"`, is redacted to
      `[]` on hidden-seat `phase == "play"` turns when `reveal_all=False` unless
      `current_player == viewer_seat`, and is `[]` when `phase == "terminal"`.
    """

    model_config = ConfigDict(extra="forbid")

    tournament_seed: int
    board_number: int
    dealer: SeatName
    ns_vulnerable: bool
    ew_vulnerable: bool
    phase: Literal["auction", "play", "terminal"]
    viewer_seat: SeatName
    reveal_all: bool
    current_player: SeatName | None
    acting_hand: SeatName | None
    hands: dict[SeatName, list[str] | None]
    hand_hcp: dict[SeatName, int | None]
    remaining_counts: dict[SeatName, int]
    initial_hands: dict[SeatName, list[str]] | None
    auction_calls: list[AuctionCallView]
    contract: ContractView | None
    ns_tricks: int
    ew_tricks: int
    current_trick: list[TrickCardView]
    completed_tricks: list[CompletedTrickView]
    legal_actions: list[LegalActionView]
    terminal_result: TerminalResultView | None


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boards: int = Field(default=2, ge=1, le=32)
    seed: int = Field(default=42, ge=0, le=_MAX_INT32)
    dds_worlds: int | None = Field(default=None, ge=1, le=64)
    double_dummy: StrictBool = False
    opponent: Literal["baseline", "tenace"] = "baseline"
    team_a_profile: ModelProfile | None = None
    team_b_profile: ModelProfile | None = None

    @field_validator("boards", "seed", "dds_worlds", mode="before")
    @classmethod
    def _reject_bool_ints(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("Boolean value is not valid for integer field")
        return value

    @model_validator(mode="after")
    def _validate_opponent_profile(self) -> MatchRequest:
        if self.opponent != "tenace" and self.team_b_profile is not None:
            raise ValueError("team_b_profile is only allowed when opponent='tenace'")
        return self


class MatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team_a_name: str
    team_b_name: str
    team_a_profile: str
    team_b_profile: str | None
    summary: MatchSummary
    report: str


@dataclass(frozen=True)
class _AuctionTrace:
    calls: list[AuctionCallView]
    finished: bool
    passed_out: bool
    declarer: Seat | None
    dummy: Seat | None
    trump_denom: int | None
    contract_view: ContractView | None


def _trace_auction(
    dealer: int,
    auction_actions: list[int],
) -> _AuctionTrace:
    bidder = dealer
    calls: list[AuctionCallView] = []
    last_contract_bid: int | None = None
    last_bidder: int | None = None
    doubled_state: Literal["undoubled", "doubled", "redoubled"] = "undoubled"
    trailing_passes = 0
    first_denom_bidder: dict[tuple[int, int], int] = {}
    finished = False
    passed_out = False

    for idx, call in enumerate(auction_actions):
        if finished:
            raise ValueError(
                f"Auction action {action_label(call)} at index {idx} occurred after auction ended"
            )
        if call == PASS_ACTION:
            trailing_passes += 1
            if last_contract_bid is None and trailing_passes == 4:
                finished = True
                passed_out = True
            elif last_contract_bid is not None and trailing_passes == 3:
                finished = True
        elif call == DBL_ACTION:
            if (
                last_contract_bid is None
                or last_bidder is None
                or doubled_state != "undoubled"
                or (bidder & 1) == (last_bidder & 1)
            ):
                raise ValueError(
                    f"Illegal Dbl by {Seat(bidder).name} at auction index {idx}"
                )
            doubled_state = "doubled"
            trailing_passes = 0
        elif call == RDBL_ACTION:
            if (
                last_contract_bid is None
                or last_bidder is None
                or doubled_state != "doubled"
                or (bidder & 1) != (last_bidder & 1)
            ):
                raise ValueError(
                    f"Illegal RDbl by {Seat(bidder).name} at auction index {idx}"
                )
            doubled_state = "redoubled"
            trailing_passes = 0
        elif FIRST_CONTRACT_BID_ACTION <= call <= MAX_ACTION:
            if last_contract_bid is not None and call <= last_contract_bid:
                raise ValueError(
                    f"Insufficient contract bid {action_label(call)} after "
                    f"{action_label(last_contract_bid)} at auction index {idx}"
                )
            denom = (call - FIRST_CONTRACT_BID_ACTION) % 5
            key = (bidder & 1, denom)
            if key not in first_denom_bidder:
                first_denom_bidder[key] = bidder
            last_contract_bid = call
            last_bidder = bidder
            doubled_state = "undoubled"
            trailing_passes = 0
        else:
            raise ValueError(f"Invalid auction action ID: {call}")

        calls.append(
            AuctionCallView(
                seat=Seat(bidder).name,  # type: ignore[arg-type]
                label=action_label(call),
                action_id=call,
            )
        )
        bidder = (bidder + 1) % 4

    if not finished:
        return _AuctionTrace(
            calls=calls,
            finished=False,
            passed_out=False,
            declarer=None,
            dummy=None,
            trump_denom=None,
            contract_view=None,
        )

    if passed_out:
        return _AuctionTrace(
            calls=calls,
            finished=True,
            passed_out=True,
            declarer=None,
            dummy=None,
            trump_denom=None,
            contract_view=ContractView(
                level=None,
                trump=None,
                doubled="",
                declarer=None,
                dummy=None,
                formatted="Passed Out",
            ),
        )

    assert last_contract_bid is not None and last_bidder is not None
    offset = last_contract_bid - FIRST_CONTRACT_BID_ACTION
    contract_level = 1 + offset // 5
    trump_denom = offset % 5
    trump_str = DENOMINATION_CHARS[trump_denom]
    declarer_seat = Seat(first_denom_bidder[(last_bidder & 1, trump_denom)])
    dummy_seat = declarer_seat.partner
    doubled_suffix: Literal["", "X", "XX"] = (
        "XX"
        if doubled_state == "redoubled"
        else "X"
        if doubled_state == "doubled"
        else ""
    )
    contract_view = ContractView(
        level=contract_level,
        trump=trump_str,  # type: ignore[arg-type]
        doubled=doubled_suffix,
        declarer=declarer_seat.name,  # type: ignore[arg-type]
        dummy=dummy_seat.name,  # type: ignore[arg-type]
        formatted=f"{contract_level}{trump_str}{doubled_suffix} by {declarer_seat.name}",
    )
    return _AuctionTrace(
        calls=calls,
        finished=True,
        passed_out=False,
        declarer=declarer_seat,
        dummy=dummy_seat,
        trump_denom=trump_denom,
        contract_view=contract_view,
    )


def _resolve_auction(
    dealer: int,
    auction_actions: list[int],
) -> tuple[bool, bool, int | None, int | None]:
    trace = _trace_auction(dealer, auction_actions)
    return (
        trace.finished,
        trace.passed_out,
        int(trace.declarer) if trace.declarer is not None else None,
        trace.trump_denom,
    )


def _build_imperfect_info_dealt_state(
    *,
    observer: Seat,
    board_number: int,
    own_cards: list[int],
    dummy_cards: list[int] | None,
    actions: list[int],
    seed: int,
) -> pyspiel.State:
    dealer, dealer_vul, non_dealer_vul = wbf_board_params(board_number)

    seen_play = False
    auction_actions: list[int] = []
    play_actions: list[int] = []
    for idx, act in enumerate(actions):
        if act >= BIDDING_ACTION_BASE:
            if seen_play:
                raise ValueError(
                    f"Bidding action {action_label(act)} at index {idx} cannot appear after card play started"
                )
            auction_actions.append(act)
        else:
            seen_play = True
            play_actions.append(act)

    finished, passed_out, declarer, trump_denom = _resolve_auction(
        dealer, auction_actions
    )
    if passed_out:
        raise ValueError("Cannot step on a terminal passed-out board")

    rng = random.Random(seed)
    own_set = set(own_cards)
    hands: dict[int, list[int]] = {int(observer): sorted(own_cards)}

    if not play_actions:
        if dummy_cards is not None:
            raise ValueError(
                "dummy_hand must not be provided before the opening lead is played"
            )
        unseen = [c for c in range(NUM_CARDS) if c not in own_set]
        rng.shuffle(unseen)
        other_seats = [p for p in range(4) if p != int(observer)]
        for offset, seat_idx in enumerate(other_seats):
            hands[seat_idx] = sorted(unseen[offset * 13 : (offset + 1) * 13])
    else:
        if not finished or declarer is None:
            raise ValueError(
                "Card play actions provided before auction completed with a contract"
            )
        dummy = (declarer + 2) % 4
        if int(observer) == dummy:
            raise ValueError(
                f"Dummy seat {Seat(dummy).name} never acts directly; "
                f"submit declarer {Seat(declarer).name} when playing from dummy"
            )
        if dummy_cards is None:
            raise ValueError(
                "dummy_hand is required once the opening lead has been played"
            )
        if len(play_actions) >= NUM_CARDS:
            raise ValueError("Cannot step on a terminal board")

        dummy_set = set(dummy_cards)
        hands[dummy] = sorted(dummy_cards)

        hidden_seats = [p for p in range(4) if p not in (int(observer), dummy)]
        h1, h2 = hidden_seats
        hidden_pool = set(range(NUM_CARDS)) - own_set - dummy_set

        remaining_known: dict[int, set[int]] = {
            int(observer): set(own_set),
            dummy: set(dummy_set),
        }
        must_hold: dict[int, set[int]] = {h1: set(), h2: set()}
        forbidden_suits: dict[int, set[Suit]] = {h1: set(), h2: set()}
        played_so_far: set[int] = set()

        leader = (declarer + 1) % 4
        for trick_start in range(0, len(play_actions), 4):
            trick_cards = play_actions[trick_start : trick_start + 4]
            led_suit = card_suit(trick_cards[0])
            trick_plays: list[tuple[Seat, int]] = []
            for offset, card in enumerate(trick_cards):
                hand_seat = (leader + offset) % 4
                if card in played_so_far:
                    raise ValueError(
                        f"Duplicate played card {action_label(card)} in action_history"
                    )
                played_so_far.add(card)
                c_suit = card_suit(card)

                if hand_seat in remaining_known:
                    if card not in remaining_known[hand_seat]:
                        raise ValueError(
                            f"Card {action_label(card)} played by {Seat(hand_seat).name} "
                            "is not present in that seat's hand"
                        )
                    if offset > 0 and c_suit != led_suit and any(
                        card_suit(rem) == led_suit for rem in remaining_known[hand_seat]
                    ):
                        raise ValueError(
                            f"Illegal revoke by {Seat(hand_seat).name}: failed to follow "
                            f"{led_suit.name} when playing {action_label(card)}"
                        )
                    remaining_known[hand_seat].remove(card)
                else:
                    if card not in hidden_pool:
                        raise ValueError(
                            f"Card {action_label(card)} played by {Seat(hand_seat).name} "
                            "conflicts with own_hand or dummy_hand"
                        )
                    if c_suit in forbidden_suits[hand_seat]:
                        raise ValueError(
                            f"Illegal revoke by {Seat(hand_seat).name}: played {action_label(card)} "
                            f"after showing out of {c_suit.name}"
                        )
                    must_hold[hand_seat].add(card)
                    if offset > 0 and c_suit != led_suit:
                        forbidden_suits[hand_seat].add(led_suit)

                trick_plays.append((Seat(hand_seat), card))

            if len(trick_plays) == 4:
                leader = int(trick_winner(trick_plays, trump_denom))

        if len(must_hold[h1]) > 13 or len(must_hold[h2]) > 13:
            raise ValueError("Hidden seat played more than 13 cards in action_history")

        unplayed_hidden = sorted(hidden_pool - must_hold[h1] - must_hold[h2])
        forced_h1: list[int] = []
        forced_h2: list[int] = []
        free_cards: list[int] = []

        for card in unplayed_hidden:
            suit = card_suit(card)
            forbid_1 = suit in forbidden_suits[h1]
            forbid_2 = suit in forbidden_suits[h2]
            if forbid_1 and forbid_2:
                raise ValueError(
                    f"Inconsistent action_history: unplayed card {action_label(card)} remains "
                    f"in {suit.name} after both hidden seats showed out"
                )
            if forbid_1:
                forced_h2.append(card)
            elif forbid_2:
                forced_h1.append(card)
            else:
                free_cards.append(card)

        need_h1 = 13 - len(must_hold[h1])
        need_h2 = 13 - len(must_hold[h2])
        if len(forced_h1) > need_h1 or len(forced_h2) > need_h2:
            raise ValueError(
                "Inconsistent action_history: hidden-seat showout constraints exceed 13-card capacity"
            )

        rng.shuffle(free_cards)
        take_h1 = need_h1 - len(forced_h1)
        hands[h1] = sorted([*must_hold[h1], *forced_h1, *free_cards[:take_h1]])
        hands[h2] = sorted([*must_hold[h2], *forced_h2, *free_cards[take_h1:]])

    board_game = pyspiel.load_game(
        "bridge",
        {
            "dealer": dealer,
            "dealer_vul": dealer_vul,
            "non_dealer_vul": non_dealer_vul,
            "use_double_dummy_result": False,
        },
    )
    board_state = board_game.new_initial_state()
    for round_idx in range(13):
        for player_idx in range(4):
            board_state.apply_action(hands[player_idx][round_idx])
    return board_state


def reconstruct_step_state(
    req: StepRequest,
) -> tuple[pyspiel.State, dict[Seat, set[Suit]] | None, bool]:
    """Reconstruct the OpenSpiel state and `prior_strategy` tracking state for `StepRequest`."""
    observer = Seat[req.seat]
    actions = [parse_action_token(tok) for tok in req.action_history]

    if req.tournament_seed is not None:
        runner = DuplicateMatchRunner(
            tournament_seed=req.tournament_seed,
            use_double_dummy_result=False,
        )
        board_state = runner._new_dealt_board_state(req.board_number)
    else:
        assert req.own_hand is not None
        own_cards = [parse_card_token(tok) for tok in req.own_hand]
        dummy_cards = (
            [parse_card_token(tok) for tok in req.dummy_hand]
            if req.dummy_hand is not None
            else None
        )
        board_state = _build_imperfect_info_dealt_state(
            observer=observer,
            board_number=req.board_number,
            own_cards=own_cards,
            dummy_cards=dummy_cards,
            actions=actions,
            seed=req.seed,
        )

    tracker = RuleFactTracker()
    last_decision_voids: dict[Seat, set[Suit]] | None = None
    had_pre_lead_play_decision = False
    had_post_lead_play_decision = False
    played_cards_so_far: set[int] = set()

    for step_idx, action in enumerate(actions):
        if board_state.is_terminal():
            raise ValueError(
                f"Board reached terminal state before action index {step_idx} ({action_label(action)})"
            )
        legal = board_state.legal_actions()
        if action not in legal:
            acting_name = Seat(board_state.current_player()).name
            raise ValueError(
                f"Illegal action {action_label(action)} ({action}) at history index "
                f"{step_idx} for seat {acting_name}"
            )
        if req.prior_strategy is not None and board_state.current_player() == int(observer):
            candidates = collapse_equivalent_actions(
                legal,
                played_cards=played_cards_so_far,
            )
            if len(candidates) > 1:
                snapshot = tracker.extract_facts(board_state, observer)
                last_decision_voids = {
                    seat_k: set(voids) for seat_k, voids in snapshot.known_voids.items()
                }
                if snapshot.phase == "play":
                    if snapshot.dummy_seat is not None:
                        had_post_lead_play_decision = True
                    else:
                        had_pre_lead_play_decision = True

        board_state.apply_action(action)
        if action < BIDDING_ACTION_BASE:
            played_cards_so_far.add(action)

    if board_state.is_terminal():
        raise ValueError("Cannot step on a terminal board")

    current_player = board_state.current_player()
    if current_player != int(observer):
        raise ValueError(
            f"Expected turn for seat {observer.name}, but current player is {Seat(current_player).name}"
        )

    if req.prior_strategy is None:
        return board_state, None, False

    if board_state.current_phase() == 1 and req.prior_strategy.phase == "play":
        raise ValueError(
            "prior_strategy with phase='play' cannot be used during the auction phase"
        )

    if req.prior_strategy.phase == "play":
        if had_post_lead_play_decision:
            return board_state, last_decision_voids, True
        if had_pre_lead_play_decision:
            return board_state, last_decision_voids, False
        dummy_visible = board_state.current_phase() == 2 and len(played_cards_so_far) >= 1
        return board_state, {seat_k: set() for seat_k in Seat}, dummy_visible

    prev_voids = (
        last_decision_voids
        if last_decision_voids is not None
        else {seat_k: set() for seat_k in Seat}
    )
    return board_state, prev_voids, False


def _sort_bridge_hand(card_ids: Iterable[int]) -> list[int]:
    """Sort card IDs in standard Bridge display order: Spades -> Hearts -> Diamonds -> Clubs, A -> 2."""
    return sorted(card_ids, key=lambda cid: (-int(card_suit(cid)), -card_rank(cid)))


def compute_table_state(req: TableStateRequest) -> TableStateResponse:
    """Compute ground-truth OpenSpiel Bridge table state with strict imperfect-information visibility."""
    runner = DuplicateMatchRunner(
        tournament_seed=req.tournament_seed,
        use_double_dummy_result=False,
    )
    board_state = runner._new_dealt_board_state(req.board_number)

    dealer_idx, dealer_vul, non_dealer_vul = wbf_board_params(req.board_number)
    dealer_seat = Seat(dealer_idx)
    dealer_is_ew = (dealer_idx & 1) == 1
    ns_vulnerable = non_dealer_vul if dealer_is_ew else dealer_vul
    ew_vulnerable = dealer_vul if dealer_is_ew else non_dealer_vul

    initial_dealt_ids: dict[Seat, list[int]] = {s: [] for s in Seat}
    for idx, entry in enumerate(board_state.full_history()[:NUM_CARDS]):
        initial_dealt_ids[Seat(idx % 4)].append(entry.action)

    remaining_ids: dict[Seat, set[int]] = {
        s: set(initial_dealt_ids[s]) for s in Seat
    }

    auction_actions: list[int] = []
    auction_trace: _AuctionTrace | None = None

    trick_leader: Seat | None = None
    current_trick_plays: list[tuple[Seat, int]] = []
    completed_tricks: list[CompletedTrickView] = []
    ns_tricks = 0
    ew_tricks = 0
    total_play_actions = 0

    for step_idx, raw_tok in enumerate(req.action_history):
        act = parse_action_token(raw_tok)
        if board_state.is_terminal():
            raise ValueError(
                f"Board reached terminal state before action index {step_idx} ({action_label(act)})"
            )
        legal = board_state.legal_actions()
        if act not in legal:
            acting_name = Seat(board_state.current_player()).name
            raise ValueError(
                f"Illegal action {action_label(act)} ({act}) at history index "
                f"{step_idx} for seat {acting_name}"
            )

        if board_state.current_phase() == 1:
            auction_actions.append(act)
            board_state.apply_action(act)

            if board_state.current_phase() == 2 and auction_trace is None:
                auction_trace = _trace_auction(dealer_idx, auction_actions)
                assert auction_trace.declarer is not None
                trick_leader = Seat((int(auction_trace.declarer) + 1) % 4)
        else:
            assert trick_leader is not None and auction_trace is not None
            hand_seat = Seat((int(trick_leader) + len(current_trick_plays)) % 4)
            if act not in remaining_ids[hand_seat]:
                raise ValueError(
                    f"Card {action_label(act)} played at history index {step_idx} "
                    f"is not in {hand_seat.name}'s hand"
                )
            remaining_ids[hand_seat].remove(act)
            current_trick_plays.append((hand_seat, act))
            total_play_actions += 1
            board_state.apply_action(act)

            if len(current_trick_plays) == 4:
                winner = trick_winner(current_trick_plays, auction_trace.trump_denom)
                if winner in (Seat.NORTH, Seat.SOUTH):
                    ns_tricks += 1
                else:
                    ew_tricks += 1
                completed_tricks.append(
                    CompletedTrickView(
                        trick_number=len(completed_tricks) + 1,
                        leader=trick_leader.name,  # type: ignore[arg-type]
                        winner=winner.name,  # type: ignore[arg-type]
                        cards=[
                            TrickCardView(
                                seat=seat_p.name,  # type: ignore[arg-type]
                                card=action_label(card_c),
                                action_id=card_c,
                            )
                            for seat_p, card_c in current_trick_plays
                        ],
                    )
                )
                trick_leader = winner
                current_trick_plays = []

    if auction_trace is None:
        auction_trace = _trace_auction(dealer_idx, auction_actions)

    contract_view = auction_trace.contract_view
    dummy_seat = auction_trace.dummy

    if board_state.is_terminal():
        phase: Literal["auction", "play", "terminal"] = "terminal"
        show_all = True
        current_player: SeatName | None = None
        acting_hand: SeatName | None = None
        legal_actions_view: list[LegalActionView] = []
        assert contract_view is not None
        returns = board_state.returns()
        terminal_result: TerminalResultView | None = TerminalResultView(
            ns_score=round(returns[0]),
            ew_score=round(returns[1]),
            contract=contract_view.formatted,
            ns_tricks=ns_tricks,
            ew_tricks=ew_tricks,
        )
    elif board_state.current_phase() == 1:
        phase = "auction"
        show_all = req.reveal_all
        current_player = Seat(board_state.current_player()).name  # type: ignore[assignment]
        acting_hand = current_player
        legal_actions_view = [
            LegalActionView(action_id=a, label=action_label(a))
            for a in board_state.legal_actions()
        ]
        terminal_result = None
    else:
        assert trick_leader is not None
        phase = "play"
        show_all = req.reveal_all
        current_player = Seat(board_state.current_player()).name  # type: ignore[assignment]
        acting_hand = Seat(
            (int(trick_leader) + len(current_trick_plays)) % 4
        ).name  # type: ignore[assignment]
        legal_actions_view = (
            [
                LegalActionView(action_id=a, label=action_label(a))
                for a in board_state.legal_actions()
            ]
            if (show_all or current_player == req.viewer_seat)
            else []
        )
        terminal_result = None

    viewer = Seat[req.viewer_seat]
    visible_seats: set[Seat] = set(Seat) if show_all else {viewer}
    if dummy_seat is not None and total_play_actions >= 1:
        visible_seats.add(dummy_seat)

    hands: dict[SeatName, list[str] | None] = {}
    hand_hcp_map: dict[SeatName, int | None] = {}
    remaining_counts: dict[SeatName, int] = {}
    for seat_item in Seat:
        s_name: SeatName = seat_item.name  # type: ignore[assignment]
        remaining_counts[s_name] = len(remaining_ids[seat_item])
        if seat_item in visible_seats:
            hands[s_name] = [
                action_label(cid) for cid in _sort_bridge_hand(remaining_ids[seat_item])
            ]
            hand_hcp_map[s_name] = hand_hcp(initial_dealt_ids[seat_item])
        else:
            hands[s_name] = None
            hand_hcp_map[s_name] = None

    initial_hands: dict[SeatName, list[str]] | None = (
        {
            seat_item.name: [  # type: ignore[misc]
                action_label(cid) for cid in _sort_bridge_hand(initial_dealt_ids[seat_item])
            ]
            for seat_item in Seat
        }
        if show_all
        else None
    )

    current_trick_view = [
        TrickCardView(
            seat=seat_p.name,  # type: ignore[arg-type]
            card=action_label(card_c),
            action_id=card_c,
        )
        for seat_p, card_c in current_trick_plays
    ]

    return TableStateResponse(
        tournament_seed=req.tournament_seed,
        board_number=req.board_number,
        dealer=dealer_seat.name,  # type: ignore[arg-type]
        ns_vulnerable=ns_vulnerable,
        ew_vulnerable=ew_vulnerable,
        phase=phase,
        viewer_seat=req.viewer_seat,
        reveal_all=req.reveal_all,
        current_player=current_player,
        acting_hand=acting_hand,
        hands=hands,
        hand_hcp=hand_hcp_map,
        remaining_counts=remaining_counts,
        initial_hands=initial_hands,
        auction_calls=auction_trace.calls,
        contract=contract_view,
        ns_tricks=ns_tricks,
        ew_tricks=ew_tricks,
        current_trick=current_trick_view,
        completed_tricks=completed_tricks,
        legal_actions=legal_actions_view,
        terminal_result=terminal_result,
    )


def _profile_cache_key(profile: ModelProfile) -> tuple[Any, ...]:
    return (
        profile.mode,
        profile.project,
        profile.strategist_location,
        profile.strategist_model,
        profile.evaluator_backend,
        profile.evaluator_location,
        profile.evaluator_model,
        profile.evaluator_endpoint_id,
        profile.decisions_auth,
        profile.decisions_violation_threshold,
    )


def create_app(
    default_profile: ModelProfile | None = None,
    default_dds_worlds: int | None = None,
) -> FastAPI:
    """Create the stateless FastAPI service for Tenace on Google Cloud Run."""
    resolved_profile = (
        default_profile if default_profile is not None else default_profile_from_env()
    )
    if default_dds_worlds is None:
        resolved_dds_worlds = default_dds_worlds_from_env()
    else:
        if (
            isinstance(default_dds_worlds, bool)
            or not isinstance(default_dds_worlds, int)
            or default_dds_worlds < 1
        ):
            raise ValueError(
                f"default_dds_worlds must be an integer >= 1, got {default_dds_worlds!r}"
            )
        resolved_dds_worlds = default_dds_worlds

    client_cache_lock = threading.Lock()
    client_cache: OrderedDict[
        tuple[Any, ...], tuple[StructuredModelClient, StructuredModelClient]
    ] = OrderedDict()

    def get_cached_clients(
        profile: ModelProfile,
        *,
        is_caller_override: bool,
    ) -> tuple[StructuredModelClient, StructuredModelClient]:
        key = _profile_cache_key(profile)
        with client_cache_lock:
            cached = client_cache.get(key)
            if cached is not None:
                client_cache.move_to_end(key)
                return cached
            try:
                built = build_clients_for_profile(profile)
            except ValueError as exc:
                status_code = 422 if is_caller_override else 500
                raise HTTPException(status_code=status_code, detail=str(exc)) from exc
            client_cache[key] = built
            client_cache.move_to_end(key)
            while len(client_cache) > _MAX_CACHED_PROFILES:
                client_cache.popitem(last=False)
            return built

    app = FastAPI(
        title="Tenace Bridge Service",
        version=SERVICE_VERSION,
        description="Stateless 3-Layer Neuro-Symbolic Contract Bridge HTTP API.",
    )
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index_page() -> FileResponse:
        return FileResponse(
            WEB_DIR / "index.html",
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/health")
    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "tenace",
            "version": SERVICE_VERSION,
            "default_profile": resolved_profile.short_label(),
            "mode": resolved_profile.mode,
            "dds_worlds": resolved_dds_worlds,
        }

    @app.post("/v1/table/state", response_model=TableStateResponse)
    def table_state_endpoint(req: TableStateRequest) -> TableStateResponse:
        try:
            return compute_table_state(req)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/step", response_model=StepResponse)
    def step_endpoint(req: StepRequest) -> StepResponse:
        try:
            state, prev_voids, dummy_seen = reconstruct_step_state(req)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        observer = Seat[req.seat]
        phase: Literal["auction", "play"] = (
            "auction" if state.current_phase() == 1 else "play"
        )

        if req.bot == "baseline":
            chosen_action = FirstLegalPassBaselineBot(observer).step(state)
            baseline_strategy = (
                req.prior_strategy
                if (req.prior_strategy is not None and req.prior_strategy.phase == phase)
                else None
            )
            return StepResponse(
                action_id=chosen_action,
                action_label=action_label(chosen_action),
                seat=req.seat,
                phase=phase,
                strategy=baseline_strategy,
                telemetry={
                    "strategist_calls": 0,
                    "evaluator_calls": 0,
                    "forced_moves": 1 if len(state.legal_actions()) == 1 else 0,
                },
            )

        active_profile = req.profile if req.profile is not None else resolved_profile
        active_dds_worlds = (
            req.dds_worlds if req.dds_worlds is not None else resolved_dds_worlds
        )
        strategist_client, evaluator_client = get_cached_clients(
            active_profile,
            is_caller_override=req.profile is not None,
        )
        agent = TenaceAgent(
            seat=observer,
            strategist_client=strategist_client,
            evaluator_client=evaluator_client,
            dds_evaluator=RuleDDSEvaluator(
                num_worlds=active_dds_worlds,
                rng_seed=req.seed,
            ),
            convention_card=SAYC_CONVENTION_CARD,
        )
        if req.prior_strategy is not None:
            agent.strategy = req.prior_strategy
            agent.strategy_history = [req.prior_strategy]
            agent._prev_voids = prev_voids
            agent._dummy_seen = dummy_seen

        try:
            chosen_action = agent.step(state)
        except _UPSTREAM_MODEL_ERRORS as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream model evaluation error: {exc}",
            ) from exc

        return StepResponse(
            action_id=chosen_action,
            action_label=action_label(chosen_action),
            seat=req.seat,
            phase=phase,
            strategy=agent.strategy,
            telemetry={
                "strategist_calls": agent.strategist_calls,
                "evaluator_calls": agent.evaluator_calls,
                "forced_moves": agent.forced_moves,
            },
        )

    @app.post("/v1/match", response_model=MatchResponse)
    def match_endpoint(req: MatchRequest) -> MatchResponse:
        active_dds_worlds = (
            req.dds_worlds if req.dds_worlds is not None else resolved_dds_worlds
        )
        profile_a = (
            req.team_a_profile
            if req.team_a_profile is not None
            else resolved_profile
        )
        strategist_a, evaluator_a = get_cached_clients(
            profile_a,
            is_caller_override=req.team_a_profile is not None,
        )

        def team_a_factory(seat: Seat) -> BridgeSeatBot:
            return TenaceAgent(
                seat=seat,
                strategist_client=strategist_a,
                evaluator_client=evaluator_a,
                dds_evaluator=RuleDDSEvaluator(
                    num_worlds=active_dds_worlds,
                    rng_seed=(req.seed * 31) + int(seat),
                ),
                convention_card=SAYC_CONVENTION_CARD,
            )

        if req.opponent == "tenace":
            profile_b = (
                req.team_b_profile if req.team_b_profile is not None else profile_a
            )
            team_a_name = "Tenace-A"
            team_b_name = "Tenace-B"
            team_b_profile_label: str | None = profile_b.short_label()
            strategist_b, evaluator_b = get_cached_clients(
                profile_b,
                is_caller_override=req.team_b_profile is not None,
            )

            def team_b_factory(seat: Seat) -> BridgeSeatBot:
                return TenaceAgent(
                    seat=seat,
                    strategist_client=strategist_b,
                    evaluator_client=evaluator_b,
                    dds_evaluator=RuleDDSEvaluator(
                        num_worlds=active_dds_worlds,
                        rng_seed=(req.seed * 37) + int(seat) + 100,
                    ),
                    convention_card=SAYC_CONVENTION_CARD,
                )
        else:
            team_a_name = "Tenace"
            team_b_name = "Baseline"
            team_b_profile_label = None
            team_b_factory = FirstLegalPassBaselineBot

        runner = DuplicateMatchRunner(
            tournament_seed=req.seed,
            use_double_dummy_result=req.double_dummy,
        )
        try:
            summary = runner.play_match(
                team_a_factory=team_a_factory,
                team_b_factory=team_b_factory,
                num_boards=req.boards,
            )
        except _UPSTREAM_MODEL_ERRORS as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream model evaluation error: {exc}",
            ) from exc

        report = format_match_report(
            summary,
            team_a_name=team_a_name,
            team_b_name=team_b_name,
            team_a_profile=profile_a.short_label(),
            team_b_profile=team_b_profile_label,
        )

        return MatchResponse(
            team_a_name=team_a_name,
            team_b_name=team_b_name,
            team_a_profile=profile_a.short_label(),
            team_b_profile=team_b_profile_label,
            summary=summary,
            report=report,
        )

    return app


def main() -> None:
    host, port = resolve_server_bind()
    uvicorn.run("tenace.server:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    main()
