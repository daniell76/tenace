from __future__ import annotations

from collections.abc import Callable
import random
from typing import Protocol, runtime_checkable

import pyspiel
from pydantic import BaseModel, ConfigDict, Field

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
    action_label,
)
from tenace.harness.scoring import score_diff_to_imps
from tenace.layer1_strategist import format_auction_sequence
from tenace.layer3_rules.tracker import _trick_winner

__all__ = [
    "BoardResult",
    "BridgeSeatBot",
    "DuplicateMatchRunner",
    "FirstLegalPassBaselineBot",
    "MatchSummary",
    "RoomTranscript",
]


@runtime_checkable
class BridgeSeatBot(Protocol):
    def reset_for_board(self) -> None: ...

    def step(self, state: pyspiel.State) -> int: ...


class FirstLegalPassBaselineBot:
    """Deterministic baseline bot: passes whenever legal in auction, otherwise plays first legal action."""

    def __init__(self, seat: Seat) -> None:
        self.seat = Seat(seat)

    def reset_for_board(self) -> None:
        return None

    def step(self, state: pyspiel.State) -> int:
        if state.is_terminal():
            raise ValueError("Cannot step on a terminal bridge state")
        if state.is_chance_node() or state.current_phase() == 0:
            raise ValueError("Cannot step during incomplete deal phase (chance node)")

        current_player = state.current_player()
        if current_player != int(self.seat):
            raise ValueError(
                f"Baseline bot seat {self.seat.name} ({int(self.seat)}) called on "
                f"seat {current_player}'s turn"
            )

        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions available")
        if PASS_ACTION in legal:
            return PASS_ACTION
        return legal[0]


class RoomTranscript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: str
    auction: str
    tricks: list[str] = Field(default_factory=list)
    ns_score: float
    strategy_traces: dict[str, list[str]] = Field(default_factory=dict)


class BoardResult(BaseModel):
    board_number: int = Field(ge=1)
    open_room_ns_score: float
    closed_room_ns_score: float
    imps_team_a: int = Field(ge=-24, le=24)
    open_room: RoomTranscript | None = None
    closed_room: RoomTranscript | None = None


class MatchSummary(BaseModel):
    tournament_seed: int
    boards: list[BoardResult] = Field(default_factory=list)
    total_imps_team_a: int = 0
    total_imps_team_b: int = 0
    telemetry: dict[str, int] = Field(default_factory=dict)


def _build_room_transcript(
    state: pyspiel.State,
    seats: dict[Seat, BridgeSeatBot],
    ns_score: float,
) -> RoomTranscript:
    public_history = state.full_history()[NUM_CARDS:]
    auction_calls: list[tuple[Seat, int]] = []
    play_actions: list[int] = []
    last_bid_seat: Seat | None = None
    last_bid_action: int | None = None
    first_denom_bidder: dict[tuple[int, int], Seat] = {}

    for entry in public_history:
        action = entry.action
        if action >= BIDDING_ACTION_BASE:
            seat = Seat(entry.player)
            auction_calls.append((seat, action))
            if FIRST_CONTRACT_BID_ACTION <= action <= MAX_ACTION:
                denom = (action - FIRST_CONTRACT_BID_ACTION) % 5
                key = (seat.partnership, denom)
                if key not in first_denom_bidder:
                    first_denom_bidder[key] = seat
                last_bid_seat = seat
                last_bid_action = action
        else:
            play_actions.append(action)

    auction_str = format_auction_sequence(auction_calls)

    if last_bid_seat is None or last_bid_action is None:
        contract_str = "Passed Out"
        declarer: Seat | None = None
        trump_denom: int | None = None
    else:
        offset = last_bid_action - FIRST_CONTRACT_BID_ACTION
        contract_level = 1 + offset // 5
        trump_denom = offset % 5
        declarer = first_denom_bidder[(last_bid_seat.partnership, trump_denom)]
        denom_str = DENOMINATION_CHARS[trump_denom]
        doubled_suffix = ""
        for _, call in reversed(auction_calls):
            if call == RDBL_ACTION:
                doubled_suffix = "XX"
                break
            if call == DBL_ACTION:
                doubled_suffix = "X"
                break
            if call >= FIRST_CONTRACT_BID_ACTION:
                break
        contract_str = f"{contract_level}{denom_str}{doubled_suffix} by {declarer.name}"

    tricks: list[str] = []
    if declarer is not None and play_actions:
        leader = Seat((int(declarer) + 1) % 4)
        for trick_idx, trick_start in enumerate(range(0, len(play_actions), 4), start=1):
            trick_cards = play_actions[trick_start : trick_start + 4]
            trick_plays: list[tuple[Seat, int]] = [
                (Seat((int(leader) + offset_idx) % 4), card)
                for offset_idx, card in enumerate(trick_cards)
            ]
            cards_fmt = " ".join(
                f"{seat_p.name[0]}:{action_label(card)}" for seat_p, card in trick_plays
            )
            if len(trick_plays) == 4:
                winner = _trick_winner(trick_plays, trump_denom)
                tricks.append(f"Trick {trick_idx}: {cards_fmt} -> {winner.name}")
                leader = winner
            else:
                tricks.append(f"Trick {trick_idx}: {cards_fmt}")

    strategy_traces: dict[str, list[str]] = {}
    for seat, bot in seats.items():
        history = getattr(bot, "strategy_history", None)
        if history:
            strategy_traces[seat.name] = [
                f"rev {s.revision} ({s.phase}): {s.plan_summary}" for s in history
            ]

    return RoomTranscript(
        contract=contract_str,
        auction=auction_str,
        tricks=tricks,
        ns_score=ns_score,
        strategy_traces=strategy_traces,
    )


def _wbf_board_params(board_number: int) -> tuple[int, bool, bool]:
    """Compute official WBF 16-board dealer & vulnerability flags matching OpenSpiel."""
    zero_idx = board_number - 1
    dealer = zero_idx % 4
    vul_code = (zero_idx + zero_idx // 4) % 4  # 0: None, 1: N/S, 2: E/W, 3: All
    ns_vul = vul_code in (1, 3)
    ew_vul = vul_code in (2, 3)
    dealer_is_ew = (dealer & 1) == 1
    dealer_vul = ew_vul if dealer_is_ew else ns_vul
    non_dealer_vul = ns_vul if dealer_is_ew else ew_vul
    return dealer, dealer_vul, non_dealer_vul


class DuplicateMatchRunner:
    """Two-room WBF Duplicate IMP match harness (Open Room & Closed Room)."""

    def __init__(
        self,
        tournament_seed: int = 42,
        use_double_dummy_result: bool = False,
    ) -> None:
        if isinstance(tournament_seed, bool) or not isinstance(tournament_seed, int):
            raise ValueError(f"tournament_seed must be an integer, got {tournament_seed!r}")
        if not isinstance(use_double_dummy_result, bool):
            raise ValueError("use_double_dummy_result must be a bool")

        self.tournament_seed = tournament_seed
        self.use_double_dummy_result = use_double_dummy_result
        self._deal_game = pyspiel.load_game("bridge(use_double_dummy_result=false)")

    def _new_dealt_board_state(self, board_number: int) -> pyspiel.State:
        state = self._deal_game.new_initial_state(
            tournament_seed=self.tournament_seed,
            board_number=board_number,
        )
        if state.is_chance_node():
            deal_rng = random.Random((self.tournament_seed << 16) ^ board_number)
            while state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes, strict=True)
                state.apply_action(deal_rng.choices(actions, weights=probs, k=1)[0])

        if not self.use_double_dummy_result:
            return state

        # Note: OpenSpiel's BridgeGame.new_initial_state(tournament_seed, board_number)
        # constructs an internal subgame with default use_double_dummy_result=false unless
        # replayed on an explicit parameterized bridge game.
        dealer, dealer_vul, non_dealer_vul = _wbf_board_params(board_number)
        dd_game = pyspiel.load_game(
            "bridge",
            {
                "dealer": dealer,
                "dealer_vul": dealer_vul,
                "non_dealer_vul": non_dealer_vul,
                "use_double_dummy_result": True,
            },
        )
        dd_state = dd_game.new_initial_state()
        for entry in state.full_history()[:52]:
            dd_state.apply_action(entry.action)
        return dd_state

    def _play_room(
        self,
        state: pyspiel.State,
        seats: dict[Seat, BridgeSeatBot],
    ) -> float:
        for bot in seats.values():
            bot.reset_for_board()

        while not state.is_terminal():
            acting_seat = Seat(state.current_player())
            legal_actions = state.legal_actions()
            action = seats[acting_seat].step(state.clone())
            if (
                isinstance(action, bool)
                or not isinstance(action, int)
                or action not in legal_actions
            ):
                raise ValueError(
                    f"Illegal action {action!r} returned by {acting_seat.name}; "
                    f"legal actions: {legal_actions}"
                )
            state.apply_action(action)

        return float(state.returns()[int(Seat.NORTH)])

    def play_match(
        self,
        team_a_factory: Callable[[Seat], BridgeSeatBot],
        team_b_factory: Callable[[Seat], BridgeSeatBot],
        num_boards: int = 4,
    ) -> MatchSummary:
        if isinstance(num_boards, bool) or not isinstance(num_boards, int) or num_boards < 1:
            raise ValueError(f"num_boards must be an integer >= 1, got {num_boards!r}")

        # Separate per-room bot instances prevent any cross-room or cross-seat memory leakage:
        # Open Room: Team A sits North/South, Team B sits East/West
        # Closed Room: Team B sits North/South, Team A sits East/West
        open_room_seats: dict[Seat, BridgeSeatBot] = {}
        closed_room_seats: dict[Seat, BridgeSeatBot] = {}
        body_exc: BaseException | None = None

        try:
            open_room_seats[Seat.NORTH] = team_a_factory(Seat.NORTH)
            open_room_seats[Seat.SOUTH] = team_a_factory(Seat.SOUTH)
            open_room_seats[Seat.EAST] = team_b_factory(Seat.EAST)
            open_room_seats[Seat.WEST] = team_b_factory(Seat.WEST)

            closed_room_seats[Seat.NORTH] = team_b_factory(Seat.NORTH)
            closed_room_seats[Seat.SOUTH] = team_b_factory(Seat.SOUTH)
            closed_room_seats[Seat.EAST] = team_a_factory(Seat.EAST)
            closed_room_seats[Seat.WEST] = team_a_factory(Seat.WEST)

            board_results: list[BoardResult] = []
            total_imps_a = 0

            for board_num in range(1, num_boards + 1):
                open_state = self._new_dealt_board_state(board_num)
                closed_state = self._new_dealt_board_state(board_num)

                open_ns = self._play_room(open_state, open_room_seats)
                open_transcript = _build_room_transcript(
                    open_state, open_room_seats, open_ns
                )

                closed_ns = self._play_room(closed_state, closed_room_seats)
                closed_transcript = _build_room_transcript(
                    closed_state, closed_room_seats, closed_ns
                )

                imps_a = score_diff_to_imps(open_ns - closed_ns)
                total_imps_a += imps_a
                board_results.append(
                    BoardResult(
                        board_number=board_num,
                        open_room_ns_score=open_ns,
                        closed_room_ns_score=closed_ns,
                        imps_team_a=imps_a,
                        open_room=open_transcript,
                        closed_room=closed_transcript,
                    )
                )

            all_bots = list(open_room_seats.values()) + list(closed_room_seats.values())
            telemetry = {
                "forced_moves": sum(int(getattr(bot, "forced_moves", 0)) for bot in all_bots),
                "strategist_calls": sum(
                    int(getattr(bot, "strategist_calls", 0)) for bot in all_bots
                ),
                "evaluator_calls": sum(
                    int(getattr(bot, "evaluator_calls", 0)) for bot in all_bots
                ),
            }

            return MatchSummary(
                tournament_seed=self.tournament_seed,
                boards=board_results,
                total_imps_team_a=total_imps_a,
                total_imps_team_b=-total_imps_a,
                telemetry=telemetry,
            )
        except BaseException as exc:
            body_exc = exc
            raise
        finally:
            all_created_bots = list(open_room_seats.values()) + list(
                closed_room_seats.values()
            )
            close_errors: list[Exception] = []
            for bot in all_created_bots:
                close_fn = getattr(bot, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception as close_exc:
                        close_errors.append(close_exc)
            if close_errors and body_exc is None:
                first_err = close_errors[0]
                for extra_err in close_errors[1:]:
                    first_err.add_note(f"Additional close() error: {extra_err!r}")
                raise first_err
            if close_errors and body_exc is not None:
                for close_err in close_errors:
                    body_exc.add_note(f"Suppressed close() error during unwind: {close_err!r}")
