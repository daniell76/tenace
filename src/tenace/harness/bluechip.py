from __future__ import annotations

from collections.abc import Callable
import contextlib
import io
import re
import shlex
import socket
import subprocess
from typing import Protocol, runtime_checkable

import pyspiel

from tenace.domain.types import (
    BIDDING_ACTION_BASE,
    DBL_ACTION,
    DENOMINATION_CHARS,
    FIRST_CONTRACT_BID_ACTION,
    MAX_ACTION,
    NUM_CARDS,
    PASS_ACTION,
    RANK_CHARS,
    RDBL_ACTION,
    SUIT_CHARS,
    Seat,
    Suit,
    card_id,
    card_rank,
    card_suit,
)
from tenace.layer3_rules.tracker import _trick_winner

__all__ = [
    "BlueChipLineTransport",
    "BlueChipSeatBot",
    "SocketBlueChipController",
    "SubprocessTcpBlueChipController",
    "format_bluechip_deal_header",
    "format_bluechip_hand",
    "openspiel_action_to_bluechip",
    "parse_bluechip_player_action",
]

_LOWER_RANKS = tuple(r.lower() for r in RANK_CHARS)
_LOWER_SUITS = tuple(s.lower() for s in SUIT_CHARS)

_PLAY_OWN_SLICE = slice(25, 25 + NUM_CARDS)
_PLAY_DUMMY_SLICE = slice(25 + NUM_CARDS, 25 + 2 * NUM_CARDS)
_AUCTION_OWN_SLICE = slice(432, 432 + NUM_CARDS)

_CONNECT_RE = re.compile(
    r'^\s*Connecting\s+"(?P<client_name>[^"]*)"\s+as\s+\S+\s+using\s+protocol\s+version\s+\d+\s*$',
    re.IGNORECASE,
)
_PLAYER_ACTION_RE = re.compile(
    r"^\s*(?P<seat>NORTH|EAST|SOUTH|WEST)\s+"
    r"(?:"
    r"(?P<pass>PASSES)"
    r"|(?P<dbl>DOUBLES)"
    r"|(?P<rdbl>REDOUBLES)"
    r"|bids\s+(?P<level>[1-7])(?P<denom>NT|C|D|H|S)"
    r"|plays\s+(?P<rank>[23456789tjqka])(?P<suit>[cdhs])"
    r")"
    r"(?:\s+.*)?$",
    re.IGNORECASE,
)
_OBS_DEALER_RE = re.compile(r"^Dealer\s+(North|East|South|West)\s*$", re.MULTILINE)
_OBS_VUL_RE = re.compile(r"^Vul:\s*(None|N/S|E/W|All)\s*$", re.MULTILINE)

_TIMING_LINE = (
    "Timing - N/S : this board  [0:05],  total  [0:05].  "
    "E/W : this board  [0:05],  total  [0:05]"
)


def format_bluechip_deal_header(
    board_number: int,
    dealer: Seat,
    ns_vul: bool,
    ew_vul: bool,
) -> str:
    if isinstance(board_number, bool) or not isinstance(board_number, int) or board_number < 1:
        raise ValueError(f"board_number must be an integer >= 1, got {board_number!r}")
    if not isinstance(ns_vul, bool) or not isinstance(ew_vul, bool):
        raise ValueError("ns_vul and ew_vul must be booleans")

    dealer_seat = Seat(dealer)
    if ns_vul and ew_vul:
        vul_text = "Both vulnerable."
    elif ns_vul:
        vul_text = "N/S vulnerable."
    elif ew_vul:
        vul_text = "E/W vulnerable."
    else:
        vul_text = "Neither vulnerable."

    return f"Board number {board_number}. Dealer {dealer_seat.name}. {vul_text}"


def format_bluechip_hand(cards: list[int]) -> str:
    card_list = list(cards)
    if len(card_list) != 13:
        raise ValueError(f"Bridge hand must contain 13 cards, got {len(card_list)}")
    if any(
        isinstance(c, bool) or not isinstance(c, int) or c < 0 or c >= NUM_CARDS
        for c in card_list
    ):
        raise ValueError("All cards must be valid integer card IDs in [0, 51]")
    if len(set(card_list)) != 13:
        raise ValueError("Bridge hand cannot contain duplicate cards")

    suits_ranks: list[list[str]] = [[] for _ in range(4)]
    for card in sorted(card_list, reverse=True):
        s_idx = int(card_suit(card))
        r_idx = card_rank(card)
        suits_ranks[s_idx].append(RANK_CHARS[r_idx])

    parts: list[str] = []
    for s_idx, suit_char in enumerate(SUIT_CHARS):
        if suits_ranks[s_idx]:
            parts.append(f"{suit_char} {' '.join(suits_ranks[s_idx])}.")
        else:
            parts.append(f"{suit_char} -.")
    return " ".join(parts)


def openspiel_action_to_bluechip(action: int) -> str:
    if isinstance(action, bool) or not isinstance(action, int) or not (0 <= action <= MAX_ACTION):
        raise ValueError(f"Action must be an integer in [0, {MAX_ACTION}], got {action!r}")

    if action == PASS_ACTION:
        return "PASSES"
    if action == DBL_ACTION:
        return "DOUBLES"
    if action == RDBL_ACTION:
        return "REDOUBLES"
    if action >= FIRST_CONTRACT_BID_ACTION:
        offset = action - FIRST_CONTRACT_BID_ACTION
        level = 1 + offset // 5
        denom = DENOMINATION_CHARS[offset % 5]
        return f"bids {level}{denom}"

    r_idx = card_rank(action)
    s_idx = int(card_suit(action))
    return f"plays {_LOWER_RANKS[r_idx]}{_LOWER_SUITS[s_idx]}"


def _parse_bluechip_player_action_with_seat(line: str) -> tuple[Seat, int]:
    if not isinstance(line, str):
        raise ValueError(f"Expected string action line, got {line!r}")

    match = _PLAYER_ACTION_RE.match(line)
    if match is None:
        raise ValueError(f"Malformed BlueChip player action line: {line!r}")

    actor_seat = Seat[match.group("seat").upper()]
    if match.group("pass"):
        return actor_seat, PASS_ACTION
    if match.group("dbl"):
        return actor_seat, DBL_ACTION
    if match.group("rdbl"):
        return actor_seat, RDBL_ACTION
    if match.group("level"):
        level = int(match.group("level"))
        denom = match.group("denom").upper()
        denom_idx = DENOMINATION_CHARS.index(denom)
        return actor_seat, FIRST_CONTRACT_BID_ACTION + (level - 1) * 5 + denom_idx

    rank_ch = match.group("rank").lower()
    suit_ch = match.group("suit").lower()
    r_idx = _LOWER_RANKS.index(rank_ch)
    s_idx = _LOWER_SUITS.index(suit_ch)
    return actor_seat, card_id(Suit(s_idx), r_idx)


def parse_bluechip_player_action(line: str) -> int:
    _actor_seat, action = _parse_bluechip_player_action_with_seat(line)
    return action


@runtime_checkable
class BlueChipLineTransport(Protocol):
    def read_line(self) -> str: ...

    def send_line(self, line: str) -> None: ...

    def close(self) -> None: ...


class SocketBlueChipController:
    """TCP Table Manager listener for a single BlueChip v18 external bridge client."""

    def __init__(self, timeout_secs: float = 60.0) -> None:
        if timeout_secs <= 0:
            raise ValueError("timeout_secs must be positive")
        self._timeout_secs = float(timeout_secs)
        self._listener: socket.socket | None = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.settimeout(self._timeout_secs)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.port: int = int(self._listener.getsockname()[1])

        self._conn: socket.socket | None = None
        self._rfile: io.TextIOWrapper | None = None

    def accept(self) -> None:
        if self._listener is None:
            raise RuntimeError("Listener socket is already closed")
        conn, _addr = self._listener.accept()
        conn.settimeout(self._timeout_secs)
        self._conn = conn
        self._rfile = conn.makefile("r", encoding="ascii", newline="\r\n")
        try:
            self._listener.close()
        finally:
            self._listener = None

    def read_line(self) -> str:
        if self._rfile is None:
            raise RuntimeError("SocketBlueChipController is not connected")
        raw = self._rfile.readline()
        if raw == "":
            raise EOFError("BlueChip peer closed TCP connection")
        return raw.rstrip("\r\n")

    def send_line(self, line: str) -> None:
        if self._conn is None:
            raise RuntimeError("SocketBlueChipController is not connected")
        self._conn.sendall((line + "\r\n").encode("ascii"))

    def close(self) -> None:
        if self._rfile is not None:
            try:
                self._rfile.close()
            except OSError:
                pass
            finally:
                self._rfile = None
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            finally:
                self._conn = None
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            finally:
                self._listener = None


class SubprocessTcpBlueChipController:
    """Spawns a BlueChip bridge subprocess (e.g. WBridge5 or BEN) and connects via loopback TCP."""

    def __init__(self, cmd_template: str, timeout_secs: float = 60.0) -> None:
        if "{port}" not in cmd_template:
            raise ValueError(
                "cmd_template must contain '{port}' placeholder for TCP callback port"
            )
        self._socket_ctrl = SocketBlueChipController(timeout_secs=timeout_secs)
        cmd_str = cmd_template.replace("{port}", str(self._socket_ctrl.port))
        argv = shlex.split(cmd_str)
        if not argv:
            self._socket_ctrl.close()
            raise ValueError("cmd_template produced empty command argv")

        self._process: subprocess.Popen[bytes] | None = None
        try:
            self._process = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._socket_ctrl.accept()
        except BaseException:
            self.close()
            raise

    @property
    def port(self) -> int:
        return self._socket_ctrl.port

    def read_line(self) -> str:
        return self._socket_ctrl.read_line()

    def send_line(self, line: str) -> None:
        self._socket_ctrl.send_line(line)

    def close(self) -> None:
        self._socket_ctrl.close()
        proc = self._process
        self._process = None
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=3.0)


def _cards_from_obs_slice(values: list[float]) -> list[int]:
    return [card for card, flag in enumerate(values) if flag > 0.5]


def _extract_dealer_and_vul(obs_str: str) -> tuple[Seat, bool, bool]:
    m_dealer = _OBS_DEALER_RE.search(obs_str)
    m_vul = _OBS_VUL_RE.search(obs_str)
    if m_dealer is None or m_vul is None:
        raise ValueError(
            f"Could not parse dealer/vulnerability from observation header: {obs_str[:120]!r}"
        )
    dealer = Seat[m_dealer.group(1).upper()]
    vul_token = m_vul.group(1)
    ns_vul = vul_token in ("N/S", "All")
    ew_vul = vul_token in ("E/W", "All")
    return dealer, ns_vul, ew_vul


def _parse_public_board_structure(
    public_history: list[pyspiel.StateHistoryItem],
) -> tuple[list[tuple[Seat, int]], list[int], Seat | None, Seat | None, int | None]:
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

    declarer: Seat | None = None
    dummy_seat: Seat | None = None
    trump_denom: int | None = None
    if last_bid_seat is not None and last_bid_action is not None:
        trump_denom = (last_bid_action - FIRST_CONTRACT_BID_ACTION) % 5
        declarer = first_denom_bidder[(last_bid_seat.partnership, trump_denom)]
        dummy_seat = declarer.partner

    return auction_calls, play_actions, declarer, dummy_seat, trump_denom


def _validate_full_card_play_game(state: pyspiel.State) -> None:
    game = state.get_game()
    params = game.get_parameters()
    if "use_double_dummy_result" in params:
        use_dd = bool(params["use_double_dummy_result"])
    else:
        use_dd = bool(
            game.get_type().parameter_specification.get("use_double_dummy_result", False)
        )
    if use_dd:
        raise ValueError(
            "BlueChipSeatBot requires full card play (use_double_dummy_result=False); "
            "BlueChip protocol v18 does not support double-dummy play skipping"
        )


def _compute_play_actors_and_next_leader(
    play_actions: list[int],
    declarer: Seat,
    trump_denom: int | None,
) -> tuple[list[Seat], Seat]:
    """Return physical hand Seat for each played card so far and the current/next trick leader."""
    leader = Seat((int(declarer) + 1) % 4)
    card_hand_seats: list[Seat] = []
    for trick_start in range(0, len(play_actions), 4):
        trick_cards = play_actions[trick_start : trick_start + 4]
        trick_plays: list[tuple[Seat, int]] = []
        for offset, card in enumerate(trick_cards):
            hand_seat = Seat((int(leader) + offset) % 4)
            card_hand_seats.append(hand_seat)
            trick_plays.append((hand_seat, card))
        if len(trick_plays) == 4:
            leader = _trick_winner(trick_plays, trump_denom)
    return card_hand_seats, leader


class BlueChipSeatBot:
    """WBF-compliant BlueChip v18 seat bot adapter for external bridge engines (WBridge5, BEN).

    Strict imperfect-information invariant: only accesses `state.observation_tensor(int(self.seat))`,
    `state.observation_string(int(self.seat))`, and public post-deal actions at indices >= 52.
    """

    def __init__(
        self,
        seat: Seat,
        controller_factory: Callable[[], BlueChipLineTransport],
        team_ns: str = "Tenace",
        team_ew: str = "Opponent",
    ) -> None:
        self.seat = Seat(seat)
        self._controller_factory = controller_factory
        self._team_ns = team_ns
        self._team_ew = team_ew

        self._controller: BlueChipLineTransport | None = None
        self._board_number = 0
        self._board_started = False
        self._board_finished = False
        self._synced_public_actions: list[int] = []
        self._dummy_sent = False

        self._ready_for_teams_re = re.compile(
            rf"^\s*{self.seat.name}\s+ready\s+for\s+teams\s*$", re.IGNORECASE
        )
        self._ready_to_start_re = re.compile(
            rf"^\s*{self.seat.name}\s+ready\s+to\s+start\s*$", re.IGNORECASE
        )
        self._ready_for_deal_re = re.compile(
            rf"^\s*{self.seat.name}\s+ready\s+for\s+deal\s*$", re.IGNORECASE
        )
        self._ready_for_cards_re = re.compile(
            rf"^\s*{self.seat.name}\s+ready\s+for\s+cards\s*$", re.IGNORECASE
        )
        self._ready_for_dummy_re = re.compile(
            rf"^\s*{self.seat.name}\s+ready\s+for\s+dummy\s*$", re.IGNORECASE
        )
        self._ready_for_other_re = re.compile(
            rf"^\s*{self.seat.name}\s+ready\s+for\s+"
            rf"(?P<other>NORTH|EAST|SOUTH|WEST|Dummy)'s\s+"
            rf"(?:bid|card\s+to\s+trick\s+\d+)\s*$",
            re.IGNORECASE,
        )

    def _expect_match(self, pattern: re.Pattern[str]) -> re.Match[str]:
        assert self._controller is not None
        line = self._controller.read_line()
        match = pattern.match(line)
        if match is None:
            raise ValueError(
                f"Unexpected BlueChip message on {self.seat.name}: {line!r} "
                f"(expected {pattern.pattern!r})"
            )
        return match

    def _ensure_connected(self) -> BlueChipLineTransport:
        if self._controller is not None:
            return self._controller

        ctrl = self._controller_factory()
        self._controller = ctrl
        try:
            line = ctrl.read_line()
            m_conn = _CONNECT_RE.match(line)
            if m_conn is None:
                raise ValueError(f"Invalid BlueChip greeting: {line!r}")
            client_name = m_conn.group("client_name")
            ctrl.send_line(f'{self.seat.name} ("{client_name}") seated')
            self._expect_match(self._ready_for_teams_re)
            ctrl.send_line(f'Teams: N/S "{self._team_ns}" E/W "{self._team_ew}"')
            self._expect_match(self._ready_to_start_re)
        except BaseException:
            self.close()
            raise
        return ctrl

    def _extract_initial_own_cards(
        self,
        state: pyspiel.State,
        play_actions: list[int],
        card_hand_seats: list[Seat],
    ) -> list[int]:
        obs = state.observation_tensor(int(self.seat))
        if not play_actions:
            return _cards_from_obs_slice(obs[_AUCTION_OWN_SLICE])

        remaining = set(_cards_from_obs_slice(obs[_PLAY_OWN_SLICE]))
        played_by_self = {
            card
            for card, hand_seat in zip(play_actions, card_hand_seats, strict=True)
            if hand_seat == self.seat
        }
        return sorted(remaining | played_by_self)

    def _extract_initial_dummy_cards(
        self,
        state: pyspiel.State,
        dummy_seat: Seat,
        play_actions: list[int],
        card_hand_seats: list[Seat],
    ) -> list[int]:
        obs = state.observation_tensor(int(self.seat))
        remaining_dummy = set(_cards_from_obs_slice(obs[_PLAY_DUMMY_SLICE]))
        played_by_dummy = {
            card
            for card, hand_seat in zip(play_actions, card_hand_seats, strict=True)
            if hand_seat == dummy_seat
        }
        return sorted(remaining_dummy | played_by_dummy)

    def _sync_state(self, state: pyspiel.State) -> tuple[Seat | None, Seat | None, int, Seat | None]:
        ctrl = self._ensure_connected()

        # CRITICAL FAIR-PLAY INVARIANT: Only inspect public post-deal actions from NUM_CARDS onward.
        public_history = state.full_history()[NUM_CARDS:]
        public_actions = [entry.action for entry in public_history]

        if public_actions[: len(self._synced_public_actions)] != self._synced_public_actions:
            raise ValueError(
                f"State history prefix mismatch on {self.seat.name}: "
                f"expected prefix {self._synced_public_actions}, got {public_actions}"
            )

        auction_calls, play_actions, declarer, dummy_seat, trump_denom = (
            _parse_public_board_structure(public_history)
        )
        card_hand_seats: list[Seat] = []
        next_leader: Seat | None = None
        if declarer is not None and (play_actions or state.current_phase() == 2):
            card_hand_seats, next_leader = _compute_play_actors_and_next_leader(
                play_actions, declarer, trump_denom
            )

        if not self._board_started:
            self._board_number += 1
            dealer, ns_vul, ew_vul = _extract_dealer_and_vul(
                state.observation_string(int(self.seat))
            )
            own_13 = self._extract_initial_own_cards(state, play_actions, card_hand_seats)
            ctrl.send_line("Start of board")
            self._expect_match(self._ready_for_deal_re)
            ctrl.send_line(format_bluechip_deal_header(self._board_number, dealer, ns_vul, ew_vul))
            self._expect_match(self._ready_for_cards_re)
            ctrl.send_line(f"{self.seat.name}'s cards: {format_bluechip_hand(own_13)}")
            self._board_started = True

        auction_len = len(auction_calls)

        def maybe_send_dummy_after_opening_lead() -> None:
            if (
                not self._dummy_sent
                and dummy_seat is not None
                and self.seat != dummy_seat
                and len(self._synced_public_actions) >= auction_len + 1
                and len(play_actions) >= 1
            ):
                self._expect_match(self._ready_for_dummy_re)
                dummy_13 = self._extract_initial_dummy_cards(
                    state, dummy_seat, play_actions, card_hand_seats
                )
                ctrl.send_line(f"Dummy's cards: {format_bluechip_hand(dummy_13)}")
                self._dummy_sent = True

        # Handle case where this seat itself made the opening lead on its previous step()!
        if len(self._synced_public_actions) == auction_len + 1:
            maybe_send_dummy_after_opening_lead()

        while len(self._synced_public_actions) < len(public_actions):
            idx = len(self._synced_public_actions)
            action = public_actions[idx]
            m_ready = self._expect_match(self._ready_for_other_re)
            other_tok = m_ready.group("other")
            if other_tok.lower() == "dummy":
                assert dummy_seat is not None
                actor_name = dummy_seat.name
            elif idx < auction_len:
                actor_name = auction_calls[idx][0].name
            else:
                play_idx = idx - auction_len
                actor_name = card_hand_seats[play_idx].name

            ctrl.send_line(f"{actor_name} {openspiel_action_to_bluechip(action)}")
            self._synced_public_actions.append(action)

            if len(self._synced_public_actions) == auction_len + 1:
                maybe_send_dummy_after_opening_lead()

        if state.is_terminal() and not self._board_finished:
            if declarer is None or len(play_actions) == NUM_CARDS:
                ctrl.send_line(_TIMING_LINE)
                self._board_finished = True
            else:
                self.close()

        return declarer, dummy_seat, len(play_actions), next_leader

    def reset_for_board(self) -> None:
        if self._board_started and not self._board_finished:
            # Previous board did not reach clean terminal sync; reconnect cleanly for safety.
            self.close()
        self._board_started = False
        self._board_finished = False
        self._synced_public_actions = []
        self._dummy_sent = False

    def on_board_end(self, state: pyspiel.State) -> None:
        if not state.is_terminal():
            raise ValueError("on_board_end requires a terminal bridge state")
        _validate_full_card_play_game(state)
        if self._board_started and not self._board_finished:
            self._sync_state(state)

    def step(self, state: pyspiel.State) -> int:
        if state.is_terminal():
            raise ValueError("Cannot step on a terminal bridge state")
        if state.is_chance_node() or state.current_phase() == 0:
            raise ValueError("Cannot step during incomplete deal phase (chance node)")
        _validate_full_card_play_game(state)
        if state.current_player() != int(self.seat):
            raise ValueError(
                f"BlueChipSeatBot seat {self.seat.name} ({int(self.seat)}) called on "
                f"seat {state.current_player()}'s turn"
            )

        _declarer, dummy_seat, cards_played, next_leader = self._sync_state(state)
        assert self._controller is not None

        if state.current_phase() == 2:
            assert next_leader is not None
            expected_actor = Seat((int(next_leader) + (cards_played % 4)) % 4)
        else:
            expected_actor = self.seat

        if state.current_phase() == 2 and cards_played % 4 == 0:
            if dummy_seat is not None and next_leader == dummy_seat:
                self._controller.send_line("Dummy to lead")
            else:
                self._controller.send_line(f"{self.seat.name} to lead")

        reply = self._controller.read_line()
        reply_seat, chosen_action = _parse_bluechip_player_action_with_seat(reply)
        if reply_seat != expected_actor:
            raise ValueError(
                f"BlueChip reply seat mismatch on {self.seat.name}: "
                f"expected actor {expected_actor.name}, got {reply_seat.name} ({reply!r})"
            )
        self._synced_public_actions.append(chosen_action)
        return chosen_action

    def close(self) -> None:
        ctrl = self._controller
        self._controller = None
        if ctrl is not None:
            ctrl.close()
