# Tenace Phase 2: Live Match CLI & Championship BlueChip Bot Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add rich per-board room transcripts & Strategist plan-trace telemetry, a full WBF-compliant BlueChip Table Manager v18 TCP adapter (for playing against **WBridge5** and **BEN**), and a `tenace-match` CLI runner supporting both offline smoke runs and live Vertex AI (`Gemini` + `DiffusionGemma`) matches.

**Architecture:** First, extend `TenaceAgent` and `DuplicateMatchRunner` (`duplicate.py`) to capture structured `RoomTranscript`s (auction, contract, trick log, per-seat `StrategyState` revisions, and API call telemetry) and deterministically `close()` external seat bots in a `finally` block. Second, implement a clean first-principles `BlueChipSeatBot` + `SubprocessTcpBlueChipController` (`bluechip.py`) speaking Table Manager Protocol v18 across all 16 WBF dealer/vulnerability rotations and declarer-driven dummy turns (avoiding upstream OpenSpiel `BlueChipBridgeBot` dealer/dummy bugs). Third, expose `uv run tenace-match` (`cli.py`) with formatted board-by-board scorecards, strategy traces, and Vertex AI / BlueChip flags.

**Tech Stack:** Python 3.13 (`uv`), `open-spiel >= 2.0.2`, `pydantic >= 2.9`, `google-genai >= 1.5`, `google-cloud-aiplatform >= 1.84`, standard library `socket` / `subprocess` / `argparse`, `pytest`.

---

## File Structure

```text
tenace/
├── pyproject.toml                         # Modify: add [project.scripts] tenace-match = "tenace.cli:main"
├── README.md                              # Modify: document tenace-match CLI + WBridge5 / BEN BlueChip usage
├── docs/superpowers/plans/
│   ├── 2026-11-10-tenace-three-layer-bridge-ai.md
│   └── 2026-11-10-tenace-phase2-cli-and-bluechip.md
├── src/tenace/
│   ├── __init__.py                        # Modify: export BlueChipSeatBot, SubprocessTcpBlueChipController, RoomTranscript
│   ├── agent.py                           # Modify: record per-board strategy_history revisions
│   ├── cli.py                             # Create: tenace-match CLI entrypoint & formatted match report renderer
│   └── harness/
│       ├── __init__.py                    # Modify: export BlueChip classes & RoomTranscript
│       ├── duplicate.py                   # Modify: capture RoomTranscripts, telemetry summary, and bot close() cleanup
│       └── bluechip.py                    # Create: BlueChip v18 protocol codec, TCP subprocess controller, BlueChipSeatBot
└── tests/
    ├── test_agent_and_duplicate_harness.py # Modify: add RoomTranscript, strategy_history & bot close() cleanup tests
    ├── test_bluechip_adapter.py            # Create: loopback TCP + in-memory BlueChip v18 multi-board & dummy-lead tests
    └── test_cli.py                         # Create: CLI argument, mock match scorecard, Vertex wiring & BlueChip CLI tests
```

---

### Task 1: Room Transcripts, Strategy Revision History & Bot Teardown in Harness

**Files:**
- Modify: `src/tenace/agent.py`
- Modify: `src/tenace/harness/duplicate.py`
- Modify: `src/tenace/harness/__init__.py`
- Modify: `tests/test_agent_and_duplicate_harness.py`

- [ ] **Step 1: Write failing tests for `TenaceAgent.strategy_history`, `RoomTranscript` capture, and `close()` lifecycle cleanup**

Append to `tests/test_agent_and_duplicate_harness.py`:

```python
from tenace.harness import RoomTranscript


def test_duplicate_runner_records_room_transcripts_strategy_history_and_closes_bots():
    closed_seats: list[Seat] = []

    class ClosableBaselineBot(FirstLegalPassBaselineBot):
        def close(self) -> None:
            closed_seats.append(self.seat)

    def make_tenace(seat: Seat) -> TenaceAgent:
        return TenaceAgent(
            seat=seat,
            strategist_client=ScriptedStrategistClient(),
            evaluator_client=ScriptedEvaluatorClient(),
            dds_evaluator=RuleDDSEvaluator(num_worlds=2, rng_seed=7),
        )

    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    summary = runner.play_match(
        team_a_factory=make_tenace,
        team_b_factory=ClosableBaselineBot,
        num_boards=1,
    )

    board = summary.boards[0]
    assert isinstance(board.open_room, RoomTranscript)
    assert isinstance(board.closed_room, RoomTranscript)
    assert board.open_room.contract != ""
    assert board.open_room.auction != ""
    assert "NORTH" in board.open_room.strategy_traces
    assert len(board.open_room.strategy_traces["NORTH"]) >= 1
    assert summary.telemetry["strategist_calls"] >= 1
    # 2 Team-B seats in Open Room (E, W) + 2 Team-B seats in Closed Room (N, S) = 4 closed bots
    assert len(closed_seats) == 4
```

- [ ] **Step 2: Run `pytest` to verify RED failure**

Run:
```bash
~/.local/bin/uv run pytest tests/test_agent_and_duplicate_harness.py -k "records_room_transcripts" -v
```
Expected: `FAIL` with `ImportError: cannot import name 'RoomTranscript' from 'tenace.harness'`.

- [ ] **Step 3: Implement `strategy_history` in `agent.py` and `RoomTranscript` + `close()` cleanup in `duplicate.py`**

In `src/tenace/agent.py`:
1. Initialize `self.strategy_history: list[StrategyState] = []` in `__init__` (line 45) and clear `self.strategy_history = []` in `reset_for_board()` (line 55).
2. Whenever `self.strategy = self.strategist.update_strategy(...)` runs (both in initial/event-driven replan and in the assumption-violation interrupt), append `self.strategy_history.append(self.strategy)`.

In `src/tenace/harness/duplicate.py`:
1. Add `RoomTranscript(BaseModel)`:
```python
class RoomTranscript(BaseModel):
    contract: str
    auction: str
    tricks: list[str] = Field(default_factory=list)
    ns_score: float
    strategy_traces: dict[str, list[str]] = Field(default_factory=dict)
```
2. Extend `BoardResult` with optional defaults so all existing callers/tests stay 100% compatible:
```python
class BoardResult(BaseModel):
    board_number: int = Field(ge=1)
    open_room_ns_score: float
    closed_room_ns_score: float
    imps_team_a: int = Field(ge=-24, le=24)
    open_room: RoomTranscript | None = None
    closed_room: RoomTranscript | None = None
```
3. Extend `MatchSummary` with `telemetry: dict[str, int] = Field(default_factory=dict)`.
4. Add a helper `_build_room_transcript(state: pyspiel.State, seats: dict[Seat, BridgeSeatBot], ns_score: float) -> RoomTranscript`:
   - Reconstructs `auction_calls` (`[(Seat(ev.player), ev.action) for ev in state.full_history()[52:] if ev.action >= 52]`) and `play_actions` (`[ev.action for ev in state.full_history()[52:] if ev.action < 52]`).
   - Uses `format_auction_sequence(auction_calls)` and `format_contract_summary(auction_calls)` (falling back to `"Passed Out"` when 4 initial passes occur).
   - Groups `play_actions` into 4-card trick strings (`"Trick 1: E:C2 S:CA W:C5 N:C4 -> SOUTH"` using leader tracking).
   - Collects `strategy_traces[seat.name] = [f"rev {s.revision} ({s.phase}): {s.plan_summary}" for s in getattr(bot, "strategy_history", [])]` for seats with non-empty `strategy_history`.
5. Wrap `play_match`'s board loop in `try: ... finally:` calling `close_fn = getattr(bot, "close", None); if callable(close_fn): close_fn()` on all 8 room bots, and sum `forced_moves`, `strategist_calls`, and `evaluator_calls` across all 8 bots into `MatchSummary.telemetry`.
6. Export `RoomTranscript` in `duplicate.__all__` and `src/tenace/harness/__init__.py`.

- [ ] **Step 4: Run `pytest` to verify GREEN**

Run:
```bash
~/.local/bin/uv run pytest -v
```
Expected: `56 passed`.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/tenace/agent.py src/tenace/harness/ tests/test_agent_and_duplicate_harness.py
git commit -m "feat(harness): capture room transcripts, strategy revision traces, and bot close lifecycle"
```

---

### Task 2: BlueChip Table Manager v18 Protocol Adapter (for WBridge5 & BEN)

**Files:**
- Create: `src/tenace/harness/bluechip.py`
- Modify: `src/tenace/harness/__init__.py`
- Modify: `src/tenace/__init__.py`
- Test: `tests/test_bluechip_adapter.py`

- [ ] **Step 1: Write failing tests with a real loopback TCP `FakeBlueChipEngine` covering handshake, Board 2 East-dealer/NS-vul rotation, dummy reveal, declarer playing dummy cards, and `DuplicateMatchRunner` integration**

Create `tests/test_bluechip_adapter.py`:

```python
from __future__ import annotations

import re
import socket
import threading

import pyspiel

from tenace.domain.types import PASS_ACTION, Seat
from tenace.harness import DuplicateMatchRunner, FirstLegalPassBaselineBot
from tenace.harness.bluechip import (
    BlueChipSeatBot,
    SocketBlueChipController,
    format_bluechip_deal_header,
    format_bluechip_hand,
    openspiel_action_to_bluechip,
    parse_bluechip_player_action,
)


class LoopbackBlueChipPeer:
    """Deterministic BlueChip v18 engine running in a background thread over real TCP."""

    def __init__(self, port: int, open_1nt_once: bool = True) -> None:
        self.port = port
        self.open_1nt_once = open_1nt_once
        self.deal_headers: list[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5.0) as sock:
            rfile = sock.makefile("r", encoding="ascii", newline="\r\n")

            def send(msg: str) -> None:
                sock.sendall((msg + "\r\n").encode("ascii"))

            def recv() -> str:
                line = rfile.readline()
                return line.strip() if line else ""

            send('Connecting "FakeWB5" as ANYPL using protocol version 18')
            seated = recv()  # e.g. NORTH ("FakeWB5") seated
            seat = seated.split()[0]
            send(f"{seat} ready for teams")
            _teams = recv()
            send(f"{seat} ready to start")

            seats_order = ["NORTH", "EAST", "SOUTH", "WEST"]
            suit_map = {"C": 0, "D": 1, "H": 2, "S": 3}
            rank_chars = "23456789tjqka"

            while True:
                line = recv()
                if not line:
                    break
                if line.lower() != "start of board":
                    continue
                send(f"{seat} ready for deal")
                deal_hdr = recv()
                self.deal_headers.append(deal_hdr)
                m_dealer = re.search(r"Dealer (NORTH|EAST|SOUTH|WEST)\.", deal_hdr)
                assert m_dealer is not None
                dealer = m_dealer.group(1)

                send(f"{seat} ready for cards")
                cards_line = recv()  # SEAT's cards: C ... D ... H ... S ...
                own_cards = self._parse_hand_line(cards_line)
                dummy_cards: list[str] = []

                # Auction loop
                turn_idx = seats_order.index(dealer)
                passes = 0
                has_bid = False
                last_bid_denom = "NT"
                first_bidder: dict[tuple[int, str], str] = {}
                winning_side = 0
                declarer = "NORTH"

                while True:
                    actor = seats_order[turn_idx]
                    if actor == seat:
                        if not has_bid and self.open_1nt_once:
                            call_msg = f"{seat} bids 1NT"
                            has_bid = True
                            passes = 0
                            last_bid_denom = "NT"
                            winning_side = turn_idx % 2
                            first_bidder.setdefault((winning_side, "NT"), actor)
                            declarer = first_bidder[(winning_side, "NT")]
                        else:
                            call_msg = f"{seat} PASSES"
                            passes += 1
                        send(call_msg)
                    else:
                        send(f"{seat} ready for {actor}'s bid")
                        obs_line = recv()
                        if "PASSES" in obs_line:
                            passes += 1
                        elif "bids " in obs_line:
                            bid_tok = obs_line.split("bids ")[1].split()[0]
                            denom = bid_tok[1:]
                            has_bid = True
                            passes = 0
                            last_bid_denom = denom
                            winning_side = turn_idx % 2
                            first_bidder.setdefault((winning_side, denom), actor)
                            declarer = first_bidder[(winning_side, denom)]
                        else:
                            passes = 0

                    turn_idx = (turn_idx + 1) % 4
                    if (not has_bid and passes == 4) or (has_bid and passes == 3):
                        break

                if not has_bid:
                    _timing = recv()
                    continue

                dummy_seat = seats_order[(seats_order.index(declarer) + 2) % 4]
                leader_idx = (seats_order.index(declarer) + 1) % 4

                for trick_no in range(1, 14):
                    trick_cards: list[tuple[int, str]] = []
                    led_suit = ""
                    for offset in range(4):
                        p_idx = (leader_idx + offset) % 4
                        p_seat = seats_order[p_idx]
                        is_our_action = (p_seat == seat and seat != dummy_seat) or (
                            p_seat == dummy_seat and seat == declarer
                        )
                        if offset == 0 and is_our_action:
                            _to_lead = recv()  # "{SEAT|Dummy} to lead"

                        if is_our_action:
                            hand_ref = dummy_cards if p_seat == dummy_seat else own_cards
                            chosen = self._pick_legal(hand_ref, led_suit)
                            hand_ref.remove(chosen)
                            actor_label = p_seat
                            send(f"{actor_label} plays {chosen}")
                            card_tok = chosen
                        else:
                            req_target = "Dummy" if p_seat == dummy_seat else p_seat
                            send(f"{seat} ready for {req_target}'s card to trick {trick_no}")
                            play_line = recv()
                            card_tok = play_line.split("plays ")[1].split()[0].lower()
                            if p_seat == dummy_seat and card_tok in dummy_cards:
                                dummy_cards.remove(card_tok)

                        if offset == 0:
                            led_suit = card_tok[1]
                        trick_cards.append((p_idx, card_tok))

                        # Right after opening lead (trick 1, offset 0), non-dummy seats request dummy!
                        if trick_no == 1 and offset == 0 and seat != dummy_seat:
                            send(f"{seat} ready for dummy")
                            dummy_line = recv()
                            dummy_cards = self._parse_hand_line(dummy_line)

                    leader_idx = self._winner(trick_cards, last_bid_denom, rank_chars, suit_map)

                _timing = recv()

    @staticmethod
    def _parse_hand_line(line: str) -> list[str]:
        body = line.split(":", 1)[1].strip()
        cards: list[str] = []
        for part in body.split("."):
            tokens = part.strip().split()
            if len(tokens) >= 2 and tokens[1] != "-":
                s_char = tokens[0].lower()
                for r_tok in tokens[1:]:
                    cards.append(r_tok.lower() + s_char)
        return cards

    @staticmethod
    def _pick_legal(hand: list[str], led_suit: str) -> str:
        if led_suit:
            in_suit = [c for c in hand if c[1] == led_suit]
            if in_suit:
                return in_suit[0]
        return hand[0]

    @staticmethod
    def _winner(trick: list[tuple[int, str]], trump: str, ranks: str, suits: dict[str, int]) -> int:
        led_suit = trick[0][1][1]
        best_p, best_c = trick[0]
        t_char = trump.lower() if trump in suits else ""
        for p_idx, card in trick[1:]:
            is_t = bool(t_char and card[1] == t_char)
            best_t = bool(t_char and best_c[1] == t_char)
            if is_t and not best_t:
                best_p, best_c = p_idx, card
            elif card[1] == best_c[1] and ranks.index(card[0]) > ranks.index(best_c[0]):
                if is_t or card[1] == led_suit:
                    best_p, best_c = p_idx, card
        return best_p


def test_bluechip_wire_formatting_helpers():
    assert format_bluechip_deal_header(1, dealer=Seat.NORTH, ns_vul=False, ew_vul=False) == (
        "Board number 1. Dealer NORTH. Neither vulnerable."
    )
    assert format_bluechip_deal_header(2, dealer=Seat.EAST, ns_vul=True, ew_vul=False) == (
        "Board number 2. Dealer EAST. N/S vulnerable."
    )
    assert format_bluechip_deal_header(4, dealer=Seat.WEST, ns_vul=True, ew_vul=True) == (
        "Board number 4. Dealer WEST. Both vulnerable."
    )
    assert openspiel_action_to_bluechip(52) == "PASSES"
    assert openspiel_action_to_bluechip(59) == "bids 1NT"
    assert openspiel_action_to_bluechip(51) == "plays as"
    assert parse_bluechip_player_action("NORTH bids 1NT Alert.") == 59
    assert parse_bluechip_player_action("SOUTH plays ck") == 44


def test_bluechip_seat_bot_plays_multi_board_duplicate_match_over_real_tcp():
    peers: list[LoopbackBlueChipPeer] = []

    def make_bluechip_bot(seat: Seat) -> BlueChipSeatBot:
        def controller_factory() -> SocketBlueChipController:
            ctrl = SocketBlueChipController(timeout_secs=5.0)
            peer = LoopbackBlueChipPeer(port=ctrl.port, open_1nt_once=True)
            peers.append(peer)
            peer.start()
            ctrl.accept()
            return ctrl

        return BlueChipSeatBot(seat=seat, controller_factory=controller_factory)

    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    summary = runner.play_match(
        team_a_factory=make_bluechip_bot,
        team_b_factory=FirstLegalPassBaselineBot,
        num_boards=2,
    )

    assert len(summary.boards) == 2
    for p in peers:
        p.join(timeout=5.0)
    # Confirm Board 1 used NORTH / Neither and Board 2 used EAST / N/S vulnerable!
    all_headers = [h for p in peers for h in p.deal_headers]
    assert any("Dealer NORTH. Neither vulnerable." in h for h in all_headers)
    assert any("Dealer EAST. N/S vulnerable." in h for h in all_headers)
```

- [ ] **Step 2: Run `pytest` to verify RED failure**

Run:
```bash
~/.local/bin/uv run pytest tests/test_bluechip_adapter.py -v
```
Expected: `FAIL` with `ModuleNotFoundError: No module named 'tenace.harness.bluechip'`.

- [ ] **Step 3: Implement `src/tenace/harness/bluechip.py` and package exports**

Create `src/tenace/harness/bluechip.py` implementing:
- `format_bluechip_deal_header(board_number: int, dealer: Seat, ns_vul: bool, ew_vul: bool) -> str` (`Neither` / `N/S` / `E/W` / `Both vulnerable.`)
- `format_bluechip_hand(cards: list[int]) -> str` (`C A K 2. D -. H Q 9. S J 4.`)
- `openspiel_action_to_bluechip(action: int) -> str` and `parse_bluechip_player_action(line: str) -> int`
- `BlueChipLineTransport(Protocol)` (`read_line() -> str`, `send_line(line: str) -> None`, `close() -> None`)
- `SocketBlueChipController` (listens on `127.0.0.1:0` ephemeral port, `.port`, `.accept()`, buffered `\r\n` `read_line()`, `send_line()`, `close()`) and `SubprocessTcpBlueChipController(cmd_template: str, timeout_secs: float = 60.0)` (substitutes `{port}` via `shlex.split`, spawns `subprocess.Popen`, accepts TCP connection, terminates process cleanly in `close()`).
- `BlueChipSeatBot(seat: Seat, controller_factory: Callable[[], BlueChipLineTransport], team_ns: str = "Tenace", team_ew: str = "Opponent")`:
  - Strictly imperfect-information safe: reads `seat`'s 13 dealt cards from `state.observation_tensor(int(seat))[432:484]` at auction start, reads dealer (`(first_auction_player)` from `state.full_history()[52].player` or `state.current_player()` at action 52) and vulnerability flags directly from `obs[4:8]` (`bridge.cc:612-615`: `obs[4..5]` = own partnership vul one-hot, `obs[6..7]` = opp partnership vul one-hot!), and reads dummy's 13 cards after opening lead (`obs[77:129]` + played lead card if dummy led? Wait: leader is `(declarer + 1) % 4`, never dummy, so at `cards_played == 1` dummy ALWAYS still holds all 13 cards in `obs[77:129]`!!).
  - Syncs intermediate public actions (`actions[self._synced_idx:]`) to the external engine, sends `Dummy's cards: ...` right after the opening lead (`cards_played == 1` when `self.seat != dummy_seat`), sends `{acting_seat_or_Dummy} to lead` when `cards_played % 4 == 0` in play phase (`Dummy to lead` if `acting_hand_seat == dummy_seat` else `{self.seat.name} to lead`), parses the returned bid/card, and on `reset_for_board()` finishes/syncs or restarts cleanly if the previous board ended while another player acted last!
  - Implements `close() -> None` to close and clear `self._controller`.

Export `BlueChipLineTransport`, `BlueChipSeatBot`, `SocketBlueChipController`, and `SubprocessTcpBlueChipController` in `bluechip.__all__`, `src/tenace/harness/__init__.py`, and `src/tenace/__init__.py`.

- [ ] **Step 4: Run `pytest` to verify GREEN**

Run:
```bash
~/.local/bin/uv run pytest -v
```
Expected: `58+ passed`.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/tenace/ tests/test_bluechip_adapter.py
git commit -m "feat(harness): add WBF-compliant BlueChip v18 external bot TCP adapter"
```

---

### Task 3: `tenace-match` CLI Runner & Formatted Scorecard Renderer

**Files:**
- Modify: `pyproject.toml`
- Create: `src/tenace/cli.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests (`--mode mock`, formatted board scorecard/strategy output, Vertex argument validation, and `--opponent bluechip`)**

Create `tests/test_cli.py`:

```python
import pytest

from tenace.cli import build_arg_parser, format_match_report, run_cli


def test_cli_mock_match_against_baseline_prints_board_transcript_and_telemetry(capsys):
    exit_code = run_cli([
        "--mode", "mock",
        "--opponent", "baseline",
        "--boards", "2",
        "--seed", "42",
        "--dds-worlds", "4",
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "TENACE DUPLICATE BRIDGE MATCH REPORT" in out
    assert "Board 1" in out
    assert "Board 2" in out
    assert "Open Room" in out
    assert "Closed Room" in out
    assert "Strategist Traces" in out
    assert "Telemetry" in out


def test_cli_requires_bluechip_cmd_when_opponent_is_bluechip():
    with pytest.raises(ValueError, match="--bluechip-cmd is required"):
        run_cli(["--mode", "mock", "--opponent", "bluechip", "--boards", "1"])


def test_cli_parser_defaults_to_vertex_and_gcp_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-bridge-gcp-proj")
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.mode == "vertex"
    assert args.project == "my-bridge-gcp-proj"
    assert args.strategist_model == "gemini-2.5-pro"
```

- [ ] **Step 2: Run `pytest` to verify RED failure**

Run:
```bash
~/.local/bin/uv run pytest tests/test_cli.py -v
```
Expected: `FAIL` with `ModuleNotFoundError: No module named 'tenace.cli'`.

- [ ] **Step 3: Implement `src/tenace/cli.py`, register `tenace-match` script in `pyproject.toml`, and update `README.md`**

1. In `pyproject.toml`, add:
```toml
[project.scripts]
tenace-match = "tenace.cli:main"
```
2. Create `src/tenace/cli.py` providing:
   - `HeuristicMockModelClient`: deterministic offline `StructuredModelClient` for `--mode mock` so `uv run tenace-match --mode mock` runs a complete 3-layer match immediately without network credentials.
   - `build_arg_parser() -> argparse.ArgumentParser` supporting:
     - `--mode {vertex,mock}` (default `"vertex"`)
     - `--project` (default `os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")`)
     - `--strategist-location` (default `"global"`), `--strategist-model` (default `"gemini-2.5-pro"`)
     - `--evaluator-backend {diffusion-gemma,gemini}` (default `"diffusion-gemma"`; `"gemini"` lets users run live Vertex matches immediately with e.g. `gemini-2.5-flash` even before provisioning a DiffusionGemma endpoint!)
     - `--evaluator-location` (default `"us-central1"`), `--evaluator-model` (default `"publishers/google/models/diffusiongemma-26b-a4b-it"`), `--evaluator-endpoint-id` (default `None`)
     - `--opponent {baseline,bluechip}` (default `"baseline"`), `--bluechip-cmd` (command template containing `{port}`, e.g. `'wine /path/to/Wbridge5.exe Autoconnect {port}'`)
     - `--boards` (default `4`), `--seed` (default `42`), `--dds-worlds` (default `16`), `--double-dummy` (flag)
   - `format_match_report(summary: MatchSummary, team_a_name: str = "Tenace", team_b_name: str = "Opponent") -> str`: renders header, per-board Open/Closed contracts, auctions, trick count/summary, Strategist revision traces, board IMP swing, total IMP winner banner, and API telemetry summary.
   - `run_cli(argv: list[str] | None = None) -> int` and `main() -> None`.
3. Update `README.md` with CLI examples for:
   - Offline smoke match (`uv run tenace-match --mode mock --boards 2`)
   - Live Vertex AI match (`uv run tenace-match --project your-gcp-project-id --boards 4`)
   - Playing directly against **WBridge5** or **BEN** via `--opponent bluechip --bluechip-cmd '...'`.

- [ ] **Step 4: Sync script entrypoint and run full `pytest` + `tenace-match --mode mock` smoke check**

Run:
```bash
~/.local/bin/uv sync --extra dev && ~/.local/bin/uv run pytest -v && ~/.local/bin/uv run tenace-match --mode mock --boards 1 --dds-worlds 4
```
Expected: All tests `PASS` and `tenace-match` prints the formatted 1-board Duplicate Match Report with exit code `0`.

- [ ] **Step 5: Commit Task 3**

```bash
git add pyproject.toml uv.lock README.md src/tenace/cli.py tests/test_cli.py
git commit -m "feat(cli): add tenace-match CLI runner and formatted duplicate match report"
```
