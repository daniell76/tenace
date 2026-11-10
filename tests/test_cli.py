from __future__ import annotations

import shlex
import sys
from typing import Any

import pytest

from tenace.cli import (
    HeuristicMockModelClient,
    build_arg_parser,
    format_match_report,
    main,
    run_cli,
)
from tenace.domain.types import StrategyState, TacticalDecision
from tenace.models.clients import StructuredModelClient


_BLUECHIP_SUBPROCESS_PEER = """
import re
import socket
import sys

port = int(sys.argv[1])
with socket.create_connection(("127.0.0.1", port), timeout=10.0) as sock:
    rfile = sock.makefile("r", encoding="ascii", newline="\\r\\n")

    def send(msg: str) -> None:
        sock.sendall((msg + "\\r\\n").encode("ascii"))

    def recv() -> str:
        line = rfile.readline()
        return line.strip() if line else ""

    def parse_hand(line: str) -> list[str]:
        body = line.split(":", 1)[1].strip()
        cards: list[str] = []
        for part in body.split("."):
            tokens = part.strip().split()
            if len(tokens) >= 2 and tokens[1] != "-":
                s_char = tokens[0].lower()
                for r_tok in tokens[1:]:
                    cards.append(r_tok.lower() + s_char)
        return cards

    def pick_legal(hand: list[str], led_suit: str) -> str:
        cands = [c for c in hand if c[1] == led_suit] if led_suit else []
        return (cands or hand)[0]

    def winner(trick: list[tuple[int, str]], trump: str) -> int:
        ranks = "23456789tjqka"
        led_suit = trick[0][1][1]
        best_p, best_c = trick[0]
        t_char = trump.lower() if trump in {"C", "D", "H", "S"} else ""
        for p_idx, card in trick[1:]:
            is_t = bool(t_char and card[1] == t_char)
            best_t = bool(t_char and best_c[1] == t_char)
            if is_t and not best_t:
                best_p, best_c = p_idx, card
            elif card[1] == best_c[1] and ranks.index(card[0]) > ranks.index(best_c[0]):
                if is_t or card[1] == led_suit:
                    best_p, best_c = p_idx, card
        return best_p

    send('Connecting "CLITestPeer" as ANYPL using protocol version 18')
    seated = recv()
    if not seated:
        sys.exit(0)
    seat = seated.split()[0]
    send(f"{seat} ready for teams")
    teams_line = recv()
    if not teams_line:
        sys.exit(0)
    expected_teams = (
        'Teams: N/S "BlueChip" E/W "Tenace"'
        if seat in ("NORTH", "SOUTH")
        else 'Teams: N/S "Tenace" E/W "BlueChip"'
    )
    if teams_line != expected_teams:
        raise RuntimeError(f"Seat {seat} expected {expected_teams!r}, got {teams_line!r}")
    send(f"{seat} ready to start")

    seats_order = ["NORTH", "EAST", "SOUTH", "WEST"]
    while True:
        line = recv()
        if not line:
            break
        if line.lower() != "start of board":
            continue
        send(f"{seat} ready for deal")
        deal_hdr = recv()
        if not deal_hdr:
            break
        m_dealer = re.search(r"Dealer (NORTH|EAST|SOUTH|WEST)\\.", deal_hdr)
        dealer = m_dealer.group(1)
        send(f"{seat} ready for cards")
        cards_line = recv()
        if not cards_line:
            break
        own_cards = parse_hand(cards_line)
        dummy_cards: list[str] = []

        turn_idx = seats_order.index(dealer)
        passes = 0
        has_bid = False
        last_bid_denom = "NT"
        first_bidder: dict[tuple[int, str], str] = {}
        declarer = "NORTH"
        disconnected = False

        while True:
            actor = seats_order[turn_idx]
            if actor == seat:
                send(f"{seat} PASSES")
                passes += 1
            else:
                send(f"{seat} ready for {actor}'s bid")
                obs = recv()
                if not obs:
                    disconnected = True
                    break
                if "PASSES" in obs:
                    passes += 1
                elif "bids " in obs:
                    bid_tok = obs.split("bids ")[1].split()[0]
                    denom = bid_tok[1:]
                    has_bid = True
                    passes = 0
                    last_bid_denom = denom
                    side = turn_idx % 2
                    first_bidder.setdefault((side, denom), actor)
                    declarer = first_bidder[(side, denom)]
                else:
                    passes = 0
            turn_idx = (turn_idx + 1) % 4
            if (not has_bid and passes == 4) or (has_bid and passes == 3):
                break

        if disconnected:
            break
        if not has_bid:
            if not recv():
                break
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
                    if not recv():
                        disconnected = True
                        break
                if is_our_action:
                    hand_ref = dummy_cards if p_seat == dummy_seat else own_cards
                    chosen = pick_legal(hand_ref, led_suit)
                    hand_ref.remove(chosen)
                    send(f"{p_seat} plays {chosen}")
                    card_tok = chosen
                else:
                    req_target = "Dummy" if p_seat == dummy_seat else p_seat
                    send(f"{seat} ready for {req_target}'s card to trick {trick_no}")
                    play_line = recv()
                    if not play_line:
                        disconnected = True
                        break
                    card_tok = play_line.split("plays ")[1].split()[0].lower()
                    if p_seat == dummy_seat and card_tok in dummy_cards:
                        dummy_cards.remove(card_tok)

                if offset == 0:
                    led_suit = card_tok[1]
                trick_cards.append((p_idx, card_tok))

                if trick_no == 1 and offset == 0 and seat != dummy_seat:
                    send(f"{seat} ready for dummy")
                    dummy_line = recv()
                    if not dummy_line:
                        disconnected = True
                        break
                    dummy_cards = parse_hand(dummy_line)
            if disconnected:
                break
            leader_idx = winner(trick_cards, last_bid_denom)

        if disconnected:
            break
        if not recv():
            break
"""


def test_cli_mock_match_against_baseline_prints_board_transcript_and_telemetry(capsys):
    exit_code = run_cli(
        [
            "--mode",
            "mock",
            "--opponent",
            "baseline",
            "--boards",
            "2",
            "--seed",
            "42",
            "--dds-worlds",
            "1",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "TENACE DUPLICATE BRIDGE MATCH REPORT" in out
    assert "Tenace vs Baseline" in out
    assert "Board 1" in out
    assert "Board 2" in out
    assert "Open Room" in out
    assert "Closed Room" in out
    assert "Strategist Traces" in out
    assert "Telemetry" in out
    assert "strategist_calls" in out
    assert "evaluator_calls" in out
    assert "forced_moves" in out


def test_heuristic_mock_model_client_is_stateless_and_validates_schemas():
    client = HeuristicMockModelClient()
    assert isinstance(client, StructuredModelClient)

    auction_state = client.generate_structured(
        system_prompt="sys",
        user_prompt=(
            "Seat: NORTH (Partner: SOUTH)\n"
            "Phase: auction\n"
            "Own Hand (16 HCP): S: A K 4 | H: K J 5 | D: Q 8 4 | C: A 9 3 2\n"
            "Auction Sequence: (no calls yet)\n"
        ),
        schema=StrategyState,
    )
    assert auction_state.phase == "auction"
    assert "15" in auction_state.plan_summary or "strong" in auction_state.plan_summary.lower()

    play_state = client.generate_structured(
        system_prompt="sys",
        user_prompt=(
            "Seat: SOUTH (Partner: NORTH)\n"
            "Phase: play\n"
            "Own Hand (9 HCP): S: Q 8 2 | H: A 7 3 | D: J 9 5 2 | C: 8 6 4\n"
            "Contract: 3NT by NORTH\n"
        ),
        schema=StrategyState,
    )
    assert play_state.phase == "play"

    decision = client.generate_structured(
        system_prompt="sys",
        user_prompt=(
            "=== Board Facts ===\n"
            "Observer: NORTH | Phase: auction\n"
            "Own Hand (16 HCP): S: A K 4 | H: K J 5 | D: Q 8 4 | C: A 9 3 2\n"
            "\n"
            "=== Candidate Options ===\n"
            "- Pass: baseline_ev=-50.0, make_prob=0.25\n"
            "- 1NT: baseline_ev=210.5, make_prob=0.75\n"
            "- 1C: baseline_ev=110.0, make_prob=0.62\n"
        ),
        schema=TacticalDecision,
    )
    assert decision.assumption_violated is False
    assert set(decision.option_scores.keys()) == {"Pass", "1NT", "1C"}
    assert decision.option_scores["1NT"] > decision.option_scores["1C"] > decision.option_scores["Pass"]
    for score in decision.option_scores.values():
        assert 0.0 <= score <= 1.0


def test_cli_requires_bluechip_cmd_when_opponent_is_bluechip():
    with pytest.raises(ValueError, match="--bluechip-cmd is required"):
        run_cli(["--mode", "mock", "--opponent", "bluechip", "--boards", "1"])

    with pytest.raises(ValueError, match="--bluechip-cmd is required"):
        run_cli(
            [
                "--mode",
                "mock",
                "--opponent",
                "bluechip",
                "--bluechip-cmd",
                "   ",
                "--boards",
                "1",
            ]
        )

    with pytest.raises(ValueError, match=r"\{port\}"):
        run_cli(
            [
                "--mode",
                "mock",
                "--opponent",
                "bluechip",
                "--bluechip-cmd",
                "wine /path/to/Wbridge5.exe Autoconnect 2000",
                "--boards",
                "1",
            ]
        )

    with pytest.raises(ValueError, match="--double-dummy cannot be combined"):
        run_cli(
            [
                "--mode",
                "mock",
                "--opponent",
                "bluechip",
                "--bluechip-cmd",
                "wine /path/to/Wbridge5.exe Autoconnect {port}",
                "--double-dummy",
                "--boards",
                "1",
            ]
        )


def test_cli_validates_positive_boards_and_dds_worlds():
    with pytest.raises(ValueError, match="--boards must be >= 1"):
        run_cli(["--mode", "mock", "--boards", "0"])

    with pytest.raises(ValueError, match="--dds-worlds must be >= 1"):
        run_cli(["--mode", "mock", "--dds-worlds", "0"])

    for bad_timeout in ("0", "-1", "nan", "inf", "-inf"):
        with pytest.raises(ValueError, match="--bluechip-timeout must be a finite number > 0"):
            run_cli(["--mode", "mock", f"--bluechip-timeout={bad_timeout}"])


def test_format_match_report_imp_margin_winner_and_tie():
    from tenace.harness import BoardResult, MatchSummary

    win_a = MatchSummary(
        tournament_seed=42,
        boards=[
            BoardResult(
                board_number=1,
                open_room_ns_score=410.0,
                closed_room_ns_score=0.0,
                imps_team_a=9,
            )
        ],
        total_imps_team_a=9,
        total_imps_team_b=-9,
        telemetry={"strategist_calls": 2, "evaluator_calls": 4, "forced_moves": 1},
    )
    report_a = format_match_report(win_a, team_a_name="Tenace", team_b_name="Baseline")
    assert "Winner: Tenace (+9 IMP margin)" in report_a
    assert "+18 IMP margin" not in report_a

    win_b = MatchSummary(
        tournament_seed=42,
        boards=[
            BoardResult(
                board_number=1,
                open_room_ns_score=-300.0,
                closed_room_ns_score=0.0,
                imps_team_a=-7,
            )
        ],
        total_imps_team_a=-7,
        total_imps_team_b=7,
        telemetry={},
    )
    report_b = format_match_report(win_b, team_a_name="Tenace", team_b_name="Opponent")
    assert "Winner: Opponent (+7 IMP margin)" in report_b
    assert "+14 IMP margin" not in report_b

    tie_summary = MatchSummary(
        tournament_seed=42,
        boards=[],
        total_imps_team_a=0,
        total_imps_team_b=0,
        telemetry={},
    )
    report_tie = format_match_report(tie_summary, team_a_name="Tenace", team_b_name="Opponent")
    assert "Result: Exact Tie" in report_tie


def test_cli_parser_defaults_to_vertex_and_gcp_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-bridge-gcp-proj")
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.mode == "vertex"
    assert args.project == "my-bridge-gcp-proj"
    assert args.strategist_location == "global"
    assert args.strategist_model == "gemini-2.5-pro"
    assert args.evaluator_backend == "diffusion-gemma"
    assert args.evaluator_location == "us-central1"
    assert args.evaluator_model == "publishers/google/models/diffusiongemma-26b-a4b-it"
    assert args.evaluator_endpoint_id is None
    assert args.opponent == "baseline"
    assert args.boards == 4
    assert args.seed == 42
    assert args.dds_worlds == 16
    assert args.double_dummy is False

    # Unset or empty GOOGLE_CLOUD_PROJECT falls back to your-gcp-project-id
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert build_arg_parser().parse_args([]).project == "your-gcp-project-id"

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    assert build_arg_parser().parse_args([]).project == "your-gcp-project-id"


def test_cli_vertex_mode_wires_gemini_and_diffusion_gemma_clients(monkeypatch):
    import tenace.cli as cli_mod

    gemini_inits: list[dict[str, Any]] = []
    diffusion_inits: list[dict[str, Any]] = []

    class FakeGeminiClient(HeuristicMockModelClient):
        def __init__(self, **kwargs: Any) -> None:
            gemini_inits.append(kwargs)

    class FakeDiffusionClient(HeuristicMockModelClient):
        def __init__(self, **kwargs: Any) -> None:
            diffusion_inits.append(kwargs)

    monkeypatch.setattr(cli_mod, "VertexGeminiClient", FakeGeminiClient)
    monkeypatch.setattr(cli_mod, "DiffusionGemmaClient", FakeDiffusionClient)

    # 1. Default --evaluator-backend diffusion-gemma with endpoint_id
    rc = run_cli(
        [
            "--mode",
            "vertex",
            "--project",
            "proj-alpha",
            "--strategist-location",
            "global",
            "--strategist-model",
            "gemini-2.5-pro",
            "--evaluator-backend",
            "diffusion-gemma",
            "--evaluator-location",
            "us-central1",
            "--evaluator-endpoint-id",
            "ep-98765",
            "--boards",
            "1",
            "--dds-worlds",
            "1",
            "--double-dummy",
        ]
    )
    assert rc == 0
    assert len(gemini_inits) == 1
    assert gemini_inits[0]["project"] == "proj-alpha"
    assert gemini_inits[0]["location"] == "global"
    assert gemini_inits[0]["model"] == "gemini-2.5-pro"
    assert len(diffusion_inits) == 1
    assert diffusion_inits[0] == {
        "project": "proj-alpha",
        "location": "us-central1",
        "endpoint_id": "ep-98765",
        "maas_model_name": "publishers/google/models/diffusiongemma-26b-a4b-it",
    }

    # 2. --evaluator-backend gemini WITHOUT overriding --evaluator-model -> defaults to gemini-2.5-flash
    gemini_inits.clear()
    diffusion_inits.clear()
    rc2 = run_cli(
        [
            "--mode",
            "vertex",
            "--project",
            "proj-beta",
            "--evaluator-backend",
            "gemini",
            "--evaluator-location",
            "global",
            "--boards",
            "1",
            "--dds-worlds",
            "1",
            "--double-dummy",
        ]
    )
    assert rc2 == 0
    assert len(diffusion_inits) == 0
    assert len(gemini_inits) == 2
    assert gemini_inits[0]["model"] == "gemini-2.5-pro"
    assert gemini_inits[1] == {
        "project": "proj-beta",
        "location": "global",
        "model": "gemini-2.5-flash",
        "temperature": 0.2,
    }

    # 3. --evaluator-backend gemini WITH explicit --evaluator-model override
    gemini_inits.clear()
    rc3 = run_cli(
        [
            "--mode",
            "vertex",
            "--project",
            "proj-gamma",
            "--evaluator-backend",
            "gemini",
            "--evaluator-model",
            "gemini-2.5-pro",
            "--boards",
            "1",
            "--dds-worlds",
            "1",
            "--double-dummy",
        ]
    )
    assert rc3 == 0
    assert len(gemini_inits) == 2
    assert gemini_inits[1]["model"] == "gemini-2.5-pro"


def test_cli_bluechip_opponent_end_to_end_with_subprocess_peer(capsys):
    cmd_template = f"{sys.executable} -c {shlex.quote(_BLUECHIP_SUBPROCESS_PEER)} {{port}}"
    exit_code = run_cli(
        [
            "--mode",
            "mock",
            "--opponent",
            "bluechip",
            "--bluechip-cmd",
            cmd_template,
            "--bluechip-timeout",
            "15.0",
            "--boards",
            "1",
            "--seed",
            "42",
            "--dds-worlds",
            "1",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "TENACE DUPLICATE BRIDGE MATCH REPORT" in out
    assert "Tenace vs BlueChip" in out
    assert "Board 1" in out
    assert "Open Room" in out
    assert "Closed Room" in out
    assert "Telemetry" in out


def test_main_exits_with_zero_on_mock_smoke(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["tenace-match", "--mode", "mock", "--boards", "1", "--dds-worlds", "2", "--double-dummy"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
