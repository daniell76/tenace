from __future__ import annotations

import shlex
import sys
from typing import Any

import pytest

from tenace.cli import (
    HeuristicMockModelClient,
    ModelProfile,
    build_arg_parser,
    build_clients_for_profile,
    format_match_report,
    main,
    resolve_model_profiles,
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

    # When Team B has 0 model telemetry, per-team telemetry lines are omitted
    single_tenace = MatchSummary(
        tournament_seed=42,
        boards=[],
        telemetry={
            "strategist_calls": 4,
            "evaluator_calls": 6,
            "forced_moves": 2,
            "team_a_strategist_calls": 4,
            "team_a_evaluator_calls": 6,
            "team_a_forced_moves": 2,
            "team_b_strategist_calls": 0,
            "team_b_evaluator_calls": 0,
            "team_b_forced_moves": 0,
        },
    )
    report_single = format_match_report(
        single_tenace,
        team_a_name="Tenace",
        team_b_name="Baseline",
        team_a_profile="L1:gemini-4-argon | L2:decisions/gemini-decision-1",
    )
    assert "Team A  : Tenace (L1:gemini-4-argon | L2:decisions/gemini-decision-1)" in report_single
    assert "Team B  :" not in report_single
    assert "Tenace: strategist_calls=" not in report_single

    # When Team B has >0 activity, both profile headers and per-team breakdown render
    h2h_summary = MatchSummary(
        tournament_seed=42,
        boards=[],
        telemetry={
            "strategist_calls": 9,
            "evaluator_calls": 15,
            "forced_moves": 5,
            "team_a_strategist_calls": 5,
            "team_a_evaluator_calls": 8,
            "team_a_forced_moves": 2,
            "team_b_strategist_calls": 4,
            "team_b_evaluator_calls": 7,
            "team_b_forced_moves": 3,
        },
    )
    report_h2h = format_match_report(
        h2h_summary,
        team_a_name="Tenace-A",
        team_b_name="Tenace-B",
        team_a_profile="L1:gemini-4-argon | L2:decisions/gemini-decision-1",
        team_b_profile="L1:gemini-4-argon | L2:diffusion-gemma/diffusiongemma-26b-a4b-it",
    )
    assert "Team A  : Tenace-A (L1:gemini-4-argon | L2:decisions/gemini-decision-1)" in report_h2h
    assert (
        "Team B  : Tenace-B (L1:gemini-4-argon | L2:diffusion-gemma/diffusiongemma-26b-a4b-it)"
        in report_h2h
    )
    assert "Tenace-A: strategist_calls=5, evaluator_calls=8, forced_moves=2" in report_h2h
    assert "Tenace-B: strategist_calls=4, evaluator_calls=7, forced_moves=3" in report_h2h


def test_cli_parser_defaults_to_vertex_and_gcp_env(monkeypatch):
    import tenace.cli as cli_mod

    assert "ModelProfile" in cli_mod.__all__
    assert "build_clients_for_profile" in cli_mod.__all__
    assert "resolve_model_profiles" in cli_mod.__all__

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-bridge-gcp-proj")
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.mode == "vertex"
    assert args.project == "my-bridge-gcp-proj"
    assert args.strategist_location == "global"
    assert args.strategist_model == "gemini-4-argon"
    assert args.evaluator_backend == "decisions"
    assert args.evaluator_location is None
    assert args.evaluator_model is None
    assert args.evaluator_endpoint_id is None
    assert args.decisions_auth == "vertex"
    assert args.decisions_violation_threshold == pytest.approx(0.70)
    assert args.opponent == "baseline"
    assert args.opponent_strategist_model is None
    assert args.opponent_evaluator_backend is None
    assert args.boards == 4
    assert args.seed == 42
    assert args.dds_worlds == 16
    assert args.double_dummy is False

    # Verify smart ModelProfile resolution across all 3 evaluator backends
    dec_profile, dec_opp = resolve_model_profiles(args)
    assert dec_opp is None
    assert dec_profile.evaluator_backend == "decisions"
    assert dec_profile.evaluator_location == "global"
    assert dec_profile.evaluator_model == "gemini-decision-1"
    assert dec_profile.short_label() == "L1:gemini-4-argon | L2:decisions/gemini-decision-1"

    diff_profile = ModelProfile(evaluator_backend="diffusion-gemma")
    assert diff_profile.evaluator_location == "us-central1"
    assert diff_profile.evaluator_model == "publishers/google/models/diffusiongemma-26b-a4b-it"
    assert (
        diff_profile.short_label()
        == "L1:gemini-4-argon | L2:diffusion-gemma/diffusiongemma-26b-a4b-it"
    )

    gem_profile = ModelProfile(evaluator_backend="gemini")
    assert gem_profile.evaluator_location == "global"
    assert gem_profile.evaluator_model == "gemini-2.5-flash"
    assert gem_profile.short_label() == "L1:gemini-4-argon | L2:gemini/gemini-2.5-flash"

    mock_profile = ModelProfile(mode="mock")
    assert (
        mock_profile.short_label()
        == "mock | L1:gemini-4-argon | L2:decisions/gemini-decision-1"
    )

    # Unset or empty GOOGLE_CLOUD_PROJECT falls back to your-gcp-project-id
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert build_arg_parser().parse_args([]).project == "your-gcp-project-id"

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    assert build_arg_parser().parse_args([]).project == "your-gcp-project-id"


def test_cli_vertex_mode_wires_gemini_and_diffusion_gemma_clients(monkeypatch):
    import tenace.cli as cli_mod

    gemini_inits: list[dict[str, Any]] = []
    diffusion_inits: list[dict[str, Any]] = []
    decisions_inits: list[dict[str, Any]] = []

    class FakeGeminiClient(HeuristicMockModelClient):
        def __init__(self, **kwargs: Any) -> None:
            gemini_inits.append(kwargs)

    class FakeDiffusionClient(HeuristicMockModelClient):
        def __init__(self, **kwargs: Any) -> None:
            diffusion_inits.append(kwargs)

    class FakeDecisionsClient(HeuristicMockModelClient):
        def __init__(self, **kwargs: Any) -> None:
            decisions_inits.append(kwargs)

    monkeypatch.setattr(cli_mod, "VertexGeminiClient", FakeGeminiClient)
    monkeypatch.setattr(cli_mod, "DiffusionGemmaClient", FakeDiffusionClient)
    monkeypatch.setattr(cli_mod, "GeminiDecisionsClient", FakeDecisionsClient)

    # 1. Default --evaluator-backend decisions with --decisions-auth vertex
    rc0 = run_cli(
        [
            "--mode",
            "vertex",
            "--project",
            "proj-decisions",
            "--boards",
            "1",
            "--dds-worlds",
            "1",
            "--double-dummy",
        ]
    )
    assert rc0 == 0
    assert len(gemini_inits) == 1
    assert gemini_inits[0] == {
        "project": "proj-decisions",
        "location": "global",
        "model": "gemini-4-argon",
    }
    assert len(decisions_inits) == 1
    assert decisions_inits[0] == {
        "project": "proj-decisions",
        "location": "global",
        "model": "gemini-decision-1",
        "vertexai": True,
        "violation_threshold": 0.70,
    }

    # 2. --evaluator-backend decisions with --decisions-auth api-key and threshold override
    gemini_inits.clear()
    decisions_inits.clear()
    rc0b = run_cli(
        [
            "--mode",
            "vertex",
            "--project",
            "proj-decisions",
            "--decisions-auth",
            "api-key",
            "--decisions-violation-threshold",
            "0.82",
            "--evaluator-model",
            "gemini-decision-exp",
            "--boards",
            "1",
            "--dds-worlds",
            "1",
            "--double-dummy",
        ]
    )
    assert rc0b == 0
    assert len(decisions_inits) == 1
    assert decisions_inits[0] == {
        "project": None,
        "location": "global",
        "model": "gemini-decision-exp",
        "vertexai": False,
        "violation_threshold": 0.82,
    }

    # 3. --evaluator-backend diffusion-gemma with endpoint_id (auto-resolving us-central1 & MaaS model)
    gemini_inits.clear()
    diffusion_inits.clear()
    decisions_inits.clear()
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

    # 4. --evaluator-backend gemini WITHOUT overriding --evaluator-model -> defaults to gemini-2.5-flash & global
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
    assert gemini_inits[0]["model"] == "gemini-4-argon"
    assert gemini_inits[1] == {
        "project": "proj-beta",
        "location": "global",
        "model": "gemini-2.5-flash",
        "temperature": 0.2,
    }

    # 5. --evaluator-backend gemini WITH explicit --evaluator-model override
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

    # 6. Validate --decisions-violation-threshold bounds (strictly between 0.0 and 1.0, finite)
    for bad_thresh in ("0.0", "1.0", "-0.2", "1.5", "nan", "inf", "-inf"):
        with pytest.raises(ValueError, match="threshold"):
            run_cli(["--mode", "mock", f"--decisions-violation-threshold={bad_thresh}"])
        with pytest.raises(ValueError, match="threshold"):
            run_cli(
                [
                    "--mode",
                    "mock",
                    "--opponent",
                    "tenace",
                    f"--opponent-decisions-violation-threshold={bad_thresh}",
                ]
            )

    bad_profile = ModelProfile.model_construct(
        mode="mock",
        project="proj",
        strategist_location="global",
        strategist_model="gemini-4-argon",
        evaluator_backend="decisions",
        evaluator_location="global",
        evaluator_model="gemini-decision-1",
        evaluator_endpoint_id=None,
        decisions_auth="vertex",
        decisions_violation_threshold=1.0,
    )
    with pytest.raises(ValueError, match="threshold"):
        build_clients_for_profile(bad_profile)


def test_cli_head_to_head_tenace_opponent_compares_two_model_profiles(monkeypatch, capsys):
    import tenace.cli as cli_mod

    gemini_inits: list[dict[str, Any]] = []
    diffusion_inits: list[dict[str, Any]] = []
    decisions_inits: list[dict[str, Any]] = []

    class FakeGeminiClient(HeuristicMockModelClient):
        def __init__(self, **kwargs: Any) -> None:
            gemini_inits.append(kwargs)

    class FakeDiffusionClient(HeuristicMockModelClient):
        def __init__(self, **kwargs: Any) -> None:
            diffusion_inits.append(kwargs)

    class FakeDecisionsClient(HeuristicMockModelClient):
        def __init__(self, **kwargs: Any) -> None:
            decisions_inits.append(kwargs)

    monkeypatch.setattr(cli_mod, "VertexGeminiClient", FakeGeminiClient)
    monkeypatch.setattr(cli_mod, "DiffusionGemmaClient", FakeDiffusionClient)
    monkeypatch.setattr(cli_mod, "GeminiDecisionsClient", FakeDecisionsClient)

    # 1. Vertex head-to-head: Team A (gemini-4-argon + decisions/gemini-decision-1)
    #    vs Team B (gemini-2.5-pro + diffusion-gemma/diffusiongemma-26b-a4b-it)
    rc_vertex = run_cli(
        [
            "--mode",
            "vertex",
            "--project",
            "proj-h2h",
            "--opponent",
            "tenace",
            "--opponent-strategist-model",
            "gemini-2.5-pro",
            "--opponent-evaluator-backend",
            "diffusion-gemma",
            "--boards",
            "1",
            "--dds-worlds",
            "1",
            "--double-dummy",
        ]
    )
    assert rc_vertex == 0
    assert len(gemini_inits) == 2
    assert gemini_inits[0]["model"] == "gemini-4-argon"
    assert gemini_inits[1]["model"] == "gemini-2.5-pro"
    assert len(decisions_inits) == 1
    assert decisions_inits[0] == {
        "project": "proj-h2h",
        "location": "global",
        "model": "gemini-decision-1",
        "vertexai": True,
        "violation_threshold": 0.70,
    }
    assert len(diffusion_inits) == 1
    assert diffusion_inits[0] == {
        "project": "proj-h2h",
        "location": "us-central1",
        "endpoint_id": None,
        "maas_model_name": "publishers/google/models/diffusiongemma-26b-a4b-it",
    }

    out_vertex = capsys.readouterr().out
    assert "Matchup : Tenace-A vs Tenace-B" in out_vertex
    assert "Team A  : Tenace-A (L1:gemini-4-argon | L2:decisions/gemini-decision-1)" in out_vertex
    assert (
        "Team B  : Tenace-B (L1:gemini-2.5-pro | L2:diffusion-gemma/diffusiongemma-26b-a4b-it)"
        in out_vertex
    )
    assert "Tenace-A: strategist_calls=" in out_vertex
    assert "Tenace-B: strategist_calls=" in out_vertex

    # 2. Mock head-to-head full card play smoke
    rc_mock = run_cli(
        [
            "--mode",
            "mock",
            "--opponent",
            "tenace",
            "--opponent-evaluator-backend",
            "diffusion-gemma",
            "--boards",
            "1",
            "--dds-worlds",
            "1",
        ]
    )
    assert rc_mock == 0
    out_mock = capsys.readouterr().out
    assert "Matchup : Tenace-A vs Tenace-B" in out_mock
    assert (
        "Team A  : Tenace-A (mock | L1:gemini-4-argon | L2:decisions/gemini-decision-1)"
        in out_mock
    )
    assert (
        "Team B  : Tenace-B (mock | L1:gemini-4-argon | L2:diffusion-gemma/diffusiongemma-26b-a4b-it)"
        in out_mock
    )
    assert "Tenace-A: strategist_calls=" in out_mock
    assert "Tenace-B: strategist_calls=" in out_mock


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


def test_cli_rejects_opponent_override_flags_when_opponent_is_not_tenace():
    opponent_flag_cases = [
        ["--opponent-strategist-location", "us-central1"],
        ["--opponent-strategist-model", "gemini-2.5-pro"],
        ["--opponent-evaluator-backend", "diffusion-gemma"],
        ["--opponent-evaluator-location", "us-central1"],
        ["--opponent-evaluator-model", "gemini-2.5-flash"],
        ["--opponent-evaluator-endpoint-id", "ep-123"],
        ["--opponent-decisions-auth", "api-key"],
        ["--opponent-decisions-violation-threshold", "0.80"],
    ]
    for flag_pair in opponent_flag_cases:
        args = build_arg_parser().parse_args(["--opponent", "baseline", *flag_pair])
        with pytest.raises(ValueError, match="--opponent tenace"):
            resolve_model_profiles(args)
        with pytest.raises(ValueError, match="--opponent tenace"):
            run_cli(["--mode", "mock", "--opponent", "baseline", "--boards", "1", *flag_pair])


def test_model_profile_endpoint_and_decisions_auth_backend_compatibility_and_short_labels():
    # 1. evaluator_endpoint_id rejected on non-diffusion-gemma backends
    for incompatible_backend in ("decisions", "gemini"):
        with pytest.raises(ValueError, match="evaluator_endpoint_id"):
            ModelProfile(
                evaluator_backend=incompatible_backend,  # type: ignore[arg-type]
                evaluator_endpoint_id="ep-123",
            )
        with pytest.raises(ValueError, match="evaluator_endpoint_id"):
            run_cli(
                [
                    "--mode",
                    "mock",
                    "--evaluator-backend",
                    incompatible_backend,
                    "--evaluator-endpoint-id",
                    "ep-123",
                    "--boards",
                    "1",
                ]
            )

    # 2. short_label() surfaces endpoint ID for diffusion-gemma and shortens slash resource names
    ep_profile = ModelProfile(
        evaluator_backend="diffusion-gemma",
        evaluator_endpoint_id="  ep-12345  ",
    )
    assert ep_profile.evaluator_endpoint_id == "ep-12345"
    assert ep_profile.short_label() == "L1:gemini-4-argon | L2:diffusion-gemma/endpoint:ep-12345"

    mock_ep_profile = ModelProfile(
        mode="mock",
        evaluator_backend="diffusion-gemma",
        evaluator_endpoint_id="ep-999",
    )
    assert (
        mock_ep_profile.short_label()
        == "mock | L1:gemini-4-argon | L2:diffusion-gemma/endpoint:ep-999"
    )

    full_path_profile = ModelProfile(
        strategist_model="projects/p/locations/global/publishers/google/models/gemini-4-carbon",
        evaluator_backend="gemini",
        evaluator_model="projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash",
    )
    assert full_path_profile.short_label() == "L1:gemini-4-carbon | L2:gemini/gemini-2.5-flash"

    # 3. decisions_auth != "vertex" rejected on non-decisions backends
    for incompatible_backend in ("diffusion-gemma", "gemini"):
        with pytest.raises(ValueError, match="decisions_auth"):
            ModelProfile(
                evaluator_backend=incompatible_backend,  # type: ignore[arg-type]
                decisions_auth="api-key",
            )

    # 4. Cross-backend inheritance in resolve_model_profiles:
    #    Team A uses decisions + api-key, Team B switches to diffusion-gemma without --opponent-decisions-auth
    parser = build_arg_parser()
    cross_args = parser.parse_args(
        [
            "--evaluator-backend",
            "decisions",
            "--decisions-auth",
            "api-key",
            "--opponent",
            "tenace",
            "--opponent-evaluator-backend",
            "diffusion-gemma",
        ]
    )
    team_a, team_b = resolve_model_profiles(cross_args)
    assert team_b is not None
    assert team_a.decisions_auth == "api-key"
    assert team_b.evaluator_backend == "diffusion-gemma"
    assert team_b.decisions_auth == "vertex"

    #    Same backend (decisions): Team B inherits Team A's api-key when omitted
    same_args = parser.parse_args(
        [
            "--evaluator-backend",
            "decisions",
            "--decisions-auth",
            "api-key",
            "--opponent",
            "tenace",
            "--opponent-strategist-model",
            "gemini-2.5-pro",
        ]
    )
    _, team_b_same = resolve_model_profiles(same_args)
    assert team_b_same is not None
    assert team_b_same.decisions_auth == "api-key"

    #    Explicit --opponent-decisions-auth api-key with --opponent-evaluator-backend diffusion-gemma fails
    bad_opp_auth_args = parser.parse_args(
        [
            "--opponent",
            "tenace",
            "--opponent-evaluator-backend",
            "diffusion-gemma",
            "--opponent-decisions-auth",
            "api-key",
        ]
    )
    with pytest.raises(ValueError, match="decisions_auth"):
        resolve_model_profiles(bad_opp_auth_args)


def test_model_profile_strips_strings_and_rejects_blank_values(monkeypatch):
    padded = ModelProfile(
        project="  my-proj  ",
        strategist_location="  global  ",
        strategist_model="  gemini-4-argon  ",
        evaluator_backend="diffusion-gemma",
        evaluator_location="  us-central1  ",
        evaluator_model="  publishers/google/models/diffusiongemma-26b-a4b-it  ",
        evaluator_endpoint_id="  ep-42  ",
    )
    assert padded.project == "my-proj"
    assert padded.strategist_location == "global"
    assert padded.strategist_model == "gemini-4-argon"
    assert padded.evaluator_location == "us-central1"
    assert padded.evaluator_model == "publishers/google/models/diffusiongemma-26b-a4b-it"
    assert padded.evaluator_endpoint_id == "ep-42"

    # None project remains allowed
    none_proj = ModelProfile(project=None)
    assert none_proj.project is None

    for blank in ("", "   "):
        with pytest.raises(ValueError, match="strategist_model"):
            ModelProfile(strategist_model=blank)
        with pytest.raises(ValueError, match="strategist_location"):
            ModelProfile(strategist_location=blank)
        with pytest.raises(ValueError, match="evaluator_model"):
            ModelProfile(evaluator_model=blank)
        with pytest.raises(ValueError, match="evaluator_location"):
            ModelProfile(evaluator_location=blank)
        with pytest.raises(ValueError, match="project"):
            ModelProfile(project=blank)
        with pytest.raises(ValueError, match="evaluator_endpoint_id"):
            ModelProfile(evaluator_backend="diffusion-gemma", evaluator_endpoint_id=blank)

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "   ")
    assert build_arg_parser().parse_args([]).project == "your-gcp-project-id"


def test_format_match_report_renders_per_team_telemetry_when_team_b_profile_present_even_with_zero_activity():
    from tenace.harness import MatchSummary

    zero_b_summary = MatchSummary(
        tournament_seed=42,
        boards=[],
        telemetry={
            "strategist_calls": 2,
            "evaluator_calls": 2,
            "forced_moves": 4,
            "team_a_strategist_calls": 2,
            "team_a_evaluator_calls": 2,
            "team_a_forced_moves": 4,
            "team_b_strategist_calls": 0,
            "team_b_evaluator_calls": 0,
            "team_b_forced_moves": 0,
        },
    )
    h2h_report = format_match_report(
        zero_b_summary,
        team_a_name="Tenace-A",
        team_b_name="Tenace-B",
        team_a_profile="L1:gemini-4-argon | L2:decisions/gemini-decision-1",
        team_b_profile="L1:gemini-4-argon | L2:diffusion-gemma/diffusiongemma-26b-a4b-it",
    )
    assert "Tenace-A: strategist_calls=2, evaluator_calls=2, forced_moves=4" in h2h_report
    assert "Tenace-B: strategist_calls=0, evaluator_calls=0, forced_moves=0" in h2h_report
