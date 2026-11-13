from __future__ import annotations

import inspect
from pathlib import Path
import tomllib
from typing import TypeVar

from fastapi.testclient import TestClient
import pyspiel
import pytest
from pydantic import BaseModel

import tenace
from tenace.agent import TenaceAgent
from tenace.cli import DEFAULT_DIFFUSION_MODEL, ModelProfile, build_clients_for_profile
from tenace.domain.types import (
    MAX_ACTION,
    NUM_CARDS,
    PASS_ACTION,
    Seat,
    StrategyState,
    Suit,
    action_label,
    card_id,
)
from tenace.harness import DuplicateMatchRunner
from tenace.layer1_strategist import SAYC_CONVENTION_CARD
from tenace.layer3_rules import RuleDDSEvaluator, RuleFactTracker
import tenace.server as server_mod
from tenace.server import (
    create_app,
    default_dds_worlds_from_env,
    default_profile_from_env,
    main,
    parse_action_token,
    parse_card_token,
    resolve_server_bind,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mock_profile(**overrides: object) -> ModelProfile:
    data: dict[str, object] = {"mode": "mock", **overrides}
    return ModelProfile(**data)  # type: ignore[arg-type]


def _dealt_hands(state: pyspiel.State) -> dict[Seat, list[int]]:
    dealt = state.full_history()[:NUM_CARDS]
    return {
        seat: [dealt[4 * r + int(seat)].action for r in range(13)]
        for seat in Seat
    }


def test_public_exports_and_console_entrypoint() -> None:
    assert hasattr(tenace, "create_app")
    assert tenace.create_app is create_app
    assert "create_app" in tenace.__all__
    assert set(server_mod.__all__) >= {
        "MatchRequest",
        "MatchResponse",
        "StepRequest",
        "StepResponse",
        "create_app",
        "default_dds_worlds_from_env",
        "default_profile_from_env",
        "main",
        "parse_action_token",
        "parse_card_token",
        "resolve_server_bind",
    }

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert scripts["tenace-serve"] == "tenace.server:main"
    assert scripts["tenace-match"] == "tenace.cli:main"


def test_parse_card_and_action_tokens_round_trip_and_reject_invalid() -> None:
    for c in range(NUM_CARDS):
        lbl = action_label(c)
        assert parse_card_token(c) == c
        assert parse_card_token(lbl) == c
        assert parse_card_token(f"  {lbl.lower()} ") == c

    for a in range(MAX_ACTION + 1):
        lbl = action_label(a)
        assert parse_action_token(a) == a
        assert parse_action_token(lbl) == a
        assert parse_action_token(f" {lbl.lower()}  ") == a

    for bad_card in (True, False, -1, 52, "Pass", "1NT", "", "Z:A", "SA"):
        with pytest.raises(ValueError):
            parse_card_token(bad_card)  # type: ignore[arg-type]

    for bad_action in (True, False, -1, 90, "", "8NT", "PassOut", "C:1"):
        with pytest.raises(ValueError):
            parse_action_token(bad_action)  # type: ignore[arg-type]


def test_env_profile_dds_worlds_and_server_bind_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "TENACE_MODE",
        "GOOGLE_CLOUD_PROJECT",
        "TENACE_STRATEGIST_LOCATION",
        "TENACE_STRATEGIST_MODEL",
        "TENACE_EVALUATOR_BACKEND",
        "TENACE_EVALUATOR_LOCATION",
        "TENACE_EVALUATOR_MODEL",
        "TENACE_EVALUATOR_ENDPOINT_ID",
        "TENACE_DECISIONS_AUTH",
        "TENACE_DECISIONS_VIOLATION_THRESHOLD",
        "TENACE_DDS_WORLDS",
        "HOST",
        "PORT",
    ):
        monkeypatch.delenv(key, raising=False)

    prof = default_profile_from_env()
    assert prof.mode == "vertex"
    assert prof.project is None
    assert prof.strategist_location == "global"
    assert prof.strategist_model == "gemini-4-argon"
    assert prof.evaluator_backend == "decisions"
    assert prof.evaluator_location == "global"
    assert prof.evaluator_model == "gemini-decision-1"
    assert prof.decisions_auth == "vertex"
    assert prof.decisions_violation_threshold == pytest.approx(0.70)
    assert default_dds_worlds_from_env() == 16
    assert resolve_server_bind() == ("0.0.0.0", 8080)

    custom_prof = default_profile_from_env(
        {
            "TENACE_MODE": " mock ",
            "GOOGLE_CLOUD_PROJECT": " example-bridge-project ",
            "TENACE_STRATEGIST_LOCATION": "us-central1",
            "TENACE_STRATEGIST_MODEL": "gemini-2.5-pro",
            "TENACE_EVALUATOR_BACKEND": "diffusion-gemma",
            "TENACE_EVALUATOR_ENDPOINT_ID": " 1234567890 ",
            "TENACE_DECISIONS_VIOLATION_THRESHOLD": "0.65",
        }
    )
    assert custom_prof.mode == "mock"
    assert custom_prof.project == "example-bridge-project"
    assert custom_prof.strategist_location == "us-central1"
    assert custom_prof.strategist_model == "gemini-2.5-pro"
    assert custom_prof.evaluator_backend == "diffusion-gemma"
    assert custom_prof.evaluator_location == "us-central1"
    assert custom_prof.evaluator_model == DEFAULT_DIFFUSION_MODEL
    assert custom_prof.evaluator_endpoint_id == "1234567890"
    assert custom_prof.decisions_violation_threshold == pytest.approx(0.65)

    assert default_dds_worlds_from_env({"TENACE_DDS_WORLDS": " 8 "}) == 8
    assert resolve_server_bind({"HOST": "127.0.0.1", "PORT": "9090"}) == (
        "127.0.0.1",
        9090,
    )

    for bad_env in (
        {"TENACE_MODE": "bogus"},
        {"TENACE_DECISIONS_VIOLATION_THRESHOLD": "not-a-float"},
        {"TENACE_DECISIONS_VIOLATION_THRESHOLD": "1.2"},
    ):
        with pytest.raises(ValueError):
            default_profile_from_env(bad_env)

    for bad_worlds in ({"TENACE_DDS_WORLDS": "0"}, {"TENACE_DDS_WORLDS": "abc"}):
        with pytest.raises(ValueError):
            default_dds_worlds_from_env(bad_worlds)

    for bad_bind in (
        {"HOST": "   "},
        {"PORT": ""},
        {"PORT": "abc"},
        {"PORT": "0"},
        {"PORT": "65536"},
    ):
        with pytest.raises(ValueError):
            resolve_server_bind(bad_bind)


def test_main_invokes_uvicorn_with_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}

    def fake_run(app_target: str, **kwargs: object) -> None:
        recorded["app_target"] = app_target
        recorded.update(kwargs)

    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "8181")
    monkeypatch.setattr("uvicorn.run", fake_run)

    main()
    assert recorded == {
        "app_target": "tenace.server:create_app",
        "factory": True,
        "host": "0.0.0.0",
        "port": 8181,
    }


def test_healthz_reports_default_profile_and_env_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TENACE_MODE", "mock")
    monkeypatch.setenv("TENACE_STRATEGIST_MODEL", "gemini-4-argon")
    monkeypatch.setenv("TENACE_EVALUATOR_BACKEND", "gemini")
    monkeypatch.setenv("TENACE_DDS_WORLDS", "6")

    client = TestClient(create_app())
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "status": "ok",
        "service": "tenace",
        "version": "0.1.0",
        "default_profile": "mock | L1:gemini-4-argon | L2:gemini/gemini-2.5-flash",
        "mode": "mock",
        "dds_worlds": 6,
    }


def test_step_imperfect_info_auction_and_benchmark_equivalence() -> None:
    app = create_app(default_profile=_mock_profile(), default_dds_worlds=1)
    client = TestClient(app)

    # Board 2 has East as dealer under WBF rotation
    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    board2 = runner._new_dealt_board_state(2)
    hands = _dealt_hands(board2)
    east_labels = [action_label(c) for c in hands[Seat.EAST]]

    resp_imp = client.post(
        "/v1/step",
        json={
            "seat": "EAST",
            "board_number": 2,
            "own_hand": east_labels,
            "action_history": [],
            "seed": 99,
        },
    )
    assert resp_imp.status_code == 200
    data_imp = resp_imp.json()
    assert data_imp["seat"] == "EAST"
    assert data_imp["phase"] == "auction"
    assert data_imp["action_id"] in board2.legal_actions()
    assert data_imp["action_label"] == action_label(data_imp["action_id"])
    assert data_imp["strategy"]["phase"] == "auction"
    assert data_imp["telemetry"] == {
        "strategist_calls": 1,
        "evaluator_calls": 1,
        "forced_moves": 0,
    }

    resp_bench = client.post(
        "/v1/step",
        json={
            "seat": "EAST",
            "board_number": 2,
            "tournament_seed": 42,
            "action_history": [],
            "seed": 99,
        },
    )
    assert resp_bench.status_code == 200
    data_bench = resp_bench.json()
    assert data_imp == data_bench


def test_step_pre_lead_play_and_post_lead_dummy_and_declarer_turns() -> None:
    app = create_app(default_profile=_mock_profile(), default_dds_worlds=4)
    client = TestClient(app)

    runner = DuplicateMatchRunner(tournament_seed=7, use_double_dummy_result=False)
    state = runner._new_dealt_board_state(1)
    hands = _dealt_hands(state)

    # North opens 1NT (59), East/South/West Pass -> East is on opening lead (dummy NOT yet visible)
    auction = ["1NT", "Pass", "Pass", "Pass"]
    for tok in auction:
        state.apply_action(parse_action_token(tok))

    pre_lead_resp = client.post(
        "/v1/step",
        json={
            "seat": "EAST",
            "board_number": 1,
            "own_hand": [action_label(c) for c in hands[Seat.EAST]],
            "action_history": auction,
            "seed": 12,
        },
    )
    assert pre_lead_resp.status_code == 200
    lead_data = pre_lead_resp.json()
    assert lead_data["phase"] == "play"
    assert lead_data["action_id"] in hands[Seat.EAST]

    # Once East leads, dummy (SOUTH) is revealed and declarer (NORTH) chooses dummy's card!
    lead_card = lead_data["action_id"]
    state.apply_action(lead_card)
    history_after_lead = [*auction, action_label(lead_card)]

    dummy_turn_resp = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": [action_label(c) for c in hands[Seat.NORTH]],
            "dummy_hand": [action_label(c) for c in hands[Seat.SOUTH]],
            "action_history": history_after_lead,
            "seed": 12,
        },
    )
    assert dummy_turn_resp.status_code == 200
    dummy_turn_data = dummy_turn_resp.json()
    assert dummy_turn_data["seat"] == "NORTH"
    assert dummy_turn_data["phase"] == "play"
    assert dummy_turn_data["action_id"] in state.legal_actions()
    assert dummy_turn_data["action_id"] in hands[Seat.SOUTH]

    # Play dummy's card and West's card so it is North's turn from North's own hand
    dummy_card = dummy_turn_data["action_id"]
    state.apply_action(dummy_card)
    west_card = state.legal_actions()[0]
    state.apply_action(west_card)
    history_card_4 = [
        *history_after_lead,
        action_label(dummy_card),
        action_label(west_card),
    ]

    # Pass prior_strategy from North's previous play turn; verify 0 Strategist calls when no new showout!
    prior_strategy = dummy_turn_data["strategy"]
    tracker = RuleFactTracker()
    facts_now = tracker.extract_facts(state, Seat.NORTH)
    west_showed_out = bool(facts_now.known_voids[Seat.WEST])

    declarer_own_resp = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": [action_label(c) for c in hands[Seat.NORTH]],
            "dummy_hand": [action_label(c) for c in hands[Seat.SOUTH]],
            "action_history": history_card_4,
            "prior_strategy": prior_strategy,
            "seed": 12,
        },
    )
    assert declarer_own_resp.status_code == 200
    own_turn_data = declarer_own_resp.json()
    assert own_turn_data["action_id"] in state.legal_actions()
    assert own_turn_data["action_id"] in hands[Seat.NORTH]
    if prior_strategy is not None and not west_showed_out and own_turn_data["telemetry"]["forced_moves"] == 0:
        assert own_turn_data["telemetry"]["strategist_calls"] == 0
        assert own_turn_data["strategy"]["revision"] == prior_strategy["revision"]


def test_step_reconstructs_hidden_seat_showout_and_triggers_replan_on_surprise() -> None:
    app = create_app(default_profile=_mock_profile(), default_dds_worlds=4)
    client = TestClient(app)

    # Construct custom deal where West is void in Clubs and ruffs/discards on Trick 1!
    # North: C:3, C:4, H:2..H:Q (2 clubs + 11 hearts = 13 cards)
    # East : C:A, C:K, D:2..D:Q (2 clubs + 11 diamonds = 13 cards)
    # South (dummy): C:2, C:5, S:2..S:Q (2 clubs + 11 spades = 13 cards)
    # West : NO clubs! C:6..C:Q (7 clubs) must ALL go into East if we test West void!
    # Let's build exact disjoint 52-card partition where West has 0 Clubs while 7 unplayed Clubs remain in East!
    north_cards = [
        card_id(Suit.CLUBS, 1),   # C:3
        card_id(Suit.CLUBS, 2),   # C:4
        *[card_id(Suit.HEARTS, r) for r in range(0, 11)],  # H:2..H:Q (11)
    ]
    south_cards = [
        card_id(Suit.CLUBS, 0),   # C:2
        card_id(Suit.CLUBS, 3),   # C:5
        *[card_id(Suit.SPADES, r) for r in range(0, 11)],  # S:2..S:Q (11)
    ]
    # Remaining 26 cards: C:6..C:A (9 clubs), D:2..D:A (13 diamonds), H:K, H:A, S:K, S:A.
    # East leads C:A (rank 12), South plays C:2, West discards D:2 (showing West is VOID in Clubs!).
    # Notice 8 unplayed Clubs (C:6..C:K) remain unseen! Imperfect-info filler MUST place all 8 unplayed
    # Clubs into East (and NONE into West), otherwise OpenSpiel would reject West's D:2 discard!
    auction = ["1NT", "Pass", "Pass", "Pass"]
    lead_ca = action_label(card_id(Suit.CLUBS, 12))

    # Step 1: North chooses dummy's play at Trick 1 Card 1 -> creates rev-1 play StrategyState
    r1 = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": [action_label(c) for c in north_cards],
            "dummy_hand": [action_label(c) for c in south_cards],
            "action_history": [*auction, lead_ca],
            "seed": 5,
        },
    )
    assert r1.status_code == 200
    strat_rev1 = r1.json()["strategy"]
    assert strat_rev1["phase"] == "play"
    assert strat_rev1["revision"] == 1
    assert r1.json()["telemetry"]["strategist_calls"] == 1

    # Step 2 (control, no showout): West follows C:6 -> North's own-hand play has 0 Strategist calls!
    r2_routine = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": [action_label(c) for c in north_cards],
            "dummy_hand": [action_label(c) for c in south_cards],
            "action_history": [*auction, lead_ca, "C:2", "C:6"],
            "prior_strategy": strat_rev1,
            "seed": 5,
        },
    )
    # Note: North holds C:3 and C:4 (contiguous -> collapsed to 1 forced move!). Let's give North
    # non-contiguous Clubs (C:3 and C:K) so North has 2 distinct tactical options!
    north_non_contig = [
        card_id(Suit.CLUBS, 1),   # C:3
        card_id(Suit.CLUBS, 11),  # C:K
        *[card_id(Suit.HEARTS, r) for r in range(0, 11)],
    ]
    r1_nc = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": [action_label(c) for c in north_non_contig],
            "dummy_hand": [action_label(c) for c in south_cards],
            "action_history": [*auction, lead_ca],
            "seed": 5,
        },
    )
    assert r1_nc.status_code == 200
    strat_nc = r1_nc.json()["strategy"]

    r2_nc_routine = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": [action_label(c) for c in north_non_contig],
            "dummy_hand": [action_label(c) for c in south_cards],
            "action_history": [*auction, lead_ca, "C:2", "C:6"],
            "prior_strategy": strat_nc,
            "seed": 5,
        },
    )
    assert r2_nc_routine.status_code == 200
    assert r2_nc_routine.json()["telemetry"] == {
        "strategist_calls": 0,
        "evaluator_calls": 1,
        "forced_moves": 0,
    }
    assert r2_nc_routine.json()["strategy"]["revision"] == 1

    # Step 3 (surprise showout!): West discards D:2 on C:A lead while 7 unplayed Clubs remain unseen!
    r2_nc_showout = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": [action_label(c) for c in north_non_contig],
            "dummy_hand": [action_label(c) for c in south_cards],
            "action_history": [*auction, lead_ca, "C:2", "D:2"],
            "prior_strategy": strat_nc,
            "seed": 5,
        },
    )
    assert r2_nc_showout.status_code == 200
    showout_body = r2_nc_showout.json()
    assert showout_body["telemetry"] == {
        "strategist_calls": 1,
        "evaluator_calls": 1,
        "forced_moves": 0,
    }
    assert showout_body["strategy"]["revision"] == 2


def test_step_stateless_loop_matches_stateful_agent_telemetry_on_board() -> None:
    profile = _mock_profile()
    app = create_app(default_profile=profile, default_dds_worlds=1)
    client = TestClient(app)

    runner = DuplicateMatchRunner(tournament_seed=19, use_double_dummy_result=False)
    state_ref = runner._new_dealt_board_state(1)
    strat_c, eval_c = build_clients_for_profile(profile)
    agents = {
        seat: TenaceAgent(
            seat=seat,
            strategist_client=strat_c,
            evaluator_client=eval_c,
            dds_evaluator=RuleDDSEvaluator(num_worlds=1, rng_seed=100),
            convention_card=SAYC_CONVENTION_CARD,
        )
        for seat in Seat
    }

    prior_by_seat: dict[Seat, dict[str, object] | None] = {s: None for s in Seat}
    history: list[int] = []
    # Seed initial 3 calls (1NT - Pass - Pass) so we step through final auction call,
    # opening lead (pre-dummy), Trick 1 (dummy reveal + declarer reuse), and Trick 2!
    for initial_call in ("1NT", "Pass", "Pass"):
        act_id = parse_action_token(initial_call)
        acting = Seat(state_ref.current_player())
        agents[acting].dds_evaluator = RuleDDSEvaluator(num_worlds=1, rng_seed=50)
        if initial_call == "1NT":
            # Prime North's stateful auction strategy so both sides start synchronized
            r_init = client.post(
                "/v1/step",
                json={
                    "seat": acting.name,
                    "board_number": 1,
                    "tournament_seed": 19,
                    "action_history": history,
                    "prior_strategy": prior_by_seat[acting],
                    "seed": 50,
                },
            )
            assert r_init.status_code == 200
            prior_by_seat[acting] = r_init.json()["strategy"]
            agents[acting].step(state_ref.clone())
        state_ref.apply_action(act_id)
        history.append(act_id)

    max_steps = 10
    steps_run = 0
    while not state_ref.is_terminal() and steps_run < max_steps:
        acting = Seat(state_ref.current_player())
        turn_seed = 200 + steps_run
        agents[acting].dds_evaluator = RuleDDSEvaluator(
            num_worlds=1,
            rng_seed=turn_seed,
        )

        resp = client.post(
            "/v1/step",
            json={
                "seat": acting.name,
                "board_number": 1,
                "tournament_seed": 19,
                "action_history": history,
                "prior_strategy": prior_by_seat[acting],
                "seed": turn_seed,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        prior_by_seat[acting] = body["strategy"]

        prev_strat_calls = agents[acting].strategist_calls
        prev_eval_calls = agents[acting].evaluator_calls
        prev_forced = agents[acting].forced_moves
        expected_action = agents[acting].step(state_ref.clone())

        assert body["action_id"] == expected_action
        assert body["telemetry"] == {
            "strategist_calls": agents[acting].strategist_calls - prev_strat_calls,
            "evaluator_calls": agents[acting].evaluator_calls - prev_eval_calls,
            "forced_moves": agents[acting].forced_moves - prev_forced,
        }
        if agents[acting].strategy is not None:
            assert body["strategy"] == agents[acting].strategy.model_dump(mode="json")

        state_ref.apply_action(expected_action)
        history.append(expected_action)
        steps_run += 1


def test_step_rejects_invalid_requests_with_422() -> None:
    app = create_app(default_profile=_mock_profile(), default_dds_worlds=2)
    client = TestClient(app)

    hand_n = [action_label(c) for c in range(0, 13)]
    hand_s = [action_label(c) for c in range(26, 39)]

    with pytest.raises(ValueError):
        server_mod.StepRequest(seat="NORTH", board_number=1, own_hand=hand_n, dds_worlds=257)

    invalid_payloads: list[dict[str, object]] = [
        # Neither own_hand nor tournament_seed
        {"seat": "NORTH", "board_number": 1},
        # Both own_hand and tournament_seed
        {"seat": "NORTH", "board_number": 1, "own_hand": hand_n, "tournament_seed": 42},
        # dummy_hand with tournament_seed
        {"seat": "NORTH", "board_number": 1, "tournament_seed": 42, "dummy_hand": hand_s},
        # Wrong hand size (<13)
        {"seat": "NORTH", "board_number": 1, "own_hand": hand_n[:12]},
        # Duplicate card in own_hand
        {"seat": "NORTH", "board_number": 1, "own_hand": [hand_n[0], *hand_n[:12]]},
        # Overlapping own_hand and dummy_hand
        {
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": hand_n,
            "dummy_hand": [hand_n[0], *hand_s[1:]],
            "action_history": ["1NT", "Pass", "Pass", "Pass", "D:5"],
        },
        # Premature dummy_hand during auction
        {
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": hand_n,
            "dummy_hand": hand_s,
            "action_history": [],
        },
        # Premature dummy_hand before opening lead
        {
            "seat": "EAST",
            "board_number": 1,
            "own_hand": [action_label(c) for c in range(13, 26)],
            "dummy_hand": hand_s,
            "action_history": ["1NT", "Pass", "Pass", "Pass"],
        },
        # Missing dummy_hand after opening lead
        {
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": hand_n,
            "action_history": ["1NT", "Pass", "Pass", "Pass", action_label(13)],
        },
        # Dummy seat submitted directly after opening lead
        {
            "seat": "SOUTH",
            "board_number": 1,
            "own_hand": hand_s,
            "dummy_hand": hand_s,
            "action_history": ["1NT", "Pass", "Pass", "Pass", action_label(13)],
        },
        # Wrong turn seat on Board 1 start (North deals, not East)
        {"seat": "EAST", "board_number": 1, "own_hand": hand_n, "action_history": []},
        # Illegal auction action (lower contract bid 1C after 1NT)
        {
            "seat": "SOUTH",
            "board_number": 1,
            "own_hand": hand_s,
            "action_history": ["1NT", "1C"],
        },
        # Bidding action after card play started
        {
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": hand_n,
            "dummy_hand": hand_s,
            "action_history": ["1NT", "Pass", "Pass", "Pass", action_label(13), "Pass"],
        },
        # Terminal passed-out board
        {
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": hand_n,
            "action_history": ["Pass", "Pass", "Pass", "Pass"],
        },
        # Extra unknown field
        {
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": hand_n,
            "unexpected_field": True,
        },
        # dds_worlds above 256 upper bound
        {
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": hand_n,
            "dds_worlds": 257,
        },
    ]

    for payload in invalid_payloads:
        resp = client.post("/v1/step", json=payload)
        assert resp.status_code == 422, f"Expected 422 for payload {payload}, got {resp.status_code}: {resp.text}"


def test_match_endpoint_baseline_and_head_to_head_and_validation() -> None:
    app = create_app(default_profile=_mock_profile(), default_dds_worlds=1)
    client = TestClient(app)

    # 1. Baseline opponent (full card play on 1 board)
    resp_base = client.post(
        "/v1/match",
        json={
            "boards": 1,
            "seed": 42,
            "dds_worlds": 1,
            "double_dummy": False,
            "opponent": "baseline",
        },
    )
    assert resp_base.status_code == 200
    base_data = resp_base.json()
    assert base_data["team_a_name"] == "Tenace"
    assert base_data["team_b_name"] == "Baseline"
    assert base_data["team_a_profile"].startswith("mock | ")
    assert base_data["team_b_profile"] is None
    assert len(base_data["summary"]["boards"]) == 1
    assert (
        base_data["summary"]["total_imps_team_a"]
        == -base_data["summary"]["total_imps_team_b"]
    )
    assert "TENACE DUPLICATE BRIDGE MATCH REPORT" in base_data["report"]

    # 2. Head-to-head Tenace opponent with distinct Team B profile
    resp_h2h = client.post(
        "/v1/match",
        json={
            "boards": 1,
            "seed": 42,
            "dds_worlds": 1,
            "double_dummy": True,
            "opponent": "tenace",
            "team_b_profile": {
                "mode": "mock",
                "strategist_model": "gemini-2.5-pro",
                "evaluator_backend": "gemini",
            },
        },
    )
    assert resp_h2h.status_code == 200
    h2h_data = resp_h2h.json()
    assert h2h_data["team_a_name"] == "Tenace-A"
    assert h2h_data["team_b_name"] == "Tenace-B"
    assert h2h_data["team_b_profile"] == "mock | L1:gemini-2.5-pro | L2:gemini/gemini-2.5-flash"
    assert "team_a_strategist_calls" in h2h_data["summary"]["telemetry"]
    assert "team_b_strategist_calls" in h2h_data["summary"]["telemetry"]

    # 3. Reject bluechip over HTTP, team_b_profile with baseline, and out-of-range boards
    for bad_match in (
        {"opponent": "bluechip"},
        {"opponent": "baseline", "team_b_profile": {"mode": "mock"}},
        {"boards": 0},
        {"boards": 33},
    ):
        bad_resp = client.post("/v1/match", json=bad_match)
        assert bad_resp.status_code == 422


def test_dockerfile_and_dockerignore_conventions() -> None:
    dockerfile_path = REPO_ROOT / "Dockerfile"
    dockerignore_path = REPO_ROOT / ".dockerignore"
    assert dockerfile_path.is_file()
    assert dockerignore_path.is_file()

    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:3.13-slim" in dockerfile
    assert "PORT=8080" in dockerfile
    assert "useradd --create-home --shell /usr/sbin/nologin tenace" in dockerfile
    assert "USER tenace" in dockerfile
    assert "EXPOSE 8080" in dockerfile
    assert 'CMD ["tenace-serve"]' in dockerfile

    dockerignore_lines = {
        line.strip().rstrip("/")
        for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert {
        ".git",
        ".venv",
        ".env*",
        "build",
        "dist",
        ".mypy_cache",
        ".ruff_cache",
        "uv.lock",
        "terraform",
        "tests",
        "docs",
    } <= dockerignore_lines


def test_healthz_is_async_and_lazy_under_vertex_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TENACE_MODE", "vertex")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def fail_if_called(_profile: ModelProfile) -> object:
        raise AssertionError("build_clients_for_profile must not run during create_app or /healthz")

    monkeypatch.setattr(server_mod, "build_clients_for_profile", fail_if_called)

    app = create_app()
    health_routes = [
        route for route in app.routes if getattr(route, "path", None) == "/healthz"
    ]
    assert len(health_routes) == 1
    assert inspect.iscoroutinefunction(health_routes[0].endpoint)

    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "vertex"


def test_error_status_codes_separate_422_client_500_server_and_502_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    TModel = TypeVar("TModel", bound=BaseModel)

    class FailingUpstreamClient:
        def generate_structured(
            self,
            system_prompt: str,
            user_prompt: str,
            schema: type[TModel],
        ) -> TModel:
            del system_prompt, user_prompt, schema
            raise ValueError("Empty response from Gemini model")

    # 1. Server misconfiguration on default_profile -> 500 (NOT 422), while invalid 422 request
    # still fails before touching default_profile client construction!
    def fail_on_server_default(profile: ModelProfile) -> tuple[object, object]:
        if profile.project == "broken-server-default":
            raise ValueError("Missing server GEMINI_API_KEY")
        if profile.project == "broken-caller-override":
            raise ValueError("Caller supplied unsupported profile project")
        return FailingUpstreamClient(), FailingUpstreamClient()

    monkeypatch.setattr(server_mod, "build_clients_for_profile", fail_on_server_default)

    broken_default_app = create_app(
        default_profile=_mock_profile(project="broken-server-default"),
        default_dds_worlds=1,
    )
    broken_client = TestClient(broken_default_app)

    # Malformed request (wrong seat turn) must reject with 422 BEFORE instantiating default_profile clients
    bad_turn = broken_client.post(
        "/v1/step",
        json={"seat": "EAST", "board_number": 1, "tournament_seed": 42},
    )
    assert bad_turn.status_code == 422

    # Valid request using broken server default_profile -> 500
    server_500_step = broken_client.post(
        "/v1/step",
        json={"seat": "NORTH", "board_number": 1, "tournament_seed": 42},
    )
    assert server_500_step.status_code == 500
    assert "Missing server GEMINI_API_KEY" in server_500_step.json()["detail"]

    server_500_match = broken_client.post(
        "/v1/match",
        json={"boards": 1, "dds_worlds": 1, "double_dummy": True},
    )
    assert server_500_match.status_code == 500

    # Caller override profile failure -> 422 on both /v1/step and /v1/match
    caller_422_step = broken_client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "tournament_seed": 42,
            "profile": {"mode": "mock", "project": "broken-caller-override"},
        },
    )
    assert caller_422_step.status_code == 422

    caller_422_match = broken_client.post(
        "/v1/match",
        json={
            "boards": 1,
            "dds_worlds": 1,
            "double_dummy": True,
            "team_a_profile": {"mode": "mock", "project": "broken-caller-override"},
        },
    )
    assert caller_422_match.status_code == 422

    caller_422_match_team_b = TestClient(
        create_app(default_profile=_mock_profile(project="upstream-fails"), default_dds_worlds=1)
    ).post(
        "/v1/match",
        json={
            "boards": 1,
            "dds_worlds": 1,
            "double_dummy": True,
            "opponent": "tenace",
            "team_b_profile": {"mode": "mock", "project": "broken-caller-override"},
        },
    )
    assert caller_422_match_team_b.status_code == 422

    # 2. Upstream model ValueError inside agent.step / runner.play_match -> 502 Bad Gateway
    upstream_app = create_app(
        default_profile=_mock_profile(project="upstream-fails"),
        default_dds_worlds=1,
    )
    upstream_client = TestClient(upstream_app)

    step_502 = upstream_client.post(
        "/v1/step",
        json={"seat": "NORTH", "board_number": 1, "tournament_seed": 42},
    )
    assert step_502.status_code == 502
    assert "Upstream model evaluation error" in step_502.json()["detail"]

    match_502 = upstream_client.post(
        "/v1/match",
        json={"boards": 1, "dds_worlds": 1, "double_dummy": True},
    )
    assert match_502.status_code == 502
    assert "Upstream model evaluation error" in match_502.json()["detail"]


def test_client_cache_reuses_clients_across_turns_and_bounds_lru_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_builder = server_mod.build_clients_for_profile
    built_projects: list[str | None] = []

    def counting_builder(profile: ModelProfile) -> tuple[object, object]:
        built_projects.append(profile.project)
        return real_builder(profile)

    monkeypatch.setattr(server_mod, "build_clients_for_profile", counting_builder)

    app = create_app(
        default_profile=_mock_profile(project="proj-default"),
        default_dds_worlds=1,
    )
    client = TestClient(app)

    # Zero builds at startup or /healthz
    assert built_projects == []
    assert client.get("/healthz").status_code == 200
    assert built_projects == []

    # Repeated /v1/step calls with default profile build clients exactly ONCE
    for action_hist, seat in (([], "NORTH"), (["Pass"], "EAST")):
        resp = client.post(
            "/v1/step",
            json={
                "seat": seat,
                "board_number": 1,
                "tournament_seed": 42,
                "action_history": action_hist,
            },
        )
        assert resp.status_code == 200
    assert built_projects == ["proj-default"]

    # /v1/match with same default profile also reuses the cached client pair!
    m_resp = client.post(
        "/v1/match",
        json={"boards": 1, "dds_worlds": 1, "double_dummy": True},
    )
    assert m_resp.status_code == 200
    assert built_projects == ["proj-default"]

    # Fill >16 distinct profiles to verify bounded LRU eviction (proj-default evicted, then rebuilt)
    for i in range(17):
        r = client.post(
            "/v1/step",
            json={
                "seat": "NORTH",
                "board_number": 1,
                "tournament_seed": 42,
                "profile": {"mode": "mock", "project": f"proj-{i}"},
            },
        )
        assert r.status_code == 200

    count_before = len(built_projects)
    # Most recent profile (proj-16) is still cached -> 0 new builds
    r_hit = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "tournament_seed": 42,
            "profile": {"mode": "mock", "project": "proj-16"},
        },
    )
    assert r_hit.status_code == 200
    assert len(built_projects) == count_before

    # Oldest profile (proj-default) was LRU-evicted -> triggers 1 rebuild
    r_miss = client.post(
        "/v1/step",
        json={"seat": "NORTH", "board_number": 1, "tournament_seed": 42},
    )
    assert r_miss.status_code == 200
    assert len(built_projects) == count_before + 1
    assert built_projects[-1] == "proj-default"


def test_step_rejects_play_prior_strategy_in_auction_and_hidden_seat_contradictions() -> None:
    app = create_app(default_profile=_mock_profile(), default_dds_worlds=1)
    client = TestClient(app)

    # 1. Stale prior_strategy with phase='play' submitted during auction phase -> 422
    stale_play_strategy = StrategyState(
        phase="play",
        plan_summary="Stale play plan from another board",
        revision=2,
    ).model_dump(mode="json")
    resp_stale = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "tournament_seed": 42,
            "action_history": [],
            "prior_strategy": stale_play_strategy,
        },
    )
    assert resp_stale.status_code == 422
    assert "phase='play'" in resp_stale.json()["detail"]

    # 2. Hidden-seat double-showout contradiction: both East and West show out of Clubs
    # while 8 unplayed Clubs (C:5..C:Q) remain outside own_hand and dummy_hand -> 422
    north_double = [
        action_label(card_id(Suit.CLUBS, 12)),  # C:A
        action_label(card_id(Suit.CLUBS, 11)),  # C:K
        *[action_label(card_id(Suit.HEARTS, r)) for r in range(0, 11)],
    ]
    south_double = [
        action_label(card_id(Suit.CLUBS, 0)),   # C:2
        action_label(card_id(Suit.CLUBS, 1)),   # C:3
        *[action_label(card_id(Suit.SPADES, r)) for r in range(0, 11)],
    ]
    resp_double_void = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": north_double,
            "dummy_hand": south_double,
            "action_history": [
                "1NT", "Pass", "Pass", "Pass",
                "C:4", "C:2", "D:2", "C:A",  # West shows out of Clubs on Trick 1
                "C:K", "D:3", "C:3", "D:4",  # East ALSO shows out of Clubs on Trick 2!
            ],
        },
    )
    assert resp_double_void.status_code == 422
    assert "both hidden seats showed out" in resp_double_void.json()["detail"]

    # 3. Hidden-seat 13-card capacity overflow: West shows out of both Clubs and Diamonds,
    # forcing 20 unplayed Clubs+Diamonds into East's remaining 11 card slots -> 422
    north_cap = [
        action_label(card_id(Suit.CLUBS, 12)),     # C:A
        action_label(card_id(Suit.DIAMONDS, 0)),   # D:2
        *[action_label(card_id(Suit.HEARTS, r)) for r in range(0, 11)],
    ]
    south_cap = [
        action_label(card_id(Suit.CLUBS, 0)),      # C:2
        action_label(card_id(Suit.DIAMONDS, 12)),  # D:A
        *[action_label(card_id(Suit.SPADES, r)) for r in range(0, 11)],
    ]
    resp_cap = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": north_cap,
            "dummy_hand": south_cap,
            "action_history": [
                "1NT", "Pass", "Pass", "Pass",
                "C:3", "C:2", "H:K", "C:A",  # West shows out of Clubs
                "D:2", "D:3", "D:A", "S:K",  # West shows out of Diamonds (South wins Trick 2)
            ],
        },
    )
    assert resp_cap.status_code == 422
    assert "exceed 13-card capacity" in resp_cap.json()["detail"]


def test_gfe_safe_health_route_and_upstream_sdk_errors_map_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai import errors as genai_errors
    from google.genai._gaos.lib import compat_errors

    app = create_app(default_profile=_mock_profile(), default_dds_worlds=1)
    client = TestClient(app)

    # Cloud Run GFE reserves /*z externally, so /health MUST work identically to /healthz
    r_health = client.get("/health")
    r_healthz = client.get("/healthz")
    assert r_health.status_code == 200
    assert r_health.json() == r_healthz.json()

    # Upstream google-genai _gaos GeminiNextGenAPIClientError / APIError -> structured 502 JSON
    class FailingGaosClient:
        def generate_structured(self, system_prompt: str, user_prompt: str, schema: type) -> object:
            raise compat_errors.GeminiNextGenAPIClientError("Method not found (404)")

    class FailingGenaiClient:
        def generate_structured(self, system_prompt: str, user_prompt: str, schema: type) -> object:
            raise genai_errors.APIError(404, {"error": {"message": "Model not found"}})

    monkeypatch.setattr(
        server_mod,
        "build_clients_for_profile",
        lambda profile: (
            (FailingGaosClient(), FailingGaosClient())
            if profile.project == "gaos-err"
            else (FailingGenaiClient(), FailingGenaiClient())
        ),
    )

    err_client = TestClient(
        create_app(default_profile=_mock_profile(project="gaos-err"), default_dds_worlds=1),
        raise_server_exceptions=False,
    )
    step_gaos = err_client.post(
        "/v1/step",
        json={"seat": "NORTH", "board_number": 1, "tournament_seed": 42},
    )
    assert step_gaos.status_code == 502
    assert "Method not found" in step_gaos.json()["detail"]

    match_genai = err_client.post(
        "/v1/match",
        json={
            "boards": 1,
            "dds_worlds": 1,
            "double_dummy": True,
            "team_a_profile": {"mode": "mock", "project": "genai-err"},
        },
    )
    assert match_genai.status_code == 502
    assert "Upstream model evaluation error" in match_genai.json()["detail"]


def test_table_state_exports_and_auction_imperfect_info_visibility() -> None:
    from tenace.domain.types import card_rank, card_suit, hand_hcp
    import tenace.harness as harness_pkg
    import tenace.harness.duplicate as duplicate_mod

    assert "TableStateRequest" in server_mod.__all__
    assert "TableStateResponse" in server_mod.__all__
    assert hasattr(server_mod, "TableStateRequest")
    assert hasattr(server_mod, "TableStateResponse")
    assert "wbf_board_params" in duplicate_mod.__all__
    assert "wbf_board_params" in harness_pkg.__all__
    assert server_mod.TableStateResponse.__doc__ is not None
    assert "contract" in server_mod.TableStateResponse.__doc__
    assert "legal_actions" in server_mod.TableStateResponse.__doc__

    app = create_app(default_profile=_mock_profile(), default_dds_worlds=1)
    client = TestClient(app)

    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    board2 = runner._new_dealt_board_state(2)
    dealt = _dealt_hands(board2)
    expected_south_sorted = [
        action_label(c)
        for c in sorted(dealt[Seat.SOUTH], key=lambda cid: (-int(card_suit(cid)), -card_rank(cid)))
    ]

    resp = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 2,
            "action_history": [],
            "viewer_seat": "SOUTH",
            "reveal_all": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dealer"] == "EAST"
    assert body["ns_vulnerable"] is True
    assert body["ew_vulnerable"] is False
    assert body["phase"] == "auction"
    assert body["current_player"] == "EAST"
    assert body["acting_hand"] == "EAST"
    assert body["hands"] == {
        "NORTH": None,
        "EAST": None,
        "SOUTH": expected_south_sorted,
        "WEST": None,
    }
    assert body["hand_hcp"] == {
        "NORTH": None,
        "EAST": None,
        "SOUTH": hand_hcp(dealt[Seat.SOUTH]),
        "WEST": None,
    }
    assert body["remaining_counts"] == {
        "NORTH": 13,
        "EAST": 13,
        "SOUTH": 13,
        "WEST": 13,
    }
    assert body["initial_hands"] is None
    assert body["auction_calls"] == []
    assert body["contract"] is None
    assert body["ns_tricks"] == 0
    assert body["ew_tricks"] == 0
    assert body["current_trick"] == []
    assert body["completed_tricks"] == []
    assert body["terminal_result"] is None
    assert len(body["legal_actions"]) == 36
    assert body["legal_actions"][0] == {"action_id": PASS_ACTION, "label": "Pass"}
    assert body["legal_actions"][1]["label"] == "1C"
    assert body["legal_actions"][-1]["label"] == "7NT"

    # With reveal_all=True, all 4 hands, HCPs, and initial_hands are exposed
    resp_reveal = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 2,
            "action_history": [],
            "viewer_seat": "SOUTH",
            "reveal_all": True,
        },
    )
    assert resp_reveal.status_code == 200
    rev_body = resp_reveal.json()
    for s in Seat:
        assert rev_body["hands"][s.name] is not None
        assert len(rev_body["hands"][s.name]) == 13
        assert rev_body["hand_hcp"][s.name] == hand_hcp(dealt[s])
        assert rev_body["initial_hands"][s.name] == rev_body["hands"][s.name]

    # Non-empty in-progress auction ["1NT", "Pass"] on Board 2 (East opens 1NT, South passes -> West's turn):
    # contract remains None, while public auction legal_actions remain populated even on West's turn!
    in_prog = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 2,
            "action_history": ["1NT", "Pass"],
            "viewer_seat": "SOUTH",
            "reveal_all": False,
        },
    )
    assert in_prog.status_code == 200
    in_prog_body = in_prog.json()
    assert in_prog_body["phase"] == "auction"
    assert in_prog_body["current_player"] == "WEST"
    assert in_prog_body["acting_hand"] == "WEST"
    assert in_prog_body["contract"] is None
    assert len(in_prog_body["auction_calls"]) == 2
    assert len(in_prog_body["legal_actions"]) > 0


def test_table_state_pre_lead_vs_post_lead_dummy_and_acting_hand() -> None:
    from tenace.domain.types import hand_hcp

    app = create_app(default_profile=_mock_profile(), default_dds_worlds=1)
    client = TestClient(app)

    # Board 3: South deals, E/W vulnerable
    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    state = runner._new_dealt_board_state(3)
    dealt = _dealt_hands(state)

    auction = ["1NT", "Pass", "Pass", "Pass"]
    for tok in auction:
        state.apply_action(parse_action_token(tok))

    # 1. Pre-lead: contract is 1NT by SOUTH (dummy NORTH), West is on opening lead;
    # dummy NORTH must STILL be hidden!
    pre_lead = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 3,
            "action_history": auction,
            "viewer_seat": "SOUTH",
            "reveal_all": False,
        },
    )
    assert pre_lead.status_code == 200
    pre_body = pre_lead.json()
    assert pre_body["dealer"] == "SOUTH"
    assert pre_body["ns_vulnerable"] is False
    assert pre_body["ew_vulnerable"] is True
    assert pre_body["phase"] == "play"
    assert pre_body["current_player"] == "WEST"
    assert pre_body["acting_hand"] == "WEST"
    assert pre_body["contract"] == {
        "level": 1,
        "trump": "NT",
        "doubled": "",
        "declarer": "SOUTH",
        "dummy": "NORTH",
        "formatted": "1NT by SOUTH",
    }
    assert pre_body["hands"]["NORTH"] is None
    assert pre_body["hand_hcp"]["NORTH"] is None
    assert pre_body["hands"]["SOUTH"] is not None
    # Hidden-seat play turn (WEST != viewer_seat SOUTH, reveal_all=False) MUST redact legal_actions!
    assert pre_body["legal_actions"] == []
    assert len(pre_body["auction_calls"]) == 4
    assert pre_body["auction_calls"][0] == {
        "seat": "SOUTH",
        "label": "1NT",
        "action_id": parse_action_token("1NT"),
    }

    # With reveal_all=True on the same pre-lead state, WEST's legal_actions ARE exposed
    pre_lead_rev = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 3,
            "action_history": auction,
            "viewer_seat": "SOUTH",
            "reveal_all": True,
        },
    )
    assert pre_lead_rev.status_code == 200
    assert len(pre_lead_rev.json()["legal_actions"]) == 13

    # Verify doubled and redoubled contract metadata formatting
    dbl_resp = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 3,
            "action_history": ["1H", "Dbl", "Pass", "Pass", "Pass"],
            "viewer_seat": "SOUTH",
        },
    )
    assert dbl_resp.status_code == 200
    assert dbl_resp.json()["contract"]["doubled"] == "X"
    assert dbl_resp.json()["contract"]["formatted"] == "1HX by SOUTH"

    rdbl_resp = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 3,
            "action_history": ["1H", "Dbl", "RDbl", "Pass", "Pass", "Pass"],
            "viewer_seat": "SOUTH",
        },
    )
    assert rdbl_resp.status_code == 200
    assert rdbl_resp.json()["contract"]["doubled"] == "XX"
    assert rdbl_resp.json()["contract"]["formatted"] == "1HXX by SOUTH"

    # Verify partnership first-denomination declarer rule (South opens 1NT, North raises 3NT -> South is declarer!)
    raise_resp = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 3,
            "action_history": ["1NT", "Pass", "3NT", "Pass", "Pass", "Pass"],
            "viewer_seat": "SOUTH",
        },
    )
    assert raise_resp.status_code == 200
    assert raise_resp.json()["contract"] == {
        "level": 3,
        "trump": "NT",
        "doubled": "",
        "declarer": "SOUTH",
        "dummy": "NORTH",
        "formatted": "3NT by SOUTH",
    }

    # 2. Post-lead: West leads first legal card -> dummy NORTH becomes visible,
    # and current_player is SOUTH (declarer) while physical acting_hand is NORTH (dummy)!
    west_lead_id = state.legal_actions()[0]
    west_lead_lbl = action_label(west_lead_id)
    state.apply_action(west_lead_id)

    post_lead = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 3,
            "action_history": [*auction, west_lead_lbl],
            "viewer_seat": "SOUTH",
            "reveal_all": False,
        },
    )
    assert post_lead.status_code == 200
    post_body = post_lead.json()
    assert post_body["phase"] == "play"
    assert post_body["current_player"] == "SOUTH"
    assert post_body["acting_hand"] == "NORTH"
    assert post_body["hands"]["NORTH"] is not None
    assert len(post_body["hands"]["NORTH"]) == 13
    assert post_body["hand_hcp"]["NORTH"] == hand_hcp(dealt[Seat.NORTH])
    assert post_body["hands"]["EAST"] is None
    assert post_body["hands"]["WEST"] is None
    assert post_body["remaining_counts"] == {
        "NORTH": 13,
        "EAST": 13,
        "SOUTH": 13,
        "WEST": 12,
    }
    assert post_body["current_trick"] == [
        {"seat": "WEST", "card": west_lead_lbl, "action_id": west_lead_id}
    ]
    legal_ids = {item["action_id"] for item in post_body["legal_actions"]}
    assert legal_ids <= set(dealt[Seat.NORTH])

    # 3. Post-lead when viewer_seat == dummy_seat (Board 1: NORTH declarer, SOUTH dummy & viewer):
    # Dummy SOUTH's cards are visible, NORTH's closed hand is hidden, and legal_actions is []
    # because current_player is NORTH (!= viewer_seat SOUTH) and reveal_all=False!
    b1_state = runner._new_dealt_board_state(1)
    for tok in auction:
        b1_state.apply_action(parse_action_token(tok))
    east_lead = b1_state.legal_actions()[0]
    b1_state.apply_action(east_lead)
    dummy_card = b1_state.legal_actions()[0]
    b1_state.apply_action(dummy_card)
    west_card = b1_state.legal_actions()[0]
    b1_state.apply_action(west_card)

    for hist_b1, expected_acting_hand in (
        ([*auction, east_lead], "SOUTH"),
        ([*auction, east_lead, dummy_card, west_card], "NORTH"),
    ):
        viewer_dummy_resp = client.post(
            "/v1/table/state",
            json={
                "tournament_seed": 42,
                "board_number": 1,
                "action_history": hist_b1,
                "viewer_seat": "SOUTH",
                "reveal_all": False,
            },
        )
        assert viewer_dummy_resp.status_code == 200
        vd_body = viewer_dummy_resp.json()
        assert vd_body["current_player"] == "NORTH"
        assert vd_body["acting_hand"] == expected_acting_hand
        assert vd_body["hands"]["SOUTH"] is not None
        assert vd_body["hands"]["NORTH"] is None
        assert vd_body["legal_actions"] == []


def test_table_state_completed_tricks_passed_out_and_terminal_deal_reveal() -> None:
    app = create_app(default_profile=_mock_profile(), default_dds_worlds=1)
    client = TestClient(app)

    # 1. Passed out board
    po_resp = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 1,
            "action_history": ["Pass", "Pass", "Pass", "Pass"],
            "viewer_seat": "SOUTH",
            "reveal_all": False,
        },
    )
    assert po_resp.status_code == 200
    po_body = po_resp.json()
    assert po_body["phase"] == "terminal"
    assert po_body["current_player"] is None
    assert po_body["acting_hand"] is None
    assert po_body["legal_actions"] == []
    assert po_body["contract"]["formatted"] == "Passed Out"
    assert po_body["terminal_result"] == {
        "ns_score": 0,
        "ew_score": 0,
        "contract": "Passed Out",
        "ns_tricks": 0,
        "ew_tricks": 0,
    }
    assert po_body["initial_hands"] is not None
    for s in Seat:
        assert len(po_body["hands"][s.name]) == 13
        assert len(po_body["initial_hands"][s.name]) == 13

    # 2. Play 1 trick and then full 13 tricks to terminal
    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    state = runner._new_dealt_board_state(1)
    history: list[int] = []
    for tok in ("1NT", "Pass", "Pass", "Pass"):
        aid = parse_action_token(tok)
        state.apply_action(aid)
        history.append(aid)

    # Play 4 cards (Trick 1)
    for _ in range(4):
        c = state.legal_actions()[0]
        state.apply_action(c)
        history.append(c)

    t1_resp = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 1,
            "action_history": history,
            "viewer_seat": "SOUTH",
        },
    )
    assert t1_resp.status_code == 200
    t1_body = t1_resp.json()
    assert t1_body["phase"] == "play"
    assert t1_body["current_trick"] == []
    assert len(t1_body["completed_tricks"]) == 1
    trick1 = t1_body["completed_tricks"][0]
    assert trick1["trick_number"] == 1
    assert trick1["leader"] == "EAST"
    assert len(trick1["cards"]) == 4
    assert t1_body["ns_tricks"] + t1_body["ew_tricks"] == 1
    assert t1_body["acting_hand"] == trick1["winner"]

    # Play remaining 48 cards to terminal
    while not state.is_terminal():
        c = state.legal_actions()[0]
        state.apply_action(c)
        history.append(c)

    term_resp = client.post(
        "/v1/table/state",
        json={
            "tournament_seed": 42,
            "board_number": 1,
            "action_history": history,
            "viewer_seat": "SOUTH",
            "reveal_all": False,
        },
    )
    assert term_resp.status_code == 200
    term_body = term_resp.json()
    assert term_body["phase"] == "terminal"
    assert term_body["current_player"] is None
    assert term_body["acting_hand"] is None
    assert term_body["legal_actions"] == []
    assert len(term_body["completed_tricks"]) == 13
    assert term_body["ns_tricks"] + term_body["ew_tricks"] == 13
    assert term_body["remaining_counts"] == {
        "NORTH": 0,
        "EAST": 0,
        "SOUTH": 0,
        "WEST": 0,
    }
    assert term_body["hands"] == {
        "NORTH": [],
        "EAST": [],
        "SOUTH": [],
        "WEST": [],
    }
    assert term_body["initial_hands"] is not None
    for s in Seat:
        assert len(term_body["initial_hands"][s.name]) == 13
        assert term_body["hand_hcp"][s.name] is not None
    assert term_body["terminal_result"] == {
        "ns_score": round(state.returns()[0]),
        "ew_score": round(state.returns()[1]),
        "contract": "1NT by NORTH",
        "ns_tricks": term_body["ns_tricks"],
        "ew_tricks": term_body["ew_tricks"],
    }


def test_table_state_rejects_invalid_requests_with_422() -> None:
    app = create_app(default_profile=_mock_profile(), default_dds_worlds=1)
    client = TestClient(app)

    invalid_payloads: list[dict[str, object]] = [
        {"board_number": 0},
        {"board_number": 1025},
        {"board_number": True},
        {"tournament_seed": -1},
        {"tournament_seed": 2_147_483_648},
        {"tournament_seed": False},
        {"reveal_all": "yes"},
        {"reveal_all": 1},
        {"viewer_seat": "CENTER"},
        {"unexpected": 1},
        {"action_history": [True]},
        {"action_history": [False]},
        {"action_history": ["9NT"]},
        {"action_history": ["1NT", "1C"]},
        {"action_history": ["S:A"]},
        {"action_history": ["Pass", "Pass", "Pass", "Pass", "Pass"]},
        {"action_history": ["1NT", "Pass", "Pass", "Pass", "S:A", "S:A"]},
    ]
    for payload in invalid_payloads:
        resp = client.post("/v1/table/state", json=payload)
        assert resp.status_code == 422, f"Expected 422 for {payload}, got {resp.status_code}"

    # Verify int32 bounds and StrictInt/StrictStr/StrictBool across /v1/step, /v1/match, and DuplicateMatchRunner
    hand_n = [action_label(c) for c in range(0, 13)]
    hand_s = [action_label(c) for c in range(26, 39)]
    for bad_step in (
        {"seat": "NORTH", "board_number": 1, "tournament_seed": 2_147_483_648},
        {"seat": "NORTH", "board_number": 1, "tournament_seed": 42, "seed": 2_147_483_648},
        {"seat": "NORTH", "board_number": 1, "tournament_seed": 42, "seed": -1},
        {"seat": "NORTH", "board_number": 1025, "tournament_seed": 42},
        {"seat": "NORTH", "board_number": 1, "tournament_seed": 42, "action_history": [True]},
        {"seat": "NORTH", "board_number": 1, "own_hand": [True, *hand_n[1:]]},
        {
            "seat": "NORTH",
            "board_number": 1,
            "own_hand": hand_n,
            "dummy_hand": [False, *hand_s[1:]],
            "action_history": ["1NT", "Pass", "Pass", "Pass", "D:5"],
        },
    ):
        r_step = client.post("/v1/step", json=bad_step)
        assert r_step.status_code == 422, f"Expected 422 for step {bad_step}, got {r_step.status_code}"

    for bad_match in (
        {"seed": 2_147_483_648},
        {"seed": -1},
        {"double_dummy": "yes"},
        {"double_dummy": 1},
    ):
        r_match = client.post("/v1/match", json=bad_match)
        assert r_match.status_code == 422, f"Expected 422 for match {bad_match}, got {r_match.status_code}"

    for bad_runner_seed in (-1, 2_147_483_648):
        with pytest.raises(ValueError):
            DuplicateMatchRunner(tournament_seed=bad_runner_seed)

    runner = DuplicateMatchRunner(tournament_seed=42)
    for bad_board in (0, -1, 2_147_483_648, True):
        with pytest.raises(ValueError):
            runner._new_dealt_board_state(bad_board)  # type: ignore[arg-type]


def test_step_baseline_bot_option_uses_zero_llm_calls_and_skips_client_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(_profile: ModelProfile) -> object:
        raise AssertionError("build_clients_for_profile must NOT run when bot='baseline'")

    monkeypatch.setattr(server_mod, "build_clients_for_profile", fail_if_called)

    # Even with vertex default profile and no GCP credentials, bot='baseline' succeeds cleanly
    app = create_app(
        default_profile=ModelProfile(mode="vertex", project=None),
        default_dds_worlds=4,
    )
    client = TestClient(app)

    # 1. Auction step -> chooses Pass with 0 LLM calls
    r_auc = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "tournament_seed": 42,
            "action_history": [],
            "bot": "baseline",
        },
    )
    assert r_auc.status_code == 200
    auc_body = r_auc.json()
    assert auc_body["action_label"] == "Pass"
    assert auc_body["action_id"] == PASS_ACTION
    assert auc_body["phase"] == "auction"
    assert auc_body["strategy"] is None
    assert auc_body["telemetry"] == {
        "strategist_calls": 0,
        "evaluator_calls": 0,
        "forced_moves": 0,
    }

    # 2. Play step -> plays first legal card, preserves prior_strategy, tracks forced_moves on Trick 13
    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    state = runner._new_dealt_board_state(1)
    history: list[int] = []
    for tok in ("1NT", "Pass", "Pass", "Pass"):
        aid = parse_action_token(tok)
        state.apply_action(aid)
        history.append(aid)
    # Advance to Trick 13 (48 play cards played -> 1 legal card remaining for leader!)
    for _ in range(48):
        c = state.legal_actions()[0]
        state.apply_action(c)
        history.append(c)

    acting_seat = Seat(state.current_player()).name
    prior_strat = StrategyState(
        phase="play",
        plan_summary="Preserve baseline strategy metadata",
        revision=1,
    ).model_dump(mode="json")

    r_play = client.post(
        "/v1/step",
        json={
            "seat": acting_seat,
            "board_number": 1,
            "tournament_seed": 42,
            "action_history": history,
            "prior_strategy": prior_strat,
            "bot": "baseline",
        },
    )
    assert r_play.status_code == 200
    play_body = r_play.json()
    assert play_body["phase"] == "play"
    assert play_body["action_id"] == state.legal_actions()[0]
    assert play_body["strategy"] == prior_strat
    assert play_body["telemetry"] == {
        "strategist_calls": 0,
        "evaluator_calls": 0,
        "forced_moves": 1,
    }

    # Stale phase='auction' prior_strategy passed into a 'play' turn resets strategy to None
    stale_auc_strat = StrategyState(
        phase="auction",
        plan_summary="Old auction strategy",
        revision=1,
    ).model_dump(mode="json")
    r_play_from_auc = client.post(
        "/v1/step",
        json={
            "seat": acting_seat,
            "board_number": 1,
            "tournament_seed": 42,
            "action_history": history,
            "prior_strategy": stale_auc_strat,
            "bot": "baseline",
        },
    )
    assert r_play_from_auc.status_code == 200
    assert r_play_from_auc.json()["phase"] == "play"
    assert r_play_from_auc.json()["strategy"] is None

    # 3. Validation still enforces 422 on wrong seat or invalid bot name
    assert (
        client.post(
            "/v1/step",
            json={"seat": "EAST", "board_number": 1, "tournament_seed": 42, "bot": "baseline"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/step",
            json={"seat": "NORTH", "board_number": 1, "tournament_seed": 42, "bot": "unknown"},
        ).status_code
        == 422
    )


def test_root_index_and_static_web_assets_served() -> None:
    app = create_app(default_profile=_mock_profile(), default_dds_worlds=1)
    client = TestClient(app)

    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "text/html" in r_root.headers["content-type"]
    assert r_root.headers.get("cache-control") == "no-cache"
    assert "<!doctype html>" in r_root.text.lower()
    assert "tenace" in r_root.text.lower()

    r_css = client.get("/static/styles.css")
    assert r_css.status_code == 200
    assert "text/css" in r_css.headers["content-type"]
    assert len(r_css.text.strip()) > 0

    r_js = client.get("/static/app.js")
    assert r_js.status_code == 200
    assert "javascript" in r_js.headers["content-type"]
    assert len(r_js.text.strip()) > 0

