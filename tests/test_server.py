from __future__ import annotations

from pathlib import Path
import tomllib

from fastapi.testclient import TestClient
import pyspiel
import pytest

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
    assert "useradd" in dockerfile
    assert "USER tenace" in dockerfile
    assert "EXPOSE 8080" in dockerfile
    assert 'CMD ["tenace-serve"]' in dockerfile

    dockerignore_lines = {
        line.strip().rstrip("/")
        for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert {".git", ".venv", "uv.lock", "terraform", "tests", "docs"} <= dockerignore_lines
