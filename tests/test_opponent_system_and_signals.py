from __future__ import annotations

from typing import Any, TypeVar

import pyspiel
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from tenace.agent import TenaceAgent
from tenace.domain.types import (
    PASS_ACTION,
    CandidateOption,
    Seat,
    SeatConstraint,
    StrategyState,
    Suit,
    TacticalDecision,
    action_label,
    card_id,
)
from tenace.harness.duplicate import DuplicateMatchRunner
from tenace.layer1_strategist.strategist import (
    StrategistEngine,
    build_strategist_system_prompt,
    extract_defensive_carding_notes,
)
from tenace.layer2_evaluator.evaluator import TacticalOptionEvaluator
from tenace.layer3_rules.dds_evaluator import (
    RuleDDSEvaluator,
    infer_symbolic_auction_constraints,
)
from tenace.layer3_rules.tracker import RuleFactTracker, collapse_equivalent_actions
from tenace.server import create_app
from tenace.skills import load_bidding_skill

T = TypeVar("T", bound=BaseModel)
LABEL_TO_ID = {action_label(a).upper(): a for a in range(90)}


class _RecordingClient:
    def __init__(self, responses: list[BaseModel]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "schema": schema,
            }
        )
        resp = self._responses.pop(0)
        assert isinstance(resp, schema)
        return resp


def _state_after_calls(seed: int, board: int, calls: list[str]) -> pyspiel.State:
    st = DuplicateMatchRunner(tournament_seed=seed, use_double_dummy_result=False)._new_dealt_board_state(board)
    for tok in calls:
        st.apply_action(LABEL_TO_ID[tok.upper()])
    return st


def test_infer_symbolic_constraints_uses_per_partnership_opponent_system() -> None:
    # Board 1 (dealer NORTH): NORTH passes, EAST opens 1C (strong Precision 16+!),
    # SOUTH passes, WEST bids 1D, then play begins.
    # Observer = SOUTH (playing CCBA) vs EAST/WEST (playing Precision).
    st_prec_opp = _state_after_calls(
        3000, 1, ["Pass", "1C", "Pass", "1D", "Pass", "1NT", "Pass", "Pass", "Pass"]
    )
    cons_prec = infer_symbolic_auction_constraints(
        st_prec_opp,
        Seat.SOUTH,
        system_id="ccba",
        opponent_system_id="precision",
    )
    assert cons_prec[Seat.EAST].min_hcp >= 16
    assert Suit.CLUBS not in cons_prec[Seat.EAST].min_suit_lengths

    # Same auction with NORTH (our CCBA partner!) opening 1C vs Precision opponents:
    # Partner's 1C MUST stay natural CCBA (min_hcp=11, NOT 16+!).
    st_our_1c = _state_after_calls(
        3000, 1, ["1C", "Pass", "1D", "Pass", "1NT", "Pass", "Pass", "Pass"]
    )
    cons_our = infer_symbolic_auction_constraints(
        st_our_1c,
        Seat.SOUTH,
        system_id="ccba",
        opponent_system_id="precision",
    )
    assert cons_our[Seat.NORTH].min_hcp == 11

    # CCBA observer SOUTH vs Acol opponents: EAST's 1NT is weak (11..15) and 1H is 4+ hearts.
    st_acol_nt = _state_after_calls(3000, 1, ["Pass", "1NT", "Pass", "Pass", "Pass"])
    cons_acol_nt = infer_symbolic_auction_constraints(
        st_acol_nt,
        Seat.SOUTH,
        system_id="ccba",
        opponent_system_id="acol",
    )
    assert cons_acol_nt[Seat.EAST].min_hcp == 11
    assert cons_acol_nt[Seat.EAST].max_hcp == 15

    st_acol_1h = _state_after_calls(3000, 1, ["Pass", "1H", "Pass", "Pass", "Pass"])
    cons_acol_1h = infer_symbolic_auction_constraints(
        st_acol_1h,
        Seat.SOUTH,
        system_id="ccba",
        opponent_system_id="acol",
    )
    assert cons_acol_1h[Seat.EAST].min_suit_lengths.get(Suit.HEARTS) == 4


def test_strategist_and_evaluator_disclose_opponent_convention_card() -> None:
    ccba_card = load_bidding_skill("ccba").to_convention_card()
    prec_card = load_bidding_skill("precision").to_convention_card()

    # All 6 built-in skills must declare defensive carding in summary_rules for disclosure!
    for sid in ("sayc", "21gf", "ccba", "precision", "acol", "polish_club"):
        card = load_bidding_skill(sid).to_convention_card()
        assert "carding" in card.summary_rules.lower(), f"Missing carding in {sid} summary_rules"

    # Asymmetric disclosure in both auction and play phases
    prompt_auc = build_strategist_system_prompt(
        ccba_card, phase="auction", opponent_convention_card=prec_card
    )
    prompt_play = build_strategist_system_prompt(
        ccba_card, phase="play", opponent_convention_card=prec_card
    )
    assert "Opponent Disclosed System:" in prompt_auc
    assert "Precision Club" in prompt_auc
    assert "16+" in prompt_auc
    assert "Opponent Disclosed System:" in prompt_play
    assert "Precision Club" in prompt_play

    # Same system deduplicates cleanly (no duplicated 2KB summary text!)
    prompt_same = build_strategist_system_prompt(
        ccba_card, phase="auction", opponent_convention_card=ccba_card
    )
    assert "Opponent Disclosed System: Same as Partnership System" in prompt_same
    assert prompt_same.count(ccba_card.summary_rules) == 1

    # L2 TacticalOptionEvaluator discloses opponent system in both auction and play
    st = _state_after_calls(3000, 1, ["Pass", "1C"])
    facts = RuleFactTracker().extract_facts(st, Seat.SOUTH)
    eval_client = _RecordingClient(
        [TacticalDecision(option_scores={"Pass": 0.9}, rationale="Pass strong 1C")]
    )
    ev = TacticalOptionEvaluator(
        eval_client,
        convention_card=ccba_card,
        opponent_convention_card=prec_card,
    )
    ev.evaluate(
        facts,
        StrategyState(phase="auction", plan_summary="Pass over Precision 1C"),
        [CandidateOption(action=PASS_ACTION, label="Pass")],
    )
    assert "Precision Club" in eval_client.calls[0]["user_prompt"]


def test_collapse_equivalent_actions_preserves_high_low_spot_signals() -> None:
    h4 = card_id(Suit.HEARTS, 2)
    h3 = card_id(Suit.HEARTS, 1)
    h2 = card_id(Suit.HEARTS, 0)
    sa = card_id(Suit.SPADES, 12)
    sk = card_id(Suit.SPADES, 11)
    d6 = card_id(Suit.DIAMONDS, 4)
    d5 = card_id(Suit.DIAMONDS, 3)

    # Default preserve_signals=False keeps full backward compatibility
    collapsed = collapse_equivalent_actions([h4, h3, h2, sa, sk, d6, d5])
    assert [c.label for c in collapsed] == ["D:6", "H:4", "S:A"]

    # With preserve_signals=True: touching spot groups expose BOTH highest & lowest card,
    # while top honor sequences (S:A K) remain collapsed!
    sig_opts = collapse_equivalent_actions(
        [h4, h3, h2, sa, sk, d6, d5],
        preserve_signals=True,
    )
    labels = [c.label for c in sig_opts]
    assert "H:4" in labels and "H:2" in labels and "H:3" not in labels
    assert "D:6" in labels and "D:5" in labels
    assert "S:A" in labels and "S:K" not in labels


def test_extract_defensive_carding_notes_detects_partner_switch_and_count_echo() -> None:
    # Seed 3600 Bd 6: E:1S - S:3H - W:4S - N:5H - E:5S
    # T1: S:H:A W:H:3 N:H:Q E:H:7 -> S; T2: S:D:3 W:D:K N:D:A E:D:7 -> N
    st = _state_after_calls(
        3600,
        6,
        ["1S", "3H", "4S", "5H", "5S", "Pass", "Pass", "Pass"],
    )
    for card in ("H:A", "H:3", "H:Q", "H:7", "D:3", "D:K", "D:A", "D:7"):
        st.apply_action(LABEL_TO_ID[card])

    facts = RuleFactTracker().extract_facts(st, Seat.NORTH)
    notes = extract_defensive_carding_notes(facts)
    joined = " ".join(notes).lower()
    assert "south" in joined
    assert "diamonds" in joined
    assert "d:3" in joined
    assert "singleton" in joined or "ruff" in joined


def test_tenace_agent_replans_on_partner_early_suit_switch_and_finds_diamond_ruff() -> None:
    # Seed 3600 Bd 6 Trick 3: NORTH won D:A after SOUTH cashed H:A and switched to D:3.
    # Even with NO void show-out yet, NORTH MUST trigger an L1 replan and when L1 sets
    # SOUTH max_suit_lengths={DIAMONDS: 1}, Layer-3 DDS MUST score D:J decisively over H:T!
    st = _state_after_calls(
        3600,
        6,
        ["1S", "3H", "4S", "5H", "5S", "Pass", "Pass", "Pass"],
    )
    for card in ("H:A", "H:3", "H:Q", "H:7", "D:3", "D:K", "D:A", "D:7"):
        st.apply_action(LABEL_TO_ID[card])

    strat_client = _RecordingClient(
        [
            StrategyState(
                phase="play",
                plan_summary="Lead D:J immediately to give South a diamond ruff to beat 5S.",
                seat_constraints={
                    Seat.SOUTH: SeatConstraint(
                        seat=Seat.SOUTH,
                        min_hcp=4,
                        max_hcp=11,
                        min_suit_lengths={Suit.HEARTS: 7},
                        max_suit_lengths={Suit.DIAMONDS: 1},
                    ),
                },
            )
        ]
    )
    eval_client = _RecordingClient(
        [
            TacticalDecision(
                option_scores={"D:J": 0.95, "D:9": 0.90, "D:4": 0.85, "H:T": 0.10},
                rationale="Return partner's diamond switch for a ruff",
            )
        ]
    )
    ccba_card = load_bidding_skill("ccba").to_convention_card()
    agent = TenaceAgent(
        seat=Seat.NORTH,
        strategist_client=strat_client,
        evaluator_client=eval_client,
        dds_evaluator=RuleDDSEvaluator(num_worlds=16, rng_seed=0),
        convention_card=ccba_card,
        opponent_convention_card=load_bidding_skill("21gf").to_convention_card(),
    )
    # Simulate NORTH already having planned at Trick 1 (dummy seen, no voids yet!)
    agent.strategy = StrategyState(
        phase="play",
        plan_summary="Play H:Q on Trick 1 to encourage hearts.",
        revision=3,
    )
    agent._dummy_seen = True
    agent._play_strategist_calls = 1
    agent._prev_voids = {s: set() for s in Seat}

    chosen = agent.step(st)
    assert agent.strategist_calls == 1, "Expected L1 defensive replan on partner's Trick-2 suit switch!"
    assert "switch" in strat_client.calls[0]["user_prompt"].lower()
    assert "Carding Observations:" in strat_client.calls[0]["user_prompt"]
    # Chosen card MUST be a diamond returning partner's ruff (beats 5S!)
    assert action_label(chosen).startswith("D:")


def test_tenace_agent_defending_exposes_low_spot_signal_choice_to_l2() -> None:
    # Build a deal where EAST declares 1NT, SOUTH leads H:A, dummy WEST plays H:5,
    # and defender NORTH holds ONLY touching hearts H:4, H:3, H:2!
    # With signal-preserving defensive carding, NORTH exposes BOTH H:4 (encourage/high-low)
    # and H:2 (discourage/odd) to L2 instead of collapsing blindly to H:4!
    n_cards = [
        card_id(Suit.HEARTS, 0),  # H:2
        card_id(Suit.HEARTS, 1),  # H:3
        card_id(Suit.HEARTS, 2),  # H:4
        *[card_id(Suit.CLUBS, r) for r in range(0, 10)],
    ]
    e_cards = [
        card_id(Suit.HEARTS, 11),  # H:K
        card_id(Suit.HEARTS, 10),  # H:Q
        card_id(Suit.HEARTS, 9),   # H:J
        *[card_id(Suit.SPADES, r) for r in range(3, 13)],
    ]
    s_cards = [
        card_id(Suit.HEARTS, 12),  # H:A (opening lead!)
        card_id(Suit.HEARTS, 8),   # H:T
        card_id(Suit.HEARTS, 7),   # H:9
        card_id(Suit.HEARTS, 6),   # H:8
        *[card_id(Suit.DIAMONDS, r) for r in range(0, 9)],
    ]
    w_cards = [
        card_id(Suit.HEARTS, 3),   # H:5
        card_id(Suit.HEARTS, 4),   # H:6
        card_id(Suit.HEARTS, 5),   # H:7
        card_id(Suit.CLUBS, 10),
        card_id(Suit.CLUBS, 11),
        card_id(Suit.CLUBS, 12),
        card_id(Suit.DIAMONDS, 9),
        card_id(Suit.DIAMONDS, 10),
        card_id(Suit.DIAMONDS, 11),
        card_id(Suit.DIAMONDS, 12),
        card_id(Suit.SPADES, 0),
        card_id(Suit.SPADES, 1),
        card_id(Suit.SPADES, 2),
    ]
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    st = game.new_initial_state()
    for r in range(13):
        for p, hand in ((0, n_cards), (1, e_cards), (2, s_cards), (3, w_cards)):
            st.apply_action(hand[r])
    # Auction: N:Pass, E:1NT, S:Pass, W:Pass, N:Pass -> E declares 1NT, S leads H:A, dummy W plays H:5
    for tok in ("Pass", "1NT", "Pass", "Pass", "Pass", "H:A", "H:5"):
        st.apply_action(LABEL_TO_ID[tok.upper()])
    assert st.current_player() == int(Seat.NORTH)

    strat_client = _RecordingClient(
        [StrategyState(phase="play", plan_summary="Play H:2 to discourage hearts.")]
    )
    eval_client = _RecordingClient(
        [TacticalDecision(option_scores={"H:2": 0.95, "H:4": 0.15}, rationale="Discourage with lowest heart")]
    )
    north = TenaceAgent(
        seat=Seat.NORTH,
        strategist_client=strat_client,
        evaluator_client=eval_client,
        dds_evaluator=RuleDDSEvaluator(num_worlds=2, rng_seed=7),
    )
    chosen = north.step(st)
    assert chosen == card_id(Suit.HEARTS, 0)  # H:2 chosen, NOT forced into H:4!
    assert "H:2" in eval_client.calls[0]["user_prompt"]
    assert "H:4" in eval_client.calls[0]["user_prompt"]


def test_server_step_accepts_opponent_bidding_system_and_rejects_conflict() -> None:
    app = create_app()
    client = TestClient(app)

    # Conflict: cannot pass both opponent_bidding_system and opponent_convention_card
    resp = client.post(
        "/v1/step",
        json={
            "seat": "SOUTH",
            "tournament_seed": 3000,
            "board_number": 1,
            "action_history": [],
            "bidding_system": "ccba",
            "opponent_bidding_system": "precision",
            "opponent_convention_card": {
                "id": "custom",
                "name": "Custom",
                "summary_rules": "Natural",
            },
            "bot": "baseline",
        },
    )
    assert resp.status_code == 422

    # Valid opponent_bidding_system with baseline bot returns 200
    ok = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "tournament_seed": 3000,
            "board_number": 1,
            "action_history": [],
            "bidding_system": "ccba",
            "opponent_bidding_system": "precision",
            "bot": "baseline",
        },
    )
    assert ok.status_code == 200

    # Valid /v1/step with tenace (mock profile) + opponent_bidding_system returns 200
    step_tenace = client.post(
        "/v1/step",
        json={
            "seat": "SOUTH",
            "tournament_seed": 3000,
            "board_number": 1,
            "action_history": ["Pass", "1C"],
            "bidding_system": "ccba",
            "opponent_bidding_system": "precision",
            "profile": {"mode": "mock"},
            "dds_worlds": 2,
            "bot": "tenace",
        },
    )
    assert step_tenace.status_code == 200

    # Valid /v1/match tenace vs tenace with asymmetric systems returns 200
    match_resp = client.post(
        "/v1/match",
        json={
            "boards": 1,
            "seed": 3000,
            "dds_worlds": 1,
            "double_dummy": True,
            "opponent": "tenace",
            "team_a_system": "ccba",
            "team_b_system": "precision",
            "team_a_profile": {"mode": "mock"},
            "team_b_profile": {"mode": "mock"},
        },
    )
    assert match_resp.status_code == 200
    assert "Precision Club" in match_resp.json()["report"]


def test_cli_allows_opponent_system_with_bluechip_for_disclosure() -> None:
    from tenace.cli import build_arg_parser, resolve_model_profiles

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--mode",
            "mock",
            "--opponent",
            "bluechip",
            "--bluechip-cmd",
            "echo {port}",
            "--system",
            "ccba",
            "--opponent-system",
            "21gf",
        ]
    )
    profile_a, profile_b = resolve_model_profiles(args)
    assert profile_a.mode == "mock"
    assert profile_b is None
    assert args.opponent_system == "21gf"
