from __future__ import annotations

from typing import Any, TypeVar

import pyspiel
import pytest
from pydantic import BaseModel

import tenace
import tenace.agent as agent_mod
import tenace.harness as harness_pkg
import tenace.harness.duplicate as duplicate_mod
import tenace.harness.scoring as scoring_mod
from tenace import (
    SAYC_CONVENTION_CARD,
    DiffusionGemmaClient,
    DiffusionOptionEvaluator,
    DuplicateMatchRunner,
    GeminiDecisionsClient,
    RoomTranscript,
    Seat,
    TacticalOptionEvaluator,
    TenaceAgent,
    VertexGeminiClient,
)
from tenace.domain import (
    FIRST_CONTRACT_BID_ACTION,
    PASS_ACTION,
    Assumption,
    StrategyState,
    Suit,
    TacticalDecision,
    card_id,
)
from tenace.harness import (
    BoardResult,
    BridgeSeatBot,
    FirstLegalPassBaselineBot,
    MatchSummary,
    RoomTranscript as HarnessRoomTranscript,
    score_diff_to_imps,
)
from tenace.layer3_rules import RuleDDSEvaluator

T = TypeVar("T", bound=BaseModel)


class RecordingStructuredClient:
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
        if not self._responses:
            raise AssertionError("No queued response left in RecordingStructuredClient")
        resp = self._responses.pop(0)
        assert isinstance(resp, schema)
        return resp


def _make_custom_dealt_state(hands: dict[int, list[int]]) -> pyspiel.State:
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state()
    for r in range(13):
        for p in range(4):
            state.apply_action(hands[p][r])
    return state


def test_public_exports_for_agent_harness_and_top_level_package():
    assert set(scoring_mod.__all__) >= {"score_diff_to_imps"}
    assert set(duplicate_mod.__all__) >= {
        "BoardResult",
        "BridgeSeatBot",
        "DuplicateMatchRunner",
        "FirstLegalPassBaselineBot",
        "MatchSummary",
        "RoomTranscript",
    }
    assert set(harness_pkg.__all__) >= {
        "BoardResult",
        "BridgeSeatBot",
        "DuplicateMatchRunner",
        "FirstLegalPassBaselineBot",
        "MatchSummary",
        "RoomTranscript",
        "score_diff_to_imps",
    }
    assert set(agent_mod.__all__) >= {"TenaceAgent"}
    assert set(tenace.__all__) >= {
        "DiffusionGemmaClient",
        "DiffusionOptionEvaluator",
        "DuplicateMatchRunner",
        "GeminiDecisionsClient",
        "RoomTranscript",
        "SAYC_CONVENTION_CARD",
        "Seat",
        "TacticalOptionEvaluator",
        "TenaceAgent",
        "VertexGeminiClient",
    }
    assert tenace.TenaceAgent is TenaceAgent
    assert tenace.Seat is Seat
    assert tenace.DuplicateMatchRunner is DuplicateMatchRunner
    assert tenace.RoomTranscript is RoomTranscript
    assert HarnessRoomTranscript is RoomTranscript
    assert duplicate_mod.RoomTranscript is RoomTranscript
    assert tenace.VertexGeminiClient is VertexGeminiClient
    assert tenace.GeminiDecisionsClient is GeminiDecisionsClient
    assert tenace.DiffusionGemmaClient is DiffusionGemmaClient
    assert tenace.TacticalOptionEvaluator is TacticalOptionEvaluator
    assert tenace.DiffusionOptionEvaluator is DiffusionOptionEvaluator
    assert tenace.SAYC_CONVENTION_CARD is SAYC_CONVENTION_CARD


def test_wbf_imp_table_conversion():
    # Official WBF IMP endpoints across the entire 0..24 scale
    expected_endpoints = [
        (0, 0),
        (10, 0),
        (15, 1),
        (20, 1),
        (40, 1),
        (50, 2),
        (80, 2),
        (90, 3),
        (120, 3),
        (130, 4),
        (160, 4),
        (170, 5),
        (210, 5),
        (220, 6),
        (260, 6),
        (270, 7),
        (310, 7),
        (320, 8),
        (360, 8),
        (370, 9),
        (420, 9),
        (430, 10),
        (490, 10),
        (500, 11),
        (590, 11),
        (600, 12),
        (740, 12),
        (750, 13),
        (890, 13),
        (900, 14),
        (1090, 14),
        (1100, 15),
        (1290, 15),
        (1300, 16),
        (1490, 16),
        (1500, 17),
        (1740, 17),
        (1750, 18),
        (1990, 18),
        (2000, 19),
        (2240, 19),
        (2250, 20),
        (2490, 20),
        (2500, 21),
        (2990, 21),
        (3000, 22),
        (3490, 22),
        (3500, 23),
        (3990, 23),
        (4000, 24),
        (5500, 24),
    ]
    for diff, expected_imp in expected_endpoints:
        assert score_diff_to_imps(diff) == expected_imp, f"Failed for +{diff}"
        assert score_diff_to_imps(float(diff)) == expected_imp
        assert score_diff_to_imps(-diff) == -expected_imp, f"Failed for -{diff}"
        assert score_diff_to_imps(float(-diff)) == -expected_imp

    # Reject non-finite values and booleans cleanly
    for invalid in (float("nan"), float("inf"), float("-inf"), True, False):
        with pytest.raises(ValueError):
            score_diff_to_imps(invalid)  # type: ignore[arg-type]


def test_tenace_agent_forced_move_zero_api_fast_path_and_deferred_dummy_seen():
    # Setup:
    # North declares 1NT (59). East leads C:A (48).
    # Dummy (South) holds ONLY ONE club: C:2 (0), plus S:3 (7) and D:3 (5), etc.
    # On Trick 1, South's play (controlled by declarer North!) has ONLY 1 legal action (C:2).
    # Crucial invariant:
    # 1. Forced Trick-1 dummy singleton takes the 0-API fast path (forced_moves == 1,
    #    strategist_calls == 0, evaluator_calls == 0).
    # 2. Even if North already held an "auction" or pre-dummy "play" strategy, _dummy_seen MUST NOT
    #    be set to True during the forced move! When North next has a non-forced decision (playing
    #    from North's own hand on Trick 1 card 4, where North holds C:3 and C:5 -> 2 non-equivalent
    #    club options since C:4 is in West), Strategist MUST run and observe dummy!
    n_cards = [
        card_id(Suit.CLUBS, 1),     # C:3
        card_id(Suit.CLUBS, 3),     # C:5 (non-equivalent to C:3 because C:4 is in West)
        *[card_id(Suit.HEARTS, r) for r in range(0, 11)],  # H:2..H:Q (11 cards)
    ]
    s_cards = [
        card_id(Suit.CLUBS, 0),     # C:2 (singleton club in dummy!)
        *[card_id(Suit.SPADES, r) for r in range(0, 12)],  # S:2..S:K (12 cards)
    ]
    e_cards = [
        card_id(Suit.CLUBS, 12),    # C:A (opening lead)
        *[card_id(Suit.DIAMONDS, r) for r in range(0, 12)],  # D:2..D:K (12 cards)
    ]
    w_cards = [
        card_id(Suit.CLUBS, 2),     # C:4 (played on Trick 1 card 3, wait: if C:4 is played before North, C:3 and C:5 become bridged!)
        # Give West C:6 (rank 4) and C:4 (rank 2) so West plays C:6 and keeps C:4 unplayed!
        card_id(Suit.CLUBS, 4),     # C:6
        *[card_id(Suit.CLUBS, r) for r in range(5, 12)],   # C:7..C:K (7 cards)
        card_id(Suit.DIAMONDS, 12), # D:A
        card_id(Suit.HEARTS, 11),   # H:K
        card_id(Suit.HEARTS, 12),   # H:A
        card_id(Suit.SPADES, 12),   # S:A
    ]

    state = _make_custom_dealt_state({0: n_cards, 1: e_cards, 2: s_cards, 3: w_cards})
    state.apply_action(59)  # 1NT by North
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)

    strat_client = RecordingStructuredClient(
        [
            StrategyState(
                phase="play",
                plan_summary="Plan after observing dummy South's 12 spades",
                target_or_threat="Establish spades",
            )
        ]
    )
    eval_client = RecordingStructuredClient(
        [
            TacticalDecision(
                assumption_violated=False,
                option_scores={"C:5": 0.85, "C:3": 0.25},
                rationale="Unblock/signal C:5",
            )
        ]
    )
    dds = RuleDDSEvaluator(num_worlds=2, rng_seed=7)
    north_agent = TenaceAgent(
        seat=Seat.NORTH,
        strategist_client=strat_client,
        evaluator_client=eval_client,
        dds_evaluator=dds,
    )
    # Pre-seed a play-phase strategy as if Strategist ran before dummy was visible,
    # isolating the `not dummy_already_seen` trigger specifically!
    north_agent.strategy = StrategyState(
        phase="play",
        plan_summary="Pre-dummy placeholder play strategy",
        revision=1,
    )

    # East leads C:A
    state.apply_action(card_id(Suit.CLUBS, 12))
    assert state.current_player() == int(Seat.NORTH)  # Declarer North acts for Dummy South

    # Dummy South has forced singleton C:2 -> 0 API calls, _dummy_seen stays False!
    forced_action = north_agent.step(state)
    assert forced_action == card_id(Suit.CLUBS, 0)
    assert north_agent.forced_moves == 1
    assert north_agent.strategist_calls == 0
    assert north_agent.evaluator_calls == 0
    assert north_agent._dummy_seen is False
    state.apply_action(forced_action)

    # West plays C:6 (keeping C:4 unplayed so North's C:3 and C:5 remain non-equivalent)
    state.apply_action(card_id(Suit.CLUBS, 4))
    assert state.current_player() == int(Seat.NORTH)

    # Now North has 2 non-equivalent candidates (C:5 and C:3) AND dummy is visible
    # -> Strategist MUST run because _dummy_seen was deferred!
    chosen = north_agent.step(state)
    assert chosen in (card_id(Suit.CLUBS, 1), card_id(Suit.CLUBS, 3))
    assert north_agent.forced_moves == 1
    assert north_agent.strategist_calls == 1
    assert north_agent.evaluator_calls == 1
    assert north_agent._dummy_seen is True
    assert north_agent.strategy is not None
    assert north_agent.strategy.revision == 2
    assert "Plan after observing dummy South" in north_agent.strategy.plan_summary


def test_tenace_agent_defers_surprise_void_during_forced_move_until_next_choice():
    # Setup:
    # North declares 1NT (59) with _dummy_seen=True and an active play strategy.
    # On Trick 1: East leads C:A, South plays C:2, West discards D:2 (SHOWS OUT OF CLUBS!),
    # and North has ONLY ONE club (C:3 -> forced move!).
    # On Trick 1 card 4, North takes the 0-API forced-move fast path (C:3).
    # On Trick 2: East leads D:K, South has D:3 and D:5 (with D:4 in West -> 2 non-equivalent choices!).
    # Strategist MUST replan on Trick 2 because West's club show-out was NOT swallowed during Trick 1's forced move!
    n_cards = [
        card_id(Suit.CLUBS, 1),     # C:3 (singleton club -> forced on Trick 1!)
        *[card_id(Suit.HEARTS, r) for r in range(0, 12)],  # H:2..H:K (12 cards)
    ]
    s_cards = [
        card_id(Suit.CLUBS, 0),     # C:2 (played Trick 1)
        card_id(Suit.DIAMONDS, 1),  # D:3 (Trick 2 choice 1)
        card_id(Suit.DIAMONDS, 3),  # D:5 (Trick 2 choice 2; D:4 is in West)
        *[card_id(Suit.SPADES, r) for r in range(0, 10)],  # S:2..S:J (10 cards)
    ]
    e_cards = [
        card_id(Suit.CLUBS, 12),    # C:A (wins Trick 1)
        *[card_id(Suit.CLUBS, r) for r in range(2, 12)],   # C:4..C:K (10 cards)
        card_id(Suit.DIAMONDS, 11), # D:K (led on Trick 2)
        card_id(Suit.DIAMONDS, 12), # D:A
    ]
    w_cards = [
        card_id(Suit.DIAMONDS, 0),  # D:2 (discarded on Trick 1 -> West shows out of CLUBS!)
        card_id(Suit.DIAMONDS, 2),  # D:4 (kept so South's D:3 and D:5 are non-equivalent)
        *[card_id(Suit.DIAMONDS, r) for r in range(4, 11)],  # D:6..D:Q (7 cards)
        card_id(Suit.HEARTS, 12),   # H:A
        card_id(Suit.SPADES, 10),   # S:Q
        card_id(Suit.SPADES, 11),   # S:K
        card_id(Suit.SPADES, 12),   # S:A
    ]

    state = _make_custom_dealt_state({0: n_cards, 1: e_cards, 2: s_cards, 3: w_cards})
    state.apply_action(59)  # 1NT by North
    for _ in range(3):
        state.apply_action(PASS_ACTION)

    strat_client = RecordingStructuredClient(
        [
            StrategyState(
                phase="play",
                plan_summary="Replanned after West showed out of clubs",
                target_or_threat="Guard diamonds",
            )
        ]
    )
    eval_client = RecordingStructuredClient(
        [
            TacticalDecision(
                assumption_violated=False,
                option_scores={"D:5": 0.8, "D:3": 0.3},
                rationale="Play D:5",
            )
        ]
    )
    north_agent = TenaceAgent(
        seat=Seat.NORTH,
        strategist_client=strat_client,
        evaluator_client=eval_client,
        dds_evaluator=RuleDDSEvaluator(num_worlds=2, rng_seed=5),
    )
    # Simulate North having already planned in play with dummy seen and empty voids
    north_agent.strategy = StrategyState(
        phase="play",
        plan_summary="Initial play plan assuming clubs 4-4",
        revision=2,
    )
    north_agent._dummy_seen = True
    north_agent._prev_voids = {s: set() for s in Seat}

    # Trick 1: East leads C:A, South plays C:2, West discards D:2 (void in clubs!)
    state.apply_action(card_id(Suit.CLUBS, 12))
    state.apply_action(card_id(Suit.CLUBS, 0))
    state.apply_action(card_id(Suit.DIAMONDS, 0))

    # North is forced to play singleton C:3 -> 0 API calls
    forced = north_agent.step(state)
    assert forced == card_id(Suit.CLUBS, 1)
    assert north_agent.forced_moves == 1
    assert north_agent.strategist_calls == 0
    assert north_agent.evaluator_calls == 0
    state.apply_action(forced)

    # Trick 2: East won C:A and leads D:K; North acts for Dummy South (choices D:5 vs D:3)
    state.apply_action(card_id(Suit.DIAMONDS, 11))
    chosen = north_agent.step(state)
    assert chosen in (card_id(Suit.DIAMONDS, 1), card_id(Suit.DIAMONDS, 3))
    assert north_agent.strategist_calls == 1
    assert north_agent.evaluator_calls == 1
    assert "WEST showed out of CLUBS" in strat_client.calls[0]["user_prompt"]


def test_tenace_agent_reruns_strategist_when_layer2_flags_assumption_violation():
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state(tournament_seed=42, board_number=1)
    # Advance auction to 7S by West so North faces a crisp 2-option choice: Pass (52) vs 7NT (89)
    state.apply_action(PASS_ACTION)  # N
    state.apply_action(PASS_ACTION)  # E
    state.apply_action(PASS_ACTION)  # S
    state.apply_action(88)           # W bids 7S -> only Pass, Dbl, 7NT legal for North!
    assert state.current_player() == int(Seat.NORTH)

    strat_client = RecordingStructuredClient(
        [
            StrategyState(
                phase="auction",
                plan_summary="Initial auction plan assuming grand slam values",
                assumptions=[Assumption(claim="Partnership holds 37 HCP", confidence=0.7)],
            ),
            StrategyState(
                phase="auction",
                plan_summary="Revised auction plan doubling or passing",
                assumptions=[Assumption(claim="Defensive misfits", confidence=0.9)],
            ),
        ]
    )
    eval_client = RecordingStructuredClient(
        [
            TacticalDecision(
                assumption_violated=True,
                violation_reason="Hand has 9 HCP, contradicting 37 HCP grand slam assumption",
                option_scores={"7NT": 0.9, "Pass": 0.1, "Dbl": 0.1},
                rationale="Initial assumption violated; request Strategist augmentation.",
            ),
            TacticalDecision(
                assumption_violated=True,  # Even if still True, interrupt runs AT MOST ONCE per step!
                violation_reason="Secondary note",
                option_scores={"Pass": 0.99, "Dbl": 0.05, "7NT": 0.0},
                rationale="Pass calmly with revised plan.",
            ),
        ]
    )
    # Use max_make_prob_drop=1.0 so all 3 options stay viable and Layer 2 preference decides cleanly
    dds = RuleDDSEvaluator(num_worlds=2, max_make_prob_drop=1.0, rng_seed=11)
    agent = TenaceAgent(
        seat=Seat.NORTH,
        strategist_client=strat_client,
        evaluator_client=eval_client,
        dds_evaluator=dds,
        convention_card=SAYC_CONVENTION_CARD,
    )

    action = agent.step(state)
    # Verify play-phase self/dummy show-outs and endgame caps avoid runaway L1 replanning
    agent_capped = TenaceAgent(
        seat=Seat.NORTH,
        strategist_client=RecordingStructuredClient([]),
        evaluator_client=RecordingStructuredClient([]),
        max_play_replans=2,
    )
    assert agent_capped.max_play_replans == 2
    assert action == PASS_ACTION
    # Initial Strategist (rev 1) + Assumption-violation Strategist interrupt (rev 2) = 2 calls
    assert agent.strategist_calls == 2
    # Initial Evaluator + Post-augmentation Evaluator = 2 calls (no infinite loop!)
    assert agent.evaluator_calls == 2
    assert agent.forced_moves == 0
    assert agent.strategy is not None
    assert agent.strategy.revision == 2
    assert agent.strategy.plan_summary == "Revised auction plan doubling or passing"
    assert [
        (s.revision, s.phase, s.plan_summary) for s in agent.strategy_history
    ] == [
        (1, "auction", "Initial auction plan assuming grand slam values"),
        (2, "auction", "Revised auction plan doubling or passing"),
    ]
    # Confirm violation_reason was forwarded into Strategist replan prompt
    second_strat_prompt = strat_client.calls[1]["user_prompt"]
    assert "Hand has 9 HCP, contradicting 37 HCP grand slam assumption" in second_strat_prompt

    # Resetting for next board clears strategy, strategy_history, voids, and _dummy_seen while keeping cumulative stats
    agent._dummy_seen = True
    agent._prev_voids = {s: {Suit.CLUBS} for s in Seat}
    agent.reset_for_board()
    assert agent.strategy is None
    assert agent.strategy_history == []
    assert agent._prev_voids is None
    assert agent._dummy_seen is False
    assert agent.strategist_calls == 2
    assert agent.evaluator_calls == 2

    agent.reset_stats()
    assert agent.strategist_calls == 0
    assert agent.evaluator_calls == 0
    assert agent.forced_moves == 0


def test_tenace_agent_rejects_wrong_seat_terminal_and_undealt_states():
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    undealt = game.new_initial_state()
    dealt = game.new_initial_state(tournament_seed=42, board_number=1)

    strat_client = RecordingStructuredClient([])
    eval_client = RecordingStructuredClient([])
    south_agent = TenaceAgent(
        seat=Seat.SOUTH,
        strategist_client=strat_client,
        evaluator_client=eval_client,
    )

    # Dealt board 1 starts on North's turn -> South agent must raise ValueError
    with pytest.raises(ValueError, match="seat|turn|player"):
        south_agent.step(dealt)

    # Undealt chance node -> ValueError
    north_agent = TenaceAgent(
        seat=Seat.NORTH,
        strategist_client=strat_client,
        evaluator_client=eval_client,
    )
    with pytest.raises(ValueError, match="deal|chance|phase"):
        north_agent.step(undealt)

    # Terminal passed-out state -> ValueError
    terminal = game.new_initial_state(tournament_seed=42, board_number=1)
    for _ in range(4):
        terminal.apply_action(PASS_ACTION)
    assert terminal.is_terminal()
    with pytest.raises(ValueError, match="terminal"):
        north_agent.step(terminal)


def test_duplicate_match_runner_plays_open_and_closed_rooms_without_memory_leakage():
    class AsymmetricContractBot:
        """Team A bids 1NT when Dealer on Board 1, otherwise passes/plays first legal card."""

        def __init__(self, seat: Seat, team_name: str) -> None:
            self.seat = seat
            self.team_name = team_name
            self.reset_calls = 0
            self.observed_hands: list[tuple[int, ...]] = []

        def reset_for_board(self) -> None:
            self.reset_calls += 1

        def step(self, state: pyspiel.State) -> int:
            if state.current_phase() == 1:
                own = tuple(
                    c
                    for c, flag in enumerate(
                        state.observation_tensor(int(self.seat))[432 : 432 + 52]
                    )
                    if flag > 0.5
                )
                if not self.observed_hands or self.observed_hands[-1] != own:
                    self.observed_hands.append(own)
                if self.team_name == "A" and len(state.full_history()) == 52:
                    return 59  # Open 1NT on first call of board
                return PASS_ACTION
            return state.legal_actions()[0]

    created_a: list[AsymmetricContractBot] = []
    created_b: list[FirstLegalPassBaselineBot] = []

    def team_a_factory(seat: Seat) -> BridgeSeatBot:
        bot = AsymmetricContractBot(seat=seat, team_name="A")
        created_a.append(bot)
        return bot

    def team_b_factory(seat: Seat) -> BridgeSeatBot:
        bot = FirstLegalPassBaselineBot(seat=seat)
        created_b.append(bot)
        return bot

    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    summary = runner.play_match(team_a_factory, team_b_factory, num_boards=2)

    assert isinstance(summary, MatchSummary)
    assert len(summary.boards) == 2
    assert [b.board_number for b in summary.boards] == [1, 2]

    for board_res in summary.boards:
        assert isinstance(board_res, BoardResult)
        expected_imp = score_diff_to_imps(
            board_res.open_room_ns_score - board_res.closed_room_ns_score
        )
        assert board_res.imps_team_a == expected_imp

    assert summary.total_imps_team_a == sum(b.imps_team_a for b in summary.boards)
    assert summary.total_imps_team_b == -summary.total_imps_team_a

    # Verify Open Room and Closed Room used separate bot instances (4 Team A bots total: N,S in Open; E,W in Closed)
    assert len(created_a) == 4
    assert len(created_b) == 4
    assert all(bot.reset_calls == 2 for bot in created_a)

    # Verify identical dealt cards on Board 1 & Board 2 between Open Room North (Team A) and Closed Room East (Team A):
    # Specifically, Open Room North and Closed Room North must hold identical hands!
    open_north_a = next(b for b in created_a if b.seat == Seat.NORTH)
    closed_east_a = next(b for b in created_a if b.seat == Seat.EAST)
    assert len(open_north_a.observed_hands) == 2
    assert len(closed_east_a.observed_hands) == 2
    # North and East on the same board MUST have disjoint hands (no card overlap)
    assert set(open_north_a.observed_hands[0]).isdisjoint(set(closed_east_a.observed_hands[0]))

    # Validate num_boards >= 1 check and double-dummy fast mode
    with pytest.raises(ValueError, match="num_boards"):
        runner.play_match(team_a_factory, team_b_factory, num_boards=0)
    with pytest.raises(ValueError, match="num_boards"):
        runner.play_match(team_a_factory, team_b_factory, num_boards=True)  # type: ignore[arg-type]

    dd_runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=True)
    dd_summary = dd_runner.play_match(team_a_factory, team_b_factory, num_boards=2)
    assert len(dd_summary.boards) == 2
    # On Board 1: Open Room Team A (North) opens 1NT (non-zero DD score), Closed Room passes out (0.0)
    assert dd_summary.boards[0].open_room_ns_score != 0.0
    assert dd_summary.boards[0].closed_room_ns_score == 0.0


def test_duplicate_runner_clones_state_and_rejects_illegal_or_bool_bot_actions():
    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)

    # 1. Custom bot mutating its input `state` via `state.apply_action(...)` inside `step()`
    #    must NOT corrupt the live room state!
    class StateMutatingBot:
        def __init__(self, seat: Seat) -> None:
            self.seat = seat

        def reset_for_board(self) -> None:
            return None

        def step(self, state: pyspiel.State) -> int:
            chosen = PASS_ACTION if PASS_ACTION in state.legal_actions() else state.legal_actions()[0]
            # Mutate the passed-in state all the way to terminal!
            while not state.is_terminal():
                state.apply_action(state.legal_actions()[0])
            return chosen

    mut_summary = runner.play_match(StateMutatingBot, FirstLegalPassBaselineBot, num_boards=1)
    assert len(mut_summary.boards) == 1
    assert mut_summary.boards[0].open_room_ns_score == 0.0
    assert mut_summary.boards[0].closed_room_ns_score == 0.0

    # 2. Custom bot returning bool (`True` == 1 in Python!), illegal revoke card, or out-of-range `999`
    #    must raise `ValueError` BEFORE calling OpenSpiel C++ `apply_action`!
    class BoolReturningBot:
        def __init__(self, seat: Seat) -> None:
            self.seat = seat

        def reset_for_board(self) -> None:
            return None

        def step(self, state: pyspiel.State) -> int:
            return True  # type: ignore[return-value]

    with pytest.raises(ValueError, match="Illegal action"):
        runner.play_match(BoolReturningBot, FirstLegalPassBaselineBot, num_boards=1)

    # Illegal card play / revoke during auction or play (e.g. card 0 during auction, or non-held card in play)
    class RevokeBot:
        def __init__(self, seat: Seat) -> None:
            self.seat = seat

        def reset_for_board(self) -> None:
            return None

        def step(self, state: pyspiel.State) -> int:
            if state.current_phase() == 1:
                if len(state.full_history()) == 52:
                    return 59  # Open 1NT so we enter play phase
                return PASS_ACTION
            # In play phase, return a card NOT in legal_actions() (suit revoke / unheld card)
            legal_set = set(state.legal_actions())
            illegal_card = next(c for c in range(52) if c not in legal_set)
            return illegal_card

    with pytest.raises(ValueError, match="Illegal action"):
        runner.play_match(RevokeBot, FirstLegalPassBaselineBot, num_boards=1)

    # Out-of-range action `999`: wrap _new_dealt_board_state so even in RED phase C++ cannot SIGABRT
    class SafeStateProxy:
        def __init__(self, inner: pyspiel.State) -> None:
            self._inner = inner

        def clone(self) -> SafeStateProxy:
            return SafeStateProxy(self._inner.clone())

        def apply_action(self, action: int) -> None:
            if isinstance(action, bool) or not isinstance(action, int) or not (0 <= action <= 89):
                raise RuntimeError(f"C++ DoApplyAction reached with unsafe action {action!r}")
            self._inner.apply_action(action)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    class OutOfRangeBot:
        def __init__(self, seat: Seat) -> None:
            self.seat = seat

        def reset_for_board(self) -> None:
            return None

        def step(self, state: pyspiel.State) -> int:
            return 999

    safe_state = SafeStateProxy(runner._new_dealt_board_state(1))
    seats = {
        Seat.NORTH: OutOfRangeBot(Seat.NORTH),
        Seat.SOUTH: FirstLegalPassBaselineBot(Seat.SOUTH),
        Seat.EAST: FirstLegalPassBaselineBot(Seat.EAST),
        Seat.WEST: FirstLegalPassBaselineBot(Seat.WEST),
    }
    with pytest.raises(ValueError, match="Illegal action"):
        runner._play_room(safe_state, seats)  # type: ignore[arg-type]


def test_first_legal_pass_baseline_bot_rejects_wrong_seat_terminal_and_undealt_states():
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    undealt = game.new_initial_state()
    dealt = game.new_initial_state(tournament_seed=42, board_number=1)

    south_bot = FirstLegalPassBaselineBot(Seat.SOUTH)
    with pytest.raises(ValueError, match="seat|turn|player"):
        south_bot.step(dealt)

    north_bot = FirstLegalPassBaselineBot(Seat.NORTH)
    with pytest.raises(ValueError, match="deal|chance|phase"):
        north_bot.step(undealt)

    terminal = game.new_initial_state(tournament_seed=42, board_number=1)
    for _ in range(4):
        terminal.apply_action(PASS_ACTION)
    with pytest.raises(ValueError, match="terminal"):
        north_bot.step(terminal)


def test_duplicate_runner_records_room_transcripts_strategy_history_and_closes_bots():
    import re
    from pydantic import ValidationError

    closed_seats: list[Seat] = []

    class ClosableBaselineBot(FirstLegalPassBaselineBot):
        def close(self) -> None:
            closed_seats.append(self.seat)

    class ScriptedStrategistClient:
        def __init__(self, seat: Seat) -> None:
            self.seat = seat
            self.calls = 0

        def generate_structured(
            self,
            system_prompt: str,
            user_prompt: str,
            schema: type[T],
        ) -> T:
            self.calls += 1
            phase = "play" if "Phase: play" in user_prompt else "auction"
            resp = StrategyState(
                phase=phase,  # type: ignore[arg-type]
                plan_summary=f"{self.seat.name} {phase} plan #{self.calls}",
                target_or_threat="Control tempo",
            )
            assert isinstance(resp, schema)
            return resp

    class ScriptedEvaluatorClient:
        def __init__(self, seat: Seat) -> None:
            self.seat = seat
            self._triggered_interrupt = False

        def generate_structured(
            self,
            system_prompt: str,
            user_prompt: str,
            schema: type[T],
        ) -> T:
            # Extract candidate labels from "- LABEL: baseline_ev=..."
            labels = re.findall(r"^- ([^:]+): baseline_ev=", user_prompt, flags=re.MULTILINE)
            # In Open Room on Board 1, North opens 1NT (if legal) and triggers one assumption violation interrupt
            if (
                self.seat == Seat.NORTH
                and "Phase: auction" in user_prompt
                and "1NT" in labels
            ):
                if not self._triggered_interrupt:
                    self._triggered_interrupt = True
                    resp = TacticalDecision(
                        assumption_violated=True,
                        violation_reason="Re-check NT range before opening 1NT",
                        option_scores={"1NT": 1.0, "Pass": 0.1},
                        rationale="Trigger interrupt once then open 1NT",
                    )
                else:
                    resp = TacticalDecision(
                        assumption_violated=False,
                        option_scores={"1NT": 1.0, "Pass": 0.1},
                        rationale="Open 1NT",
                    )
            elif "Phase: auction" in user_prompt and "Pass" in labels:
                resp = TacticalDecision(
                    assumption_violated=False,
                    option_scores={"Pass": 1.0},
                    rationale="Pass cleanly",
                )
            else:
                scores = {labels[0]: 1.0} if labels else {}
                resp = TacticalDecision(
                    assumption_violated=False,
                    option_scores=scores,
                    rationale="Play top candidate",
                )
            assert isinstance(resp, schema)
            return resp

    created_tenace: list[TenaceAgent] = []

    def make_tenace(seat: Seat) -> TenaceAgent:
        # Use max_make_prob_drop=1.0 so North's scripted 1NT bid isn't blocked by DDS make_prob gate
        agent = TenaceAgent(
            seat=seat,
            strategist_client=ScriptedStrategistClient(seat),
            evaluator_client=ScriptedEvaluatorClient(seat),
            dds_evaluator=RuleDDSEvaluator(num_worlds=2, max_make_prob_drop=1.0, rng_seed=7),
        )
        created_tenace.append(agent)
        return agent

    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    summary = runner.play_match(
        team_a_factory=make_tenace,
        team_b_factory=ClosableBaselineBot,
        num_boards=1,
    )

    # Extra fields must be forbidden on RoomTranscript
    with pytest.raises(ValidationError):
        RoomTranscript(
            contract="Passed Out",
            auction="NORTH:Pass -> EAST:Pass -> SOUTH:Pass -> WEST:Pass",
            ns_score=0.0,
            unexpected_field="nope",  # type: ignore[call-arg]
        )

    board = summary.boards[0]
    assert isinstance(board.open_room, RoomTranscript)
    assert isinstance(board.closed_room, RoomTranscript)

    # Open Room: North opened 1NT and everybody passed -> 13 played tricks!
    assert board.open_room.contract == "1NT by NORTH"
    assert board.open_room.auction == "NORTH:1NT -> EAST:Pass -> SOUTH:Pass -> WEST:Pass"
    assert board.open_room.ns_score == board.open_room_ns_score
    assert len(board.open_room.tricks) == 13
    trick1_pattern = re.compile(
        r"^Trick 1: E:[CDHS]:[2-9TJQKA] S:[CDHS]:[2-9TJQKA] W:[CDHS]:[2-9TJQKA] N:[CDHS]:[2-9TJQKA] -> (NORTH|EAST|SOUTH|WEST)$"
    )
    assert trick1_pattern.match(board.open_room.tricks[0]), board.open_room.tricks[0]

    # North had rev 1 (initial auction), rev 2 (interrupt replan), plus play-phase replan(s)
    assert "NORTH" in board.open_room.strategy_traces
    north_traces = board.open_room.strategy_traces["NORTH"]
    assert north_traces[0] == "rev 1 (auction): NORTH auction plan #1"
    assert north_traces[1] == "rev 2 (auction): NORTH auction plan #2"
    assert any(t.startswith("rev 3 (play):") for t in north_traces)
    # Baseline bots (EAST, WEST in Open Room) have no strategy_history so they do not appear in strategy_traces
    assert "EAST" not in board.open_room.strategy_traces
    assert "WEST" not in board.open_room.strategy_traces

    # Closed Room: North (ClosableBaselineBot) passed, East (Tenace) passed, South passed, West passed -> Passed Out!
    assert board.closed_room.contract == "Passed Out"
    assert board.closed_room.auction == "NORTH:Pass -> EAST:Pass -> SOUTH:Pass -> WEST:Pass"
    assert board.closed_room.tricks == []
    assert board.closed_room.ns_score == 0.0
    assert "EAST" in board.closed_room.strategy_traces
    assert "WEST" in board.closed_room.strategy_traces
    assert "NORTH" not in board.closed_room.strategy_traces

    # Telemetry sums across all 8 bots + per-team breakdown
    expected_forced = sum(a.forced_moves for a in created_tenace)
    expected_strat = sum(a.strategist_calls for a in created_tenace)
    expected_eval = sum(a.evaluator_calls for a in created_tenace)
    assert summary.telemetry == {
        "forced_moves": expected_forced,
        "strategist_calls": expected_strat,
        "evaluator_calls": expected_eval,
        "team_a_forced_moves": expected_forced,
        "team_a_strategist_calls": expected_strat,
        "team_a_evaluator_calls": expected_eval,
        "team_b_forced_moves": 0,
        "team_b_strategist_calls": 0,
        "team_b_evaluator_calls": 0,
    }
    assert summary.telemetry["strategist_calls"] >= 4
    assert summary.telemetry["evaluator_calls"] >= 4

    # 2 Team-B seats in Open Room (E, W) + 2 Team-B seats in Closed Room (N, S) = 4 closed bots
    assert len(closed_seats) == 4
    assert sorted(s.name for s in closed_seats) == ["EAST", "NORTH", "SOUTH", "WEST"]

    # Double-dummy mode produces empty tricks list even on non-passed-out contracts
    dd_closed: list[Seat] = []

    class DDClosableBot(FirstLegalPassBaselineBot):
        def close(self) -> None:
            dd_closed.append(self.seat)

    dd_runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=True)
    dd_summary = dd_runner.play_match(
        team_a_factory=make_tenace,
        team_b_factory=DDClosableBot,
        num_boards=1,
    )
    assert dd_summary.boards[0].open_room is not None
    assert dd_summary.boards[0].open_room.contract == "1NT by NORTH"
    assert dd_summary.boards[0].open_room.tricks == []
    assert len(dd_closed) == 4


def test_duplicate_runner_closes_all_bots_in_finally_when_bot_raises_mid_board():
    # Also verify doubled ("4SX by SOUTH") and redoubled ("4SXX by SOUTH") contract formatting
    # and trump ruff trick winner formatting in _build_room_transcript!
    dbl_state = duplicate_mod.pyspiel.load_game("bridge(use_double_dummy_result=true)").new_initial_state(
        tournament_seed=42, board_number=1
    )
    dbl_state.apply_action(PASS_ACTION)  # N
    dbl_state.apply_action(PASS_ACTION)  # E
    dbl_state.apply_action(73)           # S bids 4S (55 + 3*5 + 3 = 73)
    dbl_state.apply_action(53)           # W Doubles
    dbl_state.apply_action(PASS_ACTION)  # N
    dbl_state.apply_action(PASS_ACTION)  # E
    dbl_state.apply_action(PASS_ACTION)  # S
    dbl_tx = duplicate_mod._build_room_transcript(dbl_state, {}, float(dbl_state.returns()[0]))
    assert dbl_tx.contract == "4SX by SOUTH"

    closed_bots: list[tuple[str, Seat]] = []

    class ExplodingClosableBot:
        def __init__(self, seat: Seat, team: str) -> None:
            self.seat = seat
            self.team = team

        def reset_for_board(self) -> None:
            return None

        def step(self, state: pyspiel.State) -> int:
            if self.team == "A" and self.seat == Seat.NORTH:
                raise ValueError("Simulated mid-board bot failure")
            return PASS_ACTION

        def close(self) -> None:
            closed_bots.append((self.team, self.seat))

    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    with pytest.raises(ValueError, match="Simulated mid-board bot failure"):
        runner.play_match(
            team_a_factory=lambda s: ExplodingClosableBot(s, "A"),
            team_b_factory=lambda s: ExplodingClosableBot(s, "B"),
            num_boards=2,
        )

    # All 8 room bots (4 in Open Room + 4 in Closed Room) MUST have close() called!
    assert len(closed_bots) == 8

    # Even if one bot's close() raises during teardown, the remaining bots still get close() called!
    close_calls_with_failing_close: list[tuple[str, Seat]] = []

    class CloseRaisingBot(FirstLegalPassBaselineBot):
        def __init__(self, seat: Seat, team: str) -> None:
            super().__init__(seat)
            self.team = team

        def close(self) -> None:
            close_calls_with_failing_close.append((self.team, self.seat))
            if self.team == "A" and self.seat == Seat.NORTH:
                raise RuntimeError("Teardown failure on Open Room North")

    with pytest.raises(RuntimeError, match="Teardown failure on Open Room North"):
        runner.play_match(
            team_a_factory=lambda s: CloseRaisingBot(s, "A"),
            team_b_factory=lambda s: CloseRaisingBot(s, "B"),
            num_boards=1,
        )
    assert len(close_calls_with_failing_close) == 8


def test_duplicate_runner_board2_non_north_dealer_supports_infostate_resampling():
    import random

    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    state = runner._new_dealt_board_state(board_number=2)
    assert state.current_player() == int(Seat.EAST)
    state.apply_action(PASS_ACTION)  # East Pass
    state.apply_action(89)           # South 7NT
    state.apply_action(PASS_ACTION)  # West Pass
    state.apply_action(PASS_ACTION)  # North Pass
    state.apply_action(PASS_ACTION)  # East Pass
    for _ in range(3):
        state.apply_action(state.legal_actions()[0])
    assert state.current_player() == int(Seat.SOUTH)
    rng = random.Random(123)
    resampled = state.resample_from_infostate(int(Seat.SOUTH), rng.random)
    assert resampled.current_player() == int(Seat.SOUTH)


def test_duplicate_runner_records_per_team_telemetry_in_head_to_head_match():
    import re

    class TeamStrategistClient:
        def __init__(self, team_tag: str, seat: Seat) -> None:
            self.team_tag = team_tag
            self.seat = seat
            self.calls = 0

        def generate_structured(
            self,
            system_prompt: str,
            user_prompt: str,
            schema: type[T],
        ) -> T:
            self.calls += 1
            phase = "play" if "Phase: play" in user_prompt else "auction"
            resp = StrategyState(
                phase=phase,  # type: ignore[arg-type]
                plan_summary=f"{self.team_tag}:{self.seat.name} {phase} #{self.calls}",
                assumptions=[Assumption(claim="Balanced tempo", confidence=0.7)],
            )
            assert isinstance(resp, schema)
            return resp

    class TeamEvaluatorClient:
        def __init__(self, team_tag: str, seat: Seat) -> None:
            self.team_tag = team_tag
            self.seat = seat
            self._interrupted = False

        def generate_structured(
            self,
            system_prompt: str,
            user_prompt: str,
            schema: type[T],
        ) -> T:
            labels = re.findall(r"^- ([^:]+): baseline_ev=", user_prompt, flags=re.MULTILINE)
            # Trigger an extra interrupt ONLY for Team A North in auction so Team A and Team B
            # have distinct non-zero call totals across rooms!
            if (
                self.team_tag == "A"
                and self.seat == Seat.NORTH
                and "Phase: auction" in user_prompt
                and "1NT" in labels
            ):
                if not self._interrupted:
                    self._interrupted = True
                    resp = TacticalDecision(
                        assumption_violated=True,
                        violation_reason="Team A extra replan check",
                        option_scores={"1NT": 1.0, "Pass": 0.1},
                        rationale="Interrupt once then bid 1NT",
                    )
                else:
                    resp = TacticalDecision(
                        assumption_violated=False,
                        option_scores={"1NT": 1.0, "Pass": 0.1},
                        rationale="Open 1NT",
                    )
            elif "Phase: auction" in user_prompt and "Pass" in labels:
                resp = TacticalDecision(
                    assumption_violated=False,
                    option_scores={"Pass": 1.0},
                    rationale="Pass",
                )
            else:
                scores = {labels[0]: 1.0} if labels else {}
                resp = TacticalDecision(
                    assumption_violated=False,
                    option_scores=scores,
                    rationale="Top card",
                )
            assert isinstance(resp, schema)
            return resp

    created_team_a: list[TenaceAgent] = []
    created_team_b: list[TenaceAgent] = []

    def make_team_a(seat: Seat) -> TenaceAgent:
        bot = TenaceAgent(
            seat=seat,
            strategist_client=TeamStrategistClient("A", seat),
            evaluator_client=TeamEvaluatorClient("A", seat),
            dds_evaluator=RuleDDSEvaluator(num_worlds=1, max_make_prob_drop=1.0, rng_seed=31 + int(seat)),
        )
        created_team_a.append(bot)
        return bot

    def make_team_b(seat: Seat) -> TenaceAgent:
        bot = TenaceAgent(
            seat=seat,
            strategist_client=TeamStrategistClient("B", seat),
            evaluator_client=TeamEvaluatorClient("B", seat),
            dds_evaluator=RuleDDSEvaluator(num_worlds=1, max_make_prob_drop=1.0, rng_seed=137 + int(seat)),
        )
        created_team_b.append(bot)
        return bot

    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    summary = runner.play_match(make_team_a, make_team_b, num_boards=1)

    assert len(created_team_a) == 4
    assert len(created_team_b) == 4

    exp_a_forced = sum(b.forced_moves for b in created_team_a)
    exp_a_strat = sum(b.strategist_calls for b in created_team_a)
    exp_a_eval = sum(b.evaluator_calls for b in created_team_a)

    exp_b_forced = sum(b.forced_moves for b in created_team_b)
    exp_b_strat = sum(b.strategist_calls for b in created_team_b)
    exp_b_eval = sum(b.evaluator_calls for b in created_team_b)

    assert exp_a_strat > 0 and exp_b_strat > 0
    assert exp_a_strat != exp_b_strat
    assert exp_a_eval != exp_b_eval

    assert summary.telemetry == {
        "forced_moves": exp_a_forced + exp_b_forced,
        "strategist_calls": exp_a_strat + exp_b_strat,
        "evaluator_calls": exp_a_eval + exp_b_eval,
        "team_a_forced_moves": exp_a_forced,
        "team_a_strategist_calls": exp_a_strat,
        "team_a_evaluator_calls": exp_a_eval,
        "team_b_forced_moves": exp_b_forced,
        "team_b_strategist_calls": exp_b_strat,
        "team_b_evaluator_calls": exp_b_eval,
    }






def test_tenace_agent_replans_auction_when_new_non_pass_call_occurs():
    """On round 2+ of an auction after partner/opponent makes a non-Pass call
    (e.g. weak 2D -> 2S forcing, or 1H -> 2H raise), Layer-1 Strategist MUST replan
    so Layer-2 never sees a stale Round-1 'Pass initially' strategy.
    """
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state(tournament_seed=100, board_number=1)
    strat_client = RecordingStructuredClient(
        [
            StrategyState(phase="auction", plan_summary="Open weak 2D"),
            StrategyState(phase="auction", plan_summary="Respond 2NT to forcing 2S"),
        ]
    )
    eval_client = RecordingStructuredClient(
        [
            TacticalDecision(option_scores={"2D": 0.95, "Pass": 0.1}),
            TacticalDecision(option_scores={"2NT": 0.95, "Pass": 0.05}),
        ]
    )
    agent = TenaceAgent(
        seat=Seat.NORTH,
        strategist_client=strat_client,
        evaluator_client=eval_client,
        dds_evaluator=RuleDDSEvaluator(num_worlds=1, rng_seed=7),
    )
    bid1 = agent.step(state.clone())
    assert bid1 == FIRST_CONTRACT_BID_ACTION + 6  # 2D
    assert agent.strategist_calls == 1
    state.apply_action(bid1)
    state.apply_action(PASS_ACTION)  # EAST Pass
    state.apply_action(FIRST_CONTRACT_BID_ACTION + 8)  # SOUTH bids forcing 2S!
    state.apply_action(PASS_ACTION)  # WEST Pass

    # Back to NORTH: SOUTH bid 2S (non-Pass!), so Strategist MUST replan!
    bid2 = agent.step(state.clone())
    assert bid2 == FIRST_CONTRACT_BID_ACTION + 9  # 2NT
    assert agent.strategist_calls == 2
    assert "SOUTH:2S" in strat_client.calls[1]["user_prompt"]


def test_tenace_agent_skips_redundant_auction_replan_on_flat_bust_hand():
    """A flat busted hand (<=5 HCP, max suit length <=4) should NOT re-invoke expensive Layer-1
    every round when only opponents are bidding (partner hasn't acted).
    """
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state()
    # Deterministic deal where NORTH holds a 4-3-3-3 flat 0-HCP bust (2..6 only).
    n_cards = [
        card_id(Suit.CLUBS, 0), card_id(Suit.CLUBS, 1), card_id(Suit.CLUBS, 2), card_id(Suit.CLUBS, 3),
        card_id(Suit.DIAMONDS, 0), card_id(Suit.DIAMONDS, 1), card_id(Suit.DIAMONDS, 2),
        card_id(Suit.HEARTS, 0), card_id(Suit.HEARTS, 1), card_id(Suit.HEARTS, 2),
        card_id(Suit.SPADES, 0), card_id(Suit.SPADES, 1), card_id(Suit.SPADES, 2),
    ]
    used = set(n_cards)
    rem = [c for c in range(52) if c not in used]
    e_cards, s_cards, w_cards = rem[0:13], rem[13:26], rem[26:39]
    for r in range(13):
        state.apply_action(n_cards[r])
        state.apply_action(e_cards[r])
        state.apply_action(s_cards[r])
        state.apply_action(w_cards[r])

    strat_client = RecordingStructuredClient(
        [StrategyState(phase="auction", plan_summary="Pass flat 0 HCP")]
    )
    eval_client = RecordingStructuredClient(
        [
            TacticalDecision(option_scores={"Pass": 1.0}),
            TacticalDecision(option_scores={"Pass": 1.0}),
        ]
    )
    agent = TenaceAgent(
        seat=Seat.NORTH,
        strategist_client=strat_client,
        evaluator_client=eval_client,
        dds_evaluator=RuleDDSEvaluator(num_worlds=1, rng_seed=3),
    )
    assert agent.step(state.clone()) == PASS_ACTION
    assert agent.strategist_calls == 1
    state.apply_action(PASS_ACTION)  # N passes
    state.apply_action(FIRST_CONTRACT_BID_ACTION + 3)  # E opens 1S
    state.apply_action(PASS_ACTION)  # S passes
    state.apply_action(FIRST_CONTRACT_BID_ACTION + 18)  # W bids 4S

    # Back to N (flat 0 HCP, partner passed): MUST NOT burn a 2nd Strategist call!
    assert agent.step(state.clone()) == PASS_ACTION
    assert agent.strategist_calls == 1


def test_extract_recommended_auction_action_parses_single_leading_directive():
    from tenace.agent import extract_recommended_auction_action
    from tenace.domain.types import CandidateOption, DBL_ACTION, RDBL_ACTION

    cands = [
        CandidateOption(action=PASS_ACTION, label="Pass"),
        CandidateOption(action=DBL_ACTION, label="Dbl"),
        CandidateOption(action=RDBL_ACTION, label="RDbl"),
        CandidateOption(action=FIRST_CONTRACT_BID_ACTION + 6, label="2D"),
        CandidateOption(action=FIRST_CONTRACT_BID_ACTION + 7, label="2H"),
        CandidateOption(action=FIRST_CONTRACT_BID_ACTION + 19, label="4NT"),
        CandidateOption(action=FIRST_CONTRACT_BID_ACTION + 28, label="6S"),
    ]
    assert extract_recommended_auction_action(
        "Bid 2H (Jacoby Transfer to Spades) over partner's 1NT opening.", cands
    ) == FIRST_CONTRACT_BID_ACTION + 7
    assert extract_recommended_auction_action(
        "Bid 6S. East's 5S bid over 4NT shows 3 Aces.", cands
    ) == FIRST_CONTRACT_BID_ACTION + 28
    assert extract_recommended_auction_action(
        "Double 4H for penalty.", cands
    ) == DBL_ACTION
    assert extract_recommended_auction_action(
        "Bid XX (Redouble) to show 10+ HCP.", cands
    ) == RDBL_ACTION
    assert extract_recommended_auction_action(
        "Pass with 9 HCP.", cands
    ) == PASS_ACTION
    # Ambiguous disjunction -> returns None so Layer 2 decides freely
    assert extract_recommended_auction_action(
        "Bid 2D or 2H depending on agreement.", cands
    ) is None
