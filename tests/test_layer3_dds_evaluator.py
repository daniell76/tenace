import pyspiel
import pytest

import tenace.layer3_rules as layer3_rules
from tenace.domain.types import (
    FIRST_CONTRACT_BID_ACTION,
    PASS_ACTION,
    CandidateOption,
    Seat,
    SeatConstraint,
    StrategyState,
    Suit,
    TacticalDecision,
    card_id,
    hand_hcp,
)
from tenace.layer3_rules.dds_evaluator import RuleDDSEvaluator
from tenace.layer3_rules.tracker import collapse_equivalent_actions


def _make_dealt_state(
    board_number: int = 1, use_dd: bool = False
) -> pyspiel.State:
    flag = "true" if use_dd else "false"
    game = pyspiel.load_game(f"bridge(use_double_dummy_result={flag})")
    return game.new_initial_state(tournament_seed=42, board_number=board_number)


def _make_custom_dealt_state(hands: dict[int, list[int]]) -> pyspiel.State:
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state()
    for r in range(13):
        for p in range(4):
            state.apply_action(hands[p][r])
    return state


def _dealt_hand(world: pyspiel.State, seat: Seat) -> list[int]:
    hist = world.full_history()[:52]
    return [hist[4 * r + int(seat)].action for r in range(13)]


def test_init_validation_and_public_reexport():
    assert layer3_rules.RuleDDSEvaluator is RuleDDSEvaluator
    assert "RuleDDSEvaluator" in layer3_rules.__all__

    with pytest.raises(ValueError, match="num_worlds"):
        RuleDDSEvaluator(num_worlds=0)

    with pytest.raises(ValueError, match="max_rejection_factor"):
        RuleDDSEvaluator(max_rejection_factor=0)

    with pytest.raises(ValueError, match="max_make_prob_drop"):
        RuleDDSEvaluator(max_make_prob_drop=-0.01)

    with pytest.raises(ValueError, match="max_make_prob_drop"):
        RuleDDSEvaluator(max_make_prob_drop=1.01)

    with pytest.raises(ValueError, match="num_worlds"):
        RuleDDSEvaluator(num_worlds=True)  # type: ignore[arg-type]

    undealt = pyspiel.load_game("bridge(use_double_dummy_result=false)").new_initial_state()
    with pytest.raises(ValueError, match="incomplete deal phase"):
        RuleDDSEvaluator(num_worlds=2).sample_consistent_worlds(undealt, Seat.NORTH)


def test_constrained_sampler_prefers_worlds_matching_hcp_bounds():
    state = _make_dealt_state(board_number=1)
    observer = Seat.NORTH
    own_cards = [
        c
        for c, flag in enumerate(state.observation_tensor(int(observer))[432 : 432 + 52])
        if flag > 0.5
    ]

    strategy = StrategyState(
        phase="auction",
        plan_summary="East opened 1NT showing 15-17 HCP and balanced shape",
        seat_constraints={
            Seat.EAST: SeatConstraint(
                seat=Seat.EAST,
                min_hcp=15,
                max_hcp=17,
                min_suit_lengths={Suit.SPADES: 2, Suit.HEARTS: 2},
                max_suit_lengths={Suit.SPADES: 5, Suit.CLUBS: 5},
            )
        },
    )

    evaluator = RuleDDSEvaluator(num_worlds=8, max_rejection_factor=20, rng_seed=123)
    worlds = evaluator.sample_consistent_worlds(state, observer, strategy)

    assert len(worlds) == 8
    for w in worlds:
        assert sorted(_dealt_hand(w, Seat.NORTH)) == sorted(own_cards)
        east_cards = _dealt_hand(w, Seat.EAST)
        assert 15 <= hand_hcp(east_cards) <= 17
        spades = [c for c in east_cards if c % 4 == int(Suit.SPADES)]
        hearts = [c for c in east_cards if c % 4 == int(Suit.HEARTS)]
        clubs = [c for c in east_cards if c % 4 == int(Suit.CLUBS)]
        assert 2 <= len(spades) <= 5
        assert len(hearts) >= 2
        assert len(clubs) <= 5


def test_sampler_never_reads_original_hidden_deal_and_falls_back_on_impossible_constraints():
    state = _make_dealt_state(board_number=2)

    class _GuardedStateProxy:
        def __init__(self, inner: pyspiel.State):
            self._inner = inner

        def full_history(self):
            raise AssertionError(
                "Fair-play violation: sampler accessed original state.full_history()"
            )

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    # Impossible constraint: East AND West both claimed to have 30+ HCP
    impossible_strategy = StrategyState(
        phase="auction",
        plan_summary="Impossible opponent HCP claims",
        seat_constraints={
            Seat.EAST: SeatConstraint(seat=Seat.EAST, min_hcp=30, max_hcp=37),
            Seat.WEST: SeatConstraint(seat=Seat.WEST, min_hcp=30, max_hcp=37),
        },
    )

    evaluator = RuleDDSEvaluator(num_worlds=6, max_rejection_factor=3, rng_seed=99)
    worlds = evaluator.sample_consistent_worlds(
        _GuardedStateProxy(state),  # type: ignore[arg-type]
        Seat.NORTH,
        impossible_strategy,
    )
    assert len(worlds) == 6

    # Also ensure contradictory constraint on visible dummy does not block other valid constraints
    play_state = _make_dealt_state(board_number=1)
    play_state.apply_action(59)  # 1NT by North
    for _ in range(3):
        play_state.apply_action(PASS_ACTION)
    play_state.apply_action(play_state.legal_actions()[0])  # East leads -> South is dummy

    dummy_hallucination = StrategyState(
        phase="play",
        plan_summary="Stale constraint on dummy South",
        seat_constraints={
            Seat.SOUTH: SeatConstraint(seat=Seat.SOUTH, min_hcp=37, max_hcp=37),
            Seat.WEST: SeatConstraint(seat=Seat.WEST, min_hcp=8, max_hcp=14),
        },
    )
    play_worlds = evaluator.sample_consistent_worlds(
        play_state, Seat.EAST, dummy_hallucination
    )
    assert len(play_worlds) == 6
    for w in play_worlds:
        assert 8 <= hand_hcp(_dealt_hand(w, Seat.WEST)) <= 14


def test_evaluate_play_candidates_computes_baseline_ev_and_make_prob():
    # At Trick 12 in 7NT by North:
    # North holds C:2 and D:2; Dummy South holds C:A and H:A; unseen EW cards are {D:T, D:J, D:Q, D:K, D:A}.
    # - Playing C:2 reaches Dummy's C:A (Trick 12) and cashes H:A (Trick 13) -> 7NT makes in every world (make_prob=1.0).
    # - Playing D:2 loses Trick 12 to EW's high diamonds -> 7NT fails in every world (make_prob=0.0).
    n_cards = [
        *[card_id(Suit.SPADES, r) for r in range(2, 13)],  # S:4..S:A (Tricks 1..11)
        card_id(Suit.CLUBS, 0),     # C:2 (entry to Dummy's C:A + H:A)
        card_id(Suit.DIAMONDS, 0),  # D:2 (loser to EW's diamonds)
    ]
    s_cards = [
        card_id(Suit.SPADES, 1),    # S:3 (Trick 1)
        *[card_id(Suit.CLUBS, r) for r in range(1, 11)],   # C:3..C:Q (Tricks 2..11 discards)
        card_id(Suit.CLUBS, 12),    # C:A (Trick 12)
        card_id(Suit.HEARTS, 12),   # H:A (Trick 13)
    ]
    e_cards = [
        card_id(Suit.SPADES, 0),    # S:2 (opening lead Trick 1)
        *[card_id(Suit.HEARTS, r) for r in range(0, 10)],  # H:2..H:J (Tricks 2..11 discards)
        card_id(Suit.DIAMONDS, 11), # D:K (kept for Trick 12)
        card_id(Suit.DIAMONDS, 10), # D:Q (kept for Trick 13)
    ]
    w_cards = [
        card_id(Suit.HEARTS, 10),   # H:Q (Trick 1 discard)
        card_id(Suit.HEARTS, 11),   # H:K (Trick 2 discard)
        card_id(Suit.CLUBS, 11),    # C:K (Trick 3 discard)
        *[card_id(Suit.DIAMONDS, r) for r in range(1, 9)],  # D:3..D:T (Tricks 4..11 discards)
        card_id(Suit.DIAMONDS, 9),  # D:J (kept for Trick 12)
        card_id(Suit.DIAMONDS, 12), # D:A (kept for Trick 13)
    ]

    state = _make_custom_dealt_state({0: n_cards, 1: e_cards, 2: s_cards, 3: w_cards})
    state.apply_action(89)  # 7NT by North
    for _ in range(3):
        state.apply_action(PASS_ACTION)

    played: set[int] = set()

    def _play(card: int) -> None:
        state.apply_action(card)
        played.add(card)

    # Trick 1: East leads S:2, South S:3, West H:Q, North wins S:4
    _play(card_id(Suit.SPADES, 0))
    _play(card_id(Suit.SPADES, 1))
    _play(card_id(Suit.HEARTS, 10))
    _play(card_id(Suit.SPADES, 2))

    # Tricks 2..11 (10 tricks): North cashes S:5..S:A
    w_discards = [
        card_id(Suit.HEARTS, 11),
        card_id(Suit.CLUBS, 11),
        *[card_id(Suit.DIAMONDS, r) for r in range(1, 9)],
    ]
    for idx in range(10):
        _play(card_id(Suit.SPADES, 3 + idx))     # North leads spade winner
        _play(card_id(Suit.HEARTS, idx))         # East discards H:2..H:J
        _play(card_id(Suit.CLUBS, 1 + idx))      # South discards C:3..C:Q
        _play(w_discards[idx])                   # West discards

    assert state.current_player() == int(Seat.NORTH)
    candidates = collapse_equivalent_actions(state.legal_actions(), played_cards=played)
    assert len(candidates) == 2

    evaluator = RuleDDSEvaluator(num_worlds=8, rng_seed=42)
    scored = evaluator.score_candidates(state, Seat.NORTH, candidates)

    assert len(scored) == 2
    by_label = {c.label: c for c in scored}
    assert by_label["C:2"].make_prob == 1.0
    assert by_label["C:2"].baseline_ev > 1000.0
    assert by_label["D:2"].make_prob == 0.0
    assert by_label["D:2"].baseline_ev < 0.0
    # Original input candidates must not be mutated in place
    assert all(c.make_prob is None and c.baseline_ev == 0.0 for c in candidates)


def test_evaluate_auction_candidates_uses_cpp_dds_and_handles_passout():
    # Give North 13 top spades/aces so 7S makes cold, while passing out scores 0 (make_prob=0.5).
    n_cards = [card_id(Suit.SPADES, r) for r in range(13)]
    e_cards = [card_id(Suit.CLUBS, r) for r in range(13)]
    s_cards = [card_id(Suit.DIAMONDS, r) for r in range(13)]
    w_cards = [card_id(Suit.HEARTS, r) for r in range(13)]

    state = _make_custom_dealt_state({0: n_cards, 1: e_cards, 2: s_cards, 3: w_cards})
    # Suppose N, E, S passed, now West is deciding between Pass (passout -> 0) and 1H (57) vs 7C (85, down 13)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)

    bid_7h = FIRST_CONTRACT_BID_ACTION + 6 * 5 + 2  # 7H = 87 (West has all 13 hearts -> cold!)
    bid_7c = FIRST_CONTRACT_BID_ACTION + 6 * 5 + 0  # 7C = 85
    cands = collapse_equivalent_actions([PASS_ACTION, bid_7c, bid_7h])

    evaluator = RuleDDSEvaluator(num_worlds=4, rng_seed=7)
    scored = evaluator.score_candidates(state, Seat.WEST, cands)
    by_label = {c.label: c for c in scored}

    assert by_label["Pass"].baseline_ev == 0.0
    assert by_label["Pass"].make_prob == 0.5
    assert by_label["7H"].baseline_ev > 1000.0
    assert by_label["7H"].make_prob == 1.0

    # Single candidate fast-path returns unchanged list without sampling
    single = [CandidateOption(action=PASS_ACTION, label="Pass", equivalent_actions=[PASS_ACTION])]
    assert evaluator.score_candidates(state, Seat.WEST, single) == single
    assert evaluator.score_candidates(state, Seat.WEST, []) == []


def test_blunder_gate_blocks_low_make_prob_hallucination():
    evaluator = RuleDDSEvaluator(max_make_prob_drop=0.20, rng_seed=1)

    safe_line = CandidateOption(
        action=card_id(Suit.DIAMONDS, 12),
        label="D:A",
        equivalent_actions=[card_id(Suit.DIAMONDS, 12)],
        baseline_ev=620.0,
        make_prob=0.90,
    )
    close_line = CandidateOption(
        action=card_id(Suit.SPADES, 12),
        label="S:A",
        equivalent_actions=[card_id(Suit.SPADES, 12)],
        baseline_ev=600.0,
        make_prob=0.80,
    )
    blunder_line = CandidateOption(
        action=card_id(Suit.CLUBS, 0),
        label="C:2",
        equivalent_actions=[card_id(Suit.CLUBS, 0)],
        baseline_ev=-100.0,
        make_prob=0.10,
    )

    # Even if Layer 2 LLM strongly hallucinates C:2 (score 1.0) and prefers S:A (0.95) over D:A (0.10),
    # C:2 is gated out (0.10 < 0.90 - 0.20 = 0.70), while S:A survives (0.80 >= 0.70) and wins blend!
    hallucinated_decision = TacticalDecision(
        option_scores={"C:2": 1.0, "S:A": 0.95, "D:A": 0.10},
        rationale="Play C:2 for a magical pseudo-squeeze",
    )
    chosen = evaluator.select_guarded_action(
        [safe_line, close_line, blunder_line], hallucinated_decision
    )
    assert chosen == close_line.action

    # If only safe_line and blunder_line exist, safe_line wins despite 0.0 LLM score
    only_blunder_preferred = TacticalDecision(
        option_scores={"C:2": 1.0, "D:A": 0.0},
        rationale="Blunder!",
    )
    assert (
        evaluator.select_guarded_action([safe_line, blunder_line], only_blunder_preferred)
        == safe_line.action
    )

    # Without tactical decision (or empty option_scores), picks highest baseline_ev among viable
    assert evaluator.select_guarded_action([close_line, safe_line, blunder_line], None) == safe_line.action
    assert (
        evaluator.select_guarded_action(
            [close_line, safe_line, blunder_line], TacticalDecision(option_scores={})
        )
        == safe_line.action
    )


def test_select_guarded_action_edge_cases():
    evaluator = RuleDDSEvaluator(max_make_prob_drop=0.20)

    with pytest.raises(ValueError, match="No legal candidates available"):
        evaluator.select_guarded_action([])

    only = CandidateOption(action=52, label="Pass", equivalent_actions=[52])
    assert evaluator.select_guarded_action([only]) == 52

    # Candidates with make_prob=None bypass prob filtering and blend properly
    c1 = CandidateOption(action=55, label="1C", baseline_ev=100.0, make_prob=None)
    c2 = CandidateOption(action=56, label="1D", baseline_ev=100.0, make_prob=None)
    decision = TacticalDecision(option_scores={"1C": 0.2, "1D": 0.9})
    assert evaluator.select_guarded_action([c1, c2], decision) == 56
