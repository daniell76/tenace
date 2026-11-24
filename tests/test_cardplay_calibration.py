from __future__ import annotations

import pyspiel
import pytest

from tenace.agent import extract_recommended_play_action
from tenace.domain.types import (
    CandidateOption,
    Seat,
    SeatConstraint,
    StrategyState,
    Suit,
    TacticalDecision,
    action_label,
    card_id,
    card_suit,
    hand_hcp,
)
from tenace.harness.duplicate import DuplicateMatchRunner
from tenace.layer1_strategist.conventions import SAYC_CONVENTION_CARD
from tenace.layer1_strategist.strategist import build_strategist_system_prompt
from tenace.layer3_rules.dds_evaluator import (
    RuleDDSEvaluator,
    infer_symbolic_auction_constraints,
)

LABEL_TO_ID = {action_label(a).upper(): a for a in range(90)}


def _state_after_auction(seed: int, board: int, calls: list[str]) -> pyspiel.State:
    st = DuplicateMatchRunner(tournament_seed=seed, use_double_dummy_result=False)._new_dealt_board_state(board)
    for tok in calls:
        st.apply_action(LABEL_TO_ID[tok.upper()])
    return st


def test_infer_symbolic_auction_constraints_natural_and_system_aware() -> None:
    # Board 1: Dealer NORTH. Auction: 1S - P - P - X - 2S - 3C - P - 3NT - P - P - P
    st = _state_after_auction(
        3000,
        1,
        ["1S", "Pass", "Pass", "Dbl", "2S", "3C", "Pass", "3NT", "Pass", "Pass", "Pass"],
    )
    cons = infer_symbolic_auction_constraints(st, Seat.NORTH, system_id="ccba")
    assert Seat.NORTH not in cons  # observer excluded
    # SOUTH passed partner's 1S opening -> strictly weak (< 7 HCP)
    assert cons[Seat.SOUTH].max_hcp <= 7
    # WEST made a takeout/strong double then jumped to 3NT -> strong!
    assert cons[Seat.WEST].min_hcp >= 11
    # EAST bid 3C freely over 2S -> club length
    assert cons[Seat.EAST].min_suit_lengths.get(Suit.CLUBS, 0) >= 4

    # 3H preempt: Dealer NORTH passes, EAST opens 3H - P - 4NT - P - 5C - P - 5H - P - P - P
    st_pre = _state_after_auction(
        3000,
        1,
        ["Pass", "3H", "Pass", "4NT", "Pass", "5C", "Pass", "5H", "Pass", "Pass", "Pass"],
    )
    cons_pre = infer_symbolic_auction_constraints(st_pre, Seat.SOUTH, system_id="ccba")
    assert cons_pre[Seat.EAST].min_suit_lengths.get(Suit.HEARTS, 0) >= 6
    assert cons_pre[Seat.EAST].max_hcp <= 12
    assert cons_pre[Seat.NORTH].max_hcp <= 12

    # Precision 1C is strong artificial (16+ HCP, NOT clubs!)
    st_prec = _state_after_auction(3000, 1, ["1C", "Pass", "1D", "Pass", "1NT", "Pass", "Pass", "Pass"])
    cons_prec = infer_symbolic_auction_constraints(st_prec, Seat.EAST, system_id="precision")
    assert cons_prec[Seat.NORTH].min_hcp >= 16
    assert Suit.CLUBS not in cons_prec[Seat.NORTH].min_suit_lengths

    # Acol 1H allows 4-card majors
    st_acol = _state_after_auction(3000, 1, ["1H", "Pass", "Pass", "Pass"])
    cons_acol = infer_symbolic_auction_constraints(st_acol, Seat.EAST, system_id="acol")
    assert cons_acol[Seat.NORTH].min_suit_lengths.get(Suit.HEARTS, 0) == 4


def test_soft_penalty_sampling_keeps_worlds_close_to_auction_constraints() -> None:
    # Board 5 (seed=3000): Dealer NORTH holds 12 HCP (28 unseen HCP matching the 1S-P-P-X-2S-3C-P-3NT auction!)
    st = _state_after_auction(
        3000,
        5,
        ["1S", "Pass", "Pass", "Dbl", "2S", "3C", "Pass", "3NT", "Pass", "Pass", "Pass"],
    )
    # Ask for tough joint constraint where uniform 128 draws would rarely hit exact 0
    strat = StrategyState(
        phase="play",
        plan_summary="Lead H:A",
        seat_constraints={
            Seat.SOUTH: SeatConstraint(seat=Seat.SOUTH, min_hcp=0, max_hcp=5),
            Seat.WEST: SeatConstraint(seat=Seat.WEST, min_hcp=17, max_hcp=25),
            Seat.EAST: SeatConstraint(
                seat=Seat.EAST, min_hcp=5, max_hcp=12, min_suit_lengths={Suit.CLUBS: 5}
            ),
        },
    )
    ev = RuleDDSEvaluator(num_worlds=12, max_rejection_factor=40, rng_seed=42)
    worlds = ev.sample_consistent_worlds(st, Seat.NORTH, strategy=strat)
    assert len(worlds) == 12
    avg_west_hcp = sum(
        hand_hcp([w.full_history()[4 * r + int(Seat.WEST)].action for r in range(13)])
        for w in worlds
    ) / len(worlds)
    avg_south_hcp = sum(
        hand_hcp([w.full_history()[4 * r + int(Seat.SOUTH)].action for r in range(13)])
        for w in worlds
    ) / len(worlds)
    assert avg_west_hcp >= 16.5
    assert avg_south_hcp <= 5.5


def test_extract_recommended_play_action_parses_l1_card_directives() -> None:
    ha = card_id(Suit.HEARTS, 12)
    hk = card_id(Suit.HEARTS, 11)
    dk = card_id(Suit.DIAMONDS, 11)
    ct = card_id(Suit.CLUBS, 8)
    cands = [
        CandidateOption(action=ha, label="H:A"),
        CandidateOption(action=hk, label="H:K"),
        CandidateOption(action=dk, label="D:K"),
        CandidateOption(action=ct, label="C:T"),
    ]
    assert extract_recommended_play_action("Lead D:K from KJT7 against East's 5H.", cands) == dk
    assert extract_recommended_play_action("Lead H:A (or H:K) at Trick 1.", cands) == ha
    assert extract_recommended_play_action("Lead the King of Diamonds (D:K) now.", cands) == dk
    assert extract_recommended_play_action("Play C:T immediately to set up a ruff.", cands) == ct
    # Card not in candidate list -> None (never hallucinate illegal card)
    assert extract_recommended_play_action("Lead S:A to cash a trick.", cands) is None


def test_fourth_seat_conservation_prevents_needlessly_overtaking_partner_winner() -> None:
    ev = RuleDDSEvaluator(max_make_prob_drop=0.20, max_ev_drop=45.0)
    hk = CandidateOption(action=card_id(Suit.HEARTS, 11), label="H:K", baseline_ev=120.0, make_prob=0.40)
    ht = CandidateOption(action=card_id(Suit.HEARTS, 8), label="H:T", baseline_ev=120.0, make_prob=0.40)

    # Even if L2 prefers H:K (0.9) over H:T (0.3), H:Q is already winning from dummy!
    # Do NOT overtake partner/dummy's H:Q with H:K on an EV tie!
    decision = TacticalDecision(option_scores={"H:K": 0.90, "H:T": 0.30})
    chosen = ev.select_guarded_action(
        [hk, ht],
        decision,
        phase="play",
        partner_winning_trick_card=card_id(Suit.HEARTS, 10),  # dummy H:Q already winning!
    )
    assert chosen == ht.action

    # If overtaking strictly gains > 15 pts DDS EV (e.g. unblocking / entry cross), ALLOW it!
    hk_gain = hk.model_copy(update={"baseline_ev": 200.0})
    chosen_unblock = ev.select_guarded_action(
        [hk_gain, ht],
        decision,
        phase="play",
        partner_winning_trick_card=card_id(Suit.HEARTS, 10),
    )
    assert chosen_unblock == hk_gain.action


def test_opening_lead_honors_l1_directive_on_dds_tie_but_blocks_blunders() -> None:
    ev = RuleDDSEvaluator(max_make_prob_drop=0.20, max_ev_drop=45.0)
    ct = CandidateOption(action=card_id(Suit.CLUBS, 8), label="C:T", baseline_ev=-338.0, make_prob=0.25)
    dk = CandidateOption(action=card_id(Suit.DIAMONDS, 11), label="D:K", baseline_ev=-341.0, make_prob=0.25)
    bad = CandidateOption(action=card_id(Suit.SPADES, 0), label="S:2", baseline_ev=-520.0, make_prob=0.05)

    # Tiny 3-pt DDS noise gap (-338 vs -341): honor L1's explicit opening lead D:K!
    decision = TacticalDecision(option_scores={"C:T": 0.72, "D:K": 0.65, "S:2": 0.10})
    assert (
        ev.select_guarded_action(
            [ct, dk, bad],
            decision,
            phase="play",
            preferred_action=dk.action,
            play_preferred_ev_margin=25.0,
        )
        == dk.action
    )

    # Huge -182-pt blunder S:2 recommended by L1 -> MUST be rejected!
    assert (
        ev.select_guarded_action(
            [ct, dk, bad],
            decision,
            phase="play",
            preferred_action=bad.action,
            play_preferred_ev_margin=25.0,
        )
        == ct.action
    )


def test_play_strategist_prompt_includes_opening_lead_and_conservation_guidance() -> None:
    prompt_play = build_strategist_system_prompt(SAYC_CONVENTION_CARD, phase="play")
    low = prompt_play.lower()
    assert "lead" in low
    assert "overtake" in low or "already winning" in low
    assert "unsupported bare ace" in low


def test_unique_viable_play_action_fast_path() -> None:
    ev = RuleDDSEvaluator(max_make_prob_drop=0.20, max_ev_drop=45.0)
    win = CandidateOption(action=card_id(Suit.HEARTS, 12), label="H:A", baseline_ev=100.0, make_prob=0.80)
    lose = CandidateOption(action=card_id(Suit.CLUBS, 0), label="C:2", baseline_ev=-400.0, make_prob=0.10)
    close = CandidateOption(action=card_id(Suit.HEARTS, 11), label="H:K", baseline_ev=85.0, make_prob=0.78)
    assert ev.unique_viable_play_action([win, lose]) == win.action
    assert ev.unique_viable_play_action([win, close, lose]) is None


def test_passed_hand_response_infers_suit_and_penalty_double_does_not_mislocate_trumps() -> None:
    # Seed 3200 board 6 (dealer EAST): P - P - 1C - P - 1S - P - P - X - P - 2H - X - P - P - P
    st = _state_after_auction(
        3200,
        6,
        ["Pass", "Pass", "1C", "Pass", "1S", "Pass", "Pass", "Dbl", "Pass", "2H", "Dbl", "Pass", "Pass", "Pass"],
    )
    cons = infer_symbolic_auction_constraints(st, Seat.SOUTH, system_id="ccba")
    # EAST passed 1st seat, then responded 1S over partner's 1C -> MUST show 4+ SPADES!
    assert cons[Seat.EAST].min_suit_lengths.get(Suit.SPADES, 0) >= 4
    # WEST doubled 2H directly for PENALTY (not reopening takeout converted by East)!
    # EAST must NOT be falsely assigned 4+ HEARTS.
    assert Suit.HEARTS not in cons[Seat.EAST].min_suit_lengths


def test_overtight_l1_hcp_does_not_collapse_dds_worlds_to_wrong_discard() -> None:
    # Reproduce 3200:6 Trick 9 where L1 overestimates WEST HCP (19-22) while WEST actually holds 14.
    # With EAST=4+ spades & softened non-degenerate sampling, pitching winning D:K MUST score
    # strictly worse than pitching losing club C:5 (never degenerate tie!).
    st = _state_after_auction(
        3200,
        6,
        ["Pass", "Pass", "1C", "Pass", "1S", "Pass", "Pass", "Dbl", "Pass", "2H", "Dbl", "Pass", "Pass", "Pass"],
    )
    for card in [
        "H:5", "H:2", "H:3", "H:T",
        "D:5", "D:4", "D:A", "D:2",
        "H:Q", "H:4", "H:9", "H:A",
        "H:6", "H:J", "C:2", "H:K",
        "H:8", "H:7", "D:7", "S:8",
        "C:K", "C:A", "C:3", "C:4",
        "S:A", "S:9", "S:2", "S:4",
        "S:Q", "S:5", "S:6", "S:J",
        "S:3",
    ]:
        st.apply_action(LABEL_TO_ID[card])

    from tenace.layer3_rules.tracker import collapse_equivalent_actions

    played = {e.action for e in st.full_history()[52:] if e.action < 52}
    cands = collapse_equivalent_actions(st.legal_actions(), played_cards=played)
    overtight_l1 = StrategyState(
        phase="play",
        plan_summary="West doubled 2H with strong hand",
        seat_constraints={
            Seat.WEST: SeatConstraint(seat=Seat.WEST, min_hcp=19, max_hcp=22),
        },
    )
    ev = RuleDDSEvaluator(num_worlds=16, rng_seed=0)
    scored = ev.score_candidates(st, Seat.SOUTH, cands, strategy=overtight_l1, system_id="ccba")
    by_lbl = {c.label: c.baseline_ev for c in scored}
    assert by_lbl["C:5"] - by_lbl["D:K"] > 100.0


def test_play_strategist_prompt_interior_sequence_and_nt_ak_lead() -> None:
    prompt_play = build_strategist_system_prompt(SAYC_CONVENTION_CARD, phase="play")
    low = prompt_play.lower()
    assert "interior sequence" in low
    assert "kjt" in low


def test_infer_symbolic_constraints_4nt_rkcb_and_penalty_pass_double() -> None:
    # 3H - P - 4NT - P - 5C - P - 5H -> West asked keycards facing a preempt => strong (16+ HCP)!
    st_rkcb = _state_after_auction(
        3400,
        2,
        ["3H", "Pass", "4NT", "Pass", "5C", "Pass", "5H", "Pass", "Pass", "Pass"],
    )
    cons_rkcb = infer_symbolic_auction_constraints(st_rkcb, Seat.SOUTH, system_id="ccba")
    assert cons_rkcb[Seat.EAST].min_suit_lengths.get(Suit.HEARTS, 0) >= 6
    assert cons_rkcb[Seat.WEST].min_hcp >= 16
    assert Suit.CLUBS not in cons_rkcb[Seat.EAST].min_suit_lengths  # 5C is artificial RKCB answer

    # Penalty pass converting reopening takeout double:
    # NORTH:P - EAST:1H - SOUTH:2D - WEST:P - NORTH:P - EAST:Dbl - SOUTH:P - WEST:P - NORTH:P
    st_pen = _state_after_auction(
        3000,
        1,
        ["Pass", "1H", "2D", "Pass", "Pass", "Dbl", "Pass", "Pass", "Pass"],
    )
    cons_pen = infer_symbolic_auction_constraints(st_pen, Seat.SOUTH, system_id="ccba")
    # West passed East's reopening double of 2D for penalty => trump length in DIAMONDS (>= 4)
    assert cons_pen[Seat.WEST].min_suit_lengths.get(Suit.DIAMONDS, 0) >= 4

    # Unraised competitive major rebid (1S - P - P - X - 2S) implies 6+ spades
    st_reb = _state_after_auction(
        3000,
        5,
        ["1S", "Pass", "Pass", "Dbl", "2S", "3C", "Pass", "3NT", "Pass", "Pass", "Pass"],
    )
    cons_reb = infer_symbolic_auction_constraints(st_reb, Seat.WEST, system_id="ccba")
    assert cons_reb[Seat.NORTH].min_suit_lengths.get(Suit.SPADES, 0) >= 6

    # 2/1 game force 1S - P - 2H - P - 2NT - P - 3S -> East shows 5+ hearts and 3+ spades
    st_gf = _state_after_auction(
        3500,
        11,
        ["Pass", "1S", "Pass", "2H", "Pass", "2NT", "Pass", "3S", "Pass", "4S", "Pass", "Pass", "Pass"],
    )
    cons_gf = infer_symbolic_auction_constraints(st_gf, Seat.NORTH, system_id="ccba")
    assert cons_gf[Seat.EAST].min_suit_lengths.get(Suit.HEARTS, 0) >= 5
    assert cons_gf[Seat.EAST].min_suit_lengths.get(Suit.SPADES, 0) >= 3
    assert cons_gf[Seat.EAST].min_hcp >= 10


def test_opening_lead_mp_margin_guards_partscores_while_honoring_close_game_leads() -> None:
    ev = RuleDDSEvaluator(max_make_prob_drop=0.20, max_ev_drop=45.0)
    # Tight game-level Monte-Carlo variance (2/32 worlds = -37 pts, -0.08 mp): honor H:A!
    s3 = CandidateOption(action=card_id(Suit.SPADES, 1), label="S:3", baseline_ev=-182.0, make_prob=0.48)
    ha = CandidateOption(action=card_id(Suit.HEARTS, 12), label="H:A", baseline_ev=-219.0, make_prob=0.40)
    dec = TacticalDecision(option_scores={"S:3": 0.70, "H:A": 0.65})
    assert (
        ev.select_guarded_action(
            [s3, ha],
            dec,
            phase="play",
            preferred_action=ha.action,
            play_preferred_ev_margin=45.0,
            play_preferred_mp_margin=0.10,
        )
        == ha.action
    )

    # Partscore where preferred card is only -25 pts worse in EV, YET drops make_prob by -0.18 (> 0.10): reject!
    c4 = CandidateOption(action=card_id(Suit.CLUBS, 2), label="C:4", baseline_ev=20.0, make_prob=0.72)
    bad_ps = CandidateOption(action=card_id(Suit.DIAMONDS, 11), label="D:K", baseline_ev=-5.0, make_prob=0.54)
    dec_ps = TacticalDecision(option_scores={"C:4": 0.65, "D:K": 0.60})
    assert (
        ev.select_guarded_action(
            [c4, bad_ps],
            dec_ps,
            phase="play",
            preferred_action=bad_ps.action,
            play_preferred_ev_margin=45.0,
            play_preferred_mp_margin=0.10,
        )
        == c4.action
    )
