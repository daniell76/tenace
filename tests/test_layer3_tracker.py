import pyspiel
import pytest
from pydantic import ValidationError

import tenace.layer3_rules as layer3_rules
from tenace.domain.types import (
    DBL_ACTION,
    FIRST_CONTRACT_BID_ACTION,
    PASS_ACTION,
    Seat,
    Suit,
    card_id,
)
from tenace.layer3_rules.tracker import (
    BoardFactSnapshot,
    RuleFactTracker,
    collapse_equivalent_actions,
)


def _make_dealt_state() -> pyspiel.State:
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    return game.new_initial_state(tournament_seed=42, board_number=1)


def _make_custom_dealt_state(hands: dict[int, list[int]]) -> pyspiel.State:
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state()
    for r in range(13):
        for p in range(4):
            state.apply_action(hands[p][r])
    return state


def test_collapse_equivalent_actions_empty_and_bidding():
    assert collapse_equivalent_actions([]) == []

    bid_1c = FIRST_CONTRACT_BID_ACTION
    options = collapse_equivalent_actions([bid_1c, DBL_ACTION, PASS_ACTION])
    assert [opt.action for opt in options] == [PASS_ACTION, DBL_ACTION, bid_1c]
    assert [opt.label for opt in options] == ["Pass", "Dbl", "1C"]
    assert all(opt.equivalent_actions == [opt.action] for opt in options)


def test_collapse_equivalent_actions_merges_contiguous_and_bridged_cards():
    spade_q = card_id(Suit.SPADES, 10)
    spade_j = card_id(Suit.SPADES, 9)
    spade_t = card_id(Suit.SPADES, 8)
    spade_9 = card_id(Suit.SPADES, 7)
    spade_2 = card_id(Suit.SPADES, 0)

    options = collapse_equivalent_actions(
        legal_actions=[spade_2, spade_9, spade_q],
        played_cards={spade_j, spade_t},
    )
    assert len(options) == 2
    by_action = {opt.action: opt for opt in options}
    assert spade_q in by_action
    assert by_action[spade_q].equivalent_actions == [spade_q, spade_9]
    assert by_action[spade_q].label == "S:Q"
    assert by_action[spade_2].equivalent_actions == [spade_2]


def test_collapse_equivalent_actions_separates_suits_and_rejects_mixed_action_types():
    club_5 = card_id(Suit.CLUBS, 3)
    club_6 = card_id(Suit.CLUBS, 4)
    diamond_5 = card_id(Suit.DIAMONDS, 3)
    diamond_6 = card_id(Suit.DIAMONDS, 4)

    options = collapse_equivalent_actions([diamond_5, club_5, diamond_6, club_6])
    assert len(options) == 2
    by_action = {opt.action: opt for opt in options}
    assert by_action[club_6].equivalent_actions == [club_6, club_5]
    assert by_action[diamond_6].equivalent_actions == [diamond_6, diamond_5]

    with pytest.raises(ValueError, match="Cannot mix card and bidding actions"):
        collapse_equivalent_actions([club_5, PASS_ACTION])

    with pytest.raises(ValueError, match="out of valid OpenSpiel bridge range"):
        collapse_equivalent_actions([-1])

    with pytest.raises(ValueError, match="out of valid OpenSpiel bridge range"):
        collapse_equivalent_actions([90])

    with pytest.raises(ValueError, match="played_cards"):
        collapse_equivalent_actions([club_5], played_cards={52})

    with pytest.raises(ValueError, match="played_cards"):
        collapse_equivalent_actions([club_5], played_cards={-1})


def test_tracker_hides_unseen_hands_during_auction_and_reveals_dummy_after_lead():
    state = _make_dealt_state()
    tracker = RuleFactTracker()

    auction_facts = tracker.extract_facts(state, Seat.NORTH)
    assert isinstance(auction_facts, BoardFactSnapshot)
    assert auction_facts.phase == "auction"
    assert len(auction_facts.own_cards) == 13
    assert auction_facts.dummy_seat is None
    assert auction_facts.dummy_cards == []
    assert len(auction_facts.unseen_cards) == 39
    assert set(auction_facts.own_cards).isdisjoint(set(auction_facts.unseen_cards))

    state.apply_action(59)  # 1NT by North
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)

    pre_lead_east = tracker.extract_facts(state, Seat.EAST)
    assert pre_lead_east.phase == "play"
    assert pre_lead_east.declarer == Seat.NORTH
    assert pre_lead_east.dummy_seat is None
    assert pre_lead_east.trump_denom == 4
    assert pre_lead_east.contract_level == 1
    assert pre_lead_east.dummy_cards == []
    assert len(pre_lead_east.unseen_cards) == 39

    lead_card = state.legal_actions()[0]
    state.apply_action(lead_card)
    post_lead_east = tracker.extract_facts(state, Seat.EAST)
    assert post_lead_east.dummy_seat == Seat.SOUTH
    assert len(post_lead_east.dummy_cards) == 13
    assert lead_card in post_lead_east.played_cards
    assert len(post_lead_east.own_cards) == 12
    assert len(post_lead_east.unseen_cards) == 26
    assert (
        len(post_lead_east.own_cards)
        + len(post_lead_east.dummy_cards)
        + len(post_lead_east.played_cards)
        + len(post_lead_east.unseen_cards)
        == 52
    )

    # Observer == dummy_seat post-opening-lead: own_cards == dummy_cards (13 cards)
    # and unseen_cards has 38 cards (dummy does not see declarer's hand in OpenSpiel).
    post_lead_dummy = tracker.extract_facts(state, Seat.SOUTH)
    assert post_lead_dummy.dummy_seat == Seat.SOUTH
    assert len(post_lead_dummy.own_cards) == 13
    assert post_lead_dummy.own_cards == post_lead_dummy.dummy_cards
    assert len(post_lead_dummy.unseen_cards) == 38


def test_tracker_detects_suit_showout_void_and_emits_surprise():
    clubs = [card_id(Suit.CLUBS, r) for r in range(13)]
    diamonds = [card_id(Suit.DIAMONDS, r) for r in range(13)]
    hearts = [card_id(Suit.HEARTS, r) for r in range(13)]
    spades = [card_id(Suit.SPADES, r) for r in range(13)]

    state = _make_custom_dealt_state({0: clubs, 1: diamonds, 2: hearts, 3: spades})

    # N, E, S pass; W bids 1NT (59); N, E, S pass -> North leads C:A, East discards D:2
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(59)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)

    state.apply_action(card_id(Suit.CLUBS, 12))
    tracker = RuleFactTracker()
    before = tracker.extract_facts(state, Seat.NORTH)
    assert Suit.CLUBS not in before.known_voids[Seat.EAST]
    assert before.surprise_events == []

    state.apply_action(card_id(Suit.DIAMONDS, 0))
    after = tracker.extract_facts(state, Seat.NORTH, prev_voids=before.known_voids)
    assert Suit.CLUBS in after.known_voids[Seat.EAST]
    assert any("EAST showed out of CLUBS" in msg for msg in after.surprise_events)

    # Subsequent call with updated prev_voids should not re-emit the same surprise
    repeat = tracker.extract_facts(state, Seat.NORTH, prev_voids=after.known_voids)
    assert repeat.surprise_events == []


def test_tracker_trump_ruff_trick_winner_and_declarer_first_bidder_rule():
    # Give each player mixed suits so we can test trump ruff winning trick 1
    # and leading trick 2, plus declarer = first partner on winning side to bid denom.
    # Trick 1 setup:
    # North bids 1S (58), East Pass, South raises 2S (63), West Dbl (53), all Pass.
    # Declarer MUST be North (first NS player to bid Spades), so East leads Trick 1!
    # East leads C:A (48), South plays C:2 (0), West ruffs with S:2 (3), North plays C:3 (4).
    # West wins Trick 1 via trump ruff and leads Trick 2!
    # On Trick 2, West leads D:A (49), North discards H:2 (2) -> North shows out of DIAMONDS!
    n_cards = [
        card_id(Suit.CLUBS, 1),  # C:3
        card_id(Suit.HEARTS, 0),  # H:2 (no diamonds!)
        *[card_id(Suit.HEARTS, r) for r in range(1, 12)],
    ]  # 1 + 1 + 11 = 13 cards (C:3, H:2..H:K)
    e_cards = [
        card_id(Suit.CLUBS, 12),  # C:A
        *[card_id(Suit.CLUBS, r) for r in range(2, 12)],  # C:4..C:K (10 cards)
        card_id(Suit.DIAMONDS, 0),  # D:2
        card_id(Suit.DIAMONDS, 1),  # D:3
    ]  # 1 + 10 + 2 = 13 cards
    s_cards = [
        card_id(Suit.CLUBS, 0),  # C:2
        *[card_id(Suit.SPADES, r) for r in range(1, 13)],  # S:3..S:A (12 cards)
    ]  # 1 + 12 = 13 cards
    w_cards = [
        card_id(Suit.SPADES, 0),  # S:2 (no clubs -> ruffs trick 1!)
        card_id(Suit.DIAMONDS, 12),  # D:A
        *[card_id(Suit.DIAMONDS, r) for r in range(2, 12)],  # D:4..D:K (10 cards)
        card_id(Suit.HEARTS, 12),  # H:A
    ]  # 1 + 1 + 10 + 1 = 13 cards

    state = _make_custom_dealt_state({0: n_cards, 1: e_cards, 2: s_cards, 3: w_cards})
    tracker = RuleFactTracker()

    # North: 1S (58), East: Pass, South: 2S (63), West: Dbl (53), N/E/S: Pass
    state.apply_action(58)  # 1S
    state.apply_action(PASS_ACTION)
    state.apply_action(63)  # 2S
    state.apply_action(DBL_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)

    pre_lead = tracker.extract_facts(state, Seat.NORTH)
    assert pre_lead.declarer == Seat.NORTH
    assert pre_lead.dummy_seat is None
    assert pre_lead.trump_denom == int(Suit.SPADES)
    assert pre_lead.contract_level == 2

    # Trick 1: East leads C:A, South (via N) plays C:2, West ruffs S:2, North plays C:3
    state.apply_action(card_id(Suit.CLUBS, 12))  # East leads C:A
    state.apply_action(card_id(Suit.CLUBS, 0))   # South plays C:2
    state.apply_action(card_id(Suit.SPADES, 0))  # West ruffs S:2 (shows out of CLUBS)
    state.apply_action(card_id(Suit.CLUBS, 1))   # North plays C:3

    after_t1 = tracker.extract_facts(state, Seat.EAST)
    assert Suit.CLUBS in after_t1.known_voids[Seat.WEST]
    assert state.current_player() == int(Seat.WEST)  # West won Trick 1 with ruff

    # Trick 2: West leads D:A, North discards H:2 (shows out of DIAMONDS)
    state.apply_action(card_id(Suit.DIAMONDS, 12))
    state.apply_action(card_id(Suit.HEARTS, 0))

    after_t2_discard = tracker.extract_facts(
        state, Seat.EAST, prev_voids=after_t1.known_voids
    )
    assert Suit.DIAMONDS in after_t2_discard.known_voids[Seat.NORTH]
    assert after_t2_discard.surprise_events == ["NORTH showed out of DIAMONDS"]


def test_tracker_passed_out_and_completed_play_terminal_boards():
    tracker = RuleFactTracker()

    # 1. Passed-out board
    passed_state = _make_dealt_state()
    for _ in range(4):
        passed_state.apply_action(PASS_ACTION)

    passed_facts = tracker.extract_facts(passed_state, Seat.SOUTH)
    assert passed_facts.phase == "terminal"
    assert passed_facts.declarer is None
    assert passed_facts.dummy_seat is None
    assert passed_facts.trump_denom is None
    assert passed_facts.contract_level == 0
    assert len(passed_facts.own_cards) == 13
    assert passed_facts.dummy_cards == []
    assert len(passed_facts.unseen_cards) == 39

    # 2. Full 52-card played-out terminal board
    play_state = _make_dealt_state()
    play_state.apply_action(59)  # 1NT
    for _ in range(3):
        play_state.apply_action(PASS_ACTION)
    while not play_state.is_terminal():
        play_state.apply_action(play_state.legal_actions()[0])

    done_facts = tracker.extract_facts(play_state, Seat.NORTH)
    assert done_facts.phase == "terminal"
    assert done_facts.declarer == Seat.NORTH
    assert done_facts.own_cards == []
    assert done_facts.dummy_cards == []
    assert len(done_facts.played_cards) == 52
    assert done_facts.unseen_cards == []


def test_tracker_never_touches_dealt_card_prefix_and_exports_public_api():
    state = _make_dealt_state()

    class _GuardedHistory:
        def __init__(self, items):
            self._items = items

        def __iter__(self):
            raise AssertionError(
                "Fair-play violation: iterated raw full_history without [52:] slice"
            )

        def __getitem__(self, item):
            if isinstance(item, int) and item < 52:
                raise AssertionError("Fair-play violation: accessed dealt card history")
            if isinstance(item, slice):
                start = 0 if item.start is None else item.start
                if start < 52:
                    raise AssertionError(
                        f"Fair-play violation: slice start {start} < 52"
                    )
            return self._items[item]

    class _GuardedStateProxy:
        def __init__(self, inner: pyspiel.State):
            self._inner = inner

        def full_history(self):
            return _GuardedHistory(self._inner.full_history())

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    tracker = RuleFactTracker()
    facts = tracker.extract_facts(_GuardedStateProxy(state), Seat.WEST)  # type: ignore[arg-type]
    assert len(facts.own_cards) == 13

    assert layer3_rules.BoardFactSnapshot is BoardFactSnapshot
    assert layer3_rules.RuleFactTracker is RuleFactTracker
    assert layer3_rules.collapse_equivalent_actions is collapse_equivalent_actions
    assert set(layer3_rules.__all__) >= {
        "BoardFactSnapshot",
        "RuleFactTracker",
        "collapse_equivalent_actions",
    }


def test_tracker_rejects_incomplete_deal_phase_and_snapshot_validates_bounds():
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    undealt_state = game.new_initial_state()
    tracker = RuleFactTracker()

    with pytest.raises(ValueError, match="Cannot extract facts during incomplete deal phase"):
        tracker.extract_facts(undealt_state, Seat.NORTH)

    with pytest.raises(ValidationError):
        BoardFactSnapshot(
            observer=Seat.NORTH,
            phase="deal",  # type: ignore[arg-type]
            own_cards=[],
            own_hcp=0,
        )

    with pytest.raises(ValidationError):
        BoardFactSnapshot(
            observer=Seat.NORTH,
            phase="auction",
            own_cards=[52],
            own_hcp=0,
        )

    with pytest.raises(ValidationError):
        BoardFactSnapshot(
            observer=Seat.NORTH,
            phase="auction",
            own_cards=[],
            own_hcp=38,
        )

    with pytest.raises(ValidationError):
        BoardFactSnapshot(
            observer=Seat.NORTH,
            phase="play",
            own_cards=[],
            own_hcp=0,
            contract_level=8,
        )

    with pytest.raises(ValidationError):
        BoardFactSnapshot(
            observer=Seat.NORTH,
            phase="auction",
            own_cards=[],
            own_hcp=0,
            auction_calls=[(Seat.NORTH, 90)],
        )

    with pytest.raises(ValidationError):
        BoardFactSnapshot(
            observer=Seat.NORTH,
            phase="auction",
            own_cards=[],
            own_hcp=0,
            auction_calls=[(Seat.NORTH, -1)],
        )
