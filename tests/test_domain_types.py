import pyspiel

from tenace.domain.types import (
    BIDDING_ACTION_BASE,
    Seat,
    Suit,
    action_label,
    card_id,
    card_rank,
    card_suit,
    hand_hcp,
)


def test_seat_partner_and_partnership():
    assert Seat.NORTH.partner == Seat.SOUTH
    assert Seat.EAST.partner == Seat.WEST
    assert Seat.NORTH.partnership == Seat.SOUTH.partnership == 0
    assert Seat.EAST.partnership == Seat.WEST.partnership == 1


def test_card_encoding_matches_openspiel_bridge():
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state()

    ace_spades = card_id(Suit.SPADES, 12)
    two_clubs = card_id(Suit.CLUBS, 0)

    assert ace_spades == 51
    assert two_clubs == 0
    assert card_suit(ace_spades) == Suit.SPADES
    assert card_rank(ace_spades) == 12
    assert action_label(ace_spades) == "S:A"
    assert action_label(two_clubs) == "C:2"
    assert state.action_to_string(0, ace_spades) == "♠A"


def test_bidding_action_labels_and_hcp():
    assert action_label(BIDDING_ACTION_BASE + 0) == "Pass"
    assert action_label(BIDDING_ACTION_BASE + 1) == "Dbl"
    assert action_label(BIDDING_ACTION_BASE + 2) == "RDbl"
    assert action_label(BIDDING_ACTION_BASE + 3) == "1C"
    assert action_label(BIDDING_ACTION_BASE + 7) == "1NT"
    assert action_label(89) == "7NT"

    cards = [
        card_id(Suit.SPADES, 12),   # Ace = 4
        card_id(Suit.HEARTS, 11),   # King = 3
        card_id(Suit.DIAMONDS, 10), # Queen = 2
        card_id(Suit.CLUBS, 9),     # Jack = 1
        card_id(Suit.CLUBS, 0),     # Two = 0
    ]
    assert hand_hcp(cards) == 10
