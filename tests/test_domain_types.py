import pyspiel
import pytest
from pydantic import ValidationError

import tenace.domain as domain
from tenace.domain.types import (
    BIDDING_ACTION_BASE,
    Assumption,
    CandidateOption,
    Seat,
    SeatConstraint,
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
    assert hand_hcp(c for c in cards) == 10


def test_card_and_action_error_paths():
    with pytest.raises(ValueError):
        card_id(Suit.CLUBS, -1)
    with pytest.raises(ValueError):
        card_id(Suit.CLUBS, 13)
    with pytest.raises(ValueError):
        card_id(99, 0)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        card_suit(-1)
    with pytest.raises(ValueError):
        card_suit(52)

    with pytest.raises(ValueError):
        card_rank(-1)
    with pytest.raises(ValueError):
        card_rank(52)

    with pytest.raises(ValueError):
        action_label(-1)
    with pytest.raises(ValueError):
        action_label(90)

    ace_spades = card_id(Suit.SPADES, 12)
    with pytest.raises(ValueError, match="Duplicate card in hand"):
        hand_hcp([ace_spades, ace_spades])


def test_pydantic_validation_constraints():
    with pytest.raises(ValidationError):
        Assumption(claim="x", confidence=1.5)

    with pytest.raises(ValidationError):
        SeatConstraint(seat=Seat.NORTH, min_hcp=15, max_hcp=10)

    with pytest.raises(ValidationError):
        SeatConstraint(seat=Seat.NORTH, min_suit_lengths={Suit.SPADES: 14})

    with pytest.raises(ValidationError):
        SeatConstraint(seat=Seat.NORTH, max_suit_lengths={Suit.SPADES: -1})

    with pytest.raises(ValidationError):
        SeatConstraint(
            seat=Seat.NORTH,
            min_suit_lengths={Suit.SPADES: 6},
            max_suit_lengths={Suit.SPADES: 4},
        )

    with pytest.raises(ValidationError):
        SeatConstraint(seat=Seat.NORTH, likely_cards=[52])

    with pytest.raises(ValidationError):
        SeatConstraint(seat=Seat.NORTH, likely_cards=[-1])

    with pytest.raises(ValidationError):
        CandidateOption(action=99, label="invalid")

    with pytest.raises(ValidationError):
        CandidateOption(action=-1, label="invalid")

    with pytest.raises(ValidationError):
        CandidateOption(action=52, label="Pass", equivalent_actions=[99])

    with pytest.raises(ValidationError):
        CandidateOption(action=52, label="Pass", make_prob=1.5)

    with pytest.raises(ValidationError):
        CandidateOption(action=52, label="Pass", make_prob=-0.1)


def test_domain_package_reexports():
    assert domain.Seat is Seat
    assert domain.Suit is Suit
    assert domain.StrategyState is not None
    assert "StrategyState" in domain.__all__


def test_seat_constraint_coerces_bridge_card_strings_in_likely_cards():
    from tenace.domain.types import StrategyState

    # Exact payload style observed from live gemini-4-argon on seed=100 & seed=200!
    sc_colon = SeatConstraint.model_validate(
        {
            "seat": "NORTH",
            "min_hcp": 10,
            "max_hcp": 14,
            "likely_cards": ["D:K", "H:A", "C:T", "S:10", "C:2"],
        }
    )
    assert sc_colon.likely_cards == [
        card_id(Suit.DIAMONDS, 11),
        card_id(Suit.HEARTS, 12),
        card_id(Suit.CLUBS, 8),
        card_id(Suit.SPADES, 8),
        card_id(Suit.CLUBS, 0),
    ]

    sc_compact = SeatConstraint.model_validate(
        {
            "seat": "EAST",
            "likely_cards": ["DK", "CK", "CQ", "SK", "HQ", "HJ", "HT", "10D", "♠A", "♣2", 51],
        }
    )
    assert sc_compact.likely_cards == [
        card_id(Suit.DIAMONDS, 11),
        card_id(Suit.CLUBS, 11),
        card_id(Suit.CLUBS, 10),
        card_id(Suit.SPADES, 11),
        card_id(Suit.HEARTS, 10),
        card_id(Suit.HEARTS, 9),
        card_id(Suit.HEARTS, 8),
        card_id(Suit.DIAMONDS, 8),
        card_id(Suit.SPADES, 12),
        card_id(Suit.CLUBS, 0),
        51,
    ]

    # StrategyState full JSON validation accepts string card hints cleanly
    st = StrategyState.model_validate_json(
        '{"phase":"play","plan_summary":"Duck once","seat_constraints":'
        '{"WEST":{"min_hcp":12,"max_hcp":16,"likely_cards":["D:K","SA","♥Q"]}}}'
    )
    assert st.seat_constraints[Seat.WEST].likely_cards == [
        card_id(Suit.DIAMONDS, 11),
        card_id(Suit.SPADES, 12),
        card_id(Suit.HEARTS, 10),
    ]

    # Invalid card strings and out-of-range integers must still be rejected
    for bad in ([52], [-1], ["XY"], ["D:15"], ["not_a_card"]):
        with pytest.raises(ValidationError):
            SeatConstraint.model_validate({"seat": "NORTH", "likely_cards": bad})
