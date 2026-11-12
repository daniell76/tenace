from __future__ import annotations

from types import SimpleNamespace
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel, ValidationError

import tenace.layer1_strategist as layer1_pkg
import tenace.layer2_evaluator as layer2_pkg
from tenace.domain.types import (
    DBL_ACTION,
    FIRST_CONTRACT_BID_ACTION,
    PASS_ACTION,
    RDBL_ACTION,
    Assumption,
    CandidateOption,
    Seat,
    SeatConstraint,
    StrategyState,
    Suit,
    TacticalDecision,
    action_label,
    card_id,
)
from tenace.layer1_strategist import (
    SAYC_CONVENTION_CARD,
    ConventionCard,
    StrategistEngine,
    format_auction_sequence,
    format_cards_by_suit,
    format_contract_summary,
    format_known_voids,
)
from tenace.layer1_strategist.conventions import __all__ as conventions_all
from tenace.layer1_strategist.strategist import __all__ as strategist_all
from tenace.layer2_evaluator import DiffusionOptionEvaluator, TacticalOptionEvaluator
from tenace.layer2_evaluator.evaluator import __all__ as evaluator_all
from tenace.layer3_rules.tracker import BoardFactSnapshot
from tenace.models import GeminiDecisionsClient

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


def _make_facts(
    *,
    observer: Seat = Seat.SOUTH,
    phase: str = "auction",
    own_cards: list[int] | None = None,
    own_hcp: int = 14,
    dummy_seat: Seat | None = None,
    dummy_cards: list[int] | None = None,
    played_cards: list[int] | None = None,
    auction_calls: list[tuple[Seat, int]] | None = None,
    declarer: Seat | None = None,
    trump_denom: int | None = None,
    contract_level: int = 0,
    known_voids: dict[Seat, set[Suit]] | None = None,
    surprise_events: list[str] | None = None,
) -> BoardFactSnapshot:
    cards = (
        own_cards
        if own_cards is not None
        else [
            card_id(Suit.SPADES, 12),    # S:A
            card_id(Suit.SPADES, 11),    # S:K
            card_id(Suit.SPADES, 0),     # S:2
            card_id(Suit.HEARTS, 10),    # H:Q
            card_id(Suit.HEARTS, 7),     # H:9
            card_id(Suit.CLUBS, 5),      # C:7
            card_id(Suit.CLUBS, 2),      # C:4
        ]
    )
    return BoardFactSnapshot(
        observer=observer,
        phase=phase,  # type: ignore[arg-type]
        own_cards=cards,
        own_hcp=own_hcp,
        dummy_seat=dummy_seat,
        dummy_cards=dummy_cards or [],
        played_cards=played_cards or [],
        unseen_cards=[],
        auction_calls=auction_calls or [],
        declarer=declarer,
        trump_denom=trump_denom,
        contract_level=contract_level,
        known_voids=known_voids or {s: set() for s in Seat},
        surprise_events=surprise_events or [],
    )


def test_public_exports_for_layer1_and_layer2():
    assert {"ConventionCard", "SAYC_CONVENTION_CARD"}.issubset(set(conventions_all))
    expected_strategist_exports = {
        "StrategistEngine",
        "format_auction_sequence",
        "format_cards_by_suit",
        "format_contract_summary",
        "format_known_voids",
    }
    assert expected_strategist_exports.issubset(set(strategist_all))
    assert ({"ConventionCard", "SAYC_CONVENTION_CARD"} | expected_strategist_exports).issubset(
        set(layer1_pkg.__all__)
    )
    assert {"DiffusionOptionEvaluator", "TacticalOptionEvaluator"}.issubset(
        set(evaluator_all)
    )
    assert {"DiffusionOptionEvaluator", "TacticalOptionEvaluator"}.issubset(
        set(layer2_pkg.__all__)
    )
    assert DiffusionOptionEvaluator is TacticalOptionEvaluator


def test_format_contract_summary_doubled_and_redoubled():
    assert format_contract_summary(_make_facts(phase="auction")) == "(none)"

    # Earlier double overruled by a higher contract bid -> undoubled
    undoubled = _make_facts(
        phase="play",
        declarer=Seat.SOUTH,
        trump_denom=2,  # Hearts
        contract_level=4,
        auction_calls=[
            (Seat.SOUTH, FIRST_CONTRACT_BID_ACTION + 12),  # 3H
            (Seat.WEST, DBL_ACTION),
            (Seat.NORTH, FIRST_CONTRACT_BID_ACTION + 17),  # 4H
            (Seat.EAST, PASS_ACTION),
            (Seat.SOUTH, PASS_ACTION),
            (Seat.WEST, PASS_ACTION),
        ],
    )
    assert format_contract_summary(undoubled) == "4H by SOUTH"

    doubled = _make_facts(
        phase="play",
        declarer=Seat.NORTH,
        trump_denom=3,  # Spades
        contract_level=4,
        auction_calls=[
            (Seat.NORTH, FIRST_CONTRACT_BID_ACTION + 18),  # 4S
            (Seat.EAST, DBL_ACTION),
            (Seat.SOUTH, PASS_ACTION),
            (Seat.WEST, PASS_ACTION),
            (Seat.NORTH, PASS_ACTION),
        ],
    )
    assert format_contract_summary(doubled) == "4SX by NORTH"

    redoubled = _make_facts(
        phase="play",
        declarer=Seat.EAST,
        trump_denom=4,  # NT
        contract_level=3,
        auction_calls=[
            (Seat.EAST, FIRST_CONTRACT_BID_ACTION + 14),  # 3NT
            (Seat.SOUTH, DBL_ACTION),
            (Seat.WEST, RDBL_ACTION),
            (Seat.NORTH, PASS_ACTION),
            (Seat.EAST, PASS_ACTION),
            (Seat.SOUTH, PASS_ACTION),
        ],
    )
    assert format_contract_summary(redoubled) == "3NTXX by EAST"

    # Verify the other shared formatters also behave deterministically
    assert format_auction_sequence([]) == "(no calls yet)"
    assert format_known_voids({s: set() for s in Seat}) == "(none)"


def test_convention_card_immutability_and_zero_partner_state_leakage():
    assert SAYC_CONVENTION_CARD.name
    rules_lower = SAYC_CONVENTION_CARD.summary_rules.lower()
    assert "5-card" in rules_lower or "five-card" in rules_lower
    assert "15-17" in SAYC_CONVENTION_CARD.summary_rules or "15–17" in SAYC_CONVENTION_CARD.summary_rules
    assert "stayman" in rules_lower
    assert "jacoby" in rules_lower
    assert "2c" in rules_lower
    assert "weak" in rules_lower
    assert "20-21" in SAYC_CONVENTION_CARD.summary_rules or "20–21" in SAYC_CONVENTION_CARD.summary_rules
    assert "attitude" in rules_lower and "count" in rules_lower

    # Empty or whitespace strings must be rejected
    with pytest.raises(ValidationError):
        ConventionCard(name="", summary_rules="5-card majors")
    with pytest.raises(ValidationError):
        ConventionCard(name="   ", summary_rules="5-card majors")
    with pytest.raises(ValidationError):
        ConventionCard(name="SAYC", summary_rules="   ")

    # Frozen / immutable config: sharing across North & South cannot leak mutable state
    with pytest.raises(ValidationError):
        SAYC_CONVENTION_CARD.name = "Modified"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        SAYC_CONVENTION_CARD.summary_rules = "Secret signal"  # type: ignore[misc]
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        SAYC_CONVENTION_CARD.secret_partner_note = "H:Q"  # type: ignore[attr-defined]

    north_client = RecordingStructuredClient(
        [StrategyState(phase="auction", plan_summary="North plan")]
    )
    south_client = RecordingStructuredClient(
        [StrategyState(phase="auction", plan_summary="South plan")]
    )
    north_engine = StrategistEngine(north_client, convention_card=SAYC_CONVENTION_CARD)
    south_engine = StrategistEngine(south_client, convention_card=SAYC_CONVENTION_CARD)

    north_facts = _make_facts(
        observer=Seat.NORTH,
        own_cards=[card_id(Suit.HEARTS, 12)],  # H:A only in North
        own_hcp=4,
    )
    south_facts = _make_facts(
        observer=Seat.SOUTH,
        own_cards=[card_id(Suit.CLUBS, 12)],  # C:A only in South
        own_hcp=4,
    )
    north_engine.update_strategy(north_facts, None, trigger_reason="initial")
    south_engine.update_strategy(south_facts, None, trigger_reason="initial")

    south_prompt = south_client.calls[0]["user_prompt"]
    assert "North plan" not in south_prompt
    assert "H: A" not in south_prompt


def test_format_cards_by_suit_empty_and_grouped_suits():
    assert format_cards_by_suit([]) == "(none)"

    cards = [
        card_id(Suit.SPADES, 0),     # S:2
        card_id(Suit.SPADES, 12),    # S:A
        card_id(Suit.SPADES, 11),    # S:K
        card_id(Suit.HEARTS, 7),     # H:9
        card_id(Suit.HEARTS, 10),    # H:Q
        card_id(Suit.CLUBS, 2),      # C:4
        card_id(Suit.CLUBS, 5),      # C:7
    ]
    formatted = format_cards_by_suit(cards)
    assert formatted == "S: A K 2 | H: Q 9 | D: none | C: 7 4"

    with pytest.raises(ValueError):
        format_cards_by_suit([-1])
    with pytest.raises(ValueError):
        format_cards_by_suit([52])
    with pytest.raises(ValueError):
        format_cards_by_suit([True])  # type: ignore[list-item]


def test_strategist_triggers_on_initial_state_phase_transition_and_surprise():
    client = RecordingStructuredClient(
        [
            # Model returns wrong phase/revision intentionally; engine must enforce phase & revision!
            StrategyState(
                phase="play",
                plan_summary="Open 1S and invite game",
                target_or_threat="4S",
                assumptions=[Assumption(claim="Partner holds 8+ HCP", confidence=0.65)],
                seat_constraints={
                    Seat.NORTH: SeatConstraint(seat=Seat.NORTH, min_hcp=8, max_hcp=15)
                },
                revision=99,
            ),
            StrategyState(
                phase="auction",
                plan_summary="Lead conservatively before dummy hits",
                target_or_threat="Set 4S",
                revision=0,
            ),
            StrategyState(
                phase="play",
                plan_summary="Draw trumps with dummy's honors visible",
                target_or_threat="Make 4S",
                revision=0,
            ),
            StrategyState(
                phase="play",
                plan_summary="Switch to cross-ruff after East spade void",
                target_or_threat="10 tricks via ruffs",
                revision=0,
            ),
        ]
    )
    engine = StrategistEngine(client)

    # 1. Initial state (current_strategy is None) -> True
    auction_facts = _make_facts(
        observer=Seat.SOUTH,
        phase="auction",
        auction_calls=[(Seat.NORTH, FIRST_CONTRACT_BID_ACTION + 3)],  # 1S
    )
    assert engine.should_replan(auction_facts, None, dummy_already_seen=False) is True

    strat_1 = engine.update_strategy(
        auction_facts, None, trigger_reason="Initial auction planning"
    )
    assert strat_1.phase == "auction"
    assert strat_1.revision == 1
    assert strat_1.plan_summary == "Open 1S and invite game"

    # Steady auction turn with no surprise -> False
    assert engine.should_replan(auction_facts, strat_1, dummy_already_seen=False) is False

    # 2. Phase transition ("auction" -> "play" on opening lead, before dummy is revealed) -> True
    opening_lead_facts = _make_facts(
        observer=Seat.WEST,
        phase="play",
        dummy_seat=None,
        dummy_cards=[],
        declarer=Seat.NORTH,
        trump_denom=3,  # Spades
        contract_level=4,
        auction_calls=[
            (Seat.NORTH, FIRST_CONTRACT_BID_ACTION + 18),  # 4S
            (Seat.EAST, PASS_ACTION),
            (Seat.SOUTH, PASS_ACTION),
            (Seat.WEST, PASS_ACTION),
        ],
    )
    assert (
        engine.should_replan(opening_lead_facts, strat_1, dummy_already_seen=False)
        is True
    )
    strat_2 = engine.update_strategy(
        opening_lead_facts, strat_1, trigger_reason="Phase transition to play"
    )
    assert strat_2.phase == "play"
    assert strat_2.revision == 2

    # Still pre-opening-lead (dummy_seat is None) and already in play phase -> False
    assert (
        engine.should_replan(opening_lead_facts, strat_2, dummy_already_seen=False)
        is False
    )

    # 3. Post-opening-lead dummy reveal (dummy_seat is not None and not dummy_already_seen) -> True
    dummy_reveal_facts = _make_facts(
        observer=Seat.NORTH,
        phase="play",
        dummy_seat=Seat.SOUTH,
        dummy_cards=[card_id(Suit.SPADES, 10), card_id(Suit.DIAMONDS, 12)],  # S:Q, D:A
        played_cards=[card_id(Suit.HEARTS, 11)],  # H:K opening lead
        declarer=Seat.NORTH,
        trump_denom=3,
        contract_level=4,
    )
    assert (
        engine.should_replan(dummy_reveal_facts, strat_2, dummy_already_seen=False)
        is True
    )
    # Once dummy_already_seen=True and no surprise events -> False
    assert (
        engine.should_replan(dummy_reveal_facts, strat_2, dummy_already_seen=True)
        is False
    )

    strat_3 = engine.update_strategy(
        dummy_reveal_facts, strat_2, trigger_reason="Dummy revealed after opening lead"
    )
    assert strat_3.phase == "play"
    assert strat_3.revision == 3

    # Verify prompt contents passed to StructuredModelClient on play with dummy
    play_call = client.calls[2]
    assert SAYC_CONVENTION_CARD.name in play_call["system_prompt"]
    assert "Stayman" in play_call["system_prompt"]
    assert "NORTH" in play_call["user_prompt"]
    assert "SOUTH" in play_call["user_prompt"]  # partner & dummy
    assert "4S" in play_call["user_prompt"]
    assert "Dummy revealed after opening lead" in play_call["user_prompt"]
    assert "Lead conservatively before dummy hits" in play_call["user_prompt"]  # prev JSON

    # 4. Surprise event (unexpected void show-out) -> True even when dummy_already_seen=True
    surprise_facts = _make_facts(
        observer=Seat.NORTH,
        phase="play",
        dummy_seat=Seat.SOUTH,
        dummy_cards=[card_id(Suit.DIAMONDS, 12)],
        played_cards=[card_id(Suit.SPADES, 12), card_id(Suit.CLUBS, 0)],
        declarer=Seat.NORTH,
        trump_denom=3,
        contract_level=4,
        known_voids={
            Seat.NORTH: set(),
            Seat.EAST: {Suit.SPADES},
            Seat.SOUTH: set(),
            Seat.WEST: set(),
        },
        surprise_events=["EAST showed out of SPADES"],
    )
    assert (
        engine.should_replan(surprise_facts, strat_3, dummy_already_seen=True) is True
    )
    strat_4 = engine.update_strategy(
        surprise_facts, strat_3, trigger_reason="Surprise void"
    )
    assert strat_4.phase == "play"
    assert strat_4.revision == 4
    assert "EAST showed out of SPADES" in client.calls[3]["user_prompt"]
    assert "SPADES" in client.calls[3]["user_prompt"]


def test_strategist_rejects_terminal_phase():
    client = RecordingStructuredClient([])
    engine = StrategistEngine(client)
    terminal_facts = _make_facts(phase="terminal")

    with pytest.raises(ValueError, match="Cannot plan on terminal board"):
        engine.should_replan(terminal_facts, None, dummy_already_seen=False)

    with pytest.raises(ValueError, match="Cannot plan on terminal board"):
        engine.update_strategy(terminal_facts, None)


def test_layer2_evaluator_formats_candidates_and_returns_tactical_decision():
    expected_decision = TacticalDecision(
        assumption_violated=False,
        violation_reason="",
        option_scores={"S:A": 0.91, "H:Q": 0.42, "C:7": 0.25},
        rationale="Cash S:A first to test trump split before touching hearts.",
    )
    client = RecordingStructuredClient([expected_decision])
    evaluator = DiffusionOptionEvaluator(client)

    facts = _make_facts(
        observer=Seat.SOUTH,
        phase="play",
        dummy_seat=Seat.NORTH,
        dummy_cards=[card_id(Suit.SPADES, 10), card_id(Suit.HEARTS, 12)],
        played_cards=[card_id(Suit.DIAMONDS, 11)],
        declarer=Seat.SOUTH,
        trump_denom=3,  # Spades
        contract_level=4,
        known_voids={
            Seat.NORTH: set(),
            Seat.EAST: {Suit.DIAMONDS},
            Seat.SOUTH: set(),
            Seat.WEST: set(),
        },
    )
    strategy = StrategyState(
        phase="play",
        plan_summary="Draw two rounds of trumps ending in dummy",
        target_or_threat="Protect against East diamond ruff",
        assumptions=[Assumption(claim="Spades split 3-2", confidence=0.68)],
        revision=3,
    )
    candidates = [
        CandidateOption(
            action=card_id(Suit.SPADES, 12),
            label=action_label(card_id(Suit.SPADES, 12)),  # S:A
            equivalent_actions=[card_id(Suit.SPADES, 12), card_id(Suit.SPADES, 11)],
            baseline_ev=420.0,
            make_prob=0.8125,
        ),
        CandidateOption(
            action=card_id(Suit.HEARTS, 10),
            label=action_label(card_id(Suit.HEARTS, 10)),  # H:Q
            equivalent_actions=[card_id(Suit.HEARTS, 10)],
            baseline_ev=-50.25,
            make_prob=0.4375,
        ),
        CandidateOption(
            action=card_id(Suit.CLUBS, 5),
            label=action_label(card_id(Suit.CLUBS, 5)),  # C:7
            equivalent_actions=[card_id(Suit.CLUBS, 5)],
            baseline_ev=-100.0,
            make_prob=None,
        ),
    ]

    result = evaluator.evaluate(facts, strategy, candidates)
    assert result == expected_decision
    assert len(client.calls) == 1

    call = client.calls[0]
    assert call["schema"] is TacticalDecision
    user_prompt = call["user_prompt"]

    # Verify Acting Hand line when declarer plays from own hand
    assert "Acting Hand: Own (SOUTH)" in user_prompt

    # Verify strategy summary details
    assert "3" in user_prompt
    assert "Draw two rounds of trumps ending in dummy" in user_prompt
    assert "Protect against East diamond ruff" in user_prompt
    assert "Spades split 3-2" in user_prompt

    # Verify candidate formatting with exact label, baseline_ev=..., and make_prob (2 decimals or N/A)
    assert "S:A" in user_prompt
    assert "baseline_ev=420.0" in user_prompt
    assert "make_prob=0.81" in user_prompt or "0.81" in user_prompt

    assert "H:Q" in user_prompt
    assert "baseline_ev=-50.2" in user_prompt
    assert "make_prob=0.44" in user_prompt or "0.44" in user_prompt

    assert "C:7" in user_prompt
    assert "baseline_ev=-100.0" in user_prompt
    assert "N/A" in user_prompt


def test_layer2_evaluator_marks_acting_hand_dummy_when_declarer_plays_dummy_cards():
    sq = card_id(Suit.SPADES, 10)   # S:Q in dummy (NORTH)
    ha = card_id(Suit.HEARTS, 12)   # H:A in dummy (NORTH)
    expected = TacticalDecision(
        assumption_violated=False,
        option_scores={"S:Q": 0.85, "H:A": 0.55},
        rationale="Play S:Q from dummy to finesse.",
    )
    client = RecordingStructuredClient([expected, expected])
    evaluator = DiffusionOptionEvaluator(client)

    play_facts = _make_facts(
        observer=Seat.SOUTH,
        phase="play",
        dummy_seat=Seat.NORTH,
        dummy_cards=[sq, ha],
        played_cards=[card_id(Suit.DIAMONDS, 11)],
        declarer=Seat.SOUTH,
        trump_denom=3,
        contract_level=4,
    )
    strategy = StrategyState(
        phase="play",
        plan_summary="Lead trump from dummy",
        revision=2,
    )
    dummy_candidates = [
        CandidateOption(
            action=sq,
            label=action_label(sq),
            equivalent_actions=[sq],
            baseline_ev=420.0,
            make_prob=0.80,
        ),
        CandidateOption(
            action=ha,
            label=action_label(ha),
            equivalent_actions=[ha],
            baseline_ev=100.0,
            make_prob=0.60,
        ),
    ]

    evaluator.evaluate(play_facts, strategy, dummy_candidates)
    play_prompt = client.calls[0]["user_prompt"]
    assert "Acting Hand: Dummy (NORTH)" in play_prompt
    assert "Acting Hand: Own" not in play_prompt

    # Auction phase should not emit a card-play Acting Hand line
    auction_facts = _make_facts(observer=Seat.SOUTH, phase="auction")
    auction_strategy = StrategyState(phase="auction", plan_summary="Bid 1S", revision=1)
    bid_candidates = [
        CandidateOption(
            action=PASS_ACTION,
            label="Pass",
            equivalent_actions=[PASS_ACTION],
            baseline_ev=0.0,
            make_prob=0.5,
        )
    ]
    evaluator.evaluate(auction_facts, auction_strategy, bid_candidates)
    auction_prompt = client.calls[1]["user_prompt"]
    assert "Acting Hand:" not in auction_prompt


def test_layer2_evaluator_rejects_empty_candidates_and_terminal_phase():
    client = RecordingStructuredClient([])
    evaluator = DiffusionOptionEvaluator(client)
    strategy = StrategyState(phase="play", plan_summary="Plan", revision=1)

    play_facts = _make_facts(phase="play")
    with pytest.raises(ValueError, match="candidates"):
        evaluator.evaluate(play_facts, strategy, [])

    terminal_facts = _make_facts(phase="terminal")
    cand = CandidateOption(
        action=PASS_ACTION,
        label="Pass",
        equivalent_actions=[PASS_ACTION],
        baseline_ev=0.0,
        make_prob=0.5,
    )
    with pytest.raises(ValueError, match="terminal"):
        evaluator.evaluate(terminal_facts, strategy, [cand])


def test_tactical_option_evaluator_uses_direct_evaluate_options_with_gemini_decisions_client():
    decision_requests: list[dict[str, Any]] = []

    def fake_create(*, model: str, input: Any, questions: dict[str, Any]) -> Any:
        decision_requests.append({"model": model, "input": input, "questions": questions})
        ans: dict[str, Any] = {
            "cand_0": {"type": "score", "score": 3.6, "confidence": 0.90},
            "cand_1": {"type": "score", "score": 1.2, "confidence": 0.80},
        }
        if "assumption_violated" in questions:
            ans["assumption_violated"] = {"type": "noul", "noul": 0.82, "confidence": 0.82}
        return SimpleNamespace(answers=ans)

    decisions_client = GeminiDecisionsClient(
        genai_client=SimpleNamespace(decisions=SimpleNamespace(create=fake_create))
    )
    evaluator = TacticalOptionEvaluator(decisions_client)

    facts = _make_facts(
        observer=Seat.SOUTH,
        phase="play",
        dummy_seat=Seat.NORTH,
        dummy_cards=[card_id(Suit.SPADES, 10)],
        declarer=Seat.SOUTH,
        trump_denom=3,
        contract_level=4,
    )
    cands = [
        CandidateOption(
            action=card_id(Suit.SPADES, 12),
            label="S:A",
            equivalent_actions=[card_id(Suit.SPADES, 12)],
            baseline_ev=420.0,
            make_prob=0.85,
        ),
        CandidateOption(
            action=card_id(Suit.CLUBS, 5),
            label="C:7",
            equivalent_actions=[card_id(Suit.CLUBS, 5)],
            baseline_ev=-50.0,
            make_prob=0.35,
        ),
    ]

    # With assumptions -> evaluates assumption_violated NoulQuestion and flags interrupt
    strat_with_assumptions = StrategyState(
        phase="play",
        plan_summary="Draw trumps",
        assumptions=[Assumption(claim="Spades 3-2", confidence=0.68)],
        revision=1,
    )
    res_violated = evaluator.evaluate(facts, strat_with_assumptions, cands)
    assert res_violated.assumption_violated is True
    assert "0.82" in res_violated.violation_reason
    assert res_violated.option_scores == pytest.approx({"S:A": 0.90, "C:7": 0.30})
    assert "=== Board Facts ===" in decision_requests[0]["input"]
    assert "Acting Hand: Own (SOUTH)" in decision_requests[0]["input"]
    assert set(decision_requests[0]["questions"].keys()) == {
        "cand_0",
        "cand_1",
        "assumption_violated",
    }

    # Without assumptions -> passes has_assumptions=False and omits assumption_violated
    strat_no_assumptions = StrategyState(
        phase="play",
        plan_summary="Cash winners",
        assumptions=[],
        revision=2,
    )
    res_clean = evaluator.evaluate(facts, strat_no_assumptions, cands)
    assert res_clean.assumption_violated is False
    assert set(decision_requests[1]["questions"].keys()) == {"cand_0", "cand_1"}

    # StructuredModelClient fallback uses model-neutral system_prompt
    structured_client = RecordingStructuredClient([res_clean])
    fallback_evaluator = TacticalOptionEvaluator(structured_client)
    fallback_evaluator.evaluate(facts, strat_no_assumptions, cands)
    sys_prompt = structured_client.calls[0]["system_prompt"]
    assert sys_prompt.startswith("You are Tenace Layer 2 Tactical Option Evaluator.\n")
    assert "DiffusionGemma" not in sys_prompt

