from __future__ import annotations

from tenace.domain.types import (
    CandidateOption,
    StrategyState,
    TacticalDecision,
    action_label,
)
from tenace.layer1_strategist import (
    format_auction_sequence,
    format_cards_by_suit,
    format_contract_summary,
    format_known_voids,
)
from tenace.layer3_rules.tracker import BoardFactSnapshot
from tenace.models.clients import StructuredModelClient

__all__ = ["DiffusionOptionEvaluator"]


class DiffusionOptionEvaluator:
    def __init__(self, client: StructuredModelClient) -> None:
        self.client = client

    def evaluate(
        self,
        facts: BoardFactSnapshot,
        strategy: StrategyState,
        candidates: list[CandidateOption],
    ) -> TacticalDecision:
        if facts.phase == "terminal":
            raise ValueError("Cannot evaluate options on terminal board")
        if not candidates:
            raise ValueError("candidates must be non-empty")

        system_prompt = (
            "You are Tenace Layer 2 Tactical Option Evaluator (powered by DiffusionGemma).\n"
            "Score each legal candidate action in [0.0, 1.0] under `option_scores` keyed by "
            "the exact candidate label, balancing Layer 1 strategic goals with Layer 3 "
            "double-dummy baseline EV and make probability.\n"
            "Set `assumption_violated` to true ONLY if board facts contradict a critical "
            "Layer 1 strategic assumption."
        )

        dummy_summary = (
            f"{facts.dummy_seat.name} ({format_cards_by_suit(facts.dummy_cards)})"
            if facts.dummy_seat is not None
            else "(not visible)"
        )
        played_summary = (
            " ".join(action_label(c) for c in facts.played_cards)
            if facts.played_cards
            else "(none)"
        )
        assumptions_summary = (
            "; ".join(
                f"{a.claim} (conf={a.confidence:.2f})" for a in strategy.assumptions
            )
            if strategy.assumptions
            else "(none)"
        )
        candidate_lines = [
            (
                f"- {c.label}: baseline_ev={c.baseline_ev:.1f}, "
                f"make_prob={f'{c.make_prob:.2f}' if c.make_prob is not None else 'N/A'}"
            )
            for c in candidates
        ]

        user_lines = [
            "=== Board Facts ===",
            f"Observer: {facts.observer.name} | Phase: {facts.phase}",
        ]
        if facts.phase == "play":
            is_dummy_turn = (
                facts.dummy_seat is not None
                and facts.observer != facts.dummy_seat
                and candidates[0].action in set(facts.dummy_cards)
            )
            acting_hand_line = (
                f"Acting Hand: Dummy ({facts.dummy_seat.name})"
                if is_dummy_turn and facts.dummy_seat is not None
                else f"Acting Hand: Own ({facts.observer.name})"
            )
            user_lines.append(acting_hand_line)

        user_lines.extend(
            [
                f"Contract: {format_contract_summary(facts)}",
                f"Own Hand ({facts.own_hcp} HCP): {format_cards_by_suit(facts.own_cards)}",
                f"Dummy: {dummy_summary}",
                f"Auction: {format_auction_sequence(facts.auction_calls)}",
                f"Played Cards: {played_summary}",
                f"Known Voids: {format_known_voids(facts.known_voids)}",
                "",
                "=== Active Strategy ===",
                f"Revision: {strategy.revision}",
                f"Plan Summary: {strategy.plan_summary}",
                f"Target/Threat: {strategy.target_or_threat or '(none)'}",
                f"Assumptions: {assumptions_summary}",
                "",
                "=== Candidate Options ===",
                *candidate_lines,
            ]
        )
        user_prompt = "\n".join(user_lines)

        return self.client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=TacticalDecision,
        )
