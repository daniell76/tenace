from __future__ import annotations

from tenace.domain.types import (
    CandidateOption,
    StrategyState,
    TacticalDecision,
    action_label,
)
from tenace.layer1_strategist import (
    SAYC_CONVENTION_CARD,
    ConventionCard,
    format_auction_sequence,
    format_cards_by_suit,
    format_contract_summary,
    format_known_voids,
    format_play_state,
)
from tenace.layer3_rules.tracker import BoardFactSnapshot
from tenace.models.clients import StructuredModelClient

__all__ = ["DiffusionOptionEvaluator", "TacticalOptionEvaluator"]


class TacticalOptionEvaluator:
    def __init__(
        self,
        client: StructuredModelClient,
        convention_card: ConventionCard = SAYC_CONVENTION_CARD,
    ) -> None:
        self.client = client
        self.convention_card = convention_card

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

        if facts.phase == "auction":
            system_prompt = (
                "You are Tenace Layer 2 Tactical Option Evaluator (Bridge Auction).\n"
                "Score each legal auction call in [0.0, 1.0] under `option_scores` keyed by "
                "the exact candidate label following the partnership convention system.\n"
                "Discipline rules: Pass without opening strength (12+ HCP / Rule of 20, or "
                "Rule of 15 in 4th seat); avoid reckless low-level doubles or escaping doubled "
                "contracts into hopeless higher bids; NEVER bid slams (6/7-level) without ~33+ "
                "combined HCP and Ace/keycard control. NOTE: candidate baseline_ev values "
                "naively assume all opponents immediately pass without doubling — prioritize "
                "sound partnership bidding over pass-out double-dummy EV.\n"
                "Set `assumption_violated` to true ONLY if observed calls directly contradict "
                "a critical Layer 1 assumption. Keep `rationale` to one brief sentence and "
                "leave `violation_reason` empty unless `assumption_violated` is true."
            )
        else:
            system_prompt = (
                "You are Tenace Layer 2 Tactical Option Evaluator.\n"
                "Score each legal candidate action in [0.0, 1.0] under `option_scores` keyed by "
                "the exact candidate label, balancing Layer 1 strategic goals with Layer 3 "
                "double-dummy baseline EV and make probability.\n"
                "Set `assumption_violated` to true ONLY if board facts contradict a critical "
                "Layer 1 strategic assumption. Keep `rationale` to one brief sentence and leave "
                "`violation_reason` empty unless `assumption_violated` is true."
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
            ]
        )
        if facts.phase == "play":
            tricks_won_line, completed_line, current_line = format_play_state(facts)
            user_lines.extend(
                [
                    f"Tricks Won: {tricks_won_line}",
                    f"Completed Tricks: {completed_line}",
                    f"Current Trick: {current_line}",
                ]
            )
        user_lines.append("")
        if facts.phase == "auction":
            user_lines.extend(
                [
                    "=== Partnership Bidding System ===",
                    f"System: {self.convention_card.name}",
                    f"Rules: {self.convention_card.summary_rules}",
                    "Discipline: Pass without opening strength (12+ HCP / Rule of 20, or Rule of 15 in 4th seat). "
                    "Do NOT double marginal contracts lightly, run out of doubled contracts to hopeless higher levels, "
                    "or bid slams (6/7-level) without ~33+ combined HCP and Ace/keycard control.",
                    "",
                ]
            )
        user_lines.extend(
            [
                "=== Active Strategy ===",
                f"Revision: {strategy.revision}",
                f"Plan Summary: {strategy.plan_summary}",
                f"Target/Threat: {strategy.target_or_threat or '(none)'}",
                f"Assumptions: {assumptions_summary}",
                "",
            ]
        )
        if facts.phase == "auction":
            user_lines.append(
                "Note: Treat the LIVE Auction sequence above as authoritative over any earlier-round strategy summary."
            )

        direct_eval = getattr(self.client, "evaluate_options", None)
        if callable(direct_eval):
            if facts.phase == "auction":
                direct_lines = [
                    *user_lines,
                    "=== Legal Auction Calls ===",
                    ", ".join(c.label for c in candidates),
                ]
                context_text = "\n".join(direct_lines)
            else:
                context_text = "\n".join(
                    [*user_lines, "=== Candidate Options ===", *candidate_lines]
                )
            return direct_eval(
                context_text=context_text,
                candidates=candidates,
                has_assumptions=bool(strategy.assumptions),
            )

        user_prompt = "\n".join(
            [*user_lines, "=== Candidate Options ===", *candidate_lines]
        )
        return self.client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=TacticalDecision,
        )


DiffusionOptionEvaluator = TacticalOptionEvaluator
