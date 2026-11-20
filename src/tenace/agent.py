from __future__ import annotations

import pyspiel

from tenace.domain.types import Seat, StrategyState, Suit
from tenace.layer1_strategist import (
    SAYC_CONVENTION_CARD,
    ConventionCard,
    StrategistEngine,
)
from tenace.layer2_evaluator import TacticalOptionEvaluator
from tenace.layer3_rules import (
    RuleDDSEvaluator,
    RuleFactTracker,
    collapse_equivalent_actions,
)
from tenace.models.clients import StructuredModelClient

__all__ = ["TenaceAgent"]


class TenaceAgent:
    """3-Layer Neuro-Symbolic Contract Bridge Agent for a single seat."""

    def __init__(
        self,
        seat: Seat,
        strategist_client: StructuredModelClient,
        evaluator_client: StructuredModelClient,
        dds_evaluator: RuleDDSEvaluator | None = None,
        convention_card: ConventionCard = SAYC_CONVENTION_CARD,
    ) -> None:
        self.seat = Seat(seat)
        self.tracker = RuleFactTracker()
        self.strategist = StrategistEngine(
            client=strategist_client,
            convention_card=convention_card,
        )
        self.evaluator = TacticalOptionEvaluator(
            client=evaluator_client,
            convention_card=convention_card,
        )
        self.dds_evaluator = (
            dds_evaluator if dds_evaluator is not None else RuleDDSEvaluator()
        )

        self.strategy: StrategyState | None = None
        self.strategy_history: list[StrategyState] = []
        self._prev_voids: dict[Seat, set[Suit]] | None = None
        self._dummy_seen: bool = False

        self.forced_moves: int = 0
        self.strategist_calls: int = 0
        self.evaluator_calls: int = 0

    def reset_for_board(self) -> None:
        """Clear board-scoped state while preserving cumulative telemetry counters."""
        self.strategy = None
        self.strategy_history = []
        self._prev_voids = None
        self._dummy_seen = False

    def reset_stats(self) -> None:
        """Reset cumulative telemetry counters (`forced_moves`, `strategist_calls`, `evaluator_calls`)."""
        self.forced_moves = 0
        self.strategist_calls = 0
        self.evaluator_calls = 0

    def step(self, state: pyspiel.State) -> int:
        if state.is_terminal():
            raise ValueError("Cannot step on a terminal bridge state")
        if state.is_chance_node() or state.current_phase() == 0:
            raise ValueError("Cannot step during incomplete deal phase (chance node)")

        current_player = state.current_player()
        if current_player != int(self.seat):
            raise ValueError(
                f"Agent seat {self.seat.name} ({int(self.seat)}) called on "
                f"seat {current_player}'s turn"
            )

        facts = self.tracker.extract_facts(
            state,
            self.seat,
            prev_voids=self._prev_voids,
        )

        candidates = collapse_equivalent_actions(
            state.legal_actions(),
            played_cards=set(facts.played_cards),
        )
        if not candidates:
            raise ValueError("No legal actions available on non-terminal state")

        # 0-API Fast Path: single collapsed candidate (e.g. forced singleton or equivalent cards).
        # Defer updating _prev_voids and _dummy_seen so surprise show-outs and newly revealed
        # dummy cards are still surfaced to Strategist on the next genuine decision turn!
        if len(candidates) == 1:
            self.forced_moves += 1
            return candidates[0].action

        self._prev_voids = {s: set(voids) for s, voids in facts.known_voids.items()}

        # Layer 1 Event-Driven Strategist Check
        if self.strategist.should_replan(
            facts,
            self.strategy,
            dummy_already_seen=self._dummy_seen,
        ):
            trigger_reasons: list[str] = []
            if self.strategy is None:
                trigger_reasons.append("Initial board planning")
            elif self.strategy.phase != facts.phase:
                trigger_reasons.append(
                    f"Phase transition from {self.strategy.phase} to {facts.phase}"
                )
            if (
                facts.phase == "play"
                and facts.dummy_seat is not None
                and not self._dummy_seen
            ):
                trigger_reasons.append("Dummy revealed after opening lead")
            if facts.surprise_events:
                trigger_reasons.append("; ".join(facts.surprise_events))

            self.strategy = self.strategist.update_strategy(
                facts,
                self.strategy,
                trigger_reason="; ".join(trigger_reasons),
            )
            self.strategy_history.append(self.strategy)
            self.strategist_calls += 1
            if facts.dummy_seat is not None:
                self._dummy_seen = True

        assert self.strategy is not None

        # Layer 3 Constrained DDS Scoring
        scored_candidates = self.dds_evaluator.score_candidates(
            state,
            self.seat,
            candidates,
            strategy=self.strategy,
        )

        # Layer 2 Tactical Evaluation
        decision = self.evaluator.evaluate(
            facts,
            self.strategy,
            scored_candidates,
        )
        self.evaluator_calls += 1

        # Closed-Loop Assumption Violation Interrupt (bounded to at most 1 replan per step)
        if decision.assumption_violated:
            reason = (
                decision.violation_reason.strip()
                if decision.violation_reason.strip()
                else "Layer-2 flagged assumption violation"
            )
            self.strategy = self.strategist.update_strategy(
                facts,
                self.strategy,
                trigger_reason=reason,
            )
            self.strategy_history.append(self.strategy)
            self.strategist_calls += 1
            if facts.dummy_seat is not None:
                self._dummy_seen = True

            scored_candidates = self.dds_evaluator.score_candidates(
                state,
                self.seat,
                candidates,
                strategy=self.strategy,
            )
            decision = self.evaluator.evaluate(
                facts,
                self.strategy,
                scored_candidates,
            )
            self.evaluator_calls += 1

        # Layer 3 Guarded Selection (strict make_prob blunder gate applies during card play;
        # auction selection trusts Layer-2 partnership bidding scores without pass-out DDS bias)
        return self.dds_evaluator.select_guarded_action(
            scored_candidates,
            decision,
            phase=facts.phase,
        )
