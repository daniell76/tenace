from __future__ import annotations

import random

import numpy as np
import pyspiel

from tenace.domain.types import (
    BIDDING_ACTION_BASE,
    FIRST_CONTRACT_BID_ACTION,
    MAX_ACTION,
    NUM_CARDS,
    PASS_ACTION,
    CandidateOption,
    Seat,
    SeatConstraint,
    StrategyState,
    Suit,
    TacticalDecision,
    card_suit,
    hand_hcp,
)

__all__ = ["RuleDDSEvaluator"]


def _visible_dummy_seat(state: pyspiel.State) -> Seat | None:
    if state.current_phase() == 1:
        return None
    public_history = state.full_history()[NUM_CARDS:]
    last_bid_seat: Seat | None = None
    last_bid_action: int | None = None
    first_denom_bidder: dict[tuple[int, int], Seat] = {}
    has_play_action = False

    for entry in public_history:
        action = entry.action
        if action >= BIDDING_ACTION_BASE:
            seat = Seat(entry.player)
            if FIRST_CONTRACT_BID_ACTION <= action <= MAX_ACTION:
                denom = (action - FIRST_CONTRACT_BID_ACTION) % 5
                key = (seat.partnership, denom)
                if key not in first_denom_bidder:
                    first_denom_bidder[key] = seat
                last_bid_seat = seat
                last_bid_action = action
        else:
            has_play_action = True
            break

    if not has_play_action or last_bid_seat is None or last_bid_action is None:
        return None

    trump_denom = (last_bid_action - FIRST_CONTRACT_BID_ACTION) % 5
    declarer = first_denom_bidder[(last_bid_seat.partnership, trump_denom)]
    return declarer.partner


def _dealt_hand(world: pyspiel.State, seat: Seat) -> list[int]:
    dealt = world.full_history()[:NUM_CARDS]
    seat_idx = int(seat)
    return [dealt[4 * r + seat_idx].action for r in range(13)]


def _matches_constraint(cards: list[int], constraint: SeatConstraint) -> bool:
    hcp = hand_hcp(cards)
    if not (constraint.min_hcp <= hcp <= constraint.max_hcp):
        return False

    if constraint.min_suit_lengths or constraint.max_suit_lengths:
        counts: dict[Suit, int] = {s: 0 for s in Suit}
        for c in cards:
            counts[card_suit(c)] += 1
        for suit, min_len in constraint.min_suit_lengths.items():
            if counts[suit] < min_len:
                return False
        for suit, max_len in constraint.max_suit_lengths.items():
            if counts[suit] > max_len:
                return False

    return True


class RuleDDSEvaluator:
    def __init__(
        self,
        num_worlds: int = 16,
        max_rejection_factor: int = 8,
        max_make_prob_drop: float = 0.20,
        rng_seed: int | None = None,
    ) -> None:
        if isinstance(num_worlds, bool) or num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1, got {num_worlds}")
        if isinstance(max_rejection_factor, bool) or max_rejection_factor < 1:
            raise ValueError(
                f"max_rejection_factor must be >= 1, got {max_rejection_factor}"
            )
        if isinstance(max_make_prob_drop, bool) or not (
            0.0 <= max_make_prob_drop <= 1.0
        ):
            raise ValueError(
                f"max_make_prob_drop must be in [0.0, 1.0], got {max_make_prob_drop}"
            )

        self.num_worlds = num_worlds
        self.max_rejection_factor = max_rejection_factor
        self.max_make_prob_drop = max_make_prob_drop
        self._rng = random.Random(rng_seed)

    def sample_consistent_worlds(
        self,
        state: pyspiel.State,
        observer: Seat,
        strategy: StrategyState | None = None,
    ) -> list[pyspiel.State]:
        observer = Seat(observer)
        if state.current_phase() == 0:
            raise ValueError("Cannot sample worlds during incomplete deal phase")

        active_constraints: list[tuple[Seat, SeatConstraint]] = []
        if strategy is not None and strategy.seat_constraints:
            visible_dummy = _visible_dummy_seat(state)
            for seat_key, constraint in strategy.seat_constraints.items():
                target_seat = Seat(seat_key)
                if target_seat == observer or target_seat == visible_dummy:
                    continue
                active_constraints.append((target_seat, constraint))

        if not active_constraints:
            return [
                state.resample_from_infostate(int(observer), self._rng.random)
                for _ in range(self.num_worlds)
            ]

        accepted: list[pyspiel.State] = []
        fallbacks: list[pyspiel.State] = []
        max_attempts = self.num_worlds * self.max_rejection_factor

        for _ in range(max_attempts):
            world = state.resample_from_infostate(int(observer), self._rng.random)
            if all(
                _matches_constraint(_dealt_hand(world, seat), constraint)
                for seat, constraint in active_constraints
            ):
                accepted.append(world)
                if len(accepted) >= self.num_worlds:
                    return accepted
            elif len(fallbacks) < self.num_worlds:
                fallbacks.append(world)

        needed = self.num_worlds - len(accepted)
        accepted.extend(fallbacks[:needed])
        while len(accepted) < self.num_worlds:
            accepted.append(
                state.resample_from_infostate(int(observer), self._rng.random)
            )
        return accepted

    def _rollout_value(
        self,
        world: pyspiel.State,
        observer: Seat,
        candidate_action: int,
        rollout_seed: int | None = None,
    ) -> float:
        sim = world.clone()
        sim.apply_action(candidate_action)
        obs_idx = int(observer)

        if candidate_action >= BIDDING_ACTION_BASE:
            while not sim.is_terminal() and sim.current_phase() == 1:
                sim.apply_action(PASS_ACTION)
            if sim.is_terminal():
                return float(sim.returns()[obs_idx])
            possible = sim.possible_contracts()
            contract_indices = [idx for idx, active in enumerate(possible) if active]
            if contract_indices:
                return float(sim.score_for_contracts(obs_idx, [contract_indices[0]])[0])
            return float(sim.score_for_contracts(obs_idx, [sim.contract_index()])[0])

        rollout_rng = (
            random.Random(rollout_seed) if rollout_seed is not None else self._rng
        )
        while not sim.is_terminal():
            legal = sim.legal_actions()
            sim.apply_action(rollout_rng.choice(legal))
        return float(sim.returns()[obs_idx])

    def score_candidates(
        self,
        state: pyspiel.State,
        observer: Seat,
        candidates: list[CandidateOption],
        strategy: StrategyState | None = None,
    ) -> list[CandidateOption]:
        observer = Seat(observer)
        if len(candidates) <= 1:
            return list(candidates)

        worlds = self.sample_consistent_worlds(state, observer, strategy)
        world_seeds = [self._rng.getrandbits(64) for _ in worlds]

        scored: list[CandidateOption] = []
        for cand in candidates:
            payoffs = [
                self._rollout_value(
                    world,
                    observer,
                    cand.action,
                    rollout_seed=seed,
                )
                for world, seed in zip(worlds, world_seeds, strict=True)
            ]
            ev = float(np.mean(payoffs))
            make_prob = float(
                np.mean([1.0 if p > 0.0 else (0.5 if p == 0.0 else 0.0) for p in payoffs])
            )
            scored.append(
                cand.model_copy(update={"baseline_ev": ev, "make_prob": make_prob})
            )
        return scored

    def select_guarded_action(
        self,
        candidates: list[CandidateOption],
        decision: TacticalDecision | None = None,
    ) -> int:
        if not candidates:
            raise ValueError("No legal candidates available")
        if len(candidates) == 1:
            return candidates[0].action

        probs = [c.make_prob for c in candidates if c.make_prob is not None]
        if probs:
            best_prob = max(probs)
            threshold = best_prob - self.max_make_prob_drop
            viable = [
                c
                for c in candidates
                if c.make_prob is None or c.make_prob >= threshold - 1e-12
            ]
        else:
            viable = list(candidates)

        if decision is None or not decision.option_scores:
            return max(viable, key=lambda c: c.baseline_ev).action

        evs = [c.baseline_ev for c in viable]
        ev_min = min(evs)
        ev_max = max(evs)
        denom = max(ev_max - ev_min, 1.0)

        def _blended_score(cand: CandidateOption) -> float:
            norm_ev = (cand.baseline_ev - ev_min) / denom
            llm_score = decision.option_scores.get(cand.label, 0.5)
            return 0.45 * norm_ev + 0.55 * llm_score

        return max(viable, key=_blended_score).action
