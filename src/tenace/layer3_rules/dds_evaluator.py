from __future__ import annotations

import ctypes
import math
import random
import re
import threading
from typing import Any, Literal

import numpy as np
import pyspiel

from tenace.domain.types import (
    BIDDING_ACTION_BASE,
    DBL_ACTION,
    FIRST_CONTRACT_BID_ACTION,
    MAX_ACTION,
    NUM_CARDS,
    PASS_ACTION,
    RDBL_ACTION,
    CandidateOption,
    Seat,
    SeatConstraint,
    StrategyState,
    Suit,
    TacticalDecision,
    card_id,
    card_rank,
    card_suit,
    hand_hcp,
)
from tenace.layer3_rules.tracker import trick_winner

__all__ = [
    "RuleDDSEvaluator",
    "score_bridge_contract",
    "solve_play_world_scores",
]

_OBS_VUL_RE = re.compile(r"^Vul:\s*(None|N/S|E/W|All)\s*$", re.MULTILINE)


class _DdsDeal(ctypes.Structure):
    _fields_ = [
        ("trump", ctypes.c_int),
        ("first", ctypes.c_int),
        ("currentTrickSuit", ctypes.c_int * 3),
        ("currentTrickRank", ctypes.c_int * 3),
        ("remainCards", (ctypes.c_uint * 4) * 4),
    ]


class _DdsFutureTricks(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.c_int),
        ("cards", ctypes.c_int),
        ("suit", ctypes.c_int * 13),
        ("rank", ctypes.c_int * 13),
        ("equals", ctypes.c_int * 13),
        ("score", ctypes.c_int * 13),
    ]


_DDS_LOCK = threading.Lock()
_DDS_LIB: Any | None = None
_DDS_INIT_ATTEMPTED = False


def _get_dds_lib() -> Any | None:
    """Lazy-load Bo Haglund's DDS 2.9.0 C ABI bundled inside open_spiel's pyspiel shared library."""
    global _DDS_LIB, _DDS_INIT_ATTEMPTED
    if _DDS_INIT_ATTEMPTED:
        return _DDS_LIB
    with _DDS_LOCK:
        if _DDS_INIT_ATTEMPTED:
            return _DDS_LIB
        _DDS_INIT_ATTEMPTED = True
        try:
            lib_path = getattr(pyspiel, "__file__", None)
            if not lib_path:
                return None
            lib = ctypes.CDLL(lib_path)
            if not (hasattr(lib, "SolveBoard") and hasattr(lib, "SetMaxThreads")):
                return None
            lib.SetMaxThreads.argtypes = [ctypes.c_int]
            lib.SetMaxThreads.restype = None
            lib.SolveBoard.argtypes = [
                _DdsDeal,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(_DdsFutureTricks),
                ctypes.c_int,
            ]
            lib.SolveBoard.restype = ctypes.c_int
            lib.SetMaxThreads(0)
            _DDS_LIB = lib
        except OSError:
            _DDS_LIB = None
        return _DDS_LIB


def score_bridge_contract(
    level: int,
    denom: int,
    doubled: int,
    vul: bool,
    decl_tricks: int,
) -> int:
    """Return official WBF duplicate bridge score from declarer's perspective.

    Args:
        level: Contract level in 1..7.
        denom: Strain index in OpenSpiel order (0=C, 1=D, 2=H, 3=S, 4=NT).
        doubled: 0=undoubled, 1=doubled (X), 2=redoubled (XX).
        vul: True if declarer's side is vulnerable.
        decl_tricks: Total tricks won by declarer's side (0..13).
    """
    target = 6 + level
    diff = decl_tricks - target
    if diff < 0:
        down = -diff
        if doubled == 0:
            return -(100 * down if vul else 50 * down)
        mult = 1 if doubled == 1 else 2
        if vul:
            return -mult * (200 + 300 * (down - 1))
        if down == 1:
            return -mult * 100
        if down <= 3:
            return -mult * (100 + 200 * (down - 1))
        return -mult * (500 + 300 * (down - 3))

    if denom in (0, 1):
        trick_val = 20 * level
        ot_base = 20
    elif denom in (2, 3):
        trick_val = 30 * level
        ot_base = 30
    else:
        trick_val = 40 + 30 * (level - 1)
        ot_base = 30

    if doubled == 1:
        trick_val *= 2
    elif doubled == 2:
        trick_val *= 4

    score = trick_val
    score += (500 if vul else 300) if trick_val >= 100 else 50
    if level == 6:
        score += 750 if vul else 500
    elif level == 7:
        score += 1500 if vul else 1000
    if doubled == 1:
        score += 50
    elif doubled == 2:
        score += 100

    if diff > 0:
        if doubled == 0:
            score += diff * ot_base
        elif doubled == 1:
            score += diff * (200 if vul else 100)
        else:
            score += diff * (400 if vul else 200)
    return score


def _extract_contract_and_play(
    world: pyspiel.State,
) -> tuple[int, int, int, Seat, list[int]] | None:
    """Return (level, denom, doubled_state, declarer, play_actions) from a deal world."""
    public_history = world.full_history()[NUM_CARDS:]
    last_bid_seat: Seat | None = None
    last_bid_action: int | None = None
    first_denom_bidder: dict[tuple[int, int], Seat] = {}
    doubled = 0
    play_actions: list[int] = []

    for entry in public_history:
        action = entry.action
        if action >= BIDDING_ACTION_BASE:
            if FIRST_CONTRACT_BID_ACTION <= action <= MAX_ACTION:
                seat = Seat(entry.player)
                denom = (action - FIRST_CONTRACT_BID_ACTION) % 5
                key = (seat.partnership, denom)
                if key not in first_denom_bidder:
                    first_denom_bidder[key] = seat
                last_bid_seat = seat
                last_bid_action = action
                doubled = 0
            elif action == DBL_ACTION:
                doubled = 1
            elif action == RDBL_ACTION:
                doubled = 2
        else:
            play_actions.append(action)

    if last_bid_seat is None or last_bid_action is None:
        return None
    offset = last_bid_action - FIRST_CONTRACT_BID_ACTION
    level = 1 + offset // 5
    denom = offset % 5
    declarer = first_denom_bidder[(last_bid_seat.partnership, denom)]
    return level, denom, doubled, declarer, play_actions


def _is_declarer_vulnerable(world: pyspiel.State, observer: Seat, declarer: Seat) -> bool:
    obs_str = world.observation_string(int(observer))
    m_vul = _OBS_VUL_RE.search(obs_str)
    if m_vul is None:
        return False
    vul_tok = m_vul.group(1)
    if vul_tok == "All":
        return True
    if vul_tok == "None":
        return False
    decl_is_ns = declarer.partnership == 0
    return (vul_tok == "N/S" and decl_is_ns) or (vul_tok == "E/W" and not decl_is_ns)


def solve_play_world_scores(
    world: pyspiel.State,
    observer: Seat,
) -> dict[int, float] | None:
    """Solve all legal cards in a resampled play-phase `world` via C++ DDS `SolveBoard`.

    Returns a mapping `{card_action: exact_duplicate_score_for_observer}` or `None` if
    the native solver is unavailable or the state cannot be solved.
    IMPORTANT: Caller must pass a *resampled* infostate world (never the hidden true state).
    """
    lib = _get_dds_lib()
    if lib is None:
        return None
    full_hist = world.full_history()
    if len(full_hist) < NUM_CARDS:
        return None
    extracted = _extract_contract_and_play(world)
    if extracted is None:
        return None
    level, trump_denom, doubled, declarer, play_actions = extracted
    if len(play_actions) >= NUM_CARDS:
        return None

    played_set = set(play_actions)
    dl = _DdsDeal()
    # DDS strain encoding: 0=Spades, 1=Hearts, 2=Diamonds, 3=Clubs, 4=NT.
    dl.trump = (3 - trump_denom) if trump_denom < 4 else 4

    # Track completed tricks & locate current open trick leader.
    leader = Seat((int(declarer) + 1) % 4)
    completed_tricks = 0
    decl_tricks_so_far = 0
    for trick_start in range(0, len(play_actions) - (len(play_actions) % 4), 4):
        trick_cards = play_actions[trick_start : trick_start + 4]
        trick_plays = [
            (Seat((int(leader) + offset) % 4), card)
            for offset, card in enumerate(trick_cards)
        ]
        leader = trick_winner(trick_plays, trump_denom)
        completed_tricks += 1
        if leader.partnership == declarer.partnership:
            decl_tricks_so_far += 1

    in_trick = play_actions[completed_tricks * 4 :]
    dl.first = int(leader)
    for slot in range(3):
        if slot < len(in_trick):
            card = in_trick[slot]
            dl.currentTrickSuit[slot] = 3 - int(card_suit(card))
            dl.currentTrickRank[slot] = card_rank(card) + 2
        else:
            dl.currentTrickSuit[slot] = 0
            dl.currentTrickRank[slot] = 0

    for h in range(4):
        for s in range(4):
            dl.remainCards[h][s] = 0
    for deal_idx in range(NUM_CARDS):
        card = full_hist[deal_idx].action
        if card in played_set:
            continue
        seat_idx = deal_idx % 4
        dds_suit = 3 - int(card_suit(card))
        dl.remainCards[seat_idx][dds_suit] |= 1 << (card_rank(card) + 2)

    ft = _DdsFutureTricks()
    with _DDS_LOCK:
        ret = lib.SolveBoard(dl, -1, 3, 0, ctypes.byref(ft), 0)
    if ret != 1 or ft.cards <= 0:
        return None

    decl_vul = _is_declarer_vulnerable(world, observer, declarer)
    to_move_seat = Seat((int(leader) + len(in_trick)) % 4)
    to_move_is_decl = to_move_seat.partnership == declarer.partnership
    obs_is_decl = Seat(observer).partnership == declarer.partnership
    remaining_tricks = 13 - completed_tricks

    card_scores: dict[int, float] = {}
    for i in range(ft.cards):
        dds_suit = ft.suit[i]
        os_suit = Suit(3 - dds_suit)
        score_tricks = int(ft.score[i])
        decl_rem = score_tricks if to_move_is_decl else (remaining_tricks - score_tricks)
        decl_total = decl_tricks_so_far + decl_rem
        raw_pts = score_bridge_contract(level, trump_denom, doubled, decl_vul, decl_total)
        obs_pts = float(raw_pts if obs_is_decl else -raw_pts)

        top_rank = int(ft.rank[i])
        if 2 <= top_rank <= 14:
            card_scores[card_id(os_suit, top_rank - 2)] = obs_pts
        eq_mask = int(ft.equals[i])
        if eq_mask:
            for r_val in range(2, 15):
                if eq_mask & (1 << r_val):
                    card_scores[card_id(os_suit, r_val - 2)] = obs_pts

    return card_scores


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
    """Enforces hard HCP and suit-length bounds; likely_cards stays a soft Strategist hint."""
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
        max_ev_drop: float | None = 45.0,
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
        if max_ev_drop is not None and (
            isinstance(max_ev_drop, bool)
            or not isinstance(max_ev_drop, (int, float))
            or not math.isfinite(max_ev_drop)
            or max_ev_drop < 0.0
        ):
            raise ValueError(f"max_ev_drop must be a finite number >= 0.0, got {max_ev_drop!r}")

        self.num_worlds = num_worlds
        self.max_rejection_factor = max_rejection_factor
        self.max_make_prob_drop = max_make_prob_drop
        self.max_ev_drop = float(max_ev_drop) if max_ev_drop is not None else None
        self._rng = random.Random(rng_seed)

    def sample_consistent_worlds(
        self,
        state: pyspiel.State,
        observer: Seat,
        strategy: StrategyState | None = None,
    ) -> list[pyspiel.State]:
        """Samples infostate worlds enforcing hard HCP/suit-length bounds (likely_cards stays soft)."""
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
        obs_idx = int(observer)

        if worlds and all(cand.action >= BIDDING_ACTION_BASE for cand in candidates):
            contract_by_action: dict[int, int] = {}
            terminal_by_action: dict[int, float] = {}
            probe_world = worlds[0]
            for cand in candidates:
                sim = probe_world.clone()
                sim.apply_action(cand.action)
                while not sim.is_terminal() and sim.current_phase() == 1:
                    sim.apply_action(PASS_ACTION)
                if sim.is_terminal():
                    terminal_by_action[cand.action] = float(sim.returns()[obs_idx])
                else:
                    possible = sim.possible_contracts()
                    active = [idx for idx, ok in enumerate(possible) if ok]
                    contract_by_action[cand.action] = (
                        active[0] if active else sim.contract_index()
                    )

            unique_contracts = sorted(set(contract_by_action.values()))
            per_world_scores: list[dict[int, float]] = (
                [
                    dict(
                        zip(
                            unique_contracts,
                            world.score_for_contracts(obs_idx, unique_contracts),
                            strict=True,
                        )
                    )
                    for world in worlds
                ]
                if unique_contracts
                else [{} for _ in worlds]
            )

            scored_auction: list[CandidateOption] = []
            for cand in candidates:
                if cand.action in terminal_by_action:
                    term_val = terminal_by_action[cand.action]
                    payoffs = [term_val for _ in worlds]
                else:
                    c_idx = contract_by_action[cand.action]
                    payoffs = [ float(ws[c_idx]) for ws in per_world_scores ]
                ev = float(np.mean(payoffs))
                make_prob = float(
                    np.mean(
                        [1.0 if p > 0.0 else (0.5 if p == 0.0 else 0.0) for p in payoffs]
                    )
                )
                scored_auction.append(
                    cand.model_copy(update={"baseline_ev": ev, "make_prob": make_prob})
                )
            return scored_auction

        # Play phase: solve every legal card in each resampled world with ONE exact
        # Bo Haglund C++ DDS SolveBoard call (`solutions=3`) instead of noisy random rollouts.
        solved_worlds: list[dict[int, float] | None] = [
            solve_play_world_scores(world, observer) for world in worlds
        ]
        scored: list[CandidateOption] = []
        for cand in candidates:
            payoffs: list[float] = []
            for world, seed, score_map in zip(worlds, world_seeds, solved_worlds, strict=True):
                if score_map is not None and cand.action in score_map:
                    payoffs.append(score_map[cand.action])
                else:
                    payoffs.append(
                        self._rollout_value(
                            world,
                            observer,
                            cand.action,
                            rollout_seed=seed,
                        )
                    )
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
        *,
        phase: Literal["play", "auction"] = "play",
    ) -> int:
        if not candidates:
            raise ValueError("No legal candidates available")
        if len(candidates) == 1:
            return candidates[0].action

        def _safe_ev(cand: CandidateOption) -> float:
            return float(cand.baseline_ev) if math.isfinite(cand.baseline_ev) else float("-inf")

        # Auction calls are partnership communication bids (including Pass, artificial
        # conventions like Stayman/transfers/Blackwood, and doubles) — NOT final contracts
        # scored via naive [call, Pass, Pass, Pass] double-dummy rollouts. Never prune
        # Pass or conventional calls via the card-play make_prob gate, and do not mix
        # undoubled slam/double EV distortions against Layer-2's bidding scores.
        if phase == "auction" and decision is not None and decision.option_scores:
            def _auction_score(cand: CandidateOption) -> tuple[float, float]:
                raw = decision.option_scores.get(cand.label)
                if raw is None or not math.isfinite(raw):
                    llm_score = 0.0
                else:
                    llm_score = min(1.0, max(0.0, float(raw)))
                return (llm_score, _safe_ev(cand))

            return max(candidates, key=_auction_score).action

        probs = [
            c.make_prob
            for c in candidates
            if c.make_prob is not None and math.isfinite(c.make_prob)
        ]
        if probs:
            best_prob = max(probs)
            threshold = best_prob - self.max_make_prob_drop
            viable = [
                c
                for c in candidates
                if c.make_prob is None
                or not math.isfinite(c.make_prob)
                or c.make_prob >= threshold - 1e-12
            ]
        else:
            viable = list(candidates)

        if decision is None or not decision.option_scores:
            return max(viable, key=_safe_ev).action

        # Exact-DDS EV blunder guard: even when make_prob ties (e.g., all options 0.0 down 1 vs down 3,
        # or all 1.0 making +1 vs +0), block card plays costing > max_ev_drop expected points.
        if self.max_ev_drop is not None:
            finite_viable_evs = [
                float(c.baseline_ev) for c in viable if math.isfinite(c.baseline_ev)
            ]
            if finite_viable_evs:
                ev_cutoff = max(finite_viable_evs) - self.max_ev_drop - 1e-9
                gated_viable = [
                    c
                    for c in viable
                    if math.isfinite(c.baseline_ev) and float(c.baseline_ev) >= ev_cutoff
                ]
                if gated_viable:
                    viable = gated_viable

        finite_evs = [float(c.baseline_ev) for c in viable if math.isfinite(c.baseline_ev)]
        if finite_evs:
            ev_min = min(finite_evs)
            ev_max = max(finite_evs)
            denom = max(ev_max - ev_min, 1.0)
        else:
            ev_min = 0.0
            denom = 1.0

        def _blended_score(cand: CandidateOption) -> float:
            norm_ev = (
                (float(cand.baseline_ev) - ev_min) / denom
                if math.isfinite(cand.baseline_ev)
                else 0.0
            )
            raw = decision.option_scores.get(cand.label, 0.5)
            llm_score = (
                min(1.0, max(0.0, float(raw))) if math.isfinite(raw) else 0.5
            )
            return 0.45 * norm_ev + 0.55 * llm_score

        return max(viable, key=_blended_score).action
