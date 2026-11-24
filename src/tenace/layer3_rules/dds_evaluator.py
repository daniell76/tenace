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
    "infer_symbolic_auction_constraints",
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


def _constraint_penalty(cards: list[int], constraint: SeatConstraint) -> float:
    """Return 0.0 if `cards` satisfy hard HCP/suit-length bounds, else soft violation distance."""
    penalty = 0.0
    hcp = hand_hcp(cards)
    if hcp < constraint.min_hcp:
        penalty += float(constraint.min_hcp - hcp)
    elif hcp > constraint.max_hcp:
        penalty += float(hcp - constraint.max_hcp)

    if constraint.min_suit_lengths or constraint.max_suit_lengths:
        counts: dict[Suit, int] = {s: 0 for s in Suit}
        for c in cards:
            counts[card_suit(c)] += 1
        for suit, min_len in constraint.min_suit_lengths.items():
            if counts[suit] < min_len:
                penalty += 4.0 * float(min_len - counts[suit])
        for suit, max_len in constraint.max_suit_lengths.items():
            if counts[suit] > max_len:
                penalty += 4.0 * float(counts[suit] - max_len)
    return penalty


def _matches_constraint(cards: list[int], constraint: SeatConstraint) -> bool:
    """Enforces hard HCP and suit-length bounds; likely_cards stays a soft Strategist hint."""
    return _constraint_penalty(cards, constraint) == 0.0


def infer_symbolic_auction_constraints(
    state: pyspiel.State,
    observer: Seat,
    system_id: str = "sayc",
) -> dict[Seat, SeatConstraint]:
    """Infer conservative HCP and suit-length bounds strictly from public auction calls.

    FAIR-PLAY INVARIANT: Inspects ONLY `state.full_history()[NUM_CARDS:]` (public calls),
    never touching dealt card identities `[:NUM_CARDS]`. Excludes `observer` and any visible dummy.
    """
    observer = Seat(observer)
    visible_dummy = _visible_dummy_seat(state)
    sys_key = (system_id or "sayc").strip().lower()
    is_precision = sys_key == "precision"
    is_polish = sys_key == "polish_club"
    is_acol = sys_key == "acol"

    public_history = state.full_history()[NUM_CARDS:]
    calls: list[tuple[Seat, int]] = []
    for entry in public_history:
        if entry.action >= BIDDING_ACTION_BASE:
            calls.append((Seat(entry.player), int(entry.action)))
        else:
            break

    min_hcp: dict[Seat, int] = {s: 0 for s in Seat}
    max_hcp: dict[Seat, int] = {s: 37 for s in Seat}
    min_len: dict[Seat, dict[Suit, int]] = {s: {} for s in Seat}

    opening_bid: tuple[Seat, int, int, int] | None = None  # (seat, call_idx, level, denom)
    opener_side_bid_nt = False
    is_weak_preempt_open = False
    partner_raised_open_suit = False
    passed_before_open: set[Seat] = set()
    first_action_seen: set[Seat] = set()
    non_pass_bid_seen: set[Seat] = set()
    last_contract_denom_by_side: dict[int, set[int]] = {0: set(), 1: set()}
    doubled_opening_seat: set[Seat] = set()
    last_contract_bid: tuple[Seat, int, int] | None = None  # (seat, level, denom)
    passes_since_bid = 0
    side_has_contract_bid: dict[int, bool] = {0: False, 1: False}
    pending_double: tuple[Seat, int, int, int, bool] | None = None  # (doubler, level, denom, passes_since, is_takeout)
    pending_4nt_side: int | None = None

    for idx, (seat, act) in enumerate(calls):
        side = seat.partnership
        if act == PASS_ACTION:
            passes_since_bid += 1
            if pending_double is not None:
                dbl_seat, dbl_lvl, dbl_denom, dbl_passes, is_to = pending_double
                if (
                    is_to
                    and seat == dbl_seat.partner
                    and dbl_passes == 1
                    and dbl_lvl <= 3
                    and dbl_denom < 4
                ):
                    # Passed partner's takeout/reopening double for penalty -> trump stack!
                    dbl_suit = Suit(dbl_denom)
                    min_len[seat][dbl_suit] = max(min_len[seat].get(dbl_suit, 0), 4)
                pending_double = (dbl_seat, dbl_lvl, dbl_denom, dbl_passes + 1, is_to)

            if opening_bid is None:
                # Unopened 1st/2nd/3rd seat pass -> below opening strength
                if idx < 3:
                    max_hcp[seat] = min(max_hcp[seat], 12)
                passed_before_open.add(seat)
            else:
                open_seat, open_idx, open_lvl, open_denom = opening_bid
                # Uncontested pass directly facing partner's 1-level natural opening
                if (
                    seat == open_seat.partner
                    and idx == open_idx + 2
                    and calls[open_idx + 1][1] == PASS_ACTION
                    and open_lvl == 1
                    and open_denom < 4
                    and not (open_denom == 0 and (is_precision or is_polish))
                ):
                    max_hcp[seat] = min(max_hcp[seat], 6)
            first_action_seen.add(seat)
            continue

        if act == DBL_ACTION:
            if last_contract_bid is not None:
                bid_seat, bid_lvl, bid_denom = last_contract_bid
                if bid_seat.partnership != side and bid_lvl <= 3 and bid_denom < 4:
                    # Reopening/balancing double (after 2 passes: e.g. 1H - 2D - P - P - X)
                    # OR direct takeout double when doubler's side has NOT yet made a bid.
                    is_reopening = passes_since_bid == 2
                    is_fresh_takeout = passes_since_bid == 0 and not side_has_contract_bid[side]
                    is_takeout = is_reopening or is_fresh_takeout
                    pending_double = (seat, bid_lvl, bid_denom, 0, is_takeout)
                    if not is_takeout and passes_since_bid == 0 and bid_lvl <= 2:
                        # Direct-seat penalty double after our side already bid: doubler holds trumps.
                        min_len[seat][Suit(bid_denom)] = max(
                            min_len[seat].get(Suit(bid_denom), 0), 4
                        )
                else:
                    pending_double = None
            passes_since_bid = 0
            if opening_bid is not None and seat not in non_pass_bid_seen:
                open_seat, open_idx, open_lvl, _ = opening_bid
                if open_seat.partnership != side and open_lvl <= 2:
                    min_hcp[seat] = max(min_hcp[seat], 11)
                    doubled_opening_seat.add(seat)
            first_action_seen.add(seat)
            non_pass_bid_seen.add(seat)
            continue

        if act == RDBL_ACTION:
            pending_double = None
            passes_since_bid = 0
            first_action_seen.add(seat)
            non_pass_bid_seen.add(seat)
            continue

        if FIRST_CONTRACT_BID_ACTION <= act <= MAX_ACTION:
            pending_double = None
            passes_since_bid = 0
            offset = act - FIRST_CONTRACT_BID_ACTION
            level = 1 + offset // 5
            denom = offset % 5  # 0=C, 1=D, 2=H, 3=S, 4=NT
            suit_enum = Suit(denom) if denom < 4 else None
            is_rkcb_answer = (
                pending_4nt_side == side and level == 5 and denom in (0, 1, 2, 3)
            )
            if is_rkcb_answer:
                # 5C/5D/5H/5S answer to partner's 4NT keycard ask is artificial; do NOT assign suit length.
                pending_4nt_side = None
                last_contract_bid = (seat, level, denom)
                side_has_contract_bid[side] = True
                first_action_seen.add(seat)
                non_pass_bid_seen.add(seat)
                continue

            if opening_bid is None:
                opening_bid = (seat, idx, level, denom)
                if level == 1:
                    if denom == 4:  # 1NT
                        opener_side_bid_nt = True
                        if is_acol:
                            min_hcp[seat] = max(min_hcp[seat], 11)
                            max_hcp[seat] = min(max_hcp[seat], 15)
                        elif is_precision:
                            min_hcp[seat] = max(min_hcp[seat], 12)
                            max_hcp[seat] = min(max_hcp[seat], 16)
                        else:
                            min_hcp[seat] = max(min_hcp[seat], 14)
                            max_hcp[seat] = min(max_hcp[seat], 18)
                    elif denom == 0 and is_precision:
                        min_hcp[seat] = max(min_hcp[seat], 16)
                    elif denom == 0 and is_polish:
                        min_hcp[seat] = max(min_hcp[seat], 11)
                    else:
                        min_hcp[seat] = max(min_hcp[seat], 10 if idx >= 2 else 11)
                        if is_precision:
                            max_hcp[seat] = min(max_hcp[seat], 16)
                        elif is_polish:
                            max_hcp[seat] = min(max_hcp[seat], 18)
                        if suit_enum is not None:
                            if denom in (2, 3):  # 1H/1S
                                min_len[seat][suit_enum] = max(
                                    min_len[seat].get(suit_enum, 0), 4 if is_acol else 5
                                )
                            elif denom == 1 and not is_precision:
                                min_len[seat][suit_enum] = max(
                                    min_len[seat].get(suit_enum, 0), 3
                                )
                elif level == 2:
                    if denom == 4:  # 2NT
                        opener_side_bid_nt = True
                        min_hcp[seat] = max(min_hcp[seat], 19)
                        max_hcp[seat] = min(max_hcp[seat], 22)
                    elif denom == 0:
                        if is_precision or is_polish:
                            min_hcp[seat] = max(min_hcp[seat], 10)
                            max_hcp[seat] = min(max_hcp[seat], 16)
                            min_len[seat][Suit.CLUBS] = max(min_len[seat].get(Suit.CLUBS, 0), 5)
                        else:
                            min_hcp[seat] = max(min_hcp[seat], 19)
                    elif suit_enum is not None and not (denom == 1 and (is_precision or is_polish)):
                        # Weak two 2D/2H/2S (conservative 5+ suit, 4..12 HCP)
                        is_weak_preempt_open = True
                        min_hcp[seat] = max(min_hcp[seat], 4)
                        max_hcp[seat] = min(max_hcp[seat], 12)
                        min_len[seat][suit_enum] = max(min_len[seat].get(suit_enum, 0), 5)
                elif level == 3 and suit_enum is not None:
                    is_weak_preempt_open = True
                    min_hcp[seat] = max(min_hcp[seat], 3)
                    max_hcp[seat] = min(max_hcp[seat], 12)
                    min_len[seat][suit_enum] = max(min_len[seat].get(suit_enum, 0), 6)
                elif level == 4 and suit_enum is not None:
                    is_weak_preempt_open = True
                    min_hcp[seat] = max(min_hcp[seat], 3)
                    max_hcp[seat] = min(max_hcp[seat], 12)
                    min_len[seat][suit_enum] = max(min_len[seat].get(suit_enum, 0), 7)
            else:
                open_seat, open_idx, open_lvl, open_denom = opening_bid
                opp_side = 1 - side
                is_opp_cuebid = denom < 4 and denom in last_contract_denom_by_side[opp_side]
                if level == 4 and denom == 4:
                    # 4NT keycard / quantitative slam bid
                    pending_4nt_side = side
                    if seat.partnership == open_seat.partnership and is_weak_preempt_open and seat == open_seat.partner:
                        min_hcp[seat] = max(min_hcp[seat], 16)
                    elif seat.partnership == open_seat.partnership:
                        min_hcp[seat] = max(min_hcp[seat], 13)
                elif seat.partnership != open_seat.partnership:
                    # First free defensive bid over opponent opening (also applies to passed hands!)
                    if seat not in non_pass_bid_seen and not is_opp_cuebid:
                        if level == 1 and denom == 4:  # 1NT overcall
                            min_hcp[seat] = max(min_hcp[seat], 14)
                            max_hcp[seat] = min(max_hcp[seat], 18)
                        elif level == 1 and suit_enum is not None:
                            min_hcp[seat] = max(min_hcp[seat], 7)
                            min_len[seat][suit_enum] = max(min_len[seat].get(suit_enum, 0), 4)
                        elif level == 2 and suit_enum is not None and open_lvl == 1:
                            if not (denom in (0, 1) and open_denom == 4):
                                min_hcp[seat] = max(min_hcp[seat], 5)
                                min_len[seat][suit_enum] = max(
                                    min_len[seat].get(suit_enum, 0), 5
                                )
                        elif level == 3 and suit_enum is not None and not opener_side_bid_nt:
                            # Advancing partner's takeout double / 3-level jump/overcall in a natural suit
                            min_len[seat][suit_enum] = max(min_len[seat].get(suit_enum, 0), 4)
                    elif (
                        level == 3
                        and suit_enum is not None
                        and not is_opp_cuebid
                        and not opener_side_bid_nt
                    ):
                        if open_seat.partner != seat and denom < 4:
                            min_len[seat][suit_enum] = max(min_len[seat].get(suit_enum, 0), 4)
                    if level == 3 and denom == 4 and seat in doubled_opening_seat:
                        # Double then jump/bid 3NT shows a strong hand (15+ HCP)
                        min_hcp[seat] = max(min_hcp[seat], 15)
                else:
                    # Responder to opening bid (natural non-artificial slots — applies to passed hands too!)
                    if (
                        seat == open_seat.partner
                        and seat not in non_pass_bid_seen
                        and not opener_side_bid_nt
                        and open_lvl == 1
                        and not (open_denom == 0 and (is_precision or is_polish))
                    ):
                        if level == 1 and suit_enum is not None:
                            min_hcp[seat] = max(min_hcp[seat], 5)
                            min_len[seat][suit_enum] = max(min_len[seat].get(suit_enum, 0), 4)
                        elif level == 2 and suit_enum is not None and denom != open_denom and not is_opp_cuebid:
                            min_hcp[seat] = max(min_hcp[seat], 10 if (open_denom == 3 and denom == 2) else 9)
                            req_len = 5 if (open_denom == 3 and denom == 2 and not is_acol) else 4
                            min_len[seat][suit_enum] = max(min_len[seat].get(suit_enum, 0), req_len)
                    if (
                        seat == open_seat.partner
                        and open_lvl == 1
                        and open_denom in (2, 3)
                        and denom == open_denom
                        and level in (2, 3, 4)
                    ):
                        partner_raised_open_suit = True
                        min_len[seat][Suit(open_denom)] = max(
                            min_len[seat].get(Suit(open_denom), 0), 3
                        )
                    if (
                        seat == open_seat
                        and open_lvl == 2
                        and open_denom == 0
                        and not (is_precision or is_polish)
                        and level == 2
                        and denom == 4
                    ):
                        min_hcp[seat] = max(min_hcp[seat], 20)
                        max_hcp[seat] = min(max_hcp[seat], 26)
                    if (
                        seat == open_seat
                        and level == 2
                        and denom == open_denom
                        and suit_enum is not None
                        and open_lvl == 1
                        and not partner_raised_open_suit
                    ):
                        req_reb = 6 if (open_denom in (2, 3) and not is_acol) else 5
                        min_len[seat][suit_enum] = max(min_len[seat].get(suit_enum, 0), req_reb)

            if denom < 4:
                last_contract_denom_by_side[side].add(denom)
            side_has_contract_bid[side] = True
            last_contract_bid = (seat, level, denom)
            first_action_seen.add(seat)
            non_pass_bid_seen.add(seat)

    result: dict[Seat, SeatConstraint] = {}
    for seat in Seat:
        if seat == observer or seat == visible_dummy:
            continue
        lo = min_hcp[seat]
        hi = max_hcp[seat]
        if lo > hi:
            # Passed hand that later overcalled/bid aggressively: relax upper bound softly.
            hi = max(lo, 14)
        lens = min_len[seat]
        if lo > 0 or hi < 37 or lens:
            result[seat] = SeatConstraint(
                seat=seat,
                min_hcp=lo,
                max_hcp=hi,
                min_suit_lengths=lens,
            )
    return result


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
        *,
        target_worlds: int | None = None,
        system_id: str | None = None,
    ) -> list[pyspiel.State]:
        """Samples infostate worlds ranked by soft HCP/suit-length constraint fit."""
        observer = Seat(observer)
        if state.current_phase() == 0:
            raise ValueError("Cannot sample worlds during incomplete deal phase")

        desired = self.num_worlds if target_worlds is None else max(1, int(target_worlds))
        visible_dummy = _visible_dummy_seat(state)
        l1_constraints = strategy.seat_constraints if strategy is not None else None
        symbolic: dict[Seat, SeatConstraint] = (
            infer_symbolic_auction_constraints(state, observer, system_id=system_id)
            if (state.current_phase() == 2 and system_id is not None)
            else {}
        )
        symbolic_items = sorted(
            (
                (seat, c)
                for seat, c in symbolic.items()
                if seat != observer and seat != visible_dummy
            ),
            key=lambda item: int(item[0]),
        )
        l1_items: list[tuple[Seat, SeatConstraint]] = []
        if l1_constraints:
            for seat_key, c in sorted(l1_constraints.items(), key=lambda item: int(Seat(item[0]))):
                target = Seat(seat_key)
                if target != observer and target != visible_dummy:
                    l1_items.append((target, c))

        if not symbolic_items and not l1_items:
            return [
                state.resample_from_infostate(int(observer), self._rng.random)
                for _ in range(desired)
            ]

        is_play = state.current_phase() == 2
        if is_play:
            if visible_dummy is None:
                max_attempts = max(desired * self.max_rejection_factor, min(desired * 64, 2400))
            else:
                max_attempts = max(desired * self.max_rejection_factor, min(desired * 48, 1200))
        else:
            max_attempts = desired * self.max_rejection_factor

        # Weight verified public-auction symbolic constraints strictly heavier than noisy L1
        # estimates so an overtight L1 HCP guess (e.g. guessing 19-22 when only 14 remain)
        # can NEVER override a known suit length (e.g. 1S response = 4+ spades) or collapse
        # all worlds into a single degenerate corner!
        l1_weight = 0.35 if symbolic_items else 1.0
        exact: list[pyspiel.State] = []
        near_matches: list[tuple[float, int, pyspiel.State]] = []

        for draw_idx in range(max_attempts):
            world = state.resample_from_infostate(int(observer), self._rng.random)
            sym_pen = sum(
                _constraint_penalty(_dealt_hand(world, seat), constraint)
                for seat, constraint in symbolic_items
            )
            l1_pen = sum(
                _constraint_penalty(_dealt_hand(world, seat), constraint)
                for seat, constraint in l1_items
            )
            if sym_pen == 0.0 and l1_pen == 0.0:
                exact.append(world)
                if len(exact) >= desired:
                    return exact
            else:
                # Combined Bayesian soft energy + mild random tie-break/temperature to preserve
                # realistic world diversity across near-feasible deals instead of mode collapse.
                energy = (2.0 * sym_pen) + (l1_weight * l1_pen)
                near_matches.append((energy, draw_idx, world))

        # Boltzmann-style soft ordering: add bounded Gumbel noise only when ranking near-matches
        # so exact/strong symbolic worlds still dominate without pinning 16/16 identical hands.
        temp = 0.6 if symbolic_items else 0.0
        if temp > 0.0 and near_matches:
            def _soft_key(item: tuple[float, int, pyspiel.State]) -> float:
                u = max(1e-9, min(1.0 - 1e-9, self._rng.random()))
                gumbel = -math.log(-math.log(u))
                return item[0] - temp * gumbel

            scored_near = [(_soft_key(it), it[1], it[2]) for it in near_matches]
            scored_near.sort(key=lambda item: (item[0], item[1]))
            needed = desired - len(exact)
            exact.extend(world for _k, _idx, world in scored_near[:needed])
        else:
            near_matches.sort(key=lambda item: (item[0], item[1]))
            needed = desired - len(exact)
            exact.extend(world for _p, _idx, world in near_matches[:needed])
        while len(exact) < desired:
            exact.append(
                state.resample_from_infostate(int(observer), self._rng.random)
            )
        return exact

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
        *,
        system_id: str | None = None,
        target_worlds: int | None = None,
    ) -> list[CandidateOption]:
        observer = Seat(observer)
        if len(candidates) <= 1:
            return list(candidates)

        is_play_opening_lead = (
            state.current_phase() == 2
            and _visible_dummy_seat(state) is None
            and not any(c.action >= BIDDING_ACTION_BASE for c in candidates)
        )
        effective_worlds = target_worlds
        if effective_worlds is None and is_play_opening_lead and self.num_worlds >= 4:
            effective_worlds = min(self.num_worlds * 3, 48)

        if target_worlds is None and system_id is None and effective_worlds is None:
            worlds = self.sample_consistent_worlds(state, observer, strategy)
        else:
            worlds = self.sample_consistent_worlds(
                state,
                observer,
                strategy,
                target_worlds=effective_worlds,
                system_id=system_id,
            )
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

    def unique_viable_play_action(
        self,
        candidates: list[CandidateOption],
    ) -> int | None:
        """Return the sole legal action surviving both make_prob & EV blunder gates, or None."""
        if len(candidates) <= 1:
            return candidates[0].action if candidates else None
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
        if len(viable) == 1:
            return viable[0].action
        if self.max_ev_drop is not None:
            finite_evs = [float(c.baseline_ev) for c in viable if math.isfinite(c.baseline_ev)]
            if finite_evs:
                ev_cutoff = max(finite_evs) - self.max_ev_drop - 1e-9
                gated = [
                    c
                    for c in viable
                    if math.isfinite(c.baseline_ev) and float(c.baseline_ev) >= ev_cutoff
                ]
                if len(gated) == 1:
                    return gated[0].action
        return None

    def select_guarded_action(
        self,
        candidates: list[CandidateOption],
        decision: TacticalDecision | None = None,
        *,
        phase: Literal["play", "auction"] = "play",
        preferred_action: int | None = None,
        auction_tie_margin: float = 0.08,
        play_preferred_ev_margin: float | None = None,
        play_preferred_mp_margin: float | None = None,
        play_ev_weight: float = 0.45,
        play_ev_scale: float = 1.0,
        partner_winning_trick_card: int | None = None,
        overtake_ev_threshold: float = 15.0,
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
            def _norm_score(cand: CandidateOption) -> float:
                raw = decision.option_scores.get(cand.label)
                if raw is None or not math.isfinite(raw):
                    return 0.0
                return min(1.0, max(0.0, float(raw)))

            if preferred_action is not None and auction_tie_margin >= 0.0:
                best_score = max(_norm_score(c) for c in candidates)
                pref = next((c for c in candidates if c.action == preferred_action), None)
                if (
                    pref is not None
                    and pref.label in decision.option_scores
                    and math.isfinite(decision.option_scores[pref.label])
                    and _norm_score(pref) >= best_score - auction_tie_margin - 1e-12
                ):
                    return pref.action

            return max(candidates, key=lambda c: (_norm_score(c), _safe_ev(c))).action

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

        # Fourth-seat partner/dummy-winning guard: avoid burning a higher honor on top
        # of partner's already-winning trick unless DDS shows a genuine unblock/entry gain.
        if partner_winning_trick_card is not None and 0 <= partner_winning_trick_card < NUM_CARDS:
            win_suit = card_suit(partner_winning_trick_card)
            win_rank = card_rank(partner_winning_trick_card)
            under_cards = [
                c
                for c in viable
                if 0 <= c.action < NUM_CARDS
                and card_suit(c.action) == win_suit
                and card_rank(c.action) < win_rank
                and math.isfinite(c.baseline_ev)
            ]
            if under_cards:
                best_under_ev = max(float(c.baseline_ev) for c in under_cards)
                non_wasteful = [
                    c
                    for c in viable
                    if not (
                        0 <= c.action < NUM_CARDS
                        and card_suit(c.action) == win_suit
                        and card_rank(c.action) > win_rank
                        and math.isfinite(c.baseline_ev)
                        and float(c.baseline_ev) <= best_under_ev + overtake_ev_threshold + 1e-9
                    )
                ]
                if non_wasteful:
                    viable = non_wasteful

        # Opening-lead near-tie directive: when Layer-1 recommends a specific legal lead
        # inside the tight constrained DDS margin (EV and optional make_prob gate),
        # honor its single-dummy lead theory.
        if preferred_action is not None and play_preferred_ev_margin is not None:
            pref = next((c for c in viable if c.action == preferred_action), None)
            finite_v = [float(c.baseline_ev) for c in viable if math.isfinite(c.baseline_ev)]
            mp_ok = True
            if play_preferred_mp_margin is not None and pref is not None:
                finite_mp = [
                    float(c.make_prob)
                    for c in viable
                    if c.make_prob is not None and math.isfinite(c.make_prob)
                ]
                if finite_mp and pref.make_prob is not None and math.isfinite(pref.make_prob):
                    mp_ok = float(pref.make_prob) >= max(finite_mp) - play_preferred_mp_margin - 1e-12
            if (
                pref is not None
                and mp_ok
                and finite_v
                and math.isfinite(pref.baseline_ev)
                and float(pref.baseline_ev) >= max(finite_v) - play_preferred_ev_margin - 1e-9
            ):
                return pref.action

        finite_evs = [float(c.baseline_ev) for c in viable if math.isfinite(c.baseline_ev)]
        if finite_evs:
            ev_min = min(finite_evs)
            ev_max = max(finite_evs)
            denom = max(ev_max - ev_min, max(1.0, float(play_ev_scale)))
        else:
            ev_min = 0.0
            denom = max(1.0, float(play_ev_scale))

        w_ev = min(1.0, max(0.0, float(play_ev_weight)))
        w_llm = 1.0 - w_ev

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
            return w_ev * norm_ev + w_llm * llm_score

        return max(viable, key=_blended_score).action
