from __future__ import annotations

from collections import defaultdict
import re
from typing import Literal

import pyspiel
from pydantic import BaseModel, Field

from tenace.domain.types import (
    BIDDING_ACTION_BASE,
    FIRST_CONTRACT_BID_ACTION,
    MAX_ACTION,
    NUM_CARDS,
    ActionId,
    CandidateOption,
    CardId,
    Seat,
    Suit,
    action_label,
    card_id,
    card_rank,
    card_suit,
    hand_hcp,
)

__all__ = [
    "BoardFactSnapshot",
    "RuleFactTracker",
    "collapse_equivalent_actions",
    "trick_winner",
]

_PLAY_OWN_SLICE = slice(25, 25 + NUM_CARDS)
_PLAY_DUMMY_SLICE = slice(25 + NUM_CARDS, 25 + 2 * NUM_CARDS)
_AUCTION_OWN_SLICE = slice(432, 432 + NUM_CARDS)
_OBS_VUL_RE = re.compile(r"^Vul:\s*(None|N/S|E/W|All)\s*$", re.MULTILINE)


def collapse_equivalent_actions(
    legal_actions: list[int],
    played_cards: set[int] | None = None,
    *,
    preserve_signals: bool = False,
) -> list[CandidateOption]:
    if played_cards is not None and any(c < 0 or c >= NUM_CARDS for c in played_cards):
        raise ValueError("played_cards entries must be valid card IDs in [0, 51]")

    if not legal_actions:
        return []

    if any(a < 0 or a > MAX_ACTION for a in legal_actions):
        raise ValueError("Action out of valid OpenSpiel bridge range [0, 89]")

    has_bidding = any(a >= BIDDING_ACTION_BASE for a in legal_actions)
    has_cards = any(a < BIDDING_ACTION_BASE for a in legal_actions)
    if has_bidding and has_cards:
        raise ValueError("Cannot mix card and bidding actions in legal_actions")

    if has_bidding:
        return [
            CandidateOption(
                action=a,
                label=action_label(a),
                equivalent_actions=[a],
            )
            for a in sorted(set(legal_actions))
        ]

    played = played_cards or set()
    legal_set = set(legal_actions)
    by_suit: dict[Suit, list[int]] = defaultdict(list)
    for card in sorted(legal_set):
        by_suit[card_suit(card)].append(card)

    options: list[CandidateOption] = []
    for suit in sorted(by_suit):
        suit_cards = sorted(by_suit[suit], key=card_rank)
        groups: list[list[int]] = [[suit_cards[0]]]
        for card in suit_cards[1:]:
            prev_card = groups[-1][-1]
            r_prev = card_rank(prev_card)
            r_curr = card_rank(card)
            bridged = all(
                card_id(suit, r) in played or card_id(suit, r) in legal_set
                for r in range(r_prev + 1, r_curr)
            )
            if bridged:
                groups[-1].append(card)
            else:
                groups.append([card])

        for grp in groups:
            ordered = sorted(grp, reverse=True)
            # When defending with signal preservation enabled, split touching spot groups
            # (top card <= Ten, rank <= 8) into [highest, lowest] so defenders can signal
            # attitude (high encourages / low discourages) and count (high-low = even).
            if preserve_signals and len(ordered) >= 2 and card_rank(ordered[0]) <= 8:
                hi = ordered[0]
                lo = ordered[-1]
                options.append(
                    CandidateOption(
                        action=hi,
                        label=action_label(hi),
                        equivalent_actions=ordered[:-1],
                    )
                )
                options.append(
                    CandidateOption(
                        action=lo,
                        label=action_label(lo),
                        equivalent_actions=[lo],
                    )
                )
            else:
                rep = ordered[0]
                options.append(
                    CandidateOption(
                        action=rep,
                        label=action_label(rep),
                        equivalent_actions=ordered,
                    )
                )

    return options


class BoardFactSnapshot(BaseModel):
    observer: Seat
    phase: Literal["auction", "play", "terminal"]
    own_cards: list[CardId]
    own_hcp: int = Field(ge=0, le=37)
    dummy_seat: Seat | None = None
    dummy_cards: list[CardId] = Field(default_factory=list)
    played_cards: list[CardId] = Field(default_factory=list)
    unseen_cards: list[CardId] = Field(default_factory=list)
    auction_calls: list[tuple[Seat, ActionId]] = Field(default_factory=list)
    declarer: Seat | None = None
    trump_denom: int | None = Field(default=None, ge=0, le=4)
    contract_level: int = Field(default=0, ge=0, le=7)
    ns_vulnerable: bool = False
    ew_vulnerable: bool = False
    known_voids: dict[Seat, set[Suit]] = Field(default_factory=dict)
    surprise_events: list[str] = Field(default_factory=list)


def _trick_winner(trick_plays: list[tuple[Seat, int]], trump_denom: int | None) -> Seat:
    led_suit = card_suit(trick_plays[0][1])
    best_seat, best_card = trick_plays[0]
    for seat, card in trick_plays[1:]:
        suit = card_suit(card)
        best_suit = card_suit(best_card)
        is_trump = trump_denom is not None and trump_denom < 4 and int(suit) == trump_denom
        best_is_trump = (
            trump_denom is not None and trump_denom < 4 and int(best_suit) == trump_denom
        )
        if is_trump and not best_is_trump:
            best_seat, best_card = seat, card
        elif is_trump and best_is_trump and card_rank(card) > card_rank(best_card):
            best_seat, best_card = seat, card
        elif not best_is_trump and suit == led_suit and card_rank(card) > card_rank(best_card):
            best_seat, best_card = seat, card
    return best_seat


trick_winner = _trick_winner


def _cards_from_slice(values: list[float]) -> list[int]:
    return [card for card, flag in enumerate(values) if flag > 0.5]


class RuleFactTracker:
    def extract_facts(
        self,
        state: pyspiel.State,
        observer: Seat,
        prev_voids: dict[Seat, set[Suit]] | None = None,
    ) -> BoardFactSnapshot:
        observer = Seat(observer)
        raw_phase = state.current_phase()
        if raw_phase == 0:
            raise ValueError("Cannot extract facts during incomplete deal phase")
        if raw_phase == 1:
            phase: Literal["auction", "play", "terminal"] = "auction"
        elif raw_phase == 2:
            phase = "play"
        else:
            phase = "terminal"

        # CRITICAL FAIR-PLAY INVARIANT: Never access state.full_history()[:52].
        public_history = state.full_history()[NUM_CARDS:]

        auction_calls: list[tuple[Seat, int]] = []
        play_actions: list[int] = []
        last_bid_seat: Seat | None = None
        last_bid_action: int | None = None
        first_denom_bidder: dict[tuple[int, int], Seat] = {}

        for entry in public_history:
            action = entry.action
            if action >= BIDDING_ACTION_BASE:
                seat = Seat(entry.player)
                auction_calls.append((seat, action))
                if FIRST_CONTRACT_BID_ACTION <= action <= MAX_ACTION:
                    denom = (action - FIRST_CONTRACT_BID_ACTION) % 5
                    key = (seat.partnership, denom)
                    if key not in first_denom_bidder:
                        first_denom_bidder[key] = seat
                    last_bid_seat = seat
                    last_bid_action = action
            else:
                play_actions.append(action)

        declarer: Seat | None = None
        dummy_seat: Seat | None = None
        trump_denom: int | None = None
        contract_level = 0

        if raw_phase != 1 and last_bid_seat is not None and last_bid_action is not None:
            offset = last_bid_action - FIRST_CONTRACT_BID_ACTION
            contract_level = 1 + offset // 5
            trump_denom = offset % 5
            declarer = first_denom_bidder[(last_bid_seat.partnership, trump_denom)]
            if len(play_actions) > 0:
                dummy_seat = declarer.partner

        obs = state.observation_tensor(int(observer))
        if len(play_actions) > 0 and declarer is not None:
            own_cards = _cards_from_slice(obs[_PLAY_OWN_SLICE])
            dummy_cards = _cards_from_slice(obs[_PLAY_DUMMY_SLICE])
        else:
            own_cards = _cards_from_slice(obs[_AUCTION_OWN_SLICE])
            dummy_cards = []

        known_voids: dict[Seat, set[Suit]] = {s: set() for s in Seat}
        if declarer is not None and play_actions:
            leader = Seat((int(declarer) + 1) % 4)
            for trick_start in range(0, len(play_actions), 4):
                trick_cards = play_actions[trick_start : trick_start + 4]
                led_suit = card_suit(trick_cards[0])
                trick_plays: list[tuple[Seat, int]] = []
                for idx, card in enumerate(trick_cards):
                    seat_p = Seat((int(leader) + idx) % 4)
                    trick_plays.append((seat_p, card))
                    if card_suit(card) != led_suit:
                        known_voids[seat_p].add(led_suit)
                if len(trick_plays) == 4:
                    leader = _trick_winner(trick_plays, trump_denom)

        surprise_events: list[str] = []
        if prev_voids is not None:
            for s in Seat:
                prev_for_seat = prev_voids.get(s, set())
                new_voids = sorted(known_voids[s] - prev_for_seat)
                for v_suit in new_voids:
                    surprise_events.append(f"{s.name} showed out of {v_suit.name}")

        visible_or_played = set(own_cards) | set(dummy_cards) | set(play_actions)
        unseen_cards = [c for c in range(NUM_CARDS) if c not in visible_or_played]

        obs_str = state.observation_string(int(observer))
        m_vul = _OBS_VUL_RE.search(obs_str)
        vul_tok = m_vul.group(1) if m_vul is not None else "None"
        ns_vul = vul_tok in ("N/S", "All")
        ew_vul = vul_tok in ("E/W", "All")

        return BoardFactSnapshot(
            observer=observer,
            phase=phase,
            own_cards=own_cards,
            own_hcp=hand_hcp(own_cards),
            dummy_seat=dummy_seat,
            dummy_cards=dummy_cards,
            played_cards=play_actions,
            unseen_cards=unseen_cards,
            auction_calls=auction_calls,
            declarer=declarer,
            trump_denom=trump_denom,
            contract_level=contract_level,
            ns_vulnerable=ns_vul,
            ew_vulnerable=ew_vul,
            known_voids=known_voids,
            surprise_events=surprise_events,
        )
