from __future__ import annotations

from collections import defaultdict

from tenace.domain.types import (
    DBL_ACTION,
    DENOMINATION_CHARS,
    FIRST_CONTRACT_BID_ACTION,
    NUM_CARDS,
    RANK_CHARS,
    RDBL_ACTION,
    SUIT_CHARS,
    Seat,
    StrategyState,
    Suit,
    action_label,
    card_rank,
    card_suit,
)
from tenace.layer1_strategist.conventions import SAYC_CONVENTION_CARD, ConventionCard
from tenace.layer3_rules.tracker import BoardFactSnapshot, trick_winner
from tenace.models.clients import StructuredModelClient

__all__ = [
    "DEFAULT_L1_AUCTION_DIRECTIVE",
    "StrategistEngine",
    "build_strategist_system_prompt",
    "format_auction_sequence",
    "format_cards_by_suit",
    "format_contract_summary",
    "format_known_voids",
    "format_play_state",
    "format_vulnerability",
]

DEFAULT_L1_AUCTION_DIRECTIVE = (
    "During AUCTION: Start `plan_summary` by explicitly stating the single recommended call "
    "for the LIVE turn (e.g., 'Bid 2NT...' or 'Raise to 4H...' or 'Pass...'). Follow the "
    "exact Partnership System rules above for opening thresholds, artificial bids, weak/strong "
    "openings, and slam/keycard asking responses — do NOT pass qualifying opening hands "
    "merely for a singleton honor or weak suit. Always honor forcing partnership bids and "
    "relays defined by the active system, and never drop a forcing call. Factor in Duplicate "
    "IMP scoring and vulnerability: push/accept close game invites (~24-25+ combined pts with "
    "a long suit or good honors) especially when VULNERABLE (+500 game bonus only needs ~37% odds!)."
)


def build_strategist_system_prompt(
    convention_card: ConventionCard,
    phase: str = "auction",
) -> str:
    """Render the exact Layer-1 Strategist system prompt for `convention_card` and `phase`."""
    detailed_section = ""
    if phase == "auction" and convention_card.detailed_rules:
        detailed_section = (
            f"Detailed Reference Guide:\n{convention_card.detailed_rules}\n\n"
        )
    calibration_section = ""
    if convention_card.calibration_notes:
        calibration_section = (
            f"Calibration Notes (Active Hill-Climb Refinements):\n"
            f"{convention_card.calibration_notes}\n\n"
        )
    directive = (
        convention_card.l1_auction_directive
        if convention_card.l1_auction_directive
        else DEFAULT_L1_AUCTION_DIRECTIVE
    )
    play_guidance = (
        "During PLAY (Opening Lead & Card Play):\n"
        "- On the OPENING LEAD (dummy not yet revealed): start `plan_summary` with your single exact "
        "recommended card token (e.g., 'Lead H:A ...' or 'Lead D:J ...').\n"
        "- Solid AK(x) holding (BOTH Ace AND King!): against 3NT or high-level contracts, prioritize "
        "leading A (or K) from AK(x) FIRST to inspect dummy while retaining the lead — especially when "
        "declarer jumped directly to 3NT stopping your own weak suit (e.g., J98xxxx)! Conversely, "
        "avoid leading an unsupported bare Ace (A-x-x without the King) against a 4/5-level suit "
        "contract unless partner bid it or you urgently need a quick ruff/cash-out.\n"
        "- Sequence leads: from a true touching 3-card honor sequence (KQJ/QJT/JT9), lead the top "
        "honor. From an interior sequence (KJT(x), AJT(x), or QT9(x)), lead the TOP OF THE INTERIOR "
        "SEQUENCE (`J` from KJT7 or AJT, `T` from QT9x) — NEVER lead the `K` from KJT7! From a "
        "ragged 5-card suit missing the 9 (e.g., JT863), lead 4th-best low rather than the Jack.\n"
        "- Lead partner's bid/overcalled suit in competitive auctions unless holding a solid AK(x) "
        "or clear singleton ruff.\n"
        "- After winning Tricks 1-2 on defense (e.g., AK), check for a singleton shift to score a "
        "trump ruff whenever you still hold trump control or partner has an outside entry.\n"
        "- NEVER overtake partner's or dummy's already-winning trick with a higher honor unless "
        "unblocking or creating a mandatory entry to the opposite hand.\n"
        "- Populate accurate `seat_constraints` (`min_hcp`, `max_hcp`, `min_suit_lengths`) for ALL "
        "hidden seats using the public auction so the Layer-3 DDS sampler generates realistic hands.\n"
        if phase == "play"
        else f"{directive}\n"
    )
    return (
        "You are Tenace Layer 1 Strategist, an expert Contract Bridge planner.\n"
        f"Partnership System: {convention_card.name}\n"
        f"Convention Summary: {convention_card.summary_rules}\n\n"
        f"{detailed_section}"
        f"{calibration_section}"
        "Synthesize a coherent multi-trick or multi-round StrategyState based ONLY on "
        "legal public information and your own visible cards.\n"
        f"{play_guidance}"
        "Constraint syntax rules for `seat_constraints`:\n"
        "- Outer keys and inner `seat` values must use seat names: "
        '"NORTH", "EAST", "SOUTH", or "WEST".\n'
        "- Suit keys in `min_suit_lengths` and `max_suit_lengths` must use suit names: "
        '"CLUBS", "DIAMONDS", "HEARTS", or "SPADES".\n'
        "- `likely_cards` accepts canonical card strings like \"S:A\" or \"H:K\" (or omit/empty list `[]`).\n"
        "- Ensure `min_hcp <= max_hcp` (0..37) and `min_suit_lengths[s] <= max_suit_lengths[s]` (0..13)."
    )

_SUIT_DISPLAY_ORDER = (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)


def format_cards_by_suit(cards: list[int]) -> str:
    if not cards:
        return "(none)"
    if any(isinstance(c, bool) or c < 0 or c >= NUM_CARDS for c in cards):
        raise ValueError("All cards must be valid card IDs in [0, 51]")

    by_suit: dict[Suit, list[int]] = defaultdict(list)
    for card in set(cards):
        by_suit[card_suit(card)].append(card)

    parts: list[str] = []
    for suit in _SUIT_DISPLAY_ORDER:
        suit_char = SUIT_CHARS[int(suit)]
        suit_cards = sorted(by_suit.get(suit, []), key=card_rank, reverse=True)
        if suit_cards:
            ranks_str = " ".join(RANK_CHARS[card_rank(c)] for c in suit_cards)
            parts.append(f"{suit_char}: {ranks_str}")
        else:
            parts.append(f"{suit_char}: none")
    return " | ".join(parts)


def format_contract_summary(facts: BoardFactSnapshot) -> str:
    if facts.contract_level <= 0 or facts.trump_denom is None or facts.declarer is None:
        return "(none)"
    denom = DENOMINATION_CHARS[facts.trump_denom]
    doubled_suffix = ""
    for _, call in reversed(facts.auction_calls):
        if call == RDBL_ACTION:
            doubled_suffix = "XX"
            break
        if call == DBL_ACTION:
            doubled_suffix = "X"
            break
        if call >= FIRST_CONTRACT_BID_ACTION:
            break
    return f"{facts.contract_level}{denom}{doubled_suffix} by {facts.declarer.name}"


def format_auction_sequence(auction_calls: list[tuple[Seat, int]]) -> str:
    if not auction_calls:
        return "(no calls yet)"
    return " -> ".join(f"{seat.name}:{action_label(action)}" for seat, action in auction_calls)


def format_known_voids(known_voids: dict[Seat, set[Suit]]) -> str:
    entries: list[str] = []
    for seat in Seat:
        voids = sorted(known_voids.get(seat, set()), key=lambda s: int(s), reverse=True)
        if voids:
            suit_names = ", ".join(s.name for s in voids)
            entries.append(f"{seat.name}: [{suit_names}]")
    return "; ".join(entries) if entries else "(none)"


def format_vulnerability(facts: BoardFactSnapshot) -> str:
    """Return side-relative vulnerability & Duplicate IMP scoring header."""
    ns_v, ew_v = facts.ns_vulnerable, facts.ew_vulnerable
    if ns_v and ew_v:
        label = "Both (All)"
    elif ns_v:
        label = "N/S only"
    elif ew_v:
        label = "E/W only"
    else:
        label = "None (Neither)"
    obs_is_ns = facts.observer.partnership == 0
    us_vul = ns_v if obs_is_ns else ew_v
    them_vul = ew_v if obs_is_ns else ns_v
    us_str = "VULNERABLE" if us_vul else "Non-vul"
    them_str = "VULNERABLE" if them_vul else "Non-vul"
    return f"Scoring: Duplicate IMPs | Vul: {label} (Us: {us_str}, Opponents: {them_str})"


def format_play_state(facts: BoardFactSnapshot) -> tuple[str, str, str]:
    """Return `(tricks_won_line, completed_tricks_line, current_trick_line)` for play prompts."""
    if facts.declarer is None:
        return "(none)", "(none)", "(none)"

    decl_side_label = "NS" if facts.declarer.partnership == 0 else "EW"
    def_side_label = "EW" if facts.declarer.partnership == 0 else "NS"
    need = 6 + facts.contract_level if facts.contract_level > 0 else 0

    leader = Seat((int(facts.declarer) + 1) % 4)
    decl_tricks = 0
    def_tricks = 0
    completed_parts: list[str] = []
    played = facts.played_cards
    num_full = len(played) // 4

    for t_idx in range(num_full):
        cards = played[t_idx * 4 : (t_idx + 1) * 4]
        plays = [
            (Seat((int(leader) + offset) % 4), card)
            for offset, card in enumerate(cards)
        ]
        winner = trick_winner(plays, facts.trump_denom)
        if winner.partnership == facts.declarer.partnership:
            decl_tricks += 1
        else:
            def_tricks += 1
        cards_txt = " ".join(f"{s.name[0]}:{action_label(c)}" for s, c in plays)
        completed_parts.append(f"T{t_idx + 1}: {cards_txt} -> {winner.name[0]}")
        leader = winner

    in_trick = played[num_full * 4 :]
    trick_no = num_full + 1
    if in_trick:
        partial_plays = [
            (Seat((int(leader) + offset) % 4), card)
            for offset, card in enumerate(in_trick)
        ]
        led_suit = card_suit(in_trick[0])
        winning_seat = trick_winner(partial_plays, facts.trump_denom)
        winning_card = next(c for s, c in partial_plays if s == winning_seat)
        cards_txt = " ".join(f"{s.name[0]}:{action_label(c)}" for s, c in partial_plays)
        current_line = (
            f"Trick {trick_no} (in progress): {cards_txt} "
            f"[led: {led_suit.name}, currently winning: {winning_seat.name} {action_label(winning_card)}]"
        )
    else:
        current_line = f"Trick {trick_no} (new trick — {leader.name} to lead)"

    need_note = f" (target {need} to make)" if need > 0 else ""
    tricks_won_line = (
        f"Declarer ({decl_side_label}): {decl_tricks} | Defense ({def_side_label}): {def_tricks}{need_note}"
    )
    completed_line = " | ".join(completed_parts) if completed_parts else "(none yet)"
    return tricks_won_line, completed_line, current_line


class StrategistEngine:
    def __init__(
        self,
        client: StructuredModelClient,
        convention_card: ConventionCard = SAYC_CONVENTION_CARD,
    ) -> None:
        self.client = client
        self.convention_card = convention_card

    def should_replan(
        self,
        facts: BoardFactSnapshot,
        current_strategy: StrategyState | None,
        dummy_already_seen: bool,
    ) -> bool:
        if facts.phase == "terminal":
            raise ValueError("Cannot plan on terminal board")
        if current_strategy is None:
            return True
        if current_strategy.phase != facts.phase:
            return True
        if (
            facts.phase == "play"
            and facts.dummy_seat is not None
            and not dummy_already_seen
        ):
            return True
        if bool(facts.surprise_events):
            return True
        return False

    def update_strategy(
        self,
        facts: BoardFactSnapshot,
        current_strategy: StrategyState | None,
        trigger_reason: str = "",
    ) -> StrategyState:
        if facts.phase == "terminal":
            raise ValueError("Cannot plan on terminal board")

        system_prompt = build_strategist_system_prompt(
            self.convention_card,
            phase=facts.phase,
        )

        dummy_line = (
            f"{facts.dummy_seat.name} — {format_cards_by_suit(facts.dummy_cards)}"
            if facts.dummy_seat is not None
            else "(not yet revealed)"
        )
        played_line = (
            " ".join(action_label(c) for c in facts.played_cards)
            if facts.played_cards
            else "(none)"
        )
        surprise_line = (
            "; ".join(facts.surprise_events) if facts.surprise_events else "(none)"
        )
        reason_line = trigger_reason.strip() if trigger_reason.strip() else "(unspecified)"
        prev_strategy_json = (
            current_strategy.model_dump_json() if current_strategy is not None else "(none)"
        )

        lines = [
            f"Seat: {facts.observer.name} (Partner: {facts.observer.partner.name})",
            f"Phase: {facts.phase}",
            format_vulnerability(facts),
            f"Own Hand ({facts.own_hcp} HCP): {format_cards_by_suit(facts.own_cards)}",
            f"Dummy: {dummy_line}",
        ]
        if facts.phase == "play":
            tricks_won_line, completed_line, current_line = format_play_state(facts)
            lines.extend(
                [
                    f"Contract: {format_contract_summary(facts)}",
                    f"Tricks Won: {tricks_won_line}",
                    f"Completed Tricks: {completed_line}",
                    f"Current Trick: {current_line}",
                ]
            )
        lines.extend(
            [
                f"Auction Sequence: {format_auction_sequence(facts.auction_calls)}",
                f"Played Cards: {played_line}",
                f"Known Voids: {format_known_voids(facts.known_voids)}",
                f"Replan Trigger: {reason_line}",
                f"Surprise Events: {surprise_line}",
                f"Previous StrategyState JSON: {prev_strategy_json}",
            ]
        )
        user_prompt = "\n".join(lines)

        raw_strategy = self.client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=StrategyState,
        )
        next_revision = 1 if current_strategy is None else current_strategy.revision + 1
        return raw_strategy.model_copy(
            update={
                "phase": facts.phase,
                "revision": next_revision,
            }
        )
