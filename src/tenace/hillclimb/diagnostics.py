from __future__ import annotations

import re

import pyspiel

from tenace.domain.types import (
    NUM_CARDS,
    Seat,
    Suit,
    action_label,
    card_rank,
    card_suit,
    hand_hcp,
)
from tenace.harness.duplicate import wbf_board_params
from tenace.harness.scoring import score_diff_to_imps
from tenace.hillclimb.models import (
    AuctionDivergence,
    BoardDealSnapshot,
    BoardTelemetryRecord,
    MatchMode,
    RoomSnapshot,
    RunKind,
)
from tenace.layer1_strategist.strategist import format_cards_by_suit

_CONTRACT_RE = re.compile(
    r"^(?P<level>[1-7])(?P<strain>NT|[CDHS])(?P<dbl>XX|X)?\s+by\s+(?P<decl>NORTH|EAST|SOUTH|WEST)$",
    re.IGNORECASE,
)

__all__ = [
    "build_board_record",
    "classify_swing",
    "dealer_vul_labels",
    "extract_full_deal_hands",
    "find_first_auction_divergence",
    "parse_auction_tokens",
    "parse_contract",
]


def _sort_bridge_cards(cards: list[int]) -> list[int]:
    return sorted(cards, key=lambda c: (-int(card_suit(c)), -card_rank(c)))


def dealer_vul_labels(board_number: int) -> tuple[str, str]:
    """Return `(DEALER_NAME, vul_label)` for WBF board rotation."""
    dealer, dealer_vul, non_dealer_vul = wbf_board_params(board_number)
    is_ew = (dealer & 1) == 1
    ns_v = non_dealer_vul if is_ew else dealer_vul
    ew_v = dealer_vul if is_ew else non_dealer_vul
    if ns_v and ew_v:
        vul = "Both"
    elif ns_v:
        vul = "N/S"
    elif ew_v:
        vul = "E/W"
    else:
        vul = "None"
    return Seat(dealer).name, vul


def extract_full_deal_hands(
    state: pyspiel.State,
    *,
    board_number: int,
    seed: int,
) -> BoardDealSnapshot:
    """Extract full 52-card deal snapshot (`hands`, `cards_by_seat`, `hand_hcp`, `shapes`)."""
    del board_number, seed
    history = state.full_history()[:NUM_CARDS]
    if len(history) < NUM_CARDS:
        raise ValueError(
            f"State must contain {NUM_CARDS} dealt cards, found {len(history)}"
        )
    by_seat: dict[Seat, list[int]] = {seat: [] for seat in Seat}
    for idx, entry in enumerate(history):
        by_seat[Seat(idx % 4)].append(entry.action)

    hands_str: dict[str, str] = {}
    cards_by_seat: dict[str, list[str]] = {}
    hcp_map: dict[str, int] = {}
    shapes: dict[str, str] = {}
    for seat in (Seat.NORTH, Seat.EAST, Seat.SOUTH, Seat.WEST):
        cards = _sort_bridge_cards(by_seat[seat])
        hands_str[seat.name] = format_cards_by_suit(cards)
        cards_by_seat[seat.name] = [action_label(c) for c in cards]
        hcp_map[seat.name] = hand_hcp(cards)
        counts = [
            sum(1 for c in cards if card_suit(c) == s)
            for s in (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)
        ]
        shapes[seat.name] = "-".join(str(n) for n in counts)
    return BoardDealSnapshot(
        hands=hands_str,
        cards_by_seat=cards_by_seat,
        hand_hcp=hcp_map,
        shapes=shapes,
    )


def parse_auction_tokens(auction_str: str) -> list[tuple[str, str]]:
    """Parse `'NORTH:1H -> EAST:Pass -> ...'` into `[(seat, call), ...]`."""
    text = auction_str.strip()
    if not text or text == "(no calls yet)":
        return []
    tokens: list[tuple[str, str]] = []
    for raw_part in text.split("->"):
        part = raw_part.strip()
        if ":" not in part:
            continue
        seat, call = part.split(":", 1)
        tokens.append((seat.strip().upper(), call.strip()))
    return tokens


def find_first_auction_divergence(
    reference_auction: str,
    challenger_auction: str,
) -> AuctionDivergence | None:
    """Return the exact first call where the two auctions diverge, or `None` if identical."""
    ref_toks = parse_auction_tokens(reference_auction)
    chal_toks = parse_auction_tokens(challenger_auction)
    max_len = max(len(ref_toks), len(chal_toks))
    for idx in range(max_len):
        ref_pair = ref_toks[idx] if idx < len(ref_toks) else ("", "<END>")
        chal_pair = chal_toks[idx] if idx < len(chal_toks) else ("", "<END>")
        if ref_pair != chal_pair:
            seat = ref_pair[0] or chal_pair[0] or "UNKNOWN"
            return AuctionDivergence(
                call_index=idx,
                seat=seat,
                reference_call=ref_pair[1],
                challenger_call=chal_pair[1],
            )
    return None


def parse_contract(contract_str: str) -> tuple[int, str, str, str] | None:
    """Return `(level, strain, doubled_suffix, declarer)` or `None` for `Passed Out`."""
    m = _CONTRACT_RE.match(contract_str.strip())
    if m is None:
        return None
    return (
        int(m.group("level")),
        m.group("strain").upper(),
        (m.group("dbl") or "").upper(),
        m.group("decl").upper(),
    )


def _is_game_contract(parsed: tuple[int, str, str, str] | None) -> bool:
    if parsed is None:
        return False
    level, strain, _dbl, _decl = parsed
    if strain == "NT":
        return level >= 3
    if strain in ("H", "S"):
        return level >= 4
    return level >= 5


def classify_swing(
    ref_contract: str,
    chal_contract: str,
    imps_challenger: int,
    ref_ns_score: float,
    chal_ns_score: float,
) -> str:
    """Deterministic coarse swing category for clustering improvement opportunities."""
    del ref_ns_score, chal_ns_score
    if imps_challenger == 0:
        return "push"

    ref_norm = ref_contract.strip()
    chal_norm = chal_contract.strip()
    if ref_norm.upper() == chal_norm.upper():
        return "card_play_gain" if imps_challenger > 0 else "card_play_loss"

    ref_p = parse_contract(ref_norm)
    chal_p = parse_contract(chal_norm)
    if (ref_p is None) != (chal_p is None):
        return "auction_passed_out_swing"

    if (ref_p is not None and ref_p[0] >= 6) or (chal_p is not None and chal_p[0] >= 6):
        return "auction_slam_swing"

    if (ref_p is not None and ref_p[2] in ("X", "XX")) or (
        chal_p is not None and chal_p[2] in ("X", "XX")
    ):
        return "auction_doubled_penalty"

    if _is_game_contract(ref_p) != _is_game_contract(chal_p):
        return "auction_game_swing"
    if _is_game_contract(ref_p) and _is_game_contract(chal_p):
        if ref_p is not None and chal_p is not None and ref_p[1] != chal_p[1]:
            return "auction_game_swing"

    return "auction_competitive_partscore"


def build_board_record(
    *,
    run_kind: RunKind,
    batch_id: str,
    prompt_version: str,
    match_mode: MatchMode,
    tournament_seed: int,
    board_number: int,
    deal: BoardDealSnapshot,
    table_a: RoomSnapshot,
    table_b: RoomSnapshot,
    reference_cache_hit: bool = False,
    telemetry: dict[str, int] | None = None,
) -> BoardTelemetryRecord:
    """Build a scored & diagnosed `BoardTelemetryRecord`.

    In `controlled` mode: `table_a` = `4× reference`, `table_b` = `challenger N/S vs reference E/W`,
    so challenger score delta is `table_b.ns_score - table_a.ns_score`.
    In `duplicate` mode: `table_a` = Open Room (`challenger N/S`), `table_b` = Closed Room (`reference N/S`),
    so challenger score delta is `table_a.ns_score - table_b.ns_score`.
    """
    dealer, vul = dealer_vul_labels(board_number)
    if match_mode == "controlled":
        ref_room, chal_room = table_a, table_b
        diff = float(chal_room.ns_score) - float(ref_room.ns_score)
    else:
        chal_room, ref_room = table_a, table_b
        diff = float(chal_room.ns_score) - float(ref_room.ns_score)

    imps = score_diff_to_imps(diff)
    divergence = find_first_auction_divergence(ref_room.auction, chal_room.auction)
    auction_exact = divergence is None
    contract_match = (
        ref_room.contract.strip().upper() == chal_room.contract.strip().upper()
    )
    swing_cat = classify_swing(
        ref_room.contract,
        chal_room.contract,
        imps,
        ref_room.ns_score,
        chal_room.ns_score,
    )
    return BoardTelemetryRecord(
        run_kind=run_kind,
        batch_id=batch_id,
        prompt_version=prompt_version,
        match_mode=match_mode,
        tournament_seed=tournament_seed,
        board_number=board_number,
        dealer=dealer,
        vul=vul,
        deal=deal,
        table_a=table_a,
        table_b=table_b,
        score_diff=diff,
        imps_challenger=imps,
        auction_exact_match=auction_exact,
        contract_match=contract_match,
        first_divergence=divergence,
        swing_category=swing_cat,
        reference_cache_hit=reference_cache_hit,
        telemetry=dict(telemetry or {}),
    )
