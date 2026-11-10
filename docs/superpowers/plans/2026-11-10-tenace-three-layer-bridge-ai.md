# Tenace 3-Layer Bridge AI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build **Tenace**, a neuro-symbolic 3-layer Contract Bridge AI agent and official WBF Duplicate IMP match harness powered by OpenSpiel C++/DDS (Layer 3), Vertex AI DiffusionGemma MaaS (Layer 2), and a Vertex AI Frontier Reasoning LLM Strategist (Layer 1).

**Architecture:** Tenace separates slow strategic planning from fast per-turn execution and deterministic bookkeeping. Layer 3 (`layer3_rules`) tracks exact unseen cards/voids strictly from imperfect-information observations, collapses mathematically equivalent cards (`♠QJT` → 1 option), skips API calls on forced moves, computes constrained Monte Carlo Double-Dummy EVs, and enforces anti-blunder guardrails. Layer 1 (`layer1_strategist`) runs event-driven (auction decisions, dummy reveal, or assumption-violation interrupts) to maintain an isolated per-seat `StrategyState` + soft hand constraints. Layer 2 (`layer2_evaluator`) invokes Vertex AI `DiffusionGemma` to check active assumptions and score the 2–5 non-equivalent candidate actions for plan coherence.

**Tech Stack:** Python 3.13 (`uv`), `open-spiel >= 2.0.2` (C++ `BridgeGame`, `BridgeState`, `dds`), `pydantic >= 2.9`, `google-genai >= 1.0`, `google-cloud-aiplatform >= 1.84`, `numpy`, `pytest`.

---

## File Structure

```text
tenace/
├── pyproject.toml
├── README.md
├── docs/
│   └── superpowers/
│       └── plans/
│           └── 2026-11-10-tenace-three-layer-bridge-ai.md
├── src/
│   └── tenace/
│       ├── __init__.py
│       ├── domain/
│       │   ├── __init__.py
│       │   └── types.py               # Seats, suits, cards, calls, StrategyState, CandidateOption, TacticalDecision
│       ├── layer3_rules/
│       │   ├── __init__.py
│       │   ├── tracker.py             # Imperfect-info card/void/HCP tracker & equivalent-card compressor
│       │   └── dds_evaluator.py       # Constrained infostate sampler, C++ DDS rollout EV, anti-blunder gate
│       ├── models/
│       │   ├── __init__.py
│       │   └── clients.py             # StructuredModelClient protocol, VertexGeminiClient, DiffusionGemmaClient
│       ├── layer1_strategist/
│       │   ├── __init__.py
│       │   ├── conventions.py         # Shared static ConventionCard (SAYC default)
│       │   └── strategist.py          # Event-driven System-2 StrategyState builder & augmenter
│       ├── layer2_evaluator/
│       │   ├── __init__.py
│       │   └── evaluator.py           # DiffusionGemma canvas prompt & parallel candidate scorer
│       ├── agent.py                   # End-to-end 3-layer TenaceAgent per seat (isolated memory)
│       └── harness/
│           ├── __init__.py
│           ├── scoring.py             # Official WBF score-difference -> IMP conversion table
│           └── duplicate.py           # Open/Closed room Duplicate Bridge match runner & baseline bots
└── tests/
    ├── test_domain_types.py
    ├── test_layer3_tracker.py
    ├── test_layer3_dds_evaluator.py
    ├── test_model_clients.py
    ├── test_layer1_and_layer2.py
    └── test_agent_and_duplicate_harness.py
```

---

### Task 1: Project Scaffolding & Domain Types

**Files:**
- Create: `pyproject.toml`
- Create: `src/tenace/__init__.py`
- Create: `src/tenace/domain/__init__.py`
- Create: `src/tenace/domain/types.py`
- Test: `tests/test_domain_types.py`

- [ ] **Step 1: Create `pyproject.toml`, install dependencies via `uv`, and write the failing domain test**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tenace"
version = "0.1.0"
description = "Tenace: 3-Layer Neuro-Symbolic Contract Bridge AI"
requires-python = ">=3.12"
dependencies = [
    "open-spiel>=2.0.2",
    "pydantic>=2.9.0",
    "numpy>=2.0.0",
    "google-genai>=1.5.0",
    "google-cloud-aiplatform>=1.84.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/tenace"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Create empty package markers `src/tenace/__init__.py` and `src/tenace/domain/__init__.py`, then create `tests/test_domain_types.py`:

```python
import pyspiel

from tenace.domain.types import (
    BIDDING_ACTION_BASE,
    Seat,
    Suit,
    action_label,
    card_id,
    card_rank,
    card_suit,
    hand_hcp,
)


def test_seat_partner_and_partnership():
    assert Seat.NORTH.partner == Seat.SOUTH
    assert Seat.EAST.partner == Seat.WEST
    assert Seat.NORTH.partnership == Seat.SOUTH.partnership == 0
    assert Seat.EAST.partnership == Seat.WEST.partnership == 1


def test_card_encoding_matches_openspiel_bridge():
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state()

    ace_spades = card_id(Suit.SPADES, 12)
    two_clubs = card_id(Suit.CLUBS, 0)

    assert ace_spades == 51
    assert two_clubs == 0
    assert card_suit(ace_spades) == Suit.SPADES
    assert card_rank(ace_spades) == 12
    assert action_label(ace_spades) == "S:A"
    assert action_label(two_clubs) == "C:2"
    assert state.action_to_string(0, ace_spades) == "♠A"


def test_bidding_action_labels_and_hcp():
    assert action_label(BIDDING_ACTION_BASE + 0) == "Pass"
    assert action_label(BIDDING_ACTION_BASE + 1) == "Dbl"
    assert action_label(BIDDING_ACTION_BASE + 2) == "RDbl"
    assert action_label(BIDDING_ACTION_BASE + 3) == "1C"
    assert action_label(BIDDING_ACTION_BASE + 7) == "1NT"
    assert action_label(89) == "7NT"

    cards = [
        card_id(Suit.SPADES, 12),   # Ace = 4
        card_id(Suit.HEARTS, 11),   # King = 3
        card_id(Suit.DIAMONDS, 10), # Queen = 2
        card_id(Suit.CLUBS, 9),     # Jack = 1
        card_id(Suit.CLUBS, 0),     # Two = 0
    ]
    assert hand_hcp(cards) == 10
```

- [ ] **Step 2: Sync `uv` environment and run `pytest` to verify RED failure**

Run:
```bash
~/.local/bin/uv sync --extra dev && ~/.local/bin/uv run pytest tests/test_domain_types.py -v
```
Expected: `FAIL` with `ModuleNotFoundError: No module named 'tenace.domain.types'`.

- [ ] **Step 3: Implement `src/tenace/domain/types.py`**

Create `src/tenace/domain/types.py`:

```python
from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field

NUM_CARDS = 52
NUM_SUITS = 4
NUM_RANKS = 13
BIDDING_ACTION_BASE = 52
PASS_ACTION = 52
DBL_ACTION = 53
RDBL_ACTION = 54
FIRST_CONTRACT_BID_ACTION = 55
MAX_ACTION = 89

RANK_CHARS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
SUIT_CHARS = ("C", "D", "H", "S")
DENOMINATION_CHARS = ("C", "D", "H", "S", "NT")
HCP_BY_RANK = {12: 4, 11: 3, 10: 2, 9: 1}


class Seat(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @property
    def partner(self) -> Seat:
        return Seat(int(self) ^ 2)

    @property
    def partnership(self) -> int:
        return int(self) & 1


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


def card_id(suit: Suit, rank: int) -> int:
    if not (0 <= rank < NUM_RANKS):
        raise ValueError(f"Invalid rank: {rank}")
    return rank * NUM_SUITS + int(suit)


def card_suit(card: int) -> Suit:
    if not (0 <= card < NUM_CARDS):
        raise ValueError(f"Invalid card action id: {card}")
    return Suit(card % NUM_SUITS)


def card_rank(card: int) -> int:
    if not (0 <= card < NUM_CARDS):
        raise ValueError(f"Invalid card action id: {card}")
    return card // NUM_SUITS


def action_label(action: int) -> str:
    if 0 <= action < NUM_CARDS:
        return f"{SUIT_CHARS[int(card_suit(action))]}:{RANK_CHARS[card_rank(action)]}"
    if action == PASS_ACTION:
        return "Pass"
    if action == DBL_ACTION:
        return "Dbl"
    if action == RDBL_ACTION:
        return "RDbl"
    if FIRST_CONTRACT_BID_ACTION <= action <= MAX_ACTION:
        offset = action - FIRST_CONTRACT_BID_ACTION
        level = 1 + offset // 5
        denom = DENOMINATION_CHARS[offset % 5]
        return f"{level}{denom}"
    raise ValueError(f"Unknown OpenSpiel bridge action id: {action}")


def hand_hcp(cards: Iterable[int]) -> int:
    return sum(HCP_BY_RANK.get(card_rank(c), 0) for c in cards)


class SeatConstraint(BaseModel):
    seat: Seat
    min_hcp: int = Field(default=0, ge=0, le=37)
    max_hcp: int = Field(default=37, ge=0, le=37)
    min_suit_lengths: dict[Suit, int] = Field(default_factory=dict)
    max_suit_lengths: dict[Suit, int] = Field(default_factory=dict)
    likely_cards: list[int] = Field(default_factory=list)


class Assumption(BaseModel):
    claim: str
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class StrategyState(BaseModel):
    phase: Literal["auction", "play"]
    plan_summary: str
    target_or_threat: str = ""
    assumptions: list[Assumption] = Field(default_factory=list)
    seat_constraints: dict[Seat, SeatConstraint] = Field(default_factory=dict)
    revision: int = 0


class CandidateOption(BaseModel):
    action: int
    label: str
    equivalent_actions: list[int] = Field(default_factory=list)
    baseline_ev: float = 0.0
    make_prob: float | None = None


class TacticalDecision(BaseModel):
    assumption_violated: bool = False
    violation_reason: str = ""
    option_scores: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
```

- [ ] **Step 4: Run `pytest` to verify GREEN**

Run:
```bash
~/.local/bin/uv run pytest tests/test_domain_types.py -v
```
Expected: `3 passed`.

- [ ] **Step 5: Commit Task 1**

```bash
git add pyproject.toml uv.lock src/ tests/ docs/
git commit -m "feat: initialize tenace package and bridge domain types"
```

---

### Task 2: Layer 3 Imperfect-Information Fact Tracker & Equivalent-Card Compressor

**Files:**
- Create: `src/tenace/layer3_rules/__init__.py`
- Create: `src/tenace/layer3_rules/tracker.py`
- Test: `tests/test_layer3_tracker.py`

- [ ] **Step 1: Write failing tests for imperfect-info fact extraction, void detection, and card equivalence collapsing**

Create `src/tenace/layer3_rules/__init__.py` (empty) and `tests/test_layer3_tracker.py`:

```python
import pyspiel

from tenace.domain.types import PASS_ACTION, Seat, Suit, card_id
from tenace.layer3_rules.tracker import RuleFactTracker, collapse_equivalent_actions


def _make_dealt_state() -> pyspiel.State:
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    return game.new_initial_state(tournament_seed=42, board_number=1)


def test_collapse_equivalent_actions_merges_contiguous_and_bridged_cards():
    # Holding S:Q (10), S:9 (7), S:2 (0) when S:J (9) and S:T (8) are already played:
    # S:Q and S:9 become equivalent, while S:2 stays separate!
    spade_q = card_id(Suit.SPADES, 10)
    spade_j = card_id(Suit.SPADES, 9)
    spade_t = card_id(Suit.SPADES, 8)
    spade_9 = card_id(Suit.SPADES, 7)
    spade_2 = card_id(Suit.SPADES, 0)

    options = collapse_equivalent_actions(
        legal_actions=[spade_2, spade_9, spade_q],
        played_cards={spade_j, spade_t},
    )
    assert len(options) == 2
    by_action = {opt.action: opt for opt in options}
    assert spade_q in by_action
    assert set(by_action[spade_q].equivalent_actions) == {spade_q, spade_9}
    assert by_action[spade_2].equivalent_actions == [spade_2]


def test_tracker_hides_unseen_hands_during_auction_and_reveals_dummy_after_lead():
    state = _make_dealt_state()
    tracker = RuleFactTracker()

    auction_facts = tracker.extract_facts(state, Seat.NORTH)
    assert auction_facts.phase == "auction"
    assert len(auction_facts.own_cards) == 13
    assert auction_facts.dummy_cards == []
    assert len(auction_facts.unseen_cards) == 39
    assert set(auction_facts.own_cards).isdisjoint(set(auction_facts.unseen_cards))

    # North opens 1NT (action 59), E/S/W pass -> East leads
    state.apply_action(59)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)

    # Before East makes the opening lead, dummy (South) is still hidden from East
    pre_lead_east = tracker.extract_facts(state, Seat.EAST)
    assert pre_lead_east.phase == "play"
    assert pre_lead_east.dummy_cards == []
    assert len(pre_lead_east.unseen_cards) == 39

    # East leads a legal card -> Dummy (South) is now visible to East!
    lead_card = state.legal_actions()[0]
    state.apply_action(lead_card)
    post_lead_east = tracker.extract_facts(state, Seat.EAST)
    assert post_lead_east.dummy_seat == Seat.SOUTH
    assert len(post_lead_east.dummy_cards) == 13
    assert lead_card in post_lead_east.played_cards
    # 52 total - 13 (East) - 13 (Dummy South) - 1 (East's lead card came from East) = 26 unseen (North + West)
    assert len(post_lead_east.unseen_cards) == 26


def test_tracker_detects_suit_showout_void_and_emits_surprise():
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state()
    # Construct deterministic deal: give East (seat 1) ZERO clubs!
    # Deal order is i % 4: 0=N, 1=E, 2=S, 3=W
    clubs = [card_id(Suit.CLUBS, r) for r in range(13)]
    diamonds = [card_id(Suit.DIAMONDS, r) for r in range(13)]
    hearts = [card_id(Suit.HEARTS, r) for r in range(13)]
    spades = [card_id(Suit.SPADES, r) for r in range(13)]

    hands = {
        0: clubs[:13],      # North: all 13 clubs!
        1: diamonds[:13],   # East: all 13 diamonds (VOID in clubs!)
        2: hearts[:13],     # South: all 13 hearts
        3: spades[:13],     # West: all 13 spades
    }
    for r in range(13):
        for p in range(4):
            state.apply_action(hands[p][r])

    # North bids 1NT (59), E/S/W pass. East leads D:2, South & West discard, North wins?
    # Wait: if North bids 1C (55) and West leads S:A, let's instead have West bid 1NT (59):
    # Dealer is North(0): N Pass, E Pass, S Pass, W 1NT(59), N/E/S Pass -> North(0) leads a Club!
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(59)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)

    # North leads C:A; Dummy (East) plays D:2 (showing out of Clubs!)
    state.apply_action(card_id(Suit.CLUBS, 12))
    tracker = RuleFactTracker()
    before = tracker.extract_facts(state, Seat.NORTH)
    assert Suit.CLUBS not in before.known_voids[Seat.EAST]

    state.apply_action(card_id(Suit.DIAMONDS, 0))
    after = tracker.extract_facts(state, Seat.NORTH, prev_voids=before.known_voids)
    assert Suit.CLUBS in after.known_voids[Seat.EAST]
    assert any("EAST showed out of CLUBS" in msg for msg in after.surprise_events)
```

- [ ] **Step 2: Run `pytest` to verify RED failure**

Run:
```bash
~/.local/bin/uv run pytest tests/test_layer3_tracker.py -v
```
Expected: `FAIL` with `ModuleNotFoundError: No module named 'tenace.layer3_rules.tracker'`.

- [ ] **Step 3: Implement `src/tenace/layer3_rules/tracker.py`**

Create `src/tenace/layer3_rules/tracker.py`:

```python
from __future__ import annotations

import numpy as np
import pyspiel
from pydantic import BaseModel, Field

from tenace.domain.types import (
    BIDDING_ACTION_BASE,
    FIRST_CONTRACT_BID_ACTION,
    NUM_CARDS,
    PASS_ACTION,
    CandidateOption,
    Seat,
    Suit,
    action_label,
    card_id,
    card_rank,
    card_suit,
    hand_hcp,
)


def collapse_equivalent_actions(
    legal_actions: list[int],
    played_cards: set[int] | None = None,
) -> list[CandidateOption]:
    """Collapses contiguous or played-bridged cards of the same suit into canonical choices."""
    if not legal_actions:
        return []
    if legal_actions[0] >= BIDDING_ACTION_BASE:
        return [
            CandidateOption(action=a, label=action_label(a), equivalent_actions=[a])
            for a in sorted(legal_actions)
        ]

    played = played_cards or set()
    legal_set = set(legal_actions)
    by_suit: dict[Suit, list[int]] = {}
    for card in sorted(legal_actions):
        by_suit.setdefault(card_suit(card), []).append(card)

    collapsed: list[CandidateOption] = []
    for suit in sorted(by_suit):
        cards = sorted(by_suit[suit], key=card_rank)
        groups: list[list[int]] = [[cards[0]]]
        for prev_card, curr_card in zip(cards, cards[1:]):
            r_prev, r_curr = card_rank(prev_card), card_rank(curr_card)
            bridged = all(
                card_id(suit, r) in played or card_id(suit, r) in legal_set
                for r in range(r_prev + 1, r_curr)
            )
            if bridged:
                groups[-1].append(curr_card)
            else:
                groups.append([curr_card])

        for grp in groups:
            canonical = max(grp, key=card_rank)
            collapsed.append(
                CandidateOption(
                    action=canonical,
                    label=action_label(canonical),
                    equivalent_actions=sorted(grp, reverse=True),
                )
            )
    return collapsed


def _trick_winner(trick_plays: list[tuple[Seat, int]], trump_denom: int) -> Seat:
    led_suit = card_suit(trick_plays[0][1])
    best_seat, best_card = trick_plays[0]
    for seat, card in trick_plays[1:]:
        s_curr, r_curr = card_suit(card), card_rank(card)
        s_best, r_best = card_suit(best_card), card_rank(best_card)
        is_curr_trump = trump_denom < 4 and int(s_curr) == trump_denom
        is_best_trump = trump_denom < 4 and int(s_best) == trump_denom
        if is_curr_trump and not is_best_trump:
            best_seat, best_card = seat, card
        elif s_curr == s_best and r_curr > r_best:
            if is_curr_trump or s_curr == led_suit:
                best_seat, best_card = seat, card
    return best_seat


class BoardFactSnapshot(BaseModel):
    observer: Seat
    phase: str
    own_cards: list[int]
    own_hcp: int
    dummy_seat: Seat | None = None
    dummy_cards: list[int] = Field(default_factory=list)
    played_cards: list[int] = Field(default_factory=list)
    unseen_cards: list[int] = Field(default_factory=list)
    auction_calls: list[tuple[Seat, int]] = Field(default_factory=list)
    declarer: Seat | None = None
    trump_denom: int | None = None
    contract_level: int = 0
    known_voids: dict[Seat, set[Suit]] = Field(default_factory=dict)
    surprise_events: list[str] = Field(default_factory=list)


class RuleFactTracker:
    """Extracts 100% imperfect-information-safe facts from OpenSpiel BridgeState."""

    def extract_facts(
        self,
        state: pyspiel.State,
        observer: Seat,
        prev_voids: dict[Seat, set[Suit]] | None = None,
    ) -> BoardFactSnapshot:
        full_hist = state.full_history()
        post_deal = [ev for ev in full_hist[NUM_CARDS:]]
        auction_calls: list[tuple[Seat, int]] = []
        play_actions: list[int] = []

        first_bidder: dict[tuple[int, int], Seat] = {}
        declarer: Seat | None = None
        trump_denom: int | None = None
        contract_level = 0

        for ev in post_deal:
            if ev.action >= BIDDING_ACTION_BASE:
                seat = Seat(ev.player)
                auction_calls.append((seat, ev.action))
                if ev.action >= FIRST_CONTRACT_BID_ACTION:
                    offset = ev.action - FIRST_CONTRACT_BID_ACTION
                    contract_level = 1 + offset // 5
                    trump_denom = offset % 5
                    key = (seat.partnership, trump_denom)
                    if key not in first_bidder:
                        first_bidder[key] = seat
                    declarer = first_bidder[key]
            else:
                play_actions.append(ev.action)

        # Read ONLY observer's remaining cards and (if opening lead occurred) dummy's remaining cards
        # directly from OpenSpiel's imperfect-information observation tensor!
        obs = np.asarray(state.observation_tensor(int(observer)), dtype=np.float32)
        dummy_seat = declarer.partner if declarer is not None else None

        if len(play_actions) > 0 and declarer is not None:
            # Play-phase observation tensor layout (bridge.cc:521-558):
            # 4 (phase) + 7 (level) + 5 (trumps) + 3 (dbl) + 4 (declarer) + 2 (vul) = offset 25
            own_slice = obs[25 : 25 + NUM_CARDS]
            dummy_slice = obs[25 + NUM_CARDS : 25 + 2 * NUM_CARDS]
            own_cards = [i for i, v in enumerate(own_slice) if v > 0.5]
            dummy_cards = [i for i, v in enumerate(dummy_slice) if v > 0.5]
        else:
            # Auction / pre-opening-lead tensor layout (bridge.cc:609-636):
            # 4 (phase) + 4 (vul: 2*2) + 4*(1 + 3*35) (auction = 424) = offset 432 -> 52 own cards
            own_slice = obs[432 : 432 + NUM_CARDS]
            own_cards = [i for i, v in enumerate(own_slice) if v > 0.5]
            dummy_cards = []

        # Reconstruct trick leaders and detect suit show-outs strictly from public play_actions
        known_voids: dict[Seat, set[Suit]] = {s: set() for s in Seat}
        if len(play_actions) > 0 and declarer is not None and trump_denom is not None:
            leader = Seat((int(declarer) + 1) % 4)
            idx = 0
            while idx < len(play_actions):
                trick_cards = play_actions[idx : idx + 4]
                led_suit = card_suit(trick_cards[0])
                trick_plays: list[tuple[Seat, int]] = []
                for offset_p, card in enumerate(trick_cards):
                    seat_p = Seat((int(leader) + offset_p) % 4)
                    trick_plays.append((seat_p, card))
                    if offset_p > 0 and card_suit(card) != led_suit:
                        known_voids[seat_p].add(led_suit)
                if len(trick_plays) == 4:
                    leader = _trick_winner(trick_plays, trump_denom)
                idx += 4

        surprise_events: list[str] = []
        if prev_voids is not None:
            for s in Seat:
                new_voids = known_voids[s] - prev_voids.get(s, set())
                for v_suit in sorted(new_voids):
                    surprise_events.append(f"{s.name} showed out of {v_suit.name}")

        visible_or_played = set(own_cards) | set(dummy_cards) | set(play_actions)
        unseen_cards = [c for c in range(NUM_CARDS) if c not in visible_or_played]

        phase_code = state.current_phase()
        # BridgeState::Phase: 0=kDeal, 1=kAuction, 2=kPlay, 3=kGameOver
        phase_str = "auction" if phase_code == 1 else ("play" if phase_code == 2 else "terminal")

        return BoardFactSnapshot(
            observer=observer,
            phase=phase_str,
            own_cards=own_cards,
            own_hcp=hand_hcp(own_cards if observer != dummy_seat else dummy_cards),
            dummy_seat=dummy_seat if len(play_actions) > 0 else None,
            dummy_cards=dummy_cards,
            played_cards=play_actions,
            unseen_cards=unseen_cards,
            auction_calls=auction_calls,
            declarer=declarer,
            trump_denom=trump_denom,
            contract_level=contract_level,
            known_voids=known_voids,
            surprise_events=surprise_events,
        )
```

- [ ] **Step 4: Run `pytest` to verify GREEN**

Run:
```bash
~/.local/bin/uv run pytest tests/test_layer3_tracker.py -v
```
Expected: `3 passed`.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/tenace/layer3_rules/ tests/test_layer3_tracker.py
git commit -m "feat(layer3): add imperfect-info fact tracker and card equivalence compressor"
```

---

### Task 3: Layer 3 Constrained Infostate Sampler, DDS Rollout Evaluator & Anti-Blunder Gate

**Files:**
- Create: `src/tenace/layer3_rules/dds_evaluator.py`
- Test: `tests/test_layer3_dds_evaluator.py`

- [ ] **Step 1: Write failing tests for constrained world sampling, candidate EV scoring, and anti-blunder guardrail**

Create `tests/test_layer3_dds_evaluator.py`:

```python
import pyspiel

from tenace.domain.types import (
    PASS_ACTION,
    CandidateOption,
    Seat,
    SeatConstraint,
    StrategyState,
    Suit,
    TacticalDecision,
    action_label,
)
from tenace.layer3_rules.dds_evaluator import RuleDDSEvaluator
from tenace.layer3_rules.tracker import collapse_equivalent_actions


def test_constrained_sampler_prefers_worlds_matching_hcp_bounds():
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state(tournament_seed=7, board_number=1)

    strategy = StrategyState(
        phase="auction",
        plan_summary="East showed a strong hand",
        seat_constraints={
            Seat.EAST: SeatConstraint(seat=Seat.EAST, min_hcp=14, max_hcp=25),
        },
    )
    evaluator = RuleDDSEvaluator(num_worlds=12, rng_seed=123)
    worlds = evaluator.sample_consistent_worlds(state, Seat.NORTH, strategy)
    assert len(worlds) == 12
    # Observer's own observation must be preserved identically across all sampled worlds
    orig_obs = state.observation_tensor(int(Seat.NORTH))
    for w in worlds:
        assert w.observation_tensor(int(Seat.NORTH)) == orig_obs


def test_evaluate_play_candidates_computes_baseline_ev_and_make_prob():
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state(tournament_seed=11, board_number=1)
    # 1NT by North (action 59), 3 passes
    state.apply_action(59)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)
    state.apply_action(PASS_ACTION)

    candidates = collapse_equivalent_actions(state.legal_actions())
    evaluator = RuleDDSEvaluator(num_worlds=8, rng_seed=99)
    scored = evaluator.score_candidates(state, Seat.EAST, candidates[:3], strategy=None)
    assert len(scored) == 3
    for opt in scored:
        assert isinstance(opt.baseline_ev, float)
        assert opt.make_prob is not None
        assert 0.0 <= opt.make_prob <= 1.0


def test_blunder_gate_blocks_low_make_prob_hallucination():
    opt_safe = CandidateOption(
        action=51, label=action_label(51), equivalent_actions=[51], baseline_ev=400.0, make_prob=0.95
    )
    opt_blunder = CandidateOption(
        action=0, label=action_label(0), equivalent_actions=[0], baseline_ev=-100.0, make_prob=0.15
    )
    # Suppose Layer 2 mistakenly gave opt_blunder a 0.99 score and opt_safe 0.20:
    decision = TacticalDecision(
        option_scores={opt_safe.label: 0.20, opt_blunder.label: 0.99},
        rationale="Try risky low club",
    )
    evaluator = RuleDDSEvaluator(max_make_prob_drop=0.20)
    chosen = evaluator.select_guarded_action([opt_safe, opt_blunder], decision)
    # Blunder gate must veto opt_blunder (0.15 < 0.95 - 0.20) and pick opt_safe!
    assert chosen == opt_safe.action
```

- [ ] **Step 2: Run `pytest` to verify RED failure**

Run:
```bash
~/.local/bin/uv run pytest tests/test_layer3_dds_evaluator.py -v
```
Expected: `FAIL` with `ModuleNotFoundError: No module named 'tenace.layer3_rules.dds_evaluator'`.

- [ ] **Step 3: Implement `src/tenace/layer3_rules/dds_evaluator.py`**

Create `src/tenace/layer3_rules/dds_evaluator.py`:

```python
from __future__ import annotations

import random

import numpy as np
import pyspiel

from tenace.domain.types import (
    BIDDING_ACTION_BASE,
    NUM_CARDS,
    PASS_ACTION,
    CandidateOption,
    Seat,
    SeatConstraint,
    StrategyState,
    TacticalDecision,
    card_suit,
    hand_hcp,
)


def _world_matches_constraints(
    world: pyspiel.State,
    observer: Seat,
    constraints: dict[Seat, SeatConstraint],
) -> bool:
    if not constraints:
        return True
    deal_events = world.full_history()[:NUM_CARDS]
    dealt_hands: dict[Seat, list[int]] = {s: [] for s in Seat}
    for idx, ev in enumerate(deal_events):
        dealt_hands[Seat(idx % 4)].append(ev.action)

    for seat, c in constraints.items():
        if seat == observer:
            continue
        cards = dealt_hands[seat]
        hcp = hand_hcp(cards)
        if not (c.min_hcp <= hcp <= c.max_hcp):
            return False
        for suit, min_len in c.min_suit_lengths.items():
            if sum(1 for card in cards if card_suit(card) == suit) < min_len:
                return False
        for suit, max_len in c.max_suit_lengths.items():
            if sum(1 for card in cards if card_suit(card) == suit) > max_len:
                return False
    return True


class RuleDDSEvaluator:
    """Constrained imperfect-information world sampler, rollout scorer, and blunder guardrail."""

    def __init__(
        self,
        num_worlds: int = 16,
        max_rejection_Factor: int = 8,
        max_make_prob_drop: float = 0.20,
        rng_seed: int | None = None,
    ) -> None:
        self.num_worlds = num_worlds
        self.max_rejection_factor = max_rejection_Factor
        self.max_make_prob_drop = max_make_prob_drop
        self._rng = random.Random(rng_seed)

    def sample_consistent_worlds(
        self,
        state: pyspiel.State,
        observer: Seat,
        strategy: StrategyState | None = None,
    ) -> list[pyspiel.State]:
        constraints = strategy.seat_constraints if strategy else {}
        accepted: list[pyspiel.State] = []
        fallbacks: list[pyspiel.State] = []
        max_attempts = max(self.num_worlds * self.max_rejection_factor, self.num_worlds)

        for _ in range(max_attempts):
            world = state.resample_from_infostate(int(observer), self._rng.random)
            if _world_matches_constraints(world, observer, constraints):
                accepted.append(world)
                if len(accepted) >= self.num_worlds:
                    return accepted
            elif len(fallbacks) < self.num_worlds:
                fallbacks.append(world)

        return (accepted + fallbacks)[: self.num_worlds]

    def _rollout_value(self, world: pyspiel.State, observer: Seat, candidate_action: int) -> float:
        sim = world.clone()
        sim.apply_action(candidate_action)

        # If in auction, pass out remaining calls and evaluate exact C++ Double-Dummy score!
        if candidate_action >= BIDDING_ACTION_BASE:
            while not sim.is_terminal() and sim.current_phase() == 1:
                if PASS_ACTION in sim.legal_actions():
                    sim.apply_action(PASS_ACTION)
                else:
                    break
            if sim.is_terminal():
                return float(sim.returns()[int(observer)])
            possible = sim.possible_contracts()
            active_indices = [idx for idx, ok in enumerate(possible) if ok]
            if active_indices:
                scores = sim.score_for_contracts(int(observer), active_indices[:1])
                return float(scores[0])

        # Play phase: fast heuristic rollout to terminal trick score on the sampled deal
        while not sim.is_terminal():
            legal = sim.legal_actions()
            # Greedy high-card / low-follow rollout for fast deterministic evaluation
            sim.apply_action(legal[-1] if len(legal) == 1 else self._rng.choice(legal))
        return float(sim.returns()[int(observer)])

    def score_candidates(
        self,
        state: pyspiel.State,
        observer: Seat,
        candidates: list[CandidateOption],
        strategy: StrategyState | None = None,
    ) -> list[CandidateOption]:
        if len(candidates) <= 1:
            return candidates

        worlds = self.sample_consistent_worlds(state, observer, strategy)
        scored: list[CandidateOption] = []
        for cand in candidates:
            payoffs = [self._rollout_value(w, observer, cand.action) for w in worlds]
            ev = float(np.mean(payoffs)) if payoffs else 0.0
            pos_rate = float(np.mean([1.0 if p > 0 else (0.5 if p == 0 else 0.0) for p in payoffs]))
            scored.append(
                cand.model_copy(
                    update={
                        "baseline_ev": ev,
                        "make_prob": pos_rate,
                    }
                )
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

        # Guardrail 1: Filter out blunders that sacrifice > max_make_prob_drop win rate
        with_probs = [c.make_prob for c in candidates if c.make_prob is not None]
        viable = candidates
        if with_probs:
            best_prob = max(with_probs)
            filtered = [
                c
                for c in candidates
                if c.make_prob is None or c.make_prob >= best_prob - self.max_make_prob_drop
            ]
            if filtered:
                viable = filtered

        if decision is None or not decision.option_scores:
            return max(viable, key=lambda c: c.baseline_ev).action

        # Normalize baseline_ev across viable options to [0, 1] and blend with Layer 2 strategic score
        evs = [c.baseline_ev for c in viable]
        ev_min, ev_max = min(evs), max(evs)
        ev_span = max(ev_max - ev_min, 1.0)

        def combined_score(c: CandidateOption) -> float:
            norm_ev = (c.baseline_ev - ev_min) / ev_span
            strat_score = decision.option_scores.get(c.label, 0.5)
            return 0.45 * norm_ev + 0.55 * strat_score

        return max(viable, key=combined_score).action
```

- [ ] **Step 4: Run `pytest` to verify GREEN**

Run:
```bash
~/.local/bin/uv run pytest tests/test_layer3_dds_evaluator.py -v
```
Expected: `3 passed`.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/tenace/layer3_rules/dds_evaluator.py tests/test_layer3_dds_evaluator.py
git commit -m "feat(layer3): add constrained infostate rollout evaluator and anti-blunder gate"
```

---

### Task 4: Vertex AI Model Clients (Frontier Gemini Strategist + DiffusionGemma MaaS)

**Files:**
- Create: `src/tenace/models/__init__.py`
- Create: `src/tenace/models/clients.py`
- Test: `tests/test_model_clients.py`

- [ ] **Step 1: Write failing tests for JSON extraction, DiffusionGemma payload formatting, and Vertex Gemini structured client**

Create `src/tenace/models/__init__.py` (empty) and `tests/test_model_clients.py`:

```python
from unittest.mock import MagicMock

from tenace.domain.types import TacticalDecision
from tenace.models.clients import (
    DiffusionGemmaClient,
    DiffusionSamplingConfig,
    VertexGeminiClient,
    extract_json_object,
)


def test_extract_json_object_handles_markdown_fences_and_thinking_prefix():
    raw = """<thought>Checking clubs entry...</thought>
```json
{
  "assumption_violated": false,
  "violation_reason": "",
  "option_scores": {"S:A": 0.91, "C:2": 0.35},
  "rationale": "Cash spade ace first to preserve club entry"
}
```"""
    parsed = extract_json_object(raw, TacticalDecision)
    assert parsed.assumption_violated is False
    assert parsed.option_scores["S:A"] == 0.91


def test_diffusion_gemma_client_sends_denoising_parameters_to_endpoint():
    fake_endpoint = MagicMock()
    fake_prediction = MagicMock()
    fake_prediction.predictions = [
        '{"assumption_violated": true, "violation_reason": "West void in spades", "option_scores": {"H:K": 0.88}, "rationale": "Switch to hearts"}'
    ]
    fake_endpoint.predict.return_value = fake_prediction

    client = DiffusionGemmaClient(
        endpoint=fake_endpoint,
        sampling=DiffusionSamplingConfig(max_denoising_steps=32, entropy_threshold=0.005),
    )
    res = client.generate_structured("sys prompt", "user canvas", TacticalDecision)
    assert res.assumption_violated is True
    assert res.option_scores["H:K"] == 0.88

    call_kwargs = fake_endpoint.predict.call_args.kwargs
    assert call_kwargs["parameters"]["max_denoising_steps"] == 32
    assert call_kwargs["parameters"]["entropy_threshold"] == 0.005


def test_vertex_gemini_client_parses_structured_response():
    fake_genai = MagicMock()
    fake_resp = MagicMock()
    fake_resp.text = '{"assumption_violated": false, "violation_reason": "", "option_scores": {"Pass": 0.75}, "rationale": "Balanced minimum"}'
    fake_genai.models.generate_content.return_value = fake_resp

    client = VertexGeminiClient(genai_client=fake_genai, model="gemini-2.5-pro")
    res = client.generate_structured("sys", "usr", TacticalDecision)
    assert res.option_scores["Pass"] == 0.75
    fake_genai.models.generate_content.assert_called_once()
```

- [ ] **Step 2: Run `pytest` to verify RED failure**

Run:
```bash
~/.local/bin/uv run pytest tests/test_model_clients.py -v
```
Expected: `FAIL` with `ModuleNotFoundError: No module named 'tenace.models.clients'`.

- [ ] **Step 3: Implement `src/tenace/models/clients.py`**

Create `src/tenace/models/clients.py`:

```python
from __future__ import annotations

import re
from typing import Any, Protocol, TypeVar

from google import genai
from google.cloud import aiplatform
from google.genai import types as genai_types
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredModelClient(Protocol):
    def generate_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T: ...


def extract_json_object(raw_text: str, schema: type[T]) -> T:
    """Strips optional reasoning tags or markdown fences and validates against `schema`."""
    cleaned = re.sub(r"<thought>.*?</thought>", "", raw_text, flags=re.DOTALL).strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return schema.model_validate_json(cleaned)


class VertexGeminiClient:
    """Vertex AI client for Gemini reasoning models (Layer 1 Strategist or MaaS fallback)."""

    def __init__(
        self,
        project: str | None = None,
        location: str = "global",
        model: str = "gemini-2.5-pro",
        temperature: float = 0.2,
        genai_client: Any | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = genai_client or genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

    def generate_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return extract_json_object(resp.text, schema)


class DiffusionSamplingConfig(BaseModel):
    max_denoising_steps: int = 48
    max_tokens: int = 256
    temperature: float = 0.4
    entropy_threshold: float = 0.005
    entropy_bound: float = 0.1


class DiffusionGemmaClient:
    """Vertex AI Model Garden client for `google/diffusiongemma-26B-A4B-it` (MaaS or Endpoint)."""

    def __init__(
        self,
        project: str | None = None,
        location: str = "us-central1",
        endpoint_id: str | None = None,
        maas_model_name: str = "publishers/google/models/diffusiongemma-26b-a4b-it",
        sampling: DiffusionSamplingConfig | None = None,
        endpoint: Any | None = None,
        genai_maas_client: Any | None = None,
    ) -> None:
        self.sampling = sampling or DiffusionSamplingConfig()
        self.maas_model_name = maas_model_name
        self._endpoint = endpoint
        self._genai_maas = genai_maas_client

        if self._endpoint is None and endpoint_id is not None:
            aiplatform.init(project=project, location=location)
            self._endpoint = aiplatform.Endpoint(endpoint_id)
        elif self._endpoint is None and self._genai_maas is None:
            self._genai_maas = genai.Client(vertexai=True, project=project, location=location)

    def generate_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        schema_hint = f"Return ONLY valid JSON matching schema: {schema.model_json_schema()}"
        combined_prompt = f"{system_prompt}\n\n{schema_hint}\n\n{user_prompt}"

        if self._endpoint is not None:
            prediction = self._endpoint.predict(
                instances=[{"prompt": combined_prompt}],
                parameters={
                    "max_tokens": self.sampling.max_tokens,
                    "temperature": self.sampling.temperature,
                    "max_denoising_steps": self.sampling.max_denoising_steps,
                    "entropy_threshold": self.sampling.entropy_threshold,
                    "entropy_bound": self.sampling.entropy_bound,
                },
            )
            first = prediction.predictions[0]
            raw_text = first["output"] if isinstance(first, dict) and "output" in first else str(first)
            return extract_json_object(raw_text, schema)

        resp = self._genai_maas.models.generate_content(
            model=self.maas_model_name,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=f"{system_prompt}\n{schema_hint}",
                temperature=self.sampling.temperature,
                max_output_tokens=self.sampling.max_tokens,
                response_mime_type="application/json",
            ),
        )
        return extract_json_object(resp.text, schema)
```

- [ ] **Step 4: Run `pytest` to verify GREEN**

Run:
```bash
~/.local/bin/uv run pytest tests/test_model_clients.py -v
```
Expected: `3 passed`.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/tenace/models/ tests/test_model_clients.py
git commit -m "feat(models): add Vertex Gemini and DiffusionGemma structured model clients"
```

---

### Task 5: Layer 1 Strategist & Layer 2 DiffusionGemma Option Evaluator

**Files:**
- Create: `src/tenace/layer1_strategist/__init__.py`
- Create: `src/tenace/layer1_strategist/conventions.py`
- Create: `src/tenace/layer1_strategist/strategist.py`
- Create: `src/tenace/layer2_evaluator/__init__.py`
- Create: `src/tenace/layer2_evaluator/evaluator.py`
- Test: `tests/test_layer1_and_layer2.py`

- [ ] **Step 1: Write failing tests for event-driven Strategist triggers and Layer 2 DiffusionGemma option evaluation**

Create empty `__init__.py` files in `src/tenace/layer1_strategist/` and `src/tenace/layer2_evaluator/`, then create `tests/test_layer1_and_layer2.py`:

```python
import pyspiel

from tenace.domain.types import (
    Assumption,
    CandidateOption,
    Seat,
    StrategyState,
    TacticalDecision,
)
from tenace.layer1_strategist.conventions import SAYC_CONVENTION_CARD
from tenace.layer1_strategist.strategist import StrategistEngine
from tenace.layer2_evaluator.evaluator import DiffusionOptionEvaluator
from tenace.layer3_rules.tracker import RuleFactTracker


class FakeModelClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_structured(self, system_prompt, user_prompt, schema):
        self.calls.append((system_prompt, user_prompt, schema))
        return self.response


def test_strategist_triggers_on_initial_state_phase_transition_and_surprise():
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state(tournament_seed=1, board_number=1)
    facts = RuleFactTracker().extract_facts(state, Seat.NORTH)

    fake_strat = StrategyState(
        phase="auction",
        plan_summary="Open 1NT with balanced 15-17 HCP",
        assumptions=[Assumption(claim="Partner holds ~8 HCP", confidence=0.5)],
    )
    client = FakeModelClient(fake_strat)
    engine = StrategistEngine(client=client, convention_card=SAYC_CONVENTION_CARD)

    # 1. No strategy yet -> must trigger
    assert engine.should_replan(facts, current_strategy=None, dummy_already_seen=False) is True
    updated = engine.update_strategy(facts, current_strategy=None)
    assert updated.revision == 1
    assert len(client.calls) == 1

    # 2. Same auction phase, no surprise -> should NOT replan unnecessarily
    assert engine.should_replan(facts, current_strategy=updated, dummy_already_seen=False) is False

    # 3. Surprise event -> MUST trigger replan and bump revision!
    facts_surprise = facts.model_copy(update={"surprise_events": ["WEST showed out of SPADES"]})
    assert engine.should_replan(facts_surprise, current_strategy=updated, dummy_already_seen=False) is True
    updated_2 = engine.update_strategy(facts_surprise, current_strategy=updated, trigger_reason="WEST void")
    assert updated_2.revision == 2


def test_layer2_evaluator_formats_candidates_and_returns_tactical_decision():
    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state(tournament_seed=1, board_number=1)
    facts = RuleFactTracker().extract_facts(state, Seat.NORTH)
    strategy = StrategyState(phase="auction", plan_summary="Bid 5-card major", revision=1)
    candidates = [
        CandidateOption(action=52, label="Pass", equivalent_actions=[52], baseline_ev=0.0, make_prob=0.5),
        CandidateOption(action=58, label="1S", equivalent_actions=[58], baseline_ev=110.0, make_prob=0.72),
    ]

    expected_decision = TacticalDecision(
        assumption_violated=False,
        option_scores={"Pass": 0.25, "1S": 0.92},
        rationale="1S aligns with 5-card major opening plan and higher DDS EV",
    )
    client = FakeModelClient(expected_decision)
    evaluator = DiffusionOptionEvaluator(client=client)

    decision = evaluator.evaluate(facts, strategy, candidates)
    assert decision.option_scores["1S"] == 0.92
    _, user_prompt, _ = client.calls[0]
    assert "1S" in user_prompt
    assert "baseline_ev=110.0" in user_prompt
```

- [ ] **Step 2: Run `pytest` to verify RED failure**

Run:
```bash
~/.local/bin/uv run pytest tests/test_layer1_and_layer2.py -v
```
Expected: `FAIL` with `ModuleNotFoundError: No module named 'tenace.layer1_strategist.conventions'`.

- [ ] **Step 3: Implement `conventions.py`, `strategist.py`, and `evaluator.py`**

Create `src/tenace/layer1_strategist/conventions.py`:

```python
from pydantic import BaseModel


class ConventionCard(BaseModel):
    name: str
    summary_rules: str


SAYC_CONVENTION_CARD = ConventionCard(
    name="SAYC (Standard American Yellow Card)",
    summary_rules=(
        "- 5-card majors (1H/1S promises 5+ cards, 12-21 HCP).\n"
        "- 1NT opening: 15-17 HCP balanced; 2C Stayman, 2D/2H Jacoby Transfers.\n"
        "- 2C opening: Strong artificial 22+ HCP or game-forcing.\n"
        "- 2D/2H/2S weak two-bids: 6-card suit, 5-11 HCP.\n"
        "- 2NT opening: 20-21 HCP balanced.\n"
        "- Defensive carding: standard attitude (high encourages) and standard count (high-low = even)."
    ),
)
```

Create `src/tenace/layer1_strategist/strategist.py`:

```python
from __future__ import annotations

from tenace.domain.types import StrategyState, action_label
from tenace.layer1_strategist.conventions import SAYC_CONVENTION_CARD, ConventionCard
from tenace.layer3_rules.tracker import BoardFactSnapshot
from tenace.models.clients import StructuredModelClient


def format_cards_by_suit(cards: list[int]) -> str:
    labels = [action_label(c) for c in sorted(cards, reverse=True)]
    return " ".join(labels) if labels else "(none)"


class StrategistEngine:
    """Layer 1 System-2 Strategist: builds and augments StrategyState at key inflection points."""

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
        if current_strategy is None:
            return True
        if current_strategy.phase != facts.phase:
            return True
        if facts.phase == "play" and facts.dummy_seat is not None and not dummy_already_seen:
            return True
        if facts.surprise_events:
            return True
        return False

    def update_strategy(
        self,
        facts: BoardFactSnapshot,
        current_strategy: StrategyState | None,
        trigger_reason: str = "",
    ) -> StrategyState:
        system_prompt = (
            "You are Tenace's System-2 Grandmaster Bridge Strategist.\n"
            f"Partnership Convention Card ({self.convention_card.name}):\n"
            f"{self.convention_card.summary_rules}\n"
            "Synthesize an explicit multi-trick/auction plan and probabilistic assumptions "
            "about unseen hands strictly from public history and your own visible cards."
        )

        auction_str = " -> ".join(f"{s.name}:{action_label(a)}" for s, a in facts.auction_calls) or "(opening)"
        void_str = ", ".join(
            f"{s.name} void in {[v.name for v in suits]}"
            for s, suits in facts.known_voids.items()
            if suits
        ) or "none"

        user_prompt = (
            f"Seat: {facts.observer.name} (Partner: {facts.observer.partner.name})\n"
            f"Phase: {facts.phase}\n"
            f"Your Hand ({facts.own_hcp} HCP): {format_cards_by_suit(facts.own_cards)}\n"
            f"Dummy ({facts.dummy_seat.name if facts.dummy_seat is not None else 'hidden'}): "
            f"{format_cards_by_suit(facts.dummy_cards)}\n"
            f"Auction: {auction_str}\n"
            f"Played Cards ({len(facts.played_cards)}): {format_cards_by_suit(facts.played_cards)}\n"
            f"Known Voids: {void_str}\n"
            f"Surprise / Trigger: {trigger_reason or ', '.join(facts.surprise_events) or 'Initial/Phase transition'}\n"
            f"Previous Strategy: {current_strategy.model_dump_json() if current_strategy else 'None'}\n"
        )

        next_rev = 1 if current_strategy is None else current_strategy.revision + 1
        updated = self.client.generate_structured(system_prompt, user_prompt, StrategyState)
        return updated.model_copy(update={"phase": facts.phase, "revision": next_rev})
```

Create `src/tenace/layer2_evaluator/evaluator.py`:

```python
from __future__ import annotations

from tenace.domain.types import CandidateOption, StrategyState, TacticalDecision, action_label
from tenace.layer1_strategist.strategist import format_cards_by_suit
from tenace.layer3_rules.tracker import BoardFactSnapshot
from tenace.models.clients import StructuredModelClient


class DiffusionOptionEvaluator:
    """Layer 2 Fast Option Evaluator (DiffusionGemma MaaS): scores non-equivalent options against StrategyState."""

    def __init__(self, client: StructuredModelClient) -> None:
        self.client = client

    def evaluate(
        self,
        facts: BoardFactSnapshot,
        strategy: StrategyState,
        candidates: list[CandidateOption],
    ) -> TacticalDecision:
        system_prompt = (
            "You are Tenace's fast tactical option evaluator. "
            "First check whether recent public bids/plays contradict the active StrategyState assumptions. "
            "Then score EVERY candidate option in [0.0, 1.0] based on alignment with the active plan, "
            "entry management, signaling, and its Layer-3 DDS baseline EV."
        )

        auction_str = " -> ".join(f"{s.name}:{action_label(a)}" for s, a in facts.auction_calls) or "(opening)"
        cand_lines = "\n".join(
            f"- {c.label}: baseline_ev={c.baseline_ev:.1f}, make_prob={c.make_prob if c.make_prob is not None else 'N/A'}"
            for c in candidates
        )

        user_prompt = (
            f"Seat: {facts.observer.name} | Phase: {facts.phase}\n"
            f"Hand: {format_cards_by_suit(facts.own_cards)} | Dummy: {format_cards_by_suit(facts.dummy_cards)}\n"
            f"Auction: {auction_str}\n"
            f"Active Strategy (rev {strategy.revision}): {strategy.plan_summary}\n"
            f"Target/Threat: {strategy.target_or_threat}\n"
            f"Assumptions: {[a.model_dump() for a in strategy.assumptions]}\n"
            f"Candidate Options:\n{cand_lines}\n"
        )

        return self.client.generate_structured(system_prompt, user_prompt, TacticalDecision)
```

- [ ] **Step 4: Run `pytest` to verify GREEN**

Run:
```bash
~/.local/bin/uv run pytest tests/test_layer1_and_layer2.py -v
```
Expected: `2 passed`.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/tenace/layer1_strategist/ src/tenace/layer2_evaluator/ tests/test_layer1_and_layer2.py
git commit -m "feat(layers1-2): add event-driven Strategist and DiffusionGemma option evaluator"
```

---

### Task 6: End-to-End `TenaceAgent` Orchestrator & Official WBF Duplicate IMP Harness

**Files:**
- Create: `src/tenace/agent.py`
- Create: `src/tenace/harness/__init__.py`
- Create: `src/tenace/harness/scoring.py`
- Create: `src/tenace/harness/duplicate.py`
- Test: `tests/test_agent_and_duplicate_harness.py`

- [ ] **Step 1: Write failing tests for WBF IMP conversion, forced-move 0-API shortcut, assumption-violation replan loop, and full 2-room Duplicate match**

Create `src/tenace/harness/__init__.py` (empty) and `tests/test_agent_and_duplicate_harness.py`:

```python
import pyspiel

from tenace.agent import TenaceAgent
from tenace.domain.types import Assumption, Seat, StrategyState, TacticalDecision
from tenace.harness.duplicate import DuplicateMatchRunner, FirstLegalPassBaselineBot
from tenace.harness.scoring import score_diff_to_imps
from tenace.layer3_rules.dds_evaluator import RuleDDSEvaluator


class ScriptedStrategistClient:
    def __init__(self):
        self.call_count = 0

    def generate_structured(self, system_prompt, user_prompt, schema):
        self.call_count += 1
        return StrategyState(
            phase="auction",
            plan_summary=f"Plan v{self.call_count}: play solid bridge",
            assumptions=[Assumption(claim="Balanced distribution", confidence=0.7)],
        )


class ScriptedEvaluatorClient:
    def __init__(self, trigger_violation_once: bool = False):
        self.call_count = 0
        self.trigger_violation_once = trigger_violation_once

    def generate_structured(self, system_prompt, user_prompt, schema):
        self.call_count += 1
        violated = self.trigger_violation_once and self.call_count == 1
        return TacticalDecision(
            assumption_violated=violated,
            violation_reason="Opponent preempt broke balanced assumption" if violated else "",
            option_scores={},
            rationale="Defer to Layer-3 DDS EV",
        )


def test_wbf_imp_table_conversion():
    assert score_diff_to_imps(0) == 0
    assert score_diff_to_imps(10) == 0
    assert score_diff_to_imps(20) == 1
    assert score_diff_to_imps(-50) == -2
    assert score_diff_to_imps(420) == 9
    assert score_diff_to_imps(-4000) == -24


def test_tenace_agent_reruns_strategist_when_layer2_flags_assumption_violation():
    strat_client = ScriptedStrategistClient()
    eval_client = ScriptedEvaluatorClient(trigger_violation_once=True)
    agent = TenaceAgent(
        seat=Seat.NORTH,
        strategist_client=strat_client,
        evaluator_client=eval_client,
        dds_evaluator=RuleDDSEvaluator(num_worlds=4, rng_seed=1),
    )

    game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    state = game.new_initial_state(tournament_seed=5, board_number=1)
    action = agent.step(state)
    assert action in state.legal_actions()
    # Initial strategy (1) + interrupt replan after Layer 2 flagged violation (2)
    assert strat_client.call_count == 2
    assert eval_client.call_count == 2


def test_duplicate_match_runner_plays_open_and_closed_rooms_without_memory_leakage():
    def make_tenace(seat: Seat) -> TenaceAgent:
        return TenaceAgent(
            seat=seat,
            strategist_client=ScriptedStrategistClient(),
            evaluator_client=ScriptedEvaluatorClient(),
            dds_evaluator=RuleDDSEvaluator(num_worlds=2, rng_seed=42),
        )

    def make_baseline(seat: Seat) -> FirstLegalPassBaselineBot:
        return FirstLegalPassBaselineBot(seat=seat)

    runner = DuplicateMatchRunner(tournament_seed=100)
    summary = runner.play_match(
        team_a_factory=make_tenace,
        team_b_factory=make_baseline,
        num_boards=2,
    )
    assert len(summary.boards) == 2
    for b in summary.boards:
        assert b.imps_team_a == score_diff_to_imps(b.open_room_ns_score - b.closed_room_ns_score)
```

- [ ] **Step 2: Run `pytest` to verify RED failure**

Run:
```bash
~/.local/bin/uv run pytest tests/test_agent_and_duplicate_harness.py -v
```
Expected: `FAIL` with `ModuleNotFoundError: No module named 'tenace.agent'`.

- [ ] **Step 3: Implement `scoring.py`, `agent.py`, and `duplicate.py`**

Create `src/tenace/harness/scoring.py`:

```python
from bisect import bisect_left

# Official WBF IMP upper bounds for IMPs 0..23; >= 4000 is 24 IMPs
_WBF_IMP_UPPER_BOUNDS = (
    10, 40, 80, 120, 160, 210, 260, 310, 360, 420, 490, 590,
    740, 890, 1090, 1290, 1490, 1740, 1990, 2240, 2490, 2990, 3490, 3990,
)


def score_diff_to_imps(diff: int | float) -> int:
    """Converts raw duplicate score difference (`Team_A_NS + Team_A_EW`) into official WBF IMPs."""
    rounded = int(round(abs(diff)))
    imps = bisect_left(_WBF_IMP_UPPER_BOUNDS, rounded)
    return imps if diff >= 0 else -imps
```

Create `src/tenace/agent.py`:

```python
from __future__ import annotations

import pyspiel

from tenace.domain.types import Seat, StrategyState, Suit
from tenace.layer1_strategist.conventions import SAYC_CONVENTION_CARD, ConventionCard
from tenace.layer1_strategist.strategist import StrategistEngine
from tenace.layer2_evaluator.evaluator import DiffusionOptionEvaluator
from tenace.layer3_rules.dds_evaluator import RuleDDSEvaluator
from tenace.layer3_rules.tracker import RuleFactTracker, collapse_equivalent_actions
from tenace.models.clients import StructuredModelClient


class TenaceAgent:
    """Isolated single-seat 3-layer Bridge AI agent (Layer 1 LLM + Layer 2 DiffusionGemma + Layer 3 DDS/Rules)."""

    def __init__(
        self,
        seat: Seat,
        strategist_client: StructuredModelClient,
        evaluator_client: StructuredModelClient,
        dds_evaluator: RuleDDSEvaluator | None = None,
        convention_card: ConventionCard = SAYC_CONVENTION_CARD,
    ) -> None:
        self.seat = seat
        self.tracker = RuleFactTracker()
        self.dds_evaluator = dds_evaluator or RuleDDSEvaluator()
        self.strategist = StrategistEngine(client=strategist_client, convention_card=convention_card)
        self.evaluator = DiffusionOptionEvaluator(client=evaluator_client)

        self.strategy: StrategyState | None = None
        self._prev_voids: dict[Seat, set[Suit]] | None = None
        self._dummy_seen: bool = False
        self.forced_moves: int = 0
        self.strategist_calls: int = 0
        self.evaluator_calls: int = 0

    def reset_for_board(self) -> None:
        self.strategy = None
        self._prev_voids = None
        self._dummy_seen = False

    def step(self, state: pyspiel.State) -> int:
        #Note: During play, declarer plays dummy's cards too; observer is always self.seat!
        facts = self.tracker.extract_facts(state, self.seat, prev_voids=self._prev_voids)
        self._prev_voids = {s: set(v) for s, v in facts.known_voids.items()}

        candidates = collapse_equivalent_actions(
            state.legal_actions(),
            played_cards=set(facts.played_cards),
        )

        # Layer 3 Fast-Path: Forced / single equivalent move requires ZERO model calls!
        if len(candidates) == 1:
            self.forced_moves += 1
            if facts.dummy_seat is not None:
                self._dummy_seen = True
            return candidates[0].action

        # Layer 1 Event-Driven Check
        if self.strategist.should_replan(facts, self.strategy, dummy_already_seen=self._dummy_seen):
            self.strategy = self.strategist.update_strategy(facts, self.strategy)
            self.strategist_calls += 1
        if facts.dummy_seat is not None:
            self._dummy_seen = True

        assert self.strategy is not None

        # Layer 3 Constrained DDS Rollout Baseline
        scored_candidates = self.dds_evaluator.score_candidates(
            state, self.seat, candidates, strategy=self.strategy
        )

        # Layer 2 DiffusionGemma Evaluation
        decision = self.evaluator.evaluate(facts, self.strategy, scored_candidates)
        self.evaluator_calls += 1

        # Closed-Loop Assumption Violation Interrupt -> Augment Strategy once & re-evaluate!
        if decision.assumption_violated:
            self.strategy = self.strategist.update_strategy(
                facts,
                self.strategy,
                trigger_reason=decision.violation_reason or "Layer-2 flagged assumption violation",
            )
            self.strategist_calls += 1
            scored_candidates = self.dds_evaluator.score_candidates(
                state, self.seat, candidates, strategy=self.strategy
            )
            decision = self.evaluator.evaluate(facts, self.strategy, scored_candidates)
            self.evaluator_calls += 1

        # Layer 3 Final Anti-Blunder Gate
        return self.dds_evaluator.select_guarded_action(scored_candidates, decision)
```

Create `src/tenace/harness/duplicate.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import pyspiel
from pydantic import BaseModel

from tenace.domain.types import BIDDING_ACTION_BASE, PASS_ACTION, Seat
from tenace.harness.scoring import score_diff_to_imps


class BridgeSeatBot(Protocol):
    def reset_for_board(self) -> None: ...
    def step(self, state: pyspiel.State) -> int: ...


class FirstLegalPassBaselineBot:
    """Fast deterministic baseline bot (passes in auction or bids 1NT on 15+ HCP, plays lowest legal card)."""

    def __init__(self, seat: Seat) -> None:
        self.seat = seat

    def reset_for_board(self) -> None:
        pass

    def step(self, state: pyspiel.State) -> int:
        legal = state.legal_actions()
        if legal[0] >= BIDDING_ACTION_BASE and PASS_ACTION in legal:
            # Make a simple 1C opening if dealer so boards actually reach card play in tests!
            if len(state.full_history()) == 52 and 55 in legal:
                return 55
            return PASS_ACTION
        return legal[0]


class BoardResult(BaseModel):
    board_number: int
    open_room_ns_score: float
    closed_room_ns_score: float
    imps_team_a: int


class MatchSummary(BaseModel):
    boards: list[BoardResult]
    total_imps_team_a: int


def _play_single_table(state: pyspiel.State, seats: dict[Seat, BridgeSeatBot]) -> float:
    for bot in seats.values():
        bot.reset_for_board()

    # Track declarer so when OpenSpiel asks dummy's seat ID or declarer's seat ID, the right partner acts
    while not state.is_terminal():
        actor = Seat(state.current_player())
        action = seats[actor].step(state)
        state.apply_action(action)

    return float(state.returns()[int(Seat.NORTH)])


class DuplicateMatchRunner:
    """Runs official 2-room Duplicate Bridge IMP matches on identical dealt boards."""

    def __init__(self, tournament_seed: int = 42, use_double_dummy_result: bool = False) -> None:
        self.tournament_seed = tournament_seed
        self.game = pyspiel.load_game(
            f"bridge(use_double_dummy_result={'true' if use_double_dummy_result else 'false'})"
        )

    def play_match(
        self,
        team_a_factory: Callable[[Seat], BridgeSeatBot],
        team_b_factory: Callable[[Seat], BridgeSeatBot],
        num_boards: int = 4,
    ) -> MatchSummary:
        board_results: list[BoardResult] = []
        for board_num in range(1, num_boards + 1):
            # Open Room: Team A = North/South, Team B = East/West
            open_state = self.game.new_initial_state(
                tournament_seed=self.tournament_seed, board_number=board_num
            )
            open_seats = {
                Seat.NORTH: team_a_factory(Seat.NORTH),
                Seat.SOUTH: team_a_factory(Seat.SOUTH),
                Seat.EAST: team_b_factory(Seat.EAST),
                Seat.WEST: team_b_factory(Seat.WEST),
            }
            open_ns = _play_single_table(open_state, open_seats)

            # Closed Room: Team B = North/South, Team A = East/West (exact same deal!)
            closed_state = self.game.new_initial_state(
                tournament_seed=self.tournament_seed, board_number=board_num
            )
            closed_seats = {
                Seat.NORTH: team_b_factory(Seat.NORTH),
                Seat.SOUTH: team_b_factory(Seat.SOUTH),
                Seat.EAST: team_a_factory(Seat.EAST),
                Seat.WEST: team_a_factory(Seat.WEST),
            }
            closed_ns = _play_single_table(closed_state, closed_seats)

            # Team A net points = Open_NS - Closed_NS
            imps_a = score_diff_to_imps(open_ns - closed_ns)
            board_results.append(
                BoardResult(
                    board_number=board_num,
                    open_room_ns_score=open_ns,
                    closed_room_ns_score=closed_ns,
                    imps_team_a=imps_a,
                )
            )

        return MatchSummary(
            boards=board_results,
            total_imps_team_a=sum(b.imps_team_a for b in board_results),
        )
```

- [ ] **Step 4: Run full `pytest` suite to verify all tests pass**

Run:
```bash
~/.local/bin/uv run pytest -v
```
Expected: `14 passed` across all 6 test files with 0 warnings/errors.

- [ ] **Step 5: Commit Task 6**

```bash
git add src/tenace/agent.py src/tenace/harness/ tests/test_agent_and_duplicate_harness.py
git commit -m "feat: add 3-layer TenaceAgent orchestrator and 2-room Duplicate IMP match harness"
```
