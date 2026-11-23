# Hill-Climb Iter 5: 1NT Overcall Invites & 5-Level Penalty Double Discipline

**Goal:** Patch the two auction judgment leaks identified in **`seed=300` Bd 1 (`-7 IMPs`)** and **`seed=100` Bd 1 (`-8 IMPs`):**
1. **`1NT` overcall (`15–18 HCP`) game invitations (`seed=300` Bd 1):** Partner passed `1NT` holding `8 HCP` + 5-card suit + good intermediates (`T6542 / Q4 / A5 / QT95`), missing a cold `3NT+2` (`+460` vs `+150`, `-7 IMPs`). Add explicit SAYC guideline: invite `2NT` (or transfer/Stayman) with `8–9 HCP` opposite `15–17`/`15–18` `1NT` opening/overcall, bid `3NT` with `10+ HCP` (`Systems On`).
2. **High-level penalty double discipline (`seed=100` Bd 1):** `North` doubled `5♠` with a **freak `1-4-7-1` single-spade hand (`S:2`)** expecting `H:AK` + `D:AK` to cash against a known Unusual `2NT` two-suiter (costing `-200` extra when `5♠X` made `-650`). Clarify inside `SAYC_CONVENTION_CARD` / `Strategist`: avoid doubling 4/5-level distributional contracts on side-suit `A/K` tricks that are likely to be ruffed without trump control (`Qx/Jxx+`).

## Verification
- Add focused prompt check in `tests/test_layer1_and_layer2.py`, run quick test suite, commit & push, then run **Batch 3 (`seed=500, 600, 700` — 12 fresh duplicate boards)** at `--dds-worlds 12` against live `BEN`.
