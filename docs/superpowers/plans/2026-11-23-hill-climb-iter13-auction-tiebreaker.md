# Hill-Climb Iter 13: L1-Guided Auction Tie-Breaker & Explicit Jacoby Transfer Mapping

**Goal:** Eliminate the `-6` to `-13 IMP` auction tie-breaking leaks where Layer 2 (`gemini-decision-exp`) tied or scored nearly equal between two candidate calls and `select_guarded_action` broke ties using **naive pass-out `dd_table` EV** (`[call, Pass, Pass, Pass]`):

- **`seed=2200` Bd 1 (`-6 IMPs`):** West held `5` spades (`S:A7653 H:K3 D:52 C:J953`). `Layer 1` wrote `"Bid 2H (Jacoby Transfer to Spades) over partner's 1NT opening."`, yet `Layer 2` picked **`2D` (Jacoby Transfer to Hearts!)**, stranding the pair in a 4–2 heart fit (`4H-2` `-100` vs `3S=` `+140`).
- Earlier instances of the exact same pattern: `seed=400` Bd 4 (`-13 IMPs`, `L1` recommended `4NT` Blackwood, `L2` picked `4S`), `seed=1000` Bd 2 (`-11 IMPs`, `L1` recommended `6S`, `L2` picked `Pass`).

## Design (Minimal, Proportional, 100% Backwards-Compatible)
1. **`conventions.py`:** Unpack compressed slash notation `"2D/2H Jacoby Transfers to Hearts/Spades"` into explicit per-bid labels (`"2D = Jacoby Transfer to HEARTS (5+ H); 2H = Jacoby Transfer to SPADES (5+ S)"`, and same for `3D`/`3H` over `2NT`).
2. **`dds_evaluator.py`:** Add optional keyword-only `preferred_action: int | None = None` and `auction_tie_margin: float = 0.08` parameters to `select_guarded_action`. When `phase == "auction"` and `preferred_action` matches a legal candidate whose `L2` score is within `auction_tie_margin` of the highest `L2` score (i.e. near-tied on the `0–4` score rubric, `<= 0.32` pts), select `preferred_action` rather than breaking the tie with distorted pass-out `dd_table` EV. When `L2` strongly prefers another bid (`gap > 0.08`), `L2` retains full veto authority.
3. **`agent.py`:** Add pure helper `extract_recommended_auction_action(plan_summary, candidates) -> int | None` and call it **only** when `phase == "auction"` and `self.strategy` was freshly generated on the current turn (so it can never replay a stale previous-round call!). Skip when the opening sentence offers an ambiguous disjunction (`"2H or 3H"`).

## Verification (TDD: RED -> GREEN)
- Add focused unit tests in `tests/test_layer3_dds_evaluator.py` and `tests/test_agent_and_duplicate_harness.py`.
- Confirm **RED**, implement changes, confirm **GREEN**, run full test suite (`test_layer1_and_layer2.py`, `test_layer3_dds_evaluator.py`, `test_agent_and_duplicate_harness.py`, `test_server.py -k stateless_loop`), check public-repo hygiene, commit & push.
