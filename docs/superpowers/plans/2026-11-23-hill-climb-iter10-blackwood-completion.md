# Hill-Climb Iter 10: Blackwood `4NT` Step Table & Post-Ace-Ask Slam Completion

**Goal:** Eliminate the `-11 IMP` Blackwood slam pass-out bug on `seed=1000` Bd 2 (same root family as `-13 IMP` on `seed=400` Bd 4):

1. **Bd 2 (`-11 IMPs`, `5S+1` `-480` vs `6S=` `-980`):** After `2C-2D-2NT-3H(Jacoby)-3S-4NT`, East answered **`5S` (3 Aces — holding all 4 Aces combined!)**. `Layer 1` Strategist explicitly recommended **`Bid 6S`** (though noting ambiguity over whether `5S` showed 2 or 3 Aces!), yet `Layer 2` (`gemini-decision-exp`) overruled to **`Pass`** because:
   - `SAYC_CONVENTION_CARD` did not list the explicit Blackwood `4NT` response table (`5C=0/4, 5D=1, 5H=2, 5S=3 Aces`).
   - `TacticalOptionEvaluator` warned against bidding 6/7 without noting to *complete* a slam after a `4NT` Blackwood Ace check confirms 3 or 4 Aces.
   - `GeminiDecisionsClient`'s `_AUCTION_SCORE_CRITERIA[0]` broadly tagged `"slam without values/aces"` and `instructions` omitted post-Blackwood slam completions (`6M`/`6m`/`6NT`).
2. **Bd 3 (`-8 IMPs`, `4C-2` `+200` vs `3D+1` `-130`):** Over `1C - (2H) - P - (3H)`, avoid stretching to an unsupported 4-level minor contract opposite a passed partner when holding defensive Aces/stoppers (especially vulnerable).

## Changes (TDD: RED -> GREEN)
1. Add assertions in `tests/test_layer1_and_layer2.py` for the explicit Blackwood response table (`5C=0/4, 5D=1, 5H=2, 5S=3`) and post-Blackwood `6-level` completion wording in `TacticalOptionEvaluator`. Verify **RED**.
2. Update `src/tenace/layer1_strategist/conventions.py`, `src/tenace/layer2_evaluator/evaluator.py`, and `src/tenace/models/clients.py`.
3. Run targeted `pytest`, hygiene-check, commit & push to `origin/main`.
