# Hill-Climb Iter 12: Dummy/Support-Point Jump Raises & Major Suit Priority Over Minor Overcalls

**Goal:** Address the three clear SAYC auction leaks exposed on `seed=1900` (`-28 IMPs`):

1. **Bd 4 (`-12 IMPs`, `4S+3` `-710` vs `6S=` `-1430`):** After `1H - 1S`, West held `15 HCP`, `4-5-2-2` (`17` support/dummy points with 4-card spade support and two doubletons) and only single-raised to `2S` (`12-15`). In SAYC, opener counts **shortness support points** (`+1` doubleton, `+3` singleton, `+5` void) with 4-card support: jump raise `3S` with `16-18` support pts (just as `BEN` did, unlocking `6S` opposite East's `13 HCP` + heart void!).
2. **Bd 2 (`-7 IMPs`, `3C+2` `-150` vs `4S=` `-420`):** After `(1D) - 2C - (Pass) - ?`, East held `S:QJT64 H:A86 D:64 C:KT8` (`10 HCP`, `5-3-2-3`) and cuebid `2D` (`clubs limit raise`), completely burying the **5-card major** `2S`. Always prefer showing a **5+ card major** over partner's minor overcall (`2C`/`2D`) instead of a minor-suit cuebid raise!
3. **Bd 1 (`-7 IMPs`, `2S+2` `-170` vs `4D=` `+130`):** Holding shortness (`void/singleton`) in an opponent's 2-level overcall (`1D - 1NT - 2S`) with opening values (`13 HCP`, `1-4-5-3`), balance/compete via a takeout double rather than selling out cheaply to `2S` (preventing double partscore swings).

## Changes (TDD: RED -> GREEN)
1. Add assertions in `tests/test_layer1_and_layer2.py` for opener jump-raise support points (`16-18`) and major-suit priority over partner's minor overcall. Verify **RED**.
2. Update `src/tenace/layer1_strategist/conventions.py` (`SAYC_CONVENTION_CARD.summary_rules`).
3. Run targeted `pytest`, verify public-repo hygiene, commit & push to `origin/main`.
