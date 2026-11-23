# Hill-Climb Iter 7: SAYC Up-the-Line 4-Card Major Rebid Over `1C–1D`

**Goal:** Eliminate the `-12 IMP` right-siding / major-fit bug uncovered on **`seed=500` Bd 4 (`3NT-1` by W vs `3NT=` by E)**:
- Holding `A853.Q75.A98.KT7` (`13 HCP`, 4=3=3=3) after `1C – P – 1D – P`, Tenace rebid **`1NT`** bypassing `4` spades, wrong-siding `3NT` into North's killer spade lead (`QT62` opposite `KJ`).
- Standard SAYC *up-the-line* rule: over partner's `1D` response to `1C`, **always rebid a 4-card major (`1H`/`1S`) instead of bypassing it with `1NT`**, discovering 4–4 major fits and right-siding NT contracts.

## Verification
- Update `SAYC_CONVENTION_CARD` in `src/tenace/layer1_strategist/conventions.py`, assert in `tests/test_layer1_and_layer2.py`, run `pytest`, commit & push, and continue the live 100-board parallel climb (`seed=1000`, `seed=1100`, ...).
