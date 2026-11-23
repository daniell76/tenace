# Hill-Climb Iter 9: SAYC `Rule of 20` Opening & Prime-Control Discipline

**Goal:** Eliminate the `-12 IMP` passed-out vulnerable game bug uncovered on `seed=1100` Bd 4 (and tighten `1m–1M–1NT–3M` acceptance from Bd 2, `-6 IMPs`):

1. **Bd 4 (`-12 IMPs`, Passed Out vs cold vul `4H=` `-620`):** In 1st seat, West passed `S:AK8 H:87532 D:K C:QT97` (`12 raw HCP`, `5-4-3-1`, `2` quick tricks `S:AK`, **Rule of 20 = `21`**) by over-discounting singleton `D:K` and weak hearts (`87532`). Both `SAYC_CONVENTION_CARD` and Layer-2's unopened-auction discipline warned only against *over-opening* weak hands, without instructing the agent to **open** `12+ HCP` / `Rule of 20` hands (`HCP + two longest suits >= 20`, `2+` quick tricks).
2. **Bd 2 (`-6 IMPs`, `3S+1` `-170` vs `4S=` `-420`):** After `1C–1S–1NT–3S` (invitational with `6+` spades), East passed holding `12 HCP` (`2.5` quick tricks: `CAKT2`, `DK75`, `SJ7`, zero wasted quacks). Document prime-control acceptance (`A`/`K` rich + trump honor) on `3M` invites.

## Changes (TDD: RED -> GREEN)
1. Update `tests/test_layer1_and_layer2.py` asserting `SAYC_CONVENTION_CARD` contains `Rule of 20` opening guidance and `TacticalOptionEvaluator` surfaces opening discipline for unopened auctions. Run `pytest` to confirm **RED**.
2. Update `src/tenace/layer1_strategist/conventions.py` (`SAYC_CONVENTION_CARD.summary_rules`).
3. Update `src/tenace/layer1_strategist/strategist.py` (auction system prompt guidance) and `src/tenace/layer2_evaluator/evaluator.py` (unopened-auction `discipline_line`).
4. Run targeted `pytest`, verify public-repo hygiene, commit & push to `origin/main`.
