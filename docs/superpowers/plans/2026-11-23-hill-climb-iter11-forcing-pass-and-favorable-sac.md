# Hill-Climb Iter 11: Competitive `Forcing Pass` & Cuebid Sacrifice Handling

**Goal:** Fix the `-11 IMP` competitive auction leak caught on `seed=1500` Bd 2 (`4S-2` undoubled `+100` vs cold vul `4H+1` `+650`):

- **Open Room (`1H - 2S - 3S(cuebid) - 4S - Pass - Pass - Pass`):** After North made a `3S` limit+ raise cuebid establishing our side's balance of strength (vulnerable!), East jumped non-vul to `4S`. South (`13 HCP`, singleton `S:2`) passed intending a cooperative decision, and North **passed out `4S` undoubled** (`+100` instead of **`4SX-2` `+300` or cold `5H` `+650`!**). Note that `SAYC_CONVENTION_CARD`'s Iter 5 sentence cautioned against doubling distributional two-suiters on A/K side honors — North read that too conservatively and refused to double *even a known non-vul 4S sacrifice after our own game-inviting cuebid!*
- **Closed Room (`1H - 2S - 3H - 3S`):** At **favorable vulnerability** with a known 10-card `♠` fit (`2S` weak jump + 4-card support `KJ84`), raising straight to the Law of Total Tricks level (`4S`) pressures the vulnerable opponents as `BEN` did (`-300` vs `-650`).

## Changes (TDD: RED -> GREEN)
1. Add assertion in `tests/test_layer1_and_layer2.py` checking `SAYC_CONVENTION_CARD.summary_rules` distinguishes reckless doubles from doubling **known opponent sacrifices** (`never let them play undoubled after a cuebid/game force`). Verify **RED**.
2. Update `src/tenace/layer1_strategist/conventions.py` and `src/tenace/layer2_evaluator/evaluator.py`.
3. Run targeted `pytest`, hygiene-check, commit & push to `origin/main`, then launch the final 6 boards (`seed=2400` [4 bds] + `seed=2500` [2 bds]) to reach **100 / 100**.
