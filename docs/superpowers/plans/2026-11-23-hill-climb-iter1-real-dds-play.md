# Hill-Climb Iter 1: Exact C++ Double-Dummy Card Play (`SolveBoard`), Trick Context & Replan Throttle

**Goal:** Eliminate the three card-play and latency root causes uncovered in our first live duplicate match against `lorserker/ben` (`Board 2, seed=42`: identical `1NT by West` auctions in both rooms, `-6 IMP` swing purely from card-play noise `♦A` discard and dropping `♦Q` under `♦A`, and 21 `L1` strategist calls).

## Root Causes Diagnosed

1. **Random-card playouts instead of Double-Dummy Solver (`src/tenace/layer3_rules/dds_evaluator.py`):**
   - `_rollout_value` played `rollout_rng.choice(legal)` during card play, producing noisy coin-flip `baseline_ev` and `make_prob` values.
   - `open_spiel`'s bundled `pyspiel.so` already exports Bo Haglund's **DDS 2.9.0** `SolveBoard` (`extern "C"`) and `SetMaxThreads`. Calling `SolveBoard(deal, target=-1, solutions=3, mode=0, &fut, threadIndex=0)` scores **all legal candidate cards simultaneously in one C++ call per world** with zero extra Python packages or container dependencies.
   - Pair with an exact WBF duplicate scoring function (`score_bridge_contract`) verified against OpenSpiel's C++ `score_for_contracts` across non-vul/vul, undoubled/doubled/redoubled, making/overtrick/down scores.

2. **Unbounded card-play EV loss inside `select_guarded_action`:**
   - When all lines have `make_prob == 0.0` (e.g. down 1 vs down 3) or `make_prob == 1.0` (overtricks), the `make_prob` gate prunes nothing. Adding a configurable `max_ev_drop` guard (`default=45.0` pts during `phase="play"`, bypassed for `phase="auction"`) blocks gratuitous trick-losing plays (`-150` vs `-50`, dropping `♦Q` under `♦A` = `240 pt` loss) while letting Layer-2 break close single-dummy ties (`<= 45 pt` gap).

3. **Flat unlabeled `Played Cards` prompt context (`layer1_strategist/strategist.py`, `layer2_evaluator/evaluator.py`):**
   - Provide `format_play_state(facts)` showing `Tricks Won` (`Declarer vs Defense`), completed trick-by-trick sequence (`T1: N:D:6 E:D:T S:D:K W:D:4 -> SOUTH`), and explicit **`Current Trick`** (`Trick 5: S:D:A (led DIAMONDS, winning: SOUTH D:A)` or `leading new trick`).

4. **Runaway `play`-phase Strategist replanning (`src/tenace/agent.py`):**
   - `surprise_events` fired on the agent's *own* void, visible *dummy* void, and late-game tricks (e.g. trick 9 runouts), generating up to 7 slow Layer-1 calls per seat per board (`21` on 2 boards).
   - In `TenaceAgent`, keep initial play plan + dummy-reveal plan intact, ignore non-surprising self/dummy show-outs during replan gating, suppress late endgame auto-replans (`len(played_cards) >= 32`), and bound per-board `play` replans (`max_play_replans=3`, while always honoring explicit `assumption_violated` interrupt tests when within budget or preserving at least 1 interrupt slot).

## TDD Plan (`RED -> GREEN`)

1. Add failing tests in `tests/test_layer3_dds_evaluator.py`, `tests/test_layer1_and_layer2.py`, and `tests/test_agent_and_duplicate_harness.py`:
   - Exact `score_bridge_contract` cross-check against OpenSpiel `score_for_contracts` & classic WBF cases (`1NT`, `3NT`, `4SX`, `4SXX`, `6H`, `7NTXX`, non-vul/vul undertricks `down 1..6`).
   - Exact `SolveBoard` mid-hand test on **`Board 2` (`seed=42`) Closed Room Trick 5 (`S:D:A` led)**: `D:2` scores `+90` (`make_prob=1.0`), `D:Q` scores `-150` (`make_prob=0.0`) under exact deal!
   - Exact `SolveBoard` opening-lead & partial-trick test (`0`, `1`, `2`, `3` cards already played in current trick), plus interop with `score_for_contracts` in the same process.
   - `select_guarded_action` `max_ev_drop` blocks `-150` vs `-50` even when both have `make_prob=0.0` (while letting close `20 pt` tie go to Layer-2).
   - `format_play_state` renders trick-by-trick history, trick counts, and active `Current Trick` line inside both Layer-1 and Layer-2 prompts.
   - `TenaceAgent` skips unnecessary late-game/self-void replans while preserving initial, dummy-reveal, early hidden-seat show-out, and `assumption_violated` interrupt contracts.
2. Implement cleanly in `src/tenace/layer3_rules/dds_evaluator.py`, `src/tenace/layer1_strategist/strategist.py`, `src/tenace/layer2_evaluator/evaluator.py`, and `src/tenace/agent.py`.
3. Verify `uv run pytest -q` (all tests green), re-play **Board 1–2 (`seed=42`)** against real **BEN** to confirm the `-6 IMP` swing vanishes and runtime drops sharply, commit & push, then advance into the fresh duplicate board campaign vs **BEN**.
