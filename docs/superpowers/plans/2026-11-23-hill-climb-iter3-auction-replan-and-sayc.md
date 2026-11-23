# Hill-Climb Iter 3: Live Auction Replanning & Balanced SAYC Bidding Discipline

**Goal:** Eliminate the `-40 IMP` timid-pass auction bias exposed across Batch 1 (`seed=100` Bds 1–4 `-34 IMPs`, `seed=200` Bd 2 `-6 IMPs`) against live `lorserker/ben` (`GIB-BBO-8744`).

## Root Cause Found Across 5 of 5 Losing Boards
On every losing board of Batch 1, `TenaceAgent` only invoked Layer-1 (`StrategistEngine`) **once** during the auction (its Round-1 opening/initial turn). In subsequent auction rounds—even after partner opened `1H`, overcalled `1S`, raised `2H`, or bid a **forcing new suit `2S` over a weak `2D`**—`StrategistEngine` never replanned. Layer-2 (`gemini-decision-exp`, zero-output-token) evaluated Round 2+ turns against a **stale Round-1 `plan_summary`** (e.g. `"Pass initially with 9 HCP"`) alongside an unconditional `"Pass without opening strength (12+ HCP)"` discipline prompt.

## Surgical Fixes
1. **Event-driven auction replanning (`src/tenace/agent.py`):**
   - When `facts.phase == "auction"` and `self.strategy` already exists in `"auction"` phase, trigger a Layer-1 `StrategistEngine` update whenever **any non-Pass call** (`bid`, `Dbl`, `RDbl`) has occurred since `self.seat`'s previous call in `facts.auction_calls` (derived purely from public `facts.auction_calls`, preserving 100% parity with stateless `POST /v1/step`!).
   - Keep `4-pass` passouts and single-call rounds at **zero** extra API calls.
2. **Context-aware SAYC auction discipline (`src/tenace/layer2_evaluator/evaluator.py`):**
   - Separate *unopened* auction guidance (12+ HCP / Rule of 20 / Rule of 15 in 4th seat, weak twos, preempts) from *active/partner-opened* auction guidance (respond with 6+ HCP to a 1-level suit opening; NEVER pass forcing new-suit bids, weak-two responses, Stayman/transfers, or negative doubles without a 4+ trump stack; invite/bid game with 16+ HCP opposite a single major raise; cuebid/raise overcalls with a fit).
3. **Richer `SAYC_CONVENTION_CARD` summary (`src/tenace/layer1_strategist/conventions.py`):**
   - Include standard SAYC forcing/competitive mechanics (responses to 1-level openings & weak twos, single-raise game tries, overcall cuebid raises, negative doubles, Unusual 2NT, Michaels).
4. **Actionable Strategist prompt wording (`src/tenace/layer1_strategist/strategist.py`):**
   - Instruct Layer-1 in `auction` phase to lead `plan_summary` with the concrete recommended next call given the **live** auction, avoiding stale Round-1 advice bleeding into later rounds.
5. **Stateless API parity (`src/tenace/server.py`):**
   - Verified with `test_step_stateless_loop_matches_stateful_agent_telemetry_on_board`.

## Verification Plan
- `RED -> GREEN` unit tests in `tests/test_agent_and_duplicate_harness.py` and `tests/test_layer1_and_layer2.py` verifying auction replan trigger on partner's non-Pass call (and NO replan on all-Pass sequences), plus unopened vs partner-acted discipline text in Layer-2 prompts.
- Full `uv run pytest -q` regression suite (including `test_server.py` stateless parity).
- Retest **`seed=100` (4 boards)** vs live **`BEN`** to measure exact IMP gain vs `-34 IMP` baseline, then continue fresh 100-board hill climb batches (`seed=300`, `seed=400`, ...).
