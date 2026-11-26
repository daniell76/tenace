# Opponent Convention Card Disclosure & Signal-Preserving Defensive Carding

**Goal:** Enable `Tenace` to (1) explicitly read and model the **opponents' disclosed bidding & carding system** (`opponent_convention_card`) across Layer 1, Layer 2, and Layer 3 (fair-play legal public info), and (2) **send and interpret defensive playing signals** (high/low attitude & count on touching spots, plus early defensive replan & symbolic carding observations on partner's suit switch).

**Root Gaps Addressed:**
1. **Single-System Assumption:** `TenaceAgent` and `infer_symbolic_auction_constraints` applied our *own* `system_id` to all 4 seats and never disclosed the opponents' convention card to `L1`/`L2` — breaking asymmetric matches (e.g., `ccba` vs `precision` strong `1C` or `acol` weak `1NT` / 4-card majors).
2. **Touching Spot-Card Erasure:** `collapse_equivalent_actions` merged touching spots (`H:4 3 2`, `D:6 5`) into a single highest-card option, making it physically impossible for defenders to signal low (discourage / odd count) vs. high (encourage / even count).
3. **Signal-Blind Defense & Stale Trick-1 Plans:** When partner won Trick 1 and switched to a new low side suit (e.g., `s3600:b6` `S:H:A -> S:D:3` singleton ruff shift), the winning defender did not trigger an `L1` replan, leaving stale Trick-1 plans in `L2` and unconstrained `DDS` worlds.

**Implementation Plan (Strict TDD, 100% Fair-Play — Public Info Only):**
1. **Per-Partnership Opponent System (`dds_evaluator.py`, `strategist.py`, `evaluator.py`, `agent.py`, `cli.py`, `server.py`, `hillclimb/runner.py`):**
   - Add `opponent_system_id: str | None = None` to `infer_symbolic_auction_constraints`, `RuleDDSEvaluator.sample_consistent_worlds`, and `RuleDDSEvaluator.score_candidates`, selecting `our_sys` vs `opp_sys` strictly by `seat.partnership == observer.partnership`.
   - Add `opponent_convention_card: ConventionCard | None = None` to `build_strategist_system_prompt`, `StrategistEngine`, `TacticalOptionEvaluator`, and `TenaceAgent`.
   - Render concise `Opponent Disclosed System` summary in `L1` and `L2` prompts (deduplicated when identical to our own system).
   - Wire automatic mutual disclosure in `cli.py` (`--opponent tenace`, plus optional `--opponent-system` for `--opponent bluechip`), `server.py` (`StepRequest.opponent_bidding_system` / `opponent_convention_card`, and `POST /v1/match` mutual disclosure), and `hillclimb/runner.py`.
   - Ensure all 6 built-in `SKILL.md` files include their defensive carding line inside `## Summary Rules` so opponent disclosure includes both bidding and carding agreements.
2. **Signal-Preserving Defensive Carding & Partner-Switch Replan (`tracker.py`, `strategist.py`, `evaluator.py`, `agent.py`, `dds_evaluator.py`):**
   - Add `preserve_signals: bool = False` to `collapse_equivalent_actions` (default `False` for full backward compatibility): when `True`, split touching spot groups (`len >= 2`, top rank `<= T`) into `[highest, lowest]` so defenders can actively send high (encourage / even count) vs. low (discourage / odd count) signals while keeping honor sequences (`AK`, `KQ`) intact.
   - Enable `preserve_signals=True` inside `TenaceAgent.step()` strictly when **defending**, **following suit or discarding** (`len(played) % 4 != 0`), and **early/mid-play** (`len(played) < 24`).
   - Add `extract_defensive_carding_notes(facts)` (strictly from public `played_cards`!) detecting partner's early suit switch (potential singleton ruff shift) and completed high-low / low-high count echoes; display in `L1`/`L2` play prompts when non-empty.
   - Add `_partner_early_suit_switch_trigger(facts, seat)` in `agent.py`: trigger a bounded `L1` replan when defending on lead at Trick 3/4 immediately after partner won the prior trick and switched to a new non-trump side suit.
   - Ensure `L1`'s `max_suit_lengths` constraint (e.g., partner's singleton ruff `max_suit_lengths[suit]=1`) carries crisp discriminative weight in `sample_consistent_worlds` (`sym_pen` + soft `l1_pen`) so constrained DDS automatically leads partner's ruff suit.

**Verification Plan:**
1. Add comprehensive failing unit tests in `tests/test_opponent_system_and_signals.py` (`RED`).
2. Implement all components cleanly (`GREEN`).
3. Run full `uv run pytest -q` (`0` failures, `0` skipped) and public-repo hygiene check (`0` hits).
4. Run live matched-seed verification on `s3600:b6` (`5S` partner singleton `D:3` switch, old `-13 IMPs`) + live asymmetric check (`ccba` vs `precision`), then commit, push, and optionally redeploy Cloud Run.
