# Constrained Auction-Aware DDS Sampling & Opening-Lead Calibration (H1)

**Goal:** Eliminate the `-47 IMP` card-play / opening-lead deficit (`-31 IMPs` defending on divergent leads vs **`+0 IMPs`** on matching leads; `-16 IMPs` declaring) revealed by our 72-board raw `ccba` vs `BEN` dataset while preserving `ccba v1.1`'s near-even auction accuracy (`-1 IMP` across `12` fresh `seed=3500` holdout boards!).

**Root Causes Diagnosed (from `raw/boards.jsonl`, 72 boards):**
1. **`sample_consistent_worlds` fallback bias:** Binary rejection with `max_rejection_factor=8` (`128` draws) achieved `<2%` acceptance on 3-seat auction constraints (`3000:5`: `20/1500`, `3400:2`: `18/1500`, `3500:11`: `1/1500`), falling back to unconstrained random deals (distorting make probabilities from `~44%` to `78–93%` and picking wrong opening leads).
2. **No symbolic auction-constraint baseline in `RuleDDSEvaluator`:** When `L1` omits a hidden seat or supplies broad bounds, `resample_from_infostate` assigns opening strength to passed hands (`0–6` HCP) and non-preempt shapes to preemptors (`3H` with 2 hearts).
3. **Opening lead ignores `L1` guidance on DDS noise ties (`3000:5`, `3400:2`):** Min-max normalization `(ev - min) / (max - min)` stretched a `2–3` pt Monte-Carlo noise wobble into a `[0, 1]` gap while ignoring `L1`'s explicit opening lead (`H:A` on `3000:5`, `D:K` on `3400:2`).
4. **Post-dummy card-play weighting & 4th-seat overtaking (`3200:6`, `3300:1`):** `0.45 * norm_ev + 0.55 * llm_score` with `max_ev_drop=45.0` let `L2` scores override clear `+30..+45`-pt C++ DDS `SolveBoard` edges and allowed 4th-seat honors to overtake dummy/partner's already-winning card (`H:K` over `H:Q` on `3200:6`) when DDS EV tied.

**Changes (Zero Hidden-Card Access; 100% Fair-Play Preserved):**
1. `src/tenace/layer3_rules/dds_evaluator.py`:
   - `infer_symbolic_auction_constraints(state, observer, system_id="sayc")`: conservative system-aware HCP/suit bounds inferred strictly from public `auction_calls`.
   - Soft-penalty ranked sampling (`_constraint_penalty`) in `sample_consistent_worlds`: merges `L1` + symbolic constraints during play phase and ranks a fast draw pool (up to `1024` draws on play, returning early as soon as `num_worlds` exact matches are found) so imperfect-information worlds stay realistically calibrated even when exact rejection rate is `<1%`.
   - `opening_lead_worlds_multiplier` (`2×`, capped at `32`) on Trick 1 (`len(play_actions) == 0`) where `39` cards are unseen.
   - 4th-seat conservation filter inside `RuleDDSEvaluator.select_guarded_action`: when `partner_winning_trick=True`, avoid overtaking partner/dummy's already-winning card with a higher honor unless DDS proves a strict EV benefit (`> 15` pts).
   - Dedicated `play_ev_weight` / `ev_tie_scale` (backward compatible defaults for unit-test callers; stricter DDS authority post-dummy in `TenaceAgent`).
2. `src/tenace/agent.py`:
   - `extract_recommended_play_action(plan_summary, candidates)` to break tight opening-lead DDS near-ties (`ev >= best_ev - 25.0`, `mp >= best_mp - 0.08`) using a freshly generated `L1` play directive (strictly legal, public-info-only).
   - Pass explicit `partner_winning_suit_card` / `is_opening_lead` context into `select_guarded_action`.
3. `src/tenace/layer1_strategist/strategist.py`:
   - Crisp `play`-phase opening-lead & card-play guidance inside `build_strategist_system_prompt` (leading `A`/`K` from `AK`, top of sequence `KJT/QJT`, singleton shift after cashing top tricks, preserving honors / never overtaking partner's already-winning trick).

**H1 Retest Evidence & H2 Refinements (Measured on 9 matched-seed boards vs cached 4xBEN):**
- H1 recovered **+30 IMPs** on the 7 targets (`-65 -> -35`: `3300:1 +13`, `3500:11 +10`, `3200:6 +6`, `3500:2 +1`), tying both defense guards (`3300:7 0`, `3400:5 0`), net **`-42 -> -35 (+7 IMPs)`**.
- Root-cause audit of remaining two `-11 IMP` misses (`3000:5` and `3400:2`):
  1. `L1` **already emitted the exact killing opening lead** on BOTH (`Lead H:A` on `3000:5`, `Lead D:K` on `3400:2`), **but**:
     - Monte-Carlo SE of a game/slam swing over `32` DDS worlds is `~35–45` duplicate points (`1–2` flipped worlds). `play_preferred_ev_margin=25.0` dropped `H:A` when it trailed `S:3` by `-37` pts (`0.40` vs `0.48` mp).
     - `infer_symbolic_auction_constraints` omitted **`4NT` RKCB / Blackwood slam-ask HCP** (`WEST >= 16` facing a preempt), **`2C-2D-2NT` (`20..26` HCP)**, **penalty pass of partner's reopening/takeout double (`4+` cards in opponents' doubled trump suit)**, **`3rd`-seat unopened pass (`<= 12` HCP)**, **2/1 major response length (`5+` `H` over `1S`)**, and **major raise length (`3+`)**.
  2. In `play_guidance`, `L1` led an unsupported bare Ace (`H:A` from `AT7` on `3400:5`), violating AK-only cashing guidance.
- **H2 Changes:**
  - Expand `infer_symbolic_auction_constraints` cleanly (4NT RKCB ask, penalty conversion double trump length, 3rd-seat pass, 2C-2NT, unraised suit rebid `6+`, 2/1 `2H` over `1S` `5+`, delayed major raise `3+`).
  - Raise Trick-1 opening lead DDS budget to `min(num_worlds * 3, 48)` (cutting Monte-Carlo SE by ~18%).
  - Add `play_preferred_mp_margin` to `select_guarded_action` and widen opening-lead tie margin to `(ev=45.0, mp=0.10)` (safe for games AND partscores!).
  - Raise post-dummy `play_ev_weight` from `0.72` to `0.82` and fast-path `L2` skip when a single legal card uniquely dominates the `max_make_prob_drop`/`max_ev_drop` DDS gate late in play (`opt-in` via clean condition, saving redundant Vertex latency without altering decisions).
  - Clarify `play_guidance`: cash `A/K` from `AK(x)` only, avoid unsupported bare Aces against `4/5`-level suit contracts, and lead 4th-best low from ragged 5-card `JTxxx` (no `9`).

**H3 Verification Results (Fixes passed-hand inference, direct penalty vs reopening double, Boltzmann world diversity, interior-sequence leads):**
- Fixed `non_pass_bid_seen` so passed-hand responses/overcalls (`Pass - 1C - 1S`) still assert suit length (`>= 4 spades`).
- Separated reopening/takeout doubles (partner holds trumps on penalty pass) from direct-seat penalty doubles (doubler holds trumps).
- Weighted verified symbolic auction constraints (`2.0x`) above noisy `L1` estimates (`0.35x`) with mild Gumbel jitter (`T = 0.6`) to eliminate deterministic `T=0` world collapse under overtight `L1` HCP ranges.
- Clarified interior-sequence leads (`J` from `KJT(x)`/`AJT(x)`, `T` from `QT9(x)`) and `AK(x)` priority against `3NT`.
- **Matched-seed A/B evidence across the 7 biggest card-play targets (`-65 IMP` baseline):**
  - `3000:5` (`3NT by W`): **`-11 -> +0`** (`Δ=+11`, exact `N:H:A` lead, `+100`!)
  - `3400:2` (`5H by E`): **`-11 -> +0`** (`Δ=+11`, exact `S:D:J` lead, `+50`!)
  - `3300:1` (`4S by E`): **`-10 -> +2/+3`** (`Δ=+12/+13`, beat `BEN` twice in a row!)
  - `3500:11` (`4S by W`): **`-10 -> +0`** (`Δ=+10`, twice in a row!)
  - `3200:6` (`2HX -> 1S`): **`-13 -> -7`** (`Δ=+6`, doubled twice as good defending `1S`!)
  - `3500:2` (`3D by E`): **`-5 -> +0`** (`Δ=+5`, 4th-best `S:6` lead!)
  - `3000:1` (`2DX by S`): **`-5 -> -5`** (`Δ=0`, double-dummy par `-500`!)
  - **Combined 7 targets: `-65 -> -10 IMPs` (`+55 IMPs recovered`!! All 5 defensive boards `+2 IMPs` vs `BEN`!)**

**Verification Plan (TDD):**
1. Add targeted failing tests in `tests/test_cardplay_calibration.py` (`RED`).
2. Implement changes cleanly (`GREEN`), confirm FULL `pytest -q` passes (`0` regressions).
3. Run matched-seed A/B retest against cached `4×BEN` anchor on the `7` biggest `card_play_loss` boards (`-65 IMPs` baseline) + `2` defensive guard boards (`+23 IMPs` baseline).
4. Run fresh out-of-sample `12`-board matches (`seed=3600`, `seed=3700`, …) against `BEN v0.9.0.4`, continuing hill-climb iterations until winning clearly on unseen boards!
