# Auction Bidding Quality Fix (Eliminate DDS `Dbl`/Runout/`6C` Escalation) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development (`RED -> GREEN`) before modifying production code.

**Goal:** Fix three root-cause auction defects causing Tenace AI to force light openings (`Pass` pruned on 11 HCP), over-double low-level contracts (`Dbl`), and endlessly rescue/escalate all the way to slams (`6C` beating `5C` on undoubled DDS EV).

**Root Cause Analysis (Verified on Board 1, `seed=42`):**
North holds `11 HCP`, zero Aces (`S:642 H:Q95 D:KQ C:KJ642`). Both `gemini-4-argon` (L1) and `gemini-decision-1` (L2, `Pass=0.80` vs `1C=0.65`) wanted to **`Pass`**, yet `RuleDDSEvaluator.select_guarded_action` forced an escalation ladder:
1. **`Pass` pruned by `max_make_prob_drop` gate during auction:** On an unopened board, `Pass` scores `0.0` (`make_prob=0.500`), whereas `1C` makes `7` tricks in `87.5%` of random deals (`make_prob=0.875`). The card-play blunder threshold (`0.875 - 0.20 = 0.675`) pruned `Pass` completely!
2. **Infinite `Dbl` → undoubled runout escalation:** `score_candidates` evaluates `[cand, Pass, Pass, Pass]` assuming opponents never double or rescue. `Dbl` doubles opponents' penalties (`+800` vs `+194`), then escaping a doubled contract (`3NTX = -500`) to a higher undoubled bid (`4D = -150`) looks strictly profitable.
3. **Undoubled slam bonus distortion (`6C > 5C`):** Making `6C` pays `+920` vs down 1 undoubled `-50`, giving `6C` higher naive EV (`-35.0`) than `5C` (`-53.8`). Passing `baseline_ev` inside each `decisions.create` question and mixing `0.45 * norm_ev` inside `select_guarded_action` pushed marginal auctions straight to `6C`.

**Files:**
- Modify: `src/tenace/layer3_rules/dds_evaluator.py` (phase-aware `select_guarded_action(..., phase="play")`)
- Modify: `src/tenace/layer2_evaluator/evaluator.py` (`convention_card` propagation + SAYC auction context; omit naive DDS EV table from `direct_eval` auction calls)
- Modify: `src/tenace/models/clients.py` (clean SAYC auction rubric in `GeminiDecisionsClient.evaluate_options` when `is_auction=True`)
- Modify: `src/tenace/agent.py` (wire `convention_card` to `TacticalOptionEvaluator` and pass `phase=facts.phase` to `select_guarded_action`)
- Modify: `tests/test_layer3_dds_evaluator.py`, `tests/test_layer1_and_layer2.py`, `tests/test_model_clients.py`

---

### Task 1: Phase-Aware Auction Selection & SAYC-Grounded Layer-2 Evaluation

- [ ] **Step 1: Add failing `RED` unit tests** covering:
  - `select_guarded_action(..., phase="auction")` keeping `Pass` (opening), `Pass` vs `6C`, `Pass` vs `Dbl`, and artificial conventions (`2C` Stayman) without card-play `make_prob` pruning or `45%` naive EV bias.
  - `TacticalOptionEvaluator` injecting `convention_card` and bidding discipline during `auction`.
  - `GeminiDecisionsClient.evaluate_options` emitting SAYC auction rubric questions without misleading `baseline_ev=+800.0` bias on auction calls while keeping full DDS numbers on `play` tricks.
- [ ] **Step 2: Run targeted `pytest` to witness `RED`**
- [ ] **Step 3: Implement `GREEN` changes across `dds_evaluator.py`, `evaluator.py`, `clients.py`, and `agent.py`**
- [ ] **Step 4: Run full `uv run pytest -q` and live 4-seat `TenaceAgent` auction trace (Boards 1–4)**
- [ ] **Step 5: Commit, push, build `tenace:v0.2.3`, deploy `dev` via Terraform, and verify live via `127.0.0.1:8080`**
