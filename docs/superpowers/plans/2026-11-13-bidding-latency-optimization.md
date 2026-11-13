# Cloud Run Bidding & Tactical Latency Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to write failing unit tests before modifying `dds_evaluator.py`, `clients.py`, `cli.py`, `server.py`, or `Dockerfile`.

**Goal:** Cut interactive Cloud Run `/v1/step` bidding latency from ~48 seconds down to ~6–8 seconds on initial board strategy turns and ~1.5–2.2 seconds on subsequent turns, while eliminating `sh: 1: free: not found` subprocess overhead in the container.

**Root Cause Analysis (4-Phase Systematic Debugging):**
1. **Unbatched C++ Double-Dummy Solves in Auction (`~25.9s` locally / `~35s` on Cloud Run 2 vCPUs)**:
   - `RuleDDSEvaluator.score_candidates` called `_rollout_value` separately for each of the 36 legal auction calls across 16 worlds (`35 * 16 = 560` separate `score_for_contracts` calls), causing OpenSpiel's C++ `CalcDDTable` to solve the identical 52-card double-dummy table 35 times per world instead of once per world.
   - Because contract indices resulting from `[cand.action, Pass, Pass, Pass]` depend purely on the public auction history, resolving candidate contract indices once and calling `world.score_for_contracts(obs_idx, unique_contract_indices)` **once per world** reduces C++ `CalcDDTable` calls by **35x** with bit-for-bit identical EVs and make probabilities.
2. **Missing `procps` (`free`) in `python:3.13-slim` (`Dockerfile`)**:
   - Bo Haglund's C++ `dds` library invokes `free` (`procps`) to size transposition tables; missing `/usr/bin/free` in `python:3.13-slim` caused repeated `sh: 1: free: not found` shell forks in Cloud Run logs.
3. **Unbounded Hidden Thinking Tokens on Layer-2 Gemini Evaluator (`~12.1s` -> `~1.3s`)**:
   - `gemini-3-flash-preview` spent ~12.1s generating ~2,000 hidden thought tokens per tactical option scoring call. Passing `thinking_level="MINIMAL"` on `gemini-3*` Layer-2 evaluators (and `thinking_budget=0` on `gemini-2.5-flash*` Layer-2 evaluators) while keeping full deep strategic thinking on Layer-1 `gemini-4-argon` cuts Layer-2 latency ~9x and drastically lowers preview TPM pressure.

**Files:**
- Modify: `tests/test_layer3_dds_evaluator.py`
- Modify: `tests/test_model_clients.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_iac_and_container.py`
- Modify: `src/tenace/layer3_rules/dds_evaluator.py`
- Modify: `src/tenace/models/clients.py`
- Modify: `src/tenace/cli.py`
- Modify: `src/tenace/server.py`
- Modify: `Dockerfile`

---

### Task 1: Batch Auction `score_for_contracts`, Fast Layer-2 Thinking Config & Container `procps`

- [ ] **Step 1: Write failing tests (`RED`)**
  - `tests/test_layer3_dds_evaluator.py`: Verify `score_candidates` calls `score_for_contracts` at most once per world during the auction and matches per-candidate `_rollout_value` across `Pass`, contract bids, `Dbl`, `RDbl`, and passouts.
  - `tests/test_model_clients.py` & `tests/test_cli.py`: Verify `VertexGeminiClient` forwards `thinking_level` / `thinking_budget` into `ThinkingConfig`, and `build_clients_for_profile` sets fast tactical thinking on Layer-2 `gemini` evaluators (`"MINIMAL"` on `gemini-3*`, `0` on `gemini-2.5-flash*`) without altering Layer-1 Strategist thinking.
  - `tests/test_iac_and_container.py`: Verify `Dockerfile` installs `procps`.
- [ ] **Step 2: Run targeted `pytest` to witness RED failures**
- [ ] **Step 3: Implement changes in `dds_evaluator.py`, `clients.py`, `cli.py`, `server.py`, and `Dockerfile`**
- [ ] **Step 4: Run full `uv run pytest -q` to verify GREEN**
- [ ] **Step 5: Commit, push, build `tenace:v0.2.1`, apply `dev`, and measure live `/v1/step` latency via `127.0.0.1:8080`**
