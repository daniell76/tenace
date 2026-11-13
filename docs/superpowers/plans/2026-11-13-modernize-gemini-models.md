# Modernize Gemini Model Defaults (`gemini-3.8-flash` & `gemini-4-argon`) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to write failing unit tests (`RED`) before modifying production code (`GREEN`).

**Goal:** Eliminate all outdated `gemini-2.5*` and `gemini-3-*` model references across Tenace, standardize Layer-2 Gemini tactical evaluation on `gemini-3.8-flash` (or `gemini-4-argon`) with `thinking_level="LOW"`, and fix `VertexGeminiClient` open-dict JSON schema handling so `gemini-3.8-flash` (`LOW`) reliably emits valid `TacticalDecision` and `StrategyState` objects in live play.

**Root Cause & Live Verification Findings:**
1. Outdated model IDs (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-flash-preview`) were still present in `cli.py`, `web/{index.html,app.js}`, `README.md`, tests, and local `terraform.tfvars`.
2. Live Vertex AI testing in `global` confirmed `gemini-3.8-flash` and `gemini-4-argon` both support `thinking_level="LOW"`, **provided** `VertexGeminiClient.generate_structured` uses `response_mime_type="application/json"` + prompt schema instruction (`_format_schema_instruction(schema)`) rather than `response_json_schema=schema.model_json_schema()` (which malfunctions on open-key `additionalProperties` dicts like `TacticalDecision.option_scores` and `StrategyState.seat_constraints`).

**Files:**
- Modify: `tests/test_model_clients.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_server.py`
- Modify: `tests/test_web_ui.py`
- Modify: `tests/test_iac_and_container.py`
- Modify: `src/tenace/models/clients.py`
- Modify: `src/tenace/layer2_evaluator/evaluator.py`
- Modify: `src/tenace/cli.py`
- Modify: `src/tenace/web/index.html`
- Modify: `src/tenace/web/app.js`
- Modify: `README.md`

---

### Task 1: Upgrade Layer-2 Default to `gemini-3.8-flash` (`thinking_level="LOW"`) & Remove Outdated Models

- [ ] **Step 1: Write failing tests (`RED`)**
  - `tests/test_model_clients.py`: Verify `VertexGeminiClient` uses `response_mime_type="application/json"` with `_format_schema_instruction(schema)` (no `response_json_schema` open-dict bug) and forwards `thinking_level="LOW"` on `gemini-3.8-flash`.
  - `tests/test_cli.py` & `tests/test_server.py`: Verify `DEFAULT_GEMINI_EVALUATOR_MODEL == "gemini-3.8-flash"`, `build_clients_for_profile` passes `thinking_level="LOW"` on Layer-2 `gemini` evaluators (`gemini-3.8-flash` and `gemini-4-argon`) while leaving Layer-1 `gemini-4-argon` with default deep thinking (`thinking_level=None`).
  - `tests/test_web_ui.py` & `tests/test_iac_and_container.py`: Verify Web UI defaults/datalists use `gemini-3.8-flash` and `gemini-4-argon`, and enforce a repo hygiene test banning `gemini-2.5` and `gemini-3-` across `src/`, `README.md`, and `tests/`.
- [ ] **Step 2: Run targeted `pytest` to witness RED failures**
- [ ] **Step 3: Implement GREEN changes in `clients.py`, `evaluator.py`, `cli.py`, `web/{index.html,app.js}`, and `README.md`**
- [ ] **Step 4: Run full `uv run pytest -q` to verify GREEN**
- [ ] **Step 5: Commit, push, build `tenace:v0.2.2`, apply `dev` (`evaluator_model = "gemini-3.8-flash"`), and verify live `/v1/step` latency via `127.0.0.1:8080`**
