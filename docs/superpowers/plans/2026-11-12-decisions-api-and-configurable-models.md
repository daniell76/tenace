# Tenace Phase 3: Gemini Decisions API (`gemini-decision-1`) & Configurable Head-to-Head Model Benchmarking Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` and `superpowers:test-driven-development` to implement this plan task-by-task. Every step requires watching a RED test failure before writing production code, followed by `verification-before-completion`.

**Goal:** Add `GeminiDecisionsClient` (`gemini-decision-1` via `google-genai>=3.2.0` `client.decisions.create`) as a first-class zero-output-token Layer-2 Evaluator alongside `DiffusionGemmaClient` and `VertexGeminiClient`, and make Layer-1/Layer-2 models fully configurable per team (including `--opponent tenace` head-to-head model comparison and per-team telemetry) in `tenace-match`.

**Architecture:** Upgrade `google-genai>=3.2.0` and implement `GeminiDecisionsClient` in `src/tenace/models/clients.py`, issuing a single `client.decisions.create` prefill request containing 5-level `ScoreQuestion`s (`0..4` normalized to `[0.0, 1.0]`) for each legal `CandidateOption` plus an optional `NoulQuestion` (`P(true) >= violation_threshold`) for the closed-loop assumption-violation interrupt. Generalize `src/tenace/layer2_evaluator/evaluator.py` (`TacticalOptionEvaluator`, keeping `DiffusionOptionEvaluator` as a backwards-compatible alias) to pass structured `CandidateOption`s directly when supported by the client. Finally, extend `DuplicateMatchRunner` with per-team telemetry and `tenace-match` (`src/tenace/cli.py`) with `ModelProfile`, `--evaluator-backend {decisions,diffusion-gemma,gemini}`, and `--opponent tenace` head-to-head comparison flags.

**Tech Stack:** Python 3.13 (`uv`), `google-genai >= 3.2.0`, `google-cloud-aiplatform >= 1.84.0`, `open-spiel >= 2.0.2`, `pydantic >= 2.9`, `pytest`.

---

## File Structure

```text
tenace/
├── pyproject.toml                           # Modify: bump google-genai>=3.2.0
├── README.md                                # Modify: document Decisions API + head-to-head model benchmarking
├── docs/superpowers/plans/
│   └── 2026-11-12-decisions-api-and-configurable-models.md
├── src/tenace/
│   ├── __init__.py                          # Modify: export GeminiDecisionsClient, TacticalOptionEvaluator
│   ├── cli.py                               # Modify: ModelProfile, --evaluator-backend decisions, --opponent tenace
│   ├── harness/
│   │   └── duplicate.py                     # Modify: add team_a_* and team_b_* telemetry counters
│   ├── layer2_evaluator/
│   │   ├── __init__.py                      # Modify: export TacticalOptionEvaluator & DiffusionOptionEvaluator
│   │   └── evaluator.py                     # Modify: model-neutral TacticalOptionEvaluator + direct evaluate_options path
│   └── models/
│       ├── __init__.py                      # Modify: export GeminiDecisionsClient
│       └── clients.py                       # Modify: implement GeminiDecisionsClient (Vertex AI & Gemini API v1beta)
└── tests/
    ├── test_model_clients.py                # Modify: add GeminiDecisionsClient unit & validation tests
    ├── test_layer1_and_layer2.py            # Modify: test TacticalOptionEvaluator with GeminiDecisionsClient
    ├── test_agent_and_duplicate_harness.py  # Modify: test per-team telemetry in head-to-head Tenace matches
    └── test_cli.py                          # Modify: test --evaluator-backend decisions & --opponent tenace comparison
```

---

### Task 1: Upgrade `google-genai>=3.2.0` & Implement `GeminiDecisionsClient` + `TacticalOptionEvaluator`

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/tenace/models/clients.py`
- Modify: `src/tenace/models/__init__.py`
- Modify: `src/tenace/layer2_evaluator/evaluator.py`
- Modify: `src/tenace/layer2_evaluator/__init__.py`
- Modify: `src/tenace/__init__.py`
- Test: `tests/test_model_clients.py`
- Test: `tests/test_layer1_and_layer2.py`

- [ ] **Step 1: Upgrade `google-genai>=3.2.0` in `pyproject.toml` (`uv sync --extra dev`) and write failing unit tests in `tests/test_model_clients.py` and `tests/test_layer1_and_layer2.py`**
  - Test `GeminiDecisionsClient`:
    1. Builds one `client.decisions.create(model="gemini-decision-1", input=..., questions=...)` request with safe keys `"cand_0"`, `"cand_1"`, ... (`type="score"`, 5-level `0..4` rubric criteria) plus `"assumption_violated"` (`type="noul"`) when `has_assumptions=True` (and omits `"assumption_violated"` when `has_assumptions=False`).
    2. Normalizes `ScoreAnswer.score` (`0.0..4.0`) to `[0.0, 1.0]` (`score / 4.0`, clamped & finite-checked) keyed by original `CandidateOption.label` (`"S:A"`, `"1NT"`, etc.).
    3. Triggers `TacticalDecision.assumption_violated = True` when `NoulAnswer.noul >= violation_threshold` (default `0.70`), populating `violation_reason` and `rationale` with calibrated probabilities and confidence.
    4. Also satisfies `StructuredModelClient` (`generate_structured(..., schema=TacticalDecision)`) by parsing candidate lines from `user_prompt` (and rejecting unsupported schemas like `StrategyState` with a clear `ValueError`), so `GeminiDecisionsClient` works seamlessly anywhere a Layer-2 `StructuredModelClient` is expected.
    5. Validates constructor arguments (`model` non-blank, `violation_threshold` finite in `(0.0, 1.0)`, Vertex AI vs Gemini API `v1beta` `api_key` client construction, and empty/missing `answers` error handling).
  - Test `TacticalOptionEvaluator` (and `DiffusionOptionEvaluator` alias):
    1. Uses `client.evaluate_options(context_text=..., candidates=..., has_assumptions=...)` directly when available (`GeminiDecisionsClient`), passing exact `CandidateOption` objects and `bool(strategy.assumptions)`.
    2. Falls back to `client.generate_structured(...)` for `DiffusionGemmaClient`, `VertexGeminiClient`, and mock clients.

- [ ] **Step 2: Run `pytest` to confirm RED failure**
  - Run: `~/.local/bin/uv run pytest tests/test_model_clients.py tests/test_layer1_and_layer2.py -v`
  - Expected: `FAIL` with `ImportError: cannot import name 'GeminiDecisionsClient'`.

- [ ] **Step 3: Implement `GeminiDecisionsClient` and `TacticalOptionEvaluator`**
  - Export `GeminiDecisionsClient` in `tenace.models.clients`, `tenace.models`, and `tenace`.
  - Export `TacticalOptionEvaluator` (with `DiffusionOptionEvaluator = TacticalOptionEvaluator`) in `tenace.layer2_evaluator.evaluator`, `tenace.layer2_evaluator`, and `tenace`.

- [ ] **Step 4: Run `pytest` to confirm GREEN**
  - Run: `~/.local/bin/uv run pytest -v`
  - Expected: All existing + new tests pass (`0` warnings/errors from `tenace`).

- [ ] **Step 5: Commit Task 1**
  - Commit message: `feat(models,layer2): add GeminiDecisionsClient for zero-token Layer-2 option evaluation`

---

### Task 2: Configurable Model Profiles, Head-to-Head `--opponent tenace` Benchmarking & Per-Team Telemetry

**Files:**
- Modify: `src/tenace/harness/duplicate.py`
- Modify: `src/tenace/cli.py`
- Modify: `README.md`
- Test: `tests/test_agent_and_duplicate_harness.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for per-team telemetry, `--evaluator-backend decisions`, per-backend default model resolution, and `--opponent tenace` head-to-head model comparison**
  - In `tests/test_agent_and_duplicate_harness.py`:
    - Verify `MatchSummary.telemetry` records both aggregate (`strategist_calls`, `evaluator_calls`, `forced_moves`) and per-team counters (`team_a_strategist_calls`, `team_a_evaluator_calls`, `team_a_forced_moves`, `team_b_strategist_calls`, `team_b_evaluator_calls`, `team_b_forced_moves`) accurately across Open Room (Team A = N/S, Team B = E/W) and Closed Room (Team B = N/S, Team A = E/W).
  - In `tests/test_cli.py`:
    1. Verify `--evaluator-backend` supports `decisions`, `diffusion-gemma`, and `gemini`:
       - When `--evaluator-model` is omitted (`None`), resolves automatically to `"gemini-decision-1"` for `decisions`, `"publishers/google/models/diffusiongemma-26b-a4b-it"` for `diffusion-gemma`, and `"gemini-2.5-flash"` for `gemini`!
       - Explicit `--evaluator-model` overrides any backend's default.
       - Supports `--decisions-auth {vertex,api-key}` (default `"vertex"`) and `--decisions-violation-threshold` (default `0.70`, validated in `(0.0, 1.0)`).
    2. Verify `--opponent tenace` builds an independent Team B `TenaceAgent` profile using `--opponent-strategist-model`, `--opponent-strategist-location`, `--opponent-evaluator-backend`, `--opponent-evaluator-model`, `--opponent-evaluator-location`, `--opponent-evaluator-endpoint-id`, `--opponent-decisions-auth`, and `--opponent-decisions-violation-threshold` (inheriting Team A's setting for any omitted `--opponent-*` flag, and running in both `--mode mock` and `--mode vertex`).
    3. Verify `format_match_report` displays model profile tags in team headers when comparing models and displays per-team telemetry breakdown when Team B has non-zero model telemetry (`--opponent tenace`).

- [ ] **Step 2: Run `pytest` to confirm RED failure**
  - Run: `~/.local/bin/uv run pytest tests/test_cli.py tests/test_agent_and_duplicate_harness.py -v`
  - Expected: `FAIL`.

- [ ] **Step 3: Implement per-team telemetry in `duplicate.py`, configurable `ModelProfile` & `--opponent tenace` in `cli.py`, and benchmark documentation in `README.md`**
  - Keep `--evaluator-backend` default `"decisions"`? Wait: let's check existing user expectations and CLI ergonomics:
    - User said: *"keep diffusion-gemma as an option, actually we should make the model configurable, so that later we can compare the performance of different models"*.
    - Making `--evaluator-backend` choices `["decisions", "diffusion-gemma", "gemini"]` with default `"decisions"` (or `"diffusion-gemma"`? Let's check: user first asked *"can use just use this API instead of diffusion gemma?"* and then *"keep diffusion-gemma as an option, actually we should make the model configurable, so that later we can compare the performance of different models"* — so `"decisions"` (`gemini-decision-1`) as the default Layer-2 backend with `"diffusion-gemma"` and `"gemini"` kept as configurable options matches both statements cleanly!).
    - Update `README.md` with clear CLI recipes for:
      - Running with Gemini Decisions API (`--evaluator-backend decisions`, default)
      - Running with DiffusionGemma (`--evaluator-backend diffusion-gemma`) or Gemini (`--evaluator-backend gemini`)
      - Head-to-head Duplicate IMP benchmarking between two model configurations (`--opponent tenace --opponent-evaluator-backend diffusion-gemma`, or comparing `gemini-4-argon` vs `gemini-2.5-pro`).

- [ ] **Step 4: Run full `pytest` and `tenace-match` head-to-head smoke benchmark (`verification-before-completion`)**
  - Run: `~/.local/bin/uv run pytest -v && ~/.local/bin/uv run tenace-match --mode mock --opponent tenace --opponent-evaluator-backend diffusion-gemma --boards 1 --dds-worlds 2`
  - Expected: All tests pass and head-to-head match report prints both team profiles and per-team telemetry cleanly.

- [ ] **Step 5: Commit Task 2**
  - Commit message: `feat(cli,harness): add configurable model profiles and head-to-head Tenace benchmarking`
