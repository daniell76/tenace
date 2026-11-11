# Strategist Default Model Update (`gemini-4-argon`) Implementation Plan

> **For agentic workers:** Adheres strictly to Superpowers `test-driven-development` and `verification-before-completion`.

**Goal:** Switch Tenace's Layer-1 Strategist default model from `gemini-2.5-pro` to `gemini-4-argon` across the CLI, `VertexGeminiClient`, tests, and documentation.

**Architecture:** Update `DEFAULT_STRATEGIST_MODEL` in `src/tenace/cli.py` and the default `model` parameter of `VertexGeminiClient` in `src/tenace/models/clients.py` to `"gemini-4-argon"` (served via Vertex AI `global` endpoint), keeping `--evaluator-backend gemini` fallback (`DEFAULT_GEMINI_EVALUATOR_MODEL = "gemini-2.5-flash"`) distinct and overridable.

**Tech Stack:** Python 3.13 (`uv`), `google-genai`, `pytest`.

---

### Task 1: Switch Layer-1 Strategist Default Model to `gemini-4-argon`

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_model_clients.py`
- Modify: `src/tenace/cli.py`
- Modify: `src/tenace/models/clients.py`
- Modify: `README.md`

- [ ] **Step 1: Update tests first (RED)**
  - In `tests/test_cli.py`: assert `build_arg_parser().parse_args([]).strategist_model == "gemini-4-argon"` and verify `run_cli` wires `"gemini-4-argon"` as the default strategist model in Vertex mode.
  - In `tests/test_model_clients.py`: assert `VertexGeminiClient(genai_client=fake_genai)` defaults `model` to `"gemini-4-argon"`.

- [ ] **Step 2: Run `pytest` to confirm RED failure**
  - Run: `~/.local/bin/uv run pytest tests/test_cli.py tests/test_model_clients.py -v`
  - Expected: `FAIL` (`AssertionError` comparing `'gemini-2.5-pro'` vs `'gemini-4-argon'`).

- [ ] **Step 3: Implement model default update (GREEN)**
  - Set `DEFAULT_STRATEGIST_MODEL = "gemini-4-argon"` in `src/tenace/cli.py`.
  - Set `model: str = "gemini-4-argon"` in `VertexGeminiClient.__init__` (`src/tenace/models/clients.py`).
  - Update Layer-1 Strategist documentation and examples in `README.md` to `Gemini 4 Argon` (`gemini-4-argon`).

- [ ] **Step 4: Run full verification suite (`verification-before-completion`)**
  - Run: `~/.local/bin/uv run pytest -v && ~/.local/bin/uv run tenace-match --help`
  - Expected: `77 passed` and `--strategist-model` help displays `(default: gemini-4-argon)`.

- [ ] **Step 5: Commit changes**
  - Commit message: `feat(strategist): switch default Layer-1 Strategist model to gemini-4-argon`
