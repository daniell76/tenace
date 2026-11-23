# Dynamic Bidding-System Hill-Climbing Calibration Harness

> **For agentic workers:** Build with strict TDD (RED → GREEN → REFACTOR) and keep all public-repo changes strictly clean of local paths, project IDs, hostnames, API keys, or deprecated model names.

**Goal:** Design and implement `tenace.hillclimb` (`tenace-hillclimb` CLI) — a reproducible, versioned hill-climbing loop harness for calibrating any `SKILL.md` bidding system (`sayc`, `ccba`, `21gf`, `precision`, `acol`, `polish_club`, or external card) against a reference Bridge engine (`BEN` via BlueChip v18, `tenace`, or `baseline`). Expose every mutable part of the prompt explicitly, preserve full version history (`v000`, `v001`, … including rejected candidates and unified diffs), cache reference-room tables (`Table A = 4× ref`), and persist rich raw JSONL/CSV/Markdown data explaining *what changed* and *why performance improved*.

**Architecture:**
1. **`src/tenace/harness/scoring.py`**: Add official WBF 2013 continuous `20–0` VP formula `wbf_continuous_vp(imp_diff, num_boards)`.
2. **Explicit Tunable Prompt Variables** (`conventions.py`, `registry.py`, `strategist.py`, `evaluator.py`):
   - Extend `ConventionCard` with optional backward-compatible fields (`calibration_notes: str = ""`, `l1_auction_directive: str = ""`, `l2_discipline_overrides: dict[str, str] = {}`).
   - Support optional `## Calibration Notes` heading in `SKILL.md` plus lossless round-trip writer `render_skill_markdown(...)`.
   - Export default prompt-slot constants (`DEFAULT_L1_AUCTION_DIRECTIVE`, `DEFAULT_L2_DISCIPLINE_PARTNER_ACTIVE`, `DEFAULT_L2_DISCIPLINE_COMPETITIVE`, `DEFAULT_L2_DISCIPLINE_UNOPENED`) so every mutable byte sent to Layer 1 / Layer 2 is explicitly inspectable, hashable, diffable, and overridable without modifying Python source code.
3. **`src/tenace/hillclimb/` package:**
   - `models.py`: Frozen Pydantic models for `PromptVariables`, `SkillVersionMeta`, `BoardTelemetryRecord`, `IterationRecord`, `ABRetestRecord`, `ProposedPromptPatch`, `RunManifest`, rendered prompt template snapshotter, and slot-level unified diff helper.
   - `diagnostics.py`: Full 52-card deal extractor (`hands`, `hand_hcp`, `shapes`), first auction divergence finder, deterministic `swing_category` tagger (`push`, `card_play_gain`/`loss`, `auction_slam_swing`, `auction_game_swing`, `auction_doubled_penalty`, `auction_passed_out_swing`, `auction_competitive_partscore`).
   - `store.py`: Append-only, crash-safe `HillClimbStore` managing:
     - `manifest.json` & `best/SKILL.md`
     - `versions/vNNN/{SKILL.md, variables.json, rendered_prompts.json, diff_from_parent.patch, metadata.json}`
     - `raw/{boards.jsonl, iterations.jsonl, ab_retests.jsonl}`
     - `raw/reference_cache/<engine_tag>/seed_<seed>_bd_<n>.json` (eliminates redundant `4×BEN` replays across retests & multi-system benchmarks!)
     - `analysis/{report.md, changelog.md, boards.csv, versions.csv}`
   - `optimizer.py`: Pluggable `PromptProposer` protocol, `ManualPatchProposer`, and `GeminiSkillOptimizer` (`StructuredModelClient` -> `ProposedPromptPatch`) that inspects worst-swing boards and parent version history to propose minimal, falsifiable patches.
   - `runner.py`: `HillClimbEngine` supporting both `controlled` (`Table A = 4× ref`, `Table B = challenger N/S vs ref E/W`) and `duplicate` (symmetric 2-room Open/Closed WBF) evaluation modes, parallel chunk execution via external worker support, exact matched-seed A/B retest gating (`accept` / `reject` + rollback), and multi-iteration automated `run_loop(...)`.
   - `cli.py`: `tenace-hillclimb` CLI with subcommands `init`, `eval`, `propose`, `retest`, `run`, `report`, `diff`, `show-prompts`.

**Execution Plan (TDD):**
1. Write `tests/test_hillclimb.py` defining all contract tests and run `uv run pytest tests/test_hillclimb.py` to confirm **RED**.
2. Implement `scoring.py` VP scale + `conventions.py` / `registry.py` / `strategist.py` / `evaluator.py` tunable prompt exports.
3. Implement `src/tenace/hillclimb/{models,diagnostics,store,optimizer,runner,cli,__init__}.py` and register `tenace-hillclimb` in `pyproject.toml`.
4. Run full `uv run pytest -q` (GREEN), inspect diff + wheel packaging + public-repo hygiene scan, commit and push to `origin/main`.
5. Run **5 × 12 dual games (`60` boards: `seed=3000, 3100, 3200, 3300, 3400`)** of **`ccba` vs `BEN`** using the new `HillClimbStore` harness (`seed=3000` anchored against cached `Table A` + 3-way `CCBA` vs `SAYC` vs `BEN` comparison!), export raw JSONL/CSV/Markdown reports, clean `ben` shelf DBs, and summarize key differences & actionable calibration candidates for `ccba`.
