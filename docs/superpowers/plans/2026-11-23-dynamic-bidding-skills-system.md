# Dynamic Bidding-System Skills Architecture (`SKILL.md` Registry)

**Goal:** Enable **Tenace** to dynamically discover, load, hot-reload, and switch Bridge bidding systems at runtime using portable Markdown skills (`SKILL.md` with YAML frontmatter), without retraining any neural network or hardcoding system-specific logic inside prompt wrappers.

## Architecture Overview

1. **Skill Files (`src/tenace/skills/bidding/<id>/SKILL.md`):**
   Each bidding system is a self-contained directory containing a `SKILL.md` with strict YAML-style frontmatter (`id`, `name`, `version`, `description`, optional `nt_range`, `slam_control`, `author`, `tags`) and two progressive-disclosure Markdown sections:
   - **`## Summary Rules`**: Compact, high-density canonical bidding rules injected into **both** Layer 1 (`StrategistEngine`) and Layer 2 (`TacticalOptionEvaluator` / `GeminiDecisionsClient`) so the fast option evaluator always stays aligned and low-token.
   - **`## Reference Guide`** *(optional)*: Detailed opening tables, response ladders, competitive adjustments, and slam-control conventions loaded dynamically into **Layer 1 (`StrategistEngine`) only during the auction phase**.

2. **Built-in Bidding Skills (6 systems shipped inside wheel):**
   - `sayc` — SAYC (Standard American Yellow Card, default; preserves all 100-board hill-climbed rules)
   - `21gf` — Two-Over-One Game Forcing (`2/1 GF`, forcing `1NT`, `RKCB 1430`)
   - `ccba` — CCBA Natural (Chinese Contract Bridge Assoc. `2/1 GF`, semi-forcing `1NT`, inverted minors, `RKCB 1430`)
   - `precision` — Precision Club (strong artificial `1C` 16+ HCP, limited `1D/1H/1S` 11–15, `1NT` 13–15, `2C` 6+ clubs)
   - `acol` — British Acol (4-card majors, weak `1NT` 12–14 HCP, 2/1 forcing 1 round)
   - `polish_club` — Polish Club (`WJ`, 3-way `1C`, `1D` negative over `1C`, `2C` limited `5+C`, `RKCB 1430`)

3. **Zero-Dependency `SkillRegistry` (`src/tenace/skills/registry.py`):**
   - Lightweight, strict frontmatter parser (no `PyYAML` dependency required; rejects malformed/blank metadata and duplicate keys).
   - Layered discovery precedence (`later overrides earlier` by slug `id`):
     1. Built-in `src/tenace/skills/bidding/*/SKILL.md`
     2. External directories from `extra_dirs` / `TENACE_SKILLS_DIR` (`os.pathsep`-delimited)
   - **Progressive disclosure + hot reload:** `list_skills()` reads frontmatter metadata only; `load(id)` caches full parsed `BiddingSkill` keyed by `(path, mtime_ns)` so edits to external `SKILL.md` files take effect automatically on the next turn.
   - **Security boundary:** Remote HTTP endpoints pass `allow_file_path=False` with strict slug validation (`^[a-z0-9][a-z0-9_-]{0,63}$`) to prevent arbitrary filesystem path traversal; CLI (`tenace-match --system path/to/SKILL.md`) passes `allow_file_path=True` for direct local files.

4. **System-Agnostic Layer 1 & Layer 2 Prompts:**
   - Remove hardcoded SAYC rules (`12+ HCP / Rule of 20`, `5C=0/4..5S=3` straight Blackwood) from static wrapper text in `strategist.py`, `evaluator.py`, and `clients.py` so non-SAYC systems (`precision`, `ccba`, `acol`, `polish_club`, `21gf`) never receive conflicting instructions.
   - Backwards-compatible `ConventionCard` gains optional `id: str = "custom"` and `detailed_rules: str = ""` (with `SAYC_CONVENTION_CARD` sourced directly from `sayc/SKILL.md`).

5. **CLI, FastAPI Server & Web UI Integration:**
   - **CLI (`tenace-match`):** `--system`, `--opponent-system` (when `--opponent tenace`), `--skills-dir`, `--list-systems`, env `TENACE_BIDDING_SYSTEM` & `TENACE_SKILLS_DIR`.
   - **API (`tenace-serve`):**
     - `GET /v1/skills` → `{default_system, skills: [BiddingSkillMeta...]}`
     - `GET /v1/skills/{skill_id}` → full `BiddingSkill` (`summary_rules`, `detailed_rules`, metadata)
     - `POST /v1/step` → optional `bidding_system: str | None` or inline `convention_card: ConventionCard | None` (mutually exclusive)
     - `POST /v1/match` → optional `team_a_system: str | None`, `team_b_system: str | None` (`team_b_system` requires `opponent='tenace'`)
     - Preserves exact `/health` / `/healthz` JSON schema.
   - **Web UI (`src/tenace/web/`):** Bidding system `<select>` dropdowns in both Play Table and Duplicate Match Lab (Team A & Team B), plus an expandable rules viewer powered lazily by `GET /v1/skills/{id}`.

## Verification Plan (TDD: RED → GREEN)
1. Author failing unit tests in `tests/test_bidding_skills.py` covering frontmatter parsing, registry discovery, external dir overrides, hot-reload on `mtime_ns` bump, direct file loading, path-traversal rejection, `ConventionCard` conversion, and zero SAYC leakage when running `precision` / `ccba` / `acol`.
2. Verify **RED**, implement `src/tenace/skills/`, refactor `conventions.py`, `strategist.py`, `evaluator.py`, `clients.py`, `cli.py`, `server.py`, and `web/`.
3. Add focused tests in `tests/test_cli.py`, `tests/test_server.py`, `tests/test_layer1_and_layer2.py`, and `tests/test_web_ui.py`.
4. Run full `uv run pytest -q`, `node --check src/tenace/web/app.js`, public-repo hygiene scan, and commit & push to `origin/main`.
