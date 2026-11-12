# Public Repository Sanitization & Release Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement the `--project` default change test-first before rewriting repository history and pushing to GitHub.

**Goal:** Remove hardcoded GCP project defaults, workstation absolute paths, corporate package mirror lockfiles, and corporate author emails across both HEAD and Git history before publishing `daniell76/tenace` publicly over SSH.

**Architecture:** Update CLI `--project` resolution to default strictly to `$GOOGLE_CLOUD_PROJECT` (or `None` so Application Default Credentials resolve the active project), untrack `uv.lock` via `.gitignore` (keeping `uv.lock` locally unversioned while `pyproject.toml` specifies standard PyPI dependencies), convert absolute plan paths to repo-relative paths, scrub historical commits via `git filter-branch`, verify zero sensitive patterns across `git log -p`, verify full `pytest` suite, and push `main` over SSH (`git@github.com:daniell76/tenace.git`).

**Tech Stack:** Python 3.13, `pytest`, `git`, `gh` CLI.

---

### Task 1: Remove Hardcoded Default GCP Project ID via TDD

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/tenace/cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests in `tests/test_cli.py`**
  Assert that `build_arg_parser().parse_args([]).project is None` when `GOOGLE_CLOUD_PROJECT` is unset, empty (`""`), or whitespace (`"   "`).
- [ ] **Step 2: Run targeted pytest to verify RED failure**
  Run: `uv run pytest tests/test_cli.py -k "defaults_to_vertex_and_gcp_env or strips_strings" -v`
  Expected: FAIL (`AssertionError` because `project` previously fell back to a hardcoded project ID).
- [ ] **Step 3: Implement minimal fix in `src/tenace/cli.py` & `README.md`**
  Remove `DEFAULT_GCP_PROJECT` from `src/tenace/cli.py` so `default_project` resolves to `(os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip() or None`, and replace example project IDs in `README.md` with `your-gcp-project-id`.
- [ ] **Step 4: Run pytest to verify GREEN**
  Run: `uv run pytest tests/test_cli.py -v`
  Expected: PASS.

### Task 2: Untrack `uv.lock`, Sanitize Plan Paths & Scrub Git History

**Files:**
- Modify: `.gitignore`
- Modify: `docs/superpowers/plans/*.md`

- [ ] **Step 1: Add `uv.lock` to `.gitignore`, untrack `uv.lock`, and normalize `docs/superpowers/plans/*.md` to repo-relative paths**
- [ ] **Step 2: Commit cleanly and scrub all historical commits (`git filter-branch` + `git gc --prune=now`)**
  Remove `uv.lock` from historical trees, normalize historical paths/project placeholders, and rewrite commit author/committer identity to `Feng Xia <7255723+daniell76@users.noreply.github.com>`.
- [ ] **Step 3: Verify full test suite & zero internal patterns across `git log -p`**
  Run: `uv run pytest -q` and `git log -p` grep audit before creating `daniell76/tenace` and pushing over SSH.
