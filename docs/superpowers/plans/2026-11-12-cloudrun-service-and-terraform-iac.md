# Cloud Run Service, Batch Job & Multi-Environment Terraform IaC Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Containerize Tenace for Google Cloud Run with a stateless FastAPI HTTP service (`tenace-serve` exposing `/healthz`, `/v1/step`, and `/v1/match`) plus batch Cloud Run Job execution (`tenace-match`), backed by modular multi-environment Terraform IaC (`bootstrap`, `environments/dev`, `environments/prod`) with a GCS remote state backend and gitignored local variable/backend files (`terraform/local/`).

**Architecture:**
1. **Single Container Image, Dual Cloud Run Workloads**:
   - `Dockerfile` (`python:3.13-slim`, non-root user) installs `tenace` and defaults to `CMD ["tenace-serve"]` binding `0.0.0.0:${PORT:-8080}` for the **Cloud Run Service**.
   - The **Cloud Run Job** reuses the exact same Artifact Registry image with `command = ["tenace-match"]` and configurable benchmark CLI flags (`job_args`) for long-running WBF Duplicate IMP benchmarks without HTTP request timeouts.
2. **Stateless Imperfect-Information HTTP API (`src/tenace/server.py`)**:
   - `GET /healthz`: Readiness/liveness endpoint reporting service version and active default `ModelProfile.short_label()`.
   - `POST /v1/step`: Chooses the next legal bid or card for a single seat using either imperfect-information hands (`own_hand` + optional post-lead `dummy_hand` + `action_history`) or a deterministic benchmark board (`tournament_seed` + `board_number` + `action_history`), optionally accepting `prior_strategy` (`StrategyState`) so horizontally autoscaled Cloud Run instances preserve Layer-1 Strategist caching across turns with zero sticky state.
   - `POST /v1/match`: Runs a symmetric two-room WBF Duplicate IMP match (`baseline` or `tenace` head-to-head model comparison; never allows arbitrary subprocess commands over HTTP).
3. **Multi-Environment Terraform IaC + Gitignored Local Variables (`terraform/`)**:
   - `terraform/bootstrap/`: Provisions foundational Google Cloud APIs, the versioned GCS Terraform state bucket (`uniform_bucket_level_access = true`, `public_access_prevention = "enforced"`, object versioning + noncurrent retention), and the shared Artifact Registry Docker repository.
   - `terraform/modules/tenace_cloudrun/`: Reusable module provisioning a dedicated least-privilege runtime Service Account (`roles/aiplatform.user`, `roles/logging.logWriter`, optional Secret Manager secret accessor for `GEMINI_API_KEY`), `google_cloud_run_v2_service`, optional `allUsers` invoker binding (`allow_unauthenticated = false` by default), and `google_cloud_run_v2_job`.
   - `terraform/environments/dev/` & `terraform/environments/prod/`: Environment stacks using `backend "gcs" {}` partial configuration, committed `.tfvars.example` and `.tfbackend.example` templates, and gitignored local configuration under `terraform/local/{bootstrap,dev,prod}/`.

**Tech Stack:** Python 3.13, FastAPI, Uvicorn, HTTPX, OpenSpiel (`pyspiel`), Docker (`python:3.13-slim`), Terraform `>= 1.6` (`hashicorp/google ~> 6.0`, `terraform test` with `mock_provider`).

---

### Task 1: FastAPI Cloud Run Service (`tenace-serve`) & Containerization (`Dockerfile`, `.dockerignore`)

**Files:**
- Modify: `pyproject.toml`
- Create: `src/tenace/server.py`
- Modify: `src/tenace/__init__.py`
- Create: `Dockerfile`
- Create: `.dockerignore`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write failing tests in `tests/test_server.py`**
  Cover:
  - `GET /healthz` status, version, and env-configured default `ModelProfile` (`TENACE_MODE=mock`, `TENACE_STRATEGIST_MODEL`, `TENACE_EVALUATOR_BACKEND`, etc.).
  - `POST /v1/step` in imperfect-information `own_hand` auction mode, post-lead `dummy_hand` card-play mode (both declarer playing own hand and declarer playing dummy's cards), and `(tournament_seed, board_number)` benchmark mode; verify `prior_strategy` reuse skips redundant Strategist calls on routine play turns and verify strict input validation (`422` on duplicate/missing cards, premature `dummy_hand` during auction, wrong turn seat, illegal action in `action_history`, or terminal board).
  - `POST /v1/match` running both `baseline` and head-to-head `tenace` opponents in `mock` mode and returning IMP summary, formatted scorecard, and per-team telemetry.
  - `parse_action_token` / `parse_card_token` round-trip helpers and `resolve_server_bind` (`0.0.0.0`, `$PORT` validation).
- [ ] **Step 2: Run pytest to verify RED failure**
  Run: `uv run pytest tests/test_server.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'tenace.server'`).
- [ ] **Step 3: Implement `pyproject.toml` dependencies/script, `src/tenace/server.py`, `Dockerfile`, and `.dockerignore`**
  - Add `fastapi>=0.115.0` and `uvicorn>=0.32.0` to `[project].dependencies`, `httpx>=0.28.0` to `[project.optional-dependencies].dev`, and `tenace-serve = "tenace.server:main"` to `[project.scripts]`.
  - Implement `create_app`, `/healthz`, `/v1/step`, `/v1/match`, imperfect-information OpenSpiel deal/state reconstruction, and `main()` in `src/tenace/server.py`.
  - Export `create_app` in `src/tenace/__init__.py`.
  - Create production `Dockerfile` (`python:3.13-slim`, non-root `tenace` user, `PORT=8080`, `CMD ["tenace-serve"]`) and `.dockerignore`.
- [ ] **Step 4: Run pytest to verify GREEN**
  Run: `uv run pytest -v`
  Expected: PASS across all existing + new server tests.
- [ ] **Step 5: Commit Task 1**
  Commit message: `feat(server,docker): add Cloud Run FastAPI service and container image`

---

### Task 2: Multi-Environment Terraform IaC (`bootstrap`, `dev`, `prod`), GCS State Backend & Gitignored Local Configs

**Files:**
- Modify: `.gitignore`
- Create: `terraform/bootstrap/versions.tf`
- Create: `terraform/bootstrap/variables.tf`
- Create: `terraform/bootstrap/main.tf`
- Create: `terraform/bootstrap/outputs.tf`
- Create: `terraform/bootstrap/terraform.tfvars.example`
- Create: `terraform/bootstrap/tests/bootstrap.tftest.hcl`
- Create: `terraform/modules/tenace_cloudrun/versions.tf`
- Create: `terraform/modules/tenace_cloudrun/variables.tf`
- Create: `terraform/modules/tenace_cloudrun/main.tf`
- Create: `terraform/modules/tenace_cloudrun/outputs.tf`
- Create: `terraform/modules/tenace_cloudrun/tests/cloudrun.tftest.hcl`
- Create: `terraform/environments/dev/{backend.tf,versions.tf,variables.tf,main.tf,outputs.tf,backend.tfbackend.example,terraform.tfvars.example}`
- Create: `terraform/environments/prod/{backend.tf,versions.tf,variables.tf,main.tf,outputs.tf,backend.tfbackend.example,terraform.tfvars.example}`
- Create: `terraform/README.md`
- Modify: `README.md`
- Test: `tests/test_iac_and_container.py`
- Local-only (gitignored, NOT committed): `terraform/local/{bootstrap,dev,prod}/*`

- [ ] **Step 1: Write failing tests in `tests/test_iac_and_container.py` (and `.tftest.hcl` specs)**
  Assert:
  - `.gitignore` ignores `terraform/local/**`, `*.tfvars`, `*.tfbackend`, `*.tfstate*`, `**/.terraform/`, while allowing `*.tfvars.example` and `*.tfbackend.example`.
  - `Dockerfile` exposes/uses `8080` / `PORT`, runs as a non-root user, and `.dockerignore` excludes `terraform/` and `.git/`.
  - `terraform fmt -check -recursive terraform`, `terraform validate` on `bootstrap`, `environments/dev`, and `environments/prod`, and `terraform test` on `bootstrap` and `modules/tenace_cloudrun` all exit `0`.
  - Zero tracked files contain real GCP project IDs or corporate internal URLs.
- [ ] **Step 2: Run pytest to verify RED failure**
  Run: `uv run pytest tests/test_iac_and_container.py -v`
  Expected: FAIL (missing Terraform directories / `.gitignore` rules).
- [ ] **Step 3: Implement `.gitignore`, Terraform `bootstrap`, `modules/tenace_cloudrun`, `environments/{dev,prod}`, docs, and local `terraform/local/` configs**
  - Configure `backend "gcs" {}` in both `dev` and `prod` environments.
  - Populate `terraform/local/{bootstrap,dev,prod}/` locally on disk with the user's active GCP project variables while verifying `git status` ignores `terraform/local/` completely.
  - Document end-to-end Cloud Build + Terraform bootstrap/dev/prod commands in `terraform/README.md` and `README.md`.
- [ ] **Step 4: Run `terraform test`, `terraform validate`, and full `uv run pytest -v` to verify GREEN**
  Expected: All Terraform tests and Python tests pass with clean `git status`.
- [ ] **Step 5: Commit Task 2**
  Commit message: `feat(iac): add multi-env Cloud Run Terraform stacks with GCS backend`
