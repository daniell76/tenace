# Cloud Run GFE `/health` Route & Upstream SDK Error Mapping Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to write failing unit and Terraform tests before touching production files.

**Goal:** Expose a GFE-accessible `GET /health` endpoint alongside `/healthz` on Cloud Run and map upstream `google-genai` / `google-api-core` model exceptions in `/v1/step` and `/v1/match` to structured HTTP `502 Bad Gateway` responses instead of unhandled `500` errors.

**Root Cause Analysis (4-Phase Systematic Debugging):**
1. **External `/healthz` 404**: Google Cloud Run's Google Front End (GFE) reserves `/*z` paths (`/healthz`, `/varz`, `/statusz`) on external `*.run.app` URLs while internal kubelet probes hit container port `8080` directly. Adding `GET /health` (while preserving `/healthz`) and pointing Cloud Run probes + docs at `/health` makes health checks succeed both internally and over `https://*.run.app/health`.
2. **Upstream `NotFoundError` returning `500` instead of `502`**: `src/tenace/server.py` caught `ValueError` around `agent.step` and `runner.play_match`, whereas `google-genai` and `google-cloud-aiplatform` raise `google.genai.errors.APIError`, `google.genai._gaos.lib.compat_errors.GenAiError`, or `google.api_core.exceptions.GoogleAPICallError` on upstream HTTP 4xx/5xx failures. Catching those upstream SDK error classes alongside `ValueError` returns structured `502` JSON diagnostics.

**Files:**
- Modify: `tests/test_server.py`
- Modify: `terraform/modules/tenace_cloudrun/tests/cloudrun.tftest.hcl`
- Modify: `src/tenace/server.py`
- Modify: `terraform/modules/tenace_cloudrun/main.tf`
- Modify: `README.md`
- Modify: `terraform/README.md`

---

### Task 1: TDD Fix for `/health` & Upstream SDK `502` Error Mapping

- [ ] **Step 1: Write failing tests in `tests/test_server.py` and `cloudrun.tftest.hcl`**
  - Assert `GET /health` returns `200 OK` matching `/healthz`.
  - Assert `google.genai.errors.APIError` and `google.genai._gaos.lib.compat_errors.GenAiError` raised inside `agent.step` / `runner.play_match` return HTTP `502` JSON (`"Upstream model evaluation error: ..."`).
  - Assert `startup_probe` and `liveness_probe` in `cloudrun.tftest.hcl` target `"/health"`.
- [ ] **Step 2: Run `pytest tests/test_server.py` and `terraform test` to verify RED failures**
- [ ] **Step 3: Update `src/tenace/server.py`, `terraform/modules/tenace_cloudrun/main.tf`, `README.md`, and `terraform/README.md`**
- [ ] **Step 4: Run `terraform test` and full `uv run pytest -q` to verify GREEN**
- [ ] **Step 5: Commit, push to `origin/main`, rebuild Cloud Run image, apply `dev`, and verify live `/health` + `/v1/step`**
