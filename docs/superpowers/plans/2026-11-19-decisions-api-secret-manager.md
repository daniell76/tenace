# Enable `gemini-decision-1` via Secret Manager & Harden Key Hygiene Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to write failing hygiene tests (`RED`) before updating `.dockerignore` and docs (`GREEN`). NEVER print, log, or commit the API key payload.

**Goal:** Switch Tenace's Layer-2 Tactical Option Evaluator on Cloud Run (`tenace-api-dev` + `tenace-bench-dev`) to `gemini-decision-1` (zero-output-token Decisions API) authenticated via an AI Studio auth key stored in GCP Secret Manager (`tenace-gemini-api-key`), while hardening `.gitignore`, `.dockerignore`, and CI tests so `keys/` and API key literals can never leak into git or container images.

**Background & Live Verification:**
- Live probe with an AI Studio auth key (using `GeminiDecisionsClient(vertexai=False)` so only key auth is sent without dual credentials) scored 4 options in `297 ms` and completed a full 34-option cached-strategy auction turn (8-world C++ DDS + `gemini-decision-1`) in **`1,161 ms`** (~4x faster than `gemini-3.8-flash` and ~42x faster than `v0.2.0`).
- `terraform/modules/tenace_cloudrun/main.tf` already grants `roles/secretmanager.secretAccessor` and injects `GEMINI_API_KEY` from `var.gemini_api_key_secret_id` (`version = "latest"`) into both the Cloud Run Service and Cloud Run Job when `decisions_auth = "api-key"`.
- Gap discovered: `.gitignore` ignores `keys/`, but `.dockerignore` did **not** exclude `keys`, meaning `gcloud builds submit` could upload local key files into Cloud Build context unless hardened.

**Files:**
- Modify: `tests/test_iac_and_container.py`
- Modify: `tests/test_server.py`
- Modify: `.dockerignore`
- Keep/Commit: `.gitignore` (includes `keys/`)
- Modify: `terraform/README.md`
- Modify: `README.md`
- Update locally (gitignored): `terraform/local/dev/terraform.tfvars`, `terraform/local/prod/terraform.tfvars`

---

### Task 1: Harden `keys/` & API-Key Hygiene Tests + `.dockerignore` + Docs

- [ ] **Step 1: Add failing hygiene tests (`RED`)**
  - `tests/test_iac_and_container.py`: Assert `.gitignore` ignores `keys/ai-studio-apikey.txt`, `.dockerignore` excludes `keys`, no tracked file starts with `keys/`, no tracked file contains `AQ.*` / `AIza*` API keys, and `terraform/README.md` documents `gcloud secrets create` + `gcloud secrets versions add`.
  - `tests/test_server.py`: Include `"keys"` in `test_dockerfile_and_dockerignore_conventions`.
- [ ] **Step 2: Run targeted `pytest` to witness RED failures**
- [ ] **Step 3: Apply GREEN edits to `.dockerignore`, `terraform/README.md`, and `README.md`**
- [ ] **Step 4: Run targeted & full `pytest` to verify GREEN, audit diff, commit & push**

### Task 2: Provision Secret Manager Secret & Roll Out `dev` Cloud Run

- [ ] **Step 1: Create `tenace-gemini-api-key` in Secret Manager (`your-gcp-project-id`) and upload key via stdin (`tr -d '\r\n' < keys/ai-studio-apikey.txt | gcloud secrets versions add ... --data-file=-`)**
- [ ] **Step 2: Update gitignored `terraform/local/dev/terraform.tfvars` (and `prod`) with `evaluator_backend = "decisions"`, `evaluator_model = "gemini-decision-1"`, `decisions_auth = "api-key"`, `gemini_api_key_secret_id = "tenace-gemini-api-key"`**
- [ ] **Step 3: Run `terraform apply` on `dev` to bind IAM + inject `GEMINI_API_KEY` from Secret Manager**
- [ ] **Step 4: Start `gcloud run services proxy` on `127.0.0.1:8080` and verify live `/health` + Turn 1 & Turn 2 `/v1/step` latency (`verification-before-completion`)**
