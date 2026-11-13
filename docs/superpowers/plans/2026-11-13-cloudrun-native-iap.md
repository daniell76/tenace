# Cloud Run Native Identity-Aware Proxy (IAP) Support Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to write failing native `terraform test` assertions before updating Terraform module and environment files.

**Goal:** Enable Google Cloud Run Native IAP (`iap_enabled` via `hashicorp/google-beta`) on `tenace-api-<env>` so authorized users can open the interactive Tenace Web UI directly at `https://tenace-api-<env>-....run.app/` in any browser with standard Google Sign-In and zero SSH port forwarding.

**Architecture:**
- Add `iap.googleapis.com` to `terraform/bootstrap/main.tf`.
- Add `hashicorp/google-beta` (`>= 6.50.0, < 7.0.0`) alongside `hashicorp/google` in `modules/tenace_cloudrun` and `environments/{dev,prod}`.
- Expose `iap_enabled` (`bool`, default `false`) and `iap_accessor_members` (`list(string)`, default `[]`) across the module and environment stacks.
- When `iap_enabled = true`:
  - Configure `google_cloud_run_v2_service.api` (`provider = google-beta`, `launch_stage = "BETA"`, `iap_enabled = true`) and forbid combining `iap_enabled = true` with `allow_unauthenticated = true`.
  - Grant `roles/run.invoker` on the Cloud Run service to the IAP Service Agent (`service-${project_number}@gcp-sa-iap.iam.gserviceaccount.com`) and bind optional per-service `roles/iap.httpsResourceAccessor` members via `google_iap_web_cloud_run_service_iam_member`.

**Files:**
- Modify: `terraform/bootstrap/tests/bootstrap.tftest.hcl`
- Modify: `terraform/modules/tenace_cloudrun/tests/cloudrun.tftest.hcl`
- Modify: `terraform/bootstrap/main.tf`
- Modify: `terraform/modules/tenace_cloudrun/{versions.tf,variables.tf,main.tf}`
- Modify: `terraform/environments/dev/{versions.tf,variables.tf,main.tf,terraform.tfvars.example}`
- Modify: `terraform/environments/prod/{versions.tf,variables.tf,main.tf,terraform.tfvars.example}`
- Modify: `terraform/README.md`
- Modify: `README.md`
- Local-only (gitignored): `terraform/local/dev/terraform.tfvars`

---

### Task 1: Add Native Cloud Run IAP Support via TDD & Roll Out to `dev`

- [ ] **Step 1: Write failing tests in `bootstrap.tftest.hcl` and `cloudrun.tftest.hcl`**
  - Assert `iap.googleapis.com` is in `google_project_service.apis`.
  - Assert `iap_enabled` defaults to `false` (`launch_stage == "GA"`), sets `iap_enabled == true` and `launch_stage == "BETA"` with IAP service agent invoker + `google_iap_web_cloud_run_service_iam_member` when enabled, and rejects `iap_enabled = true` combined with `allow_unauthenticated = true` or invalid IAM member syntax.
- [ ] **Step 2: Run `terraform test` to verify RED failures**
- [ ] **Step 3: Implement `bootstrap`, `modules/tenace_cloudrun`, `environments/{dev,prod}`, docs, and local `terraform/local/dev/terraform.tfvars`**
- [ ] **Step 4: Run `terraform fmt -check`, `terraform test`, and `uv run pytest tests/test_iac_and_container.py -v` to verify GREEN**
- [ ] **Step 5: Commit, push to `origin/main`, apply `bootstrap` + `dev`, and verify live `302` IAP redirect on `tenace-api-dev`**
