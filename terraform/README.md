# Tenace Terraform Infrastructure (`bootstrap`, `dev`, `prod`)

This directory provisions Google Cloud infrastructure for **Tenace** across isolated environments (`dev`, `prod`) backed by a versioned GCS Terraform remote state bucket, Artifact Registry, least-privilege IAM, a stateless **Cloud Run Service** (`tenace-serve`), and a companion **Cloud Run Job** (`tenace-match`) for long-running WBF Duplicate IMP benchmarks.

## Directory Layout

```text
terraform/
├── bootstrap/                  # One-time GCP APIs, versioned GCS tfstate bucket & Artifact Registry
│   ├── main.tf
│   ├── outputs.tf
│   ├── terraform.tfvars.example
│   ├── variables.tf
│   ├── versions.tf
│   └── tests/bootstrap.tftest.hcl
├── modules/
│   └── tenace_cloudrun/        # Reusable Cloud Run Service + Job + least-privilege IAM module
│       ├── main.tf
│       ├── outputs.tf
│       ├── variables.tf
│       ├── versions.tf
│       └── tests/cloudrun.tftest.hcl
├── environments/
│   ├── dev/                    # Dev stack (GCS state prefix: tenace/dev, deletion_protection=false)
│   │   ├── backend.tf
│   │   ├── backend.tfbackend.example
│   │   ├── main.tf
│   │   ├── outputs.tf
│   │   ├── terraform.tfvars.example
│   │   ├── variables.tf
│   │   └── versions.tf
│   └── prod/                   # Prod stack (GCS state prefix: tenace/prod, deletion_protection=true)
│       ├── backend.tf
│       ├── backend.tfbackend.example
│       ├── main.tf
│       ├── outputs.tf
│       ├── terraform.tfvars.example
│       ├── variables.tf
│       └── versions.tf
└── local/                      # GITIGNORED local variable & backend files (never committed!)
    ├── bootstrap/terraform.tfvars
    ├── dev/{backend.tfbackend,terraform.tfvars}
    └── prod/{backend.tfbackend,terraform.tfvars}
```

## Gitignored Local Configuration (`terraform/local/`)

Committed files in this repository never contain real GCP project IDs, state bucket names, or secrets. Instead, copy the `.example` templates into `terraform/local/` (which is excluded by `.gitignore` along with `*.tfvars`, `*.tfbackend`, and `*.tfstate*`):

```bash
mkdir -p terraform/local/{bootstrap,dev,prod}

cp terraform/bootstrap/terraform.tfvars.example terraform/local/bootstrap/terraform.tfvars
cp terraform/environments/dev/backend.tfbackend.example terraform/local/dev/backend.tfbackend
cp terraform/environments/dev/terraform.tfvars.example terraform/local/dev/terraform.tfvars
cp terraform/environments/prod/backend.tfbackend.example terraform/local/prod/backend.tfbackend
cp terraform/environments/prod/terraform.tfvars.example terraform/local/prod/terraform.tfvars
```

Edit the files under `terraform/local/` to replace `your-gcp-project-id` and `your-gcp-project-id-tfstate` with your target Google Cloud project and GCS bucket. If your project already has a versioned GCS bucket for Terraform state, set `create_tfstate_bucket = false` in `terraform/local/bootstrap/terraform.tfvars`.

## Step-by-Step Deployment

### 1. Bootstrap APIs, State Bucket & Artifact Registry

```bash
terraform -chdir=terraform/bootstrap init
terraform -chdir=terraform/bootstrap apply \
  -var-file=../local/bootstrap/terraform.tfvars
```

### 2. Build & Push Container Image with Cloud Build

From the repository root, build the multi-workload Tenace image (`Dockerfile`) and push it to Artifact Registry:

```bash
gcloud builds submit \
  --project your-gcp-project-id \
  --tag us-central1-docker.pkg.dev/your-gcp-project-id/tenace/tenace:latest \
  .
```

### 3. Deploy `dev` Environment

```bash
terraform -chdir=terraform/environments/dev init \
  -backend-config=../../local/dev/backend.tfbackend

terraform -chdir=terraform/environments/dev apply \
  -var-file=../../local/dev/terraform.tfvars
```

### 4. Deploy `prod` Environment

For production, pin an immutable image tag or digest in `terraform/local/prod/terraform.tfvars` (`deletion_protection` defaults to `true` and `max_instance_count` defaults to `10`):

```bash
terraform -chdir=terraform/environments/prod init \
  -backend-config=../../local/prod/backend.tfbackend

terraform -chdir=terraform/environments/prod apply \
  -var-file=../../local/prod/terraform.tfvars
```

## Invoking the Cloud Run Service & Benchmark Job

By default, `allow_unauthenticated = false` so Cloud Run enforces IAM authentication (`roles/run.invoker`). To enable direct browser Google Sign-In on the interactive Web UI (`GET /`) without `gcloud run services proxy` or SSH port forwarding, set `iap_enabled = true` (and optional `iap_accessor_members = ["user:you@example.com"]`) in your local `terraform.tfvars`.

### Authenticated HTTP Endpoints (`/health`, `/v1/step`, `/v1/match`)

```bash
SERVICE_URL="$(terraform -chdir=terraform/environments/dev output -raw service_uri)"
AUTH_HEADER="Authorization: Bearer $(gcloud auth print-identity-token)"

# 1. Health & active ModelProfile check
curl -sS -H "${AUTH_HEADER}" "${SERVICE_URL}/health"

# 2. Stateless single-turn action selection (/v1/step)
#    Supports imperfect-information own_hand (+ post-lead dummy_hand & prior_strategy)
#    or deterministic benchmark boards via tournament_seed:
curl -sS -X POST "${SERVICE_URL}/v1/step" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d '{
    "seat": "NORTH",
    "board_number": 1,
    "own_hand": ["S:A","S:K","S:Q","H:A","H:K","D:A","D:Q","D:J","C:K","C:Q","C:5","C:3","C:2"],
    "action_history": []
  }'

# 3. Short synchronous two-room WBF Duplicate IMP match (/v1/match)
curl -sS -X POST "${SERVICE_URL}/v1/match" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d '{
    "boards": 2,
    "seed": 42,
    "dds_worlds": 8,
    "opponent": "baseline"
  }'
```

### Executing Batch WBF Duplicate Benchmarks (`Cloud Run Job`)

For multi-board benchmarks without HTTP request timeouts, trigger the companion Cloud Run Job (`tenace-bench-dev` or `tenace-bench-prod`). Both `tenace-serve` and `tenace-match` share the same `TENACE_*` model and `TENACE_DDS_WORLDS` environment variables injected by Terraform (`job_args` defaults to `["--boards", "4"]` in `dev` and `["--boards", "8"]` in `prod`):

```bash
gcloud run jobs execute tenace-bench-dev \
  --project your-gcp-project-id \
  --region us-central1 \
  --wait
```

Override CLI flags for a specific benchmark run (for example, head-to-head model evaluation) using `--args`:

```bash
gcloud run jobs execute tenace-bench-dev \
  --project your-gcp-project-id \
  --region us-central1 \
  --args="--boards,8,--opponent,tenace,--opponent-evaluator-backend,diffusion-gemma" \
  --wait
```

## Optional Secret Manager Wiring (`decisions_auth = "api-key"`)

By default, Tenace authenticates to Vertex AI (`gemini-4-argon` and `gemini-decision-1`) via the dedicated Cloud Run runtime Service Account (`roles/aiplatform.user`) with zero API keys. To authenticate `gemini-decision-1` using an AI Studio auth key without ever hardcoding keys in scripts or `.tfvars`, store the key in Secret Manager and pass `gemini_api_key_secret_id`:

```bash
# 1. Create the secret once (automatic replication)
gcloud secrets create tenace-gemini-api-key \
  --project your-gcp-project-id \
  --replication-policy automatic

# 2. Upload the key payload from a gitignored local file (strip trailing newlines)
tr -d '\r\n' < keys/ai-studio-apikey.txt \
  | gcloud secrets versions add tenace-gemini-api-key \
      --project your-gcp-project-id \
      --data-file=-
```

Then configure your gitignored `terraform/local/<env>/terraform.tfvars` (Terraform grants `roles/secretmanager.secretAccessor` to the Cloud Run runtime service account and injects `GEMINI_API_KEY` at runtime):

```hcl
evaluator_backend        = "decisions"
evaluator_model          = "gemini-decision-1"
evaluator_location       = "global"
decisions_auth           = "api-key"
gemini_api_key_secret_id = "tenace-gemini-api-key"
```

## Offline Validation & Native Terraform Tests

Run formatting checks and mocked provider unit tests (`mock_provider "google" {}`) locally without GCP credentials:

```bash
terraform fmt -check -recursive terraform
terraform -chdir=terraform/bootstrap init -backend=false
terraform -chdir=terraform/bootstrap test
terraform -chdir=terraform/modules/tenace_cloudrun init -backend=false
terraform -chdir=terraform/modules/tenace_cloudrun test
```
