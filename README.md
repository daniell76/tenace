# Tenace: 3-Layer Neuro-Symbolic Contract Bridge AI

**Tenace** is a fair-play, three-layer neuro-symbolic Contract Bridge agent built on [OpenSpiel](https://github.com/google-deepmind/open_spiel) (`bridge`), configurable **Vertex AI Gemini** event-driven strategic planning (`gemini-4-argon` default), and configurable **Layer-2 Tactical Option Evaluators** (**Gemini Decisions API** `gemini-decision-1` zero-output-token prefill scoring, **DiffusionGemma** parallel denoising, or **Gemini 2.5 Flash**), grounded by constrained double-dummy infostate rollouts and an anti-blunder gate.

## Architecture

```text
OpenSpiel Bridge State (Single-Seat Imperfect Information)
   │
   ▼
Layer 3 — Symbolic Rule Fact Tracker & Action Collapser (`tenace.layer3_rules`)
   ├── Extracts legal public history, own hand, visible dummy, trick voids & surprise show-outs
   ├── Collapses contiguous/bridged rank-equivalent cards (0-API Fast Path if 1 candidate!)
   │
   ▼
Layer 1 — Event-Driven Strategist (`tenace.layer1_strategist`, Configurable Gemini)
   ├── Default: `gemini-4-argon` (supports `gemini-4-carbon`, `gemini-2.5-pro`, etc.)
   ├── Runs ONLY on board start, phase transition, dummy reveal, surprise void, or Layer-2 interrupt
   ├── Produces structured `StrategyState` (plan, assumptions, hard seat HCP/suit-length constraints)
   │
   ▼
Layer 3 — Constrained Infostate DDS Sampler (`RuleDDSEvaluator`)
   ├── Resamples consistent hidden deals (`state.resample_from_infostate`) filtered by `SeatConstraint`s
   ├── Computes `baseline_ev` & `make_prob` per candidate action
   │
   ▼
Layer 2 — Tactical Option Evaluator (`tenace.layer2_evaluator.TacticalOptionEvaluator`)
   ├── Configurable backends:
   │     • `decisions` (default): `gemini-decision-1` via `client.decisions.create` (0 output tokens!)
   │     • `diffusion-gemma`: `DiffusionGemma` (`diffusiongemma-26b-a4b-it` MaaS or Vertex Endpoint)
   │     • `gemini`: `VertexGeminiClient` (`gemini-2.5-flash` structured JSON scoring)
   ├── Scores all legal candidates simultaneously in `[0.0, 1.0]`
   ├── Closed-Loop Interrupt: if `assumption_violated=True`, triggers 1 immediate Layer-1 replan + re-eval
   │
   ▼
Layer 3 — Anti-Blunder Gate (`select_guarded_action`)
   └── Prunes options dropping `make_prob` by more than `max_make_prob_drop`, blending EV + tactical score
```

### Strict Fair-Play Invariants
- **Zero Partner Memory Sharing**: Each seat (`NORTH`, `EAST`, `SOUTH`, `WEST`) instantiates an isolated `TenaceAgent`. Partnership coordination happens strictly via an immutable `ConventionCard` (default `SAYC_CONVENTION_CARD`) and public auction/card history.
- **No Hidden-Deal Leakage**: `RuleFactTracker` never inspects `state.full_history()[:52]` (the chance deal actions), and `DuplicateMatchRunner` instantiates separate bot instances across the Open Room and Closed Room.

## Installation & Offline Test Suite

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pytest -v
```

All unit and integration tests run 100% offline using deterministic OpenSpiel deals and structured mock model clients.

## `tenace-match` CLI Runner

Tenace ships with the `tenace-match` CLI entrypoint to run two-room WBF Duplicate IMP matches and print a formatted board-by-board scorecard with Open/Closed Room contracts, auctions, trick logs, Layer-1 Strategist revision traces, and per-team API telemetry.

### 1. Offline 3-Layer Smoke Match (`--mode mock`)

Run the full 3-layer neuro-symbolic pipeline locally without network credentials using `HeuristicMockModelClient` + C++ constrained double-dummy rollouts:

```bash
uv run tenace-match --mode mock --boards 2 --dds-worlds 8
```

### 2. Live Vertex AI Match (Configurable Layer-1 & Layer-2 Backends)

Authenticate with Google Cloud Application Default Credentials (`gcloud auth application-default login`) and run against live models (`--evaluator-location` and `--evaluator-model` auto-resolve per backend when omitted):

```bash
# Default: Layer-1 Gemini 4 Argon (gemini-4-argon, global)
#        + Layer-2 Gemini Decisions API (gemini-decision-1, global, 0 output tokens!)
uv run tenace-match --project your-gcp-project-id --boards 4

# Gemini Decisions API via Gemini API key ($GEMINI_API_KEY / v1beta) + custom violation threshold:
uv run tenace-match \
  --project your-gcp-project-id \
  --evaluator-backend decisions \
  --decisions-auth api-key \
  --decisions-violation-threshold 0.75 \
  --boards 4

# Layer-2 DiffusionGemma (auto-resolves location=us-central1 & MaaS model name):
uv run tenace-match \
  --project your-gcp-project-id \
  --evaluator-backend diffusion-gemma \
  --boards 4

# Optional dedicated Vertex AI Endpoint ID for DiffusionGemma:
uv run tenace-match \
  --project your-gcp-project-id \
  --evaluator-backend diffusion-gemma \
  --evaluator-endpoint-id 1234567890 \
  --boards 4

# Layer-2 Gemini structured JSON evaluator (auto-resolves gemini-2.5-flash in global):
uv run tenace-match \
  --project your-gcp-project-id \
  --evaluator-backend gemini \
  --boards 4
```

### 3. Head-to-Head Model Benchmarking (`--opponent tenace`)

Pit two distinct Tenace model profiles directly against each other in a symmetric two-room WBF Duplicate IMP match (`Tenace-A` vs `Tenace-B`, with per-team API call telemetry):

```bash
# Compare Layer-2 Evaluators: Gemini Decisions API (Team A) vs DiffusionGemma (Team B)
uv run tenace-match \
  --project your-gcp-project-id \
  --evaluator-backend decisions \
  --opponent tenace \
  --opponent-evaluator-backend diffusion-gemma \
  --boards 8

# Compare Layer-1 Strategists: gemini-4-argon (Team A) vs gemini-2.5-pro (Team B)
# (Omitted --opponent-* flags automatically inherit Team A's configuration!)
uv run tenace-match \
  --project your-gcp-project-id \
  --strategist-model gemini-4-argon \
  --opponent tenace \
  --opponent-strategist-model gemini-2.5-pro \
  --boards 8
```

### 4. Playing Championship Engines (WBridge5 & BEN) via BlueChip Protocol v18

Tenace includes a WBF-compliant BlueChip Bridge Protocol v18 Table Manager TCP adapter (`BlueChipSeatBot` + `SubprocessTcpBlueChipController`) so Tenace can play automated duplicate matches directly against external championship engines such as **WBridge5** (via Wine) or **BEN** (Bridge Engine Neural). Pass `{port}` in `--bluechip-cmd` for the per-seat loopback callback port:

```bash
# Against WBridge5 via Wine:
uv run tenace-match \
  --project your-gcp-project-id \
  --opponent bluechip \
  --bluechip-cmd 'wine /path/to/Wbridge5.exe Autoconnect {port}' \
  --boards 4

# Against BEN (Bridge Engine Neural) table manager client:
uv run tenace-match \
  --project your-gcp-project-id \
  --opponent bluechip \
  --bluechip-cmd 'python /path/to/ben/src/table_manager_client.py --host 127.0.0.1 --port {port}' \
  --boards 4
```

## Cloud Run Service (`tenace-serve`), Batch Job & Multi-Environment Terraform IaC

Tenace packages both the stateless FastAPI HTTP service (`tenace-serve`) and the batch benchmark CLI (`tenace-match`) into a single non-root container (`Dockerfile`) managed across `bootstrap`, `dev`, and `prod` environments via Terraform (`terraform/`).

### Local HTTP Server Quickstart

```bash
# Start locally in offline mock mode on 0.0.0.0:8080:
TENACE_MODE=mock uv run tenace-serve
```

### Stateless HTTP API Endpoints

- `GET /health` (and `/healthz`): Readiness/liveness check returning service version, execution mode, `dds_worlds`, and active default `ModelProfile` (use `/health` on external Cloud Run `*.run.app` URLs).
- `POST /v1/step`: Selects the next legal bid or card for a single seat using either imperfect-information hands (`own_hand` + optional post-lead `dummy_hand` + `action_history`) or a deterministic benchmark board (`tournament_seed` + `board_number` + `action_history`). Accepts and returns `StrategyState` (`prior_strategy` -> `strategy`) so horizontally autoscaled Cloud Run instances preserve Layer-1 Strategist caching across turns with zero sticky state.
- `POST /v1/match`: Runs a synchronous two-room WBF Duplicate IMP match (`baseline` or head-to-head `tenace` model comparison) and returns structured IMP results, telemetry, and scorecard text.

### Deploying to Cloud Run (`bootstrap` -> `dev` -> `prod`)

Environment-specific `.tfvars` and GCS `.tfbackend` files are stored locally under `terraform/local/` (gitignored so real GCP project IDs and state buckets are never committed). See [`terraform/README.md`](terraform/README.md) for the complete layout and options:

```bash
# 1. Copy example templates into gitignored terraform/local/ and set your GCP project ID:
mkdir -p terraform/local/{bootstrap,dev,prod}
cp terraform/bootstrap/terraform.tfvars.example terraform/local/bootstrap/terraform.tfvars
cp terraform/environments/dev/backend.tfbackend.example terraform/local/dev/backend.tfbackend
cp terraform/environments/dev/terraform.tfvars.example terraform/local/dev/terraform.tfvars
cp terraform/environments/prod/backend.tfbackend.example terraform/local/prod/backend.tfbackend
cp terraform/environments/prod/terraform.tfvars.example terraform/local/prod/terraform.tfvars

# 2. Bootstrap GCP APIs, versioned GCS remote state bucket & Artifact Registry:
terraform -chdir=terraform/bootstrap init
terraform -chdir=terraform/bootstrap apply -var-file=../local/bootstrap/terraform.tfvars

# 3. Build and push the container image via Cloud Build:
gcloud builds submit \
  --project your-gcp-project-id \
  --tag us-central1-docker.pkg.dev/your-gcp-project-id/tenace/tenace:latest \
  .

# 4. Deploy dev (and similarly prod) backed by GCS remote state:
terraform -chdir=terraform/environments/dev init \
  -backend-config=../../local/dev/backend.tfbackend
terraform -chdir=terraform/environments/dev apply \
  -var-file=../../local/dev/terraform.tfvars
```

### Calling the Deployed Cloud Run Service & Benchmark Job

```bash
SERVICE_URL="$(terraform -chdir=terraform/environments/dev output -raw service_uri)"
AUTH_HEADER="Authorization: Bearer $(gcloud auth print-identity-token)"

# Readiness & active ModelProfile
curl -sS -H "${AUTH_HEADER}" "${SERVICE_URL}/health"

# Single-seat imperfect-information bid/play decision
curl -sS -X POST "${SERVICE_URL}/v1/step" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d '{
    "seat": "NORTH",
    "board_number": 1,
    "own_hand": ["S:A","S:K","S:Q","H:A","H:K","D:A","D:Q","D:J","C:K","C:Q","C:5","C:3","C:2"],
    "action_history": []
  }'

# Synchronous duplicate match
curl -sS -X POST "${SERVICE_URL}/v1/match" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d '{"boards": 2, "seed": 42, "dds_worlds": 8, "opponent": "baseline"}'

# Long-running batch WBF Duplicate IMP benchmark via Cloud Run Job
gcloud run jobs execute tenace-bench-dev \
  --project your-gcp-project-id \
  --region us-central1 \
  --wait
```

## Python API Quickstart

You can also wire `TenaceAgent` and `DuplicateMatchRunner` directly in Python:

```python
from tenace import (
    DiffusionGemmaClient,
    DuplicateMatchRunner,
    GeminiDecisionsClient,
    SAYC_CONVENTION_CARD,
    Seat,
    TenaceAgent,
    VertexGeminiClient,
)
from tenace.harness import FirstLegalPassBaselineBot
from tenace.layer3_rules import RuleDDSEvaluator

PROJECT_ID = "your-gcp-project-id"

# Layer 1: Event-driven strategic planner (Gemini 4 Argon on Vertex AI)
strategist_client = VertexGeminiClient(
    project=PROJECT_ID,
    location="global",
    model="gemini-4-argon",
    temperature=0.2,
)

# Layer 2 (Option A, default): Zero-output-token Gemini Decisions API (gemini-decision-1)
evaluator_client = GeminiDecisionsClient(
    project=PROJECT_ID,
    location="global",
    model="gemini-decision-1",
    violation_threshold=0.70,
)

# Layer 2 (Option B): Parallel denoising tactical evaluator (DiffusionGemma via Endpoint or MaaS)
# evaluator_client = DiffusionGemmaClient(
#     project=PROJECT_ID,
#     location="us-central1",
#     endpoint_id="1234567890",  # Optional dedicated Vertex AI Endpoint ID
# )


def make_tenace_seat(seat: Seat) -> TenaceAgent:
    return TenaceAgent(
        seat=seat,
        strategist_client=strategist_client,
        evaluator_client=evaluator_client,
        dds_evaluator=RuleDDSEvaluator(num_worlds=16, max_make_prob_drop=0.20),
        convention_card=SAYC_CONVENTION_CARD,
    )


runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
summary = runner.play_match(
    team_a_factory=make_tenace_seat,
    team_b_factory=FirstLegalPassBaselineBot,
    num_boards=4,
)

print(f"Total IMPs — Tenace: {summary.total_imps_team_a:+d}, Baseline: {summary.total_imps_team_b:+d}")
for board in summary.boards:
    print(
        f"Board {board.board_number}: Open NS={board.open_room_ns_score:+.0f}, "
        f"Closed NS={board.closed_room_ns_score:+.0f} -> {board.imps_team_a:+d} IMPs"
    )
```
