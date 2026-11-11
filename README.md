# Tenace: 3-Layer Neuro-Symbolic Contract Bridge AI

**Tenace** is a fair-play, three-layer neuro-symbolic Contract Bridge agent built on [OpenSpiel](https://github.com/google-deepmind/open_spiel) (`bridge`), **Vertex AI Gemini 4 Argon** (`gemini-4-argon`, event-driven strategic planning), and **DiffusionGemma** (parallel denoising tactical option evaluation), grounded by constrained double-dummy infostate rollouts and an anti-blunder gate.

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
Layer 1 — Event-Driven Strategist (`tenace.layer1_strategist`, Gemini 4 Argon)
   ├── Runs ONLY on board start, phase transition, dummy reveal, surprise void, or Layer-2 interrupt
   ├── Produces structured `StrategyState` (plan, assumptions, hard seat HCP/suit-length constraints)
   │
   ▼
Layer 3 — Constrained Infostate DDS Sampler (`RuleDDSEvaluator`)
   ├── Resamples consistent hidden deals (`state.resample_from_infostate`) filtered by `SeatConstraint`s
   ├── Computes `baseline_ev` & `make_prob` per candidate action
   │
   ▼
Layer 2 — Tactical Option Evaluator (`tenace.layer2_evaluator`, DiffusionGemma)
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

Tenace ships with the `tenace-match` CLI entrypoint to run two-room WBF Duplicate IMP matches and print a formatted board-by-board scorecard with Open/Closed Room contracts, auctions, trick logs, Layer-1 Strategist revision traces, and API telemetry.

### 1. Offline 3-Layer Smoke Match (`--mode mock`)

Run the full 3-layer neuro-symbolic pipeline locally without network credentials using `HeuristicMockModelClient` + C++ constrained double-dummy rollouts:

```bash
uv run tenace-match --mode mock --boards 2 --dds-worlds 8
```

### 2. Live Vertex AI Match (Gemini 4 Argon + DiffusionGemma)

Authenticate with Google Cloud Application Default Credentials (`gcloud auth application-default login`) and run against live Vertex AI models:

```bash
# Default: Layer-1 Gemini 4 Argon (gemini-4-argon, global) + Layer-2 DiffusionGemma (us-central1)
uv run tenace-match --project your-gcp-project-id --boards 4

# Optional dedicated Vertex AI Endpoint ID for DiffusionGemma:
uv run tenace-match --project your-gcp-project-id --evaluator-endpoint-id 1234567890 --boards 4

# Gemini-only Layer-2 Evaluator fallback (defaults to gemini-2.5-flash when --evaluator-model is omitted):
uv run tenace-match --project your-gcp-project-id --evaluator-backend gemini --evaluator-location global --boards 4
```

### 3. Playing Championship Engines (WBridge5 & BEN) via BlueChip Protocol v18

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

## Python API Quickstart

You can also wire `TenaceAgent` and `DuplicateMatchRunner` directly in Python:

```python
from tenace import (
    DiffusionGemmaClient,
    DuplicateMatchRunner,
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

# Layer 2: Parallel denoising tactical evaluator (DiffusionGemma via Vertex Endpoint or MaaS)
evaluator_client = DiffusionGemmaClient(
    project=PROJECT_ID,
    location="us-central1",
    # Optional dedicated Vertex AI Endpoint ID; omit to use MaaS model name:
    # endpoint_id="1234567890",
)


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
