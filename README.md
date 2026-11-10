# Tenace: 3-Layer Neuro-Symbolic Contract Bridge AI

**Tenace** is a fair-play, three-layer neuro-symbolic Contract Bridge agent built on [OpenSpiel](https://github.com/google-deepmind/open_spiel) (`bridge`), **Vertex AI Gemini 2.5 Pro** (event-driven strategic planning), and **DiffusionGemma** (parallel denoising tactical option evaluation), grounded by constrained double-dummy infostate rollouts and an anti-blunder gate.

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
Layer 1 — Event-Driven Strategist (`tenace.layer1_strategist`, Gemini 2.5 Pro)
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

## Live Vertex AI Gemini + DiffusionGemma Quickstart

Authenticate with Google Cloud Application Default Credentials (`gcloud auth application-default login`) and run a two-room WBF Duplicate IMP match against the deterministic pass/first-legal baseline:

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

# Layer 1: Event-driven strategic planner (Gemini 2.5 Pro on Vertex AI)
strategist_client = VertexGeminiClient(
    project=PROJECT_ID,
    location="global",
    model="gemini-2.5-pro",
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
