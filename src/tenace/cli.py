from __future__ import annotations

import argparse
import math
import os
import re
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tenace.agent import TenaceAgent
from tenace.domain.types import (
    Assumption,
    Seat,
    StrategyState,
    TacticalDecision,
)
from tenace.harness import (
    BlueChipSeatBot,
    BridgeSeatBot,
    DuplicateMatchRunner,
    FirstLegalPassBaselineBot,
    MatchSummary,
    RoomTranscript,
    SubprocessTcpBlueChipController,
)
from tenace.layer1_strategist import SAYC_CONVENTION_CARD
from tenace.layer3_rules import RuleDDSEvaluator
from tenace.models.clients import (
    DiffusionGemmaClient,
    GeminiDecisionsClient,
    StructuredModelClient,
    VertexGeminiClient,
)

T = TypeVar("T", bound=BaseModel)

DEFAULT_GCP_PROJECT = "your-gcp-project-id"
DEFAULT_STRATEGIST_MODEL = "gemini-4-argon"
DEFAULT_DECISIONS_MODEL = "gemini-decision-1"
DEFAULT_DIFFUSION_MODEL = "publishers/google/models/diffusiongemma-26b-a4b-it"
DEFAULT_GEMINI_EVALUATOR_MODEL = "gemini-2.5-flash"

_OPPONENT_TENACE_FLAGS: tuple[tuple[str, str], ...] = (
    ("opponent_strategist_location", "--opponent-strategist-location"),
    ("opponent_strategist_model", "--opponent-strategist-model"),
    ("opponent_evaluator_backend", "--opponent-evaluator-backend"),
    ("opponent_evaluator_location", "--opponent-evaluator-location"),
    ("opponent_evaluator_model", "--opponent-evaluator-model"),
    ("opponent_evaluator_endpoint_id", "--opponent-evaluator-endpoint-id"),
    ("opponent_decisions_auth", "--opponent-decisions-auth"),
    ("opponent_decisions_violation_threshold", "--opponent-decisions-violation-threshold"),
)

_HCP_RE = re.compile(r"Own Hand\s*\((\d+)\s*HCP\)", re.IGNORECASE)
_PHASE_RE = re.compile(r"Phase:\s*(auction|play)\b", re.IGNORECASE)
_CANDIDATE_LINE_RE = re.compile(
    r"^-\s+(?P<label>[^:]+):\s*baseline_ev=(?P<ev>[-+]?\d+(?:\.\d+)?),\s*"
    r"make_prob=(?P<mp>N/A|[-+]?\d+(?:\.\d+)?)\s*$",
    re.MULTILINE,
)

__all__ = [
    "DEFAULT_DECISIONS_MODEL",
    "DEFAULT_DIFFUSION_MODEL",
    "DEFAULT_GCP_PROJECT",
    "DEFAULT_GEMINI_EVALUATOR_MODEL",
    "DEFAULT_STRATEGIST_MODEL",
    "HeuristicMockModelClient",
    "ModelProfile",
    "build_arg_parser",
    "build_clients_for_profile",
    "format_match_report",
    "main",
    "resolve_model_profiles",
    "run_cli",
]


def _default_evaluator_location(backend: str) -> str:
    return "us-central1" if backend == "diffusion-gemma" else "global"


def _default_evaluator_model(backend: str) -> str:
    if backend == "decisions":
        return DEFAULT_DECISIONS_MODEL
    if backend == "diffusion-gemma":
        return DEFAULT_DIFFUSION_MODEL
    if backend == "gemini":
        return DEFAULT_GEMINI_EVALUATOR_MODEL
    raise ValueError(f"Unsupported evaluator_backend: {backend!r}")


def _validate_violation_threshold(value: Any, *, flag_name: str = "decisions_violation_threshold") -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not (0.0 < value < 1.0)
    ):
        raise ValueError(
            f"{flag_name} must be a finite number strictly between 0.0 and 1.0, got {value!r}"
        )
    return float(value)


def _strip_required_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value.strip()


def _strip_optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank when provided")
    return value.strip()


class ModelProfile(BaseModel):
    """Configurable Layer-1 Strategist and Layer-2 Tactical Evaluator profile for a team."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["vertex", "mock"] = "vertex"
    project: str | None = DEFAULT_GCP_PROJECT
    strategist_location: str = "global"
    strategist_model: str = DEFAULT_STRATEGIST_MODEL
    evaluator_backend: Literal["decisions", "diffusion-gemma", "gemini"] = "decisions"
    evaluator_location: str = "global"
    evaluator_model: str = DEFAULT_DECISIONS_MODEL
    evaluator_endpoint_id: str | None = None
    decisions_auth: Literal["vertex", "api-key"] = "vertex"
    decisions_violation_threshold: float = 0.70

    @model_validator(mode="before")
    @classmethod
    def _resolve_backend_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        resolved = dict(data)
        raw_backend = resolved.get("evaluator_backend")
        backend = "decisions" if raw_backend is None else raw_backend
        if backend in ("decisions", "diffusion-gemma", "gemini"):
            if resolved.get("evaluator_location") is None:
                resolved["evaluator_location"] = _default_evaluator_location(backend)
            if resolved.get("evaluator_model") is None:
                resolved["evaluator_model"] = _default_evaluator_model(backend)
        return resolved

    @field_validator(
        "strategist_location",
        "strategist_model",
        "evaluator_location",
        "evaluator_model",
    )
    @classmethod
    def _validate_required_strings(cls, value: str, info: Any) -> str:
        return _strip_required_str(value, info.field_name)

    @field_validator("project", "evaluator_endpoint_id")
    @classmethod
    def _validate_optional_strings(cls, value: str | None, info: Any) -> str | None:
        return _strip_optional_str(value, info.field_name)

    @field_validator("decisions_violation_threshold")
    @classmethod
    def _check_threshold(cls, value: float) -> float:
        return _validate_violation_threshold(value)

    @model_validator(mode="after")
    def _check_backend_compatibility(self) -> ModelProfile:
        if self.evaluator_endpoint_id is not None and self.evaluator_backend != "diffusion-gemma":
            raise ValueError(
                "evaluator_endpoint_id is only supported when "
                f"evaluator_backend='diffusion-gemma', got {self.evaluator_backend!r}"
            )
        if self.decisions_auth != "vertex" and self.evaluator_backend != "decisions":
            raise ValueError(
                f"decisions_auth={self.decisions_auth!r} is only supported when "
                f"evaluator_backend='decisions', got {self.evaluator_backend!r}"
            )
        return self

    def _short_eval_target(self) -> str:
        if self.evaluator_backend == "diffusion-gemma" and self.evaluator_endpoint_id is not None:
            return f"endpoint:{self.evaluator_endpoint_id}"
        return self.evaluator_model.rsplit("/", 1)[-1]

    def short_label(self) -> str:
        short_strat = self.strategist_model.rsplit("/", 1)[-1]
        core = f"L1:{short_strat} | L2:{self.evaluator_backend}/{self._short_eval_target()}"
        if self.mode == "mock":
            return f"mock | {core}"
        return core


class HeuristicMockModelClient:
    """Stateless, deterministic StructuredModelClient for offline `--mode mock` runs.

    Zero cross-seat or cross-board state is stored on this instance so sharing or
    instantiating per seat strictly preserves Bridge fair-play invariants.
    """

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        del system_prompt
        if schema is StrategyState:
            return self._build_strategy_state(user_prompt)  # type: ignore[return-value]
        if schema is TacticalDecision:
            return self._build_tactical_decision(user_prompt)  # type: ignore[return-value]
        raise ValueError(f"Unsupported schema in HeuristicMockModelClient: {schema!r}")

    @staticmethod
    def _parse_hcp(user_prompt: str) -> int:
        match = _HCP_RE.search(user_prompt)
        return int(match.group(1)) if match is not None else 10

    @staticmethod
    def _parse_phase(user_prompt: str) -> str:
        match = _PHASE_RE.search(user_prompt)
        if match is not None:
            return match.group(1).lower()
        return "play" if "Phase: play" in user_prompt else "auction"

    def _build_strategy_state(self, user_prompt: str) -> StrategyState:
        phase = self._parse_phase(user_prompt)
        hcp = self._parse_hcp(user_prompt)

        if phase == "play":
            state = StrategyState(
                phase="play",
                plan_summary=(
                    f"Establish long tricks and preserve entries ({hcp} HCP) "
                    "while respecting anti-blunder DDS safety"
                ),
                target_or_threat="Maximize expected contract score and guard high honors",
                assumptions=[
                    Assumption(
                        claim="Unseen honors split near a-priori infostate frequencies",
                        confidence=0.65,
                    )
                ],
                revision=1,
            )
            return state

        if hcp >= 15:
            plan = (
                f"Strong/balanced opening range ({hcp} HCP); "
                "bid constructive SAYC game or notrump contract"
            )
            target = "Explore 3NT or major-suit game"
        elif hcp >= 12:
            plan = (
                f"Opening values ({hcp} HCP); "
                "open longest suit or support partner constructively"
            )
            target = "Locate 8-card major fit or sound partscore/game"
        else:
            plan = (
                f"Sub-opening hand ({hcp} HCP); "
                "pass unless partner forces or a safe competitive call stands out"
            )
            target = "Avoid minus-EV overbidding"

        return StrategyState(
            phase="auction",
            plan_summary=plan,
            target_or_threat=target,
            assumptions=[
                Assumption(
                    claim="Partner bids standard SAYC HCP and suit-length ranges",
                    confidence=0.70,
                )
            ],
            revision=1,
        )

    def _build_tactical_decision(self, user_prompt: str) -> TacticalDecision:
        phase = self._parse_phase(user_prompt)
        hcp = self._parse_hcp(user_prompt)

        parsed: list[tuple[str, float, float, int]] = []
        for idx, match in enumerate(_CANDIDATE_LINE_RE.finditer(user_prompt)):
            label = match.group("label").strip()
            ev = float(match.group("ev"))
            mp_raw = match.group("mp")
            mp = 0.5 if mp_raw == "N/A" else float(mp_raw)
            parsed.append((label, ev, mp, idx))

        if not parsed:
            return TacticalDecision(
                assumption_violated=False,
                option_scores={},
                rationale="Heuristic mock fallback (no candidate lines parsed)",
            )

        def _sort_key(item: tuple[str, float, float, int]) -> tuple[float, float, float, int]:
            label, ev, mp, idx = item
            # Sensible Bridge tie-breaker when EV/make_prob are identical:
            # prefer constructive low-level bids on opening hands, Pass on sub-opening hands.
            tie_bonus = 0.0
            if phase == "auction":
                if hcp < 12 and label == "Pass":
                    tie_bonus = 1.0
                elif hcp >= 12 and label != "Pass" and label not in ("Dbl", "RDbl"):
                    tie_bonus = 1.0 / max(1, int(label[0]) if label[0].isdigit() else 4)
            return (ev, mp, tie_bonus, -idx)

        ranked = sorted(parsed, key=_sort_key)
        n = len(ranked)
        option_scores: dict[str, float] = {}
        if n == 1:
            option_scores[ranked[0][0]] = 0.85
        else:
            for rank_pos, (label, _ev, _mp, _idx) in enumerate(ranked):
                score = 0.25 + 0.70 * (rank_pos / (n - 1))
                option_scores[label] = round(score, 4)

        best_label, best_ev, best_mp, _ = ranked[-1]
        return TacticalDecision(
            assumption_violated=False,
            option_scores=option_scores,
            rationale=(
                f"Heuristic mock ranked {best_label} highest "
                f"(baseline_ev={best_ev:.1f}, make_prob={best_mp:.2f})"
            ),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    default_project = (os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip() or DEFAULT_GCP_PROJECT

    parser = argparse.ArgumentParser(
        prog="tenace-match",
        description="Run a two-room WBF Duplicate IMP Bridge match with Tenace.",
    )
    parser.add_argument(
        "--mode",
        choices=["vertex", "mock"],
        default="vertex",
        help="Model backend mode: live Vertex AI ('vertex') or offline deterministic ('mock').",
    )
    parser.add_argument(
        "--project",
        default=default_project,
        help="Google Cloud project ID for Vertex AI (defaults to $GOOGLE_CLOUD_PROJECT).",
    )
    parser.add_argument(
        "--strategist-location",
        default="global",
        help="Vertex AI region for Layer-1 Strategist (default: global).",
    )
    parser.add_argument(
        "--strategist-model",
        default=DEFAULT_STRATEGIST_MODEL,
        help=f"Vertex AI model name for Layer-1 Strategist (default: {DEFAULT_STRATEGIST_MODEL}).",
    )
    parser.add_argument(
        "--evaluator-backend",
        choices=["decisions", "diffusion-gemma", "gemini"],
        default="decisions",
        help=(
            "Layer-2 Evaluator backend: 'decisions' (default, zero-output-token Gemini Decisions API), "
            "'diffusion-gemma', or 'gemini'."
        ),
    )
    parser.add_argument(
        "--evaluator-location",
        default=None,
        help=(
            "Vertex AI region for Layer-2 Evaluator (auto-resolves to 'global' for decisions/gemini "
            "and 'us-central1' for diffusion-gemma when omitted)."
        ),
    )
    parser.add_argument(
        "--evaluator-model",
        default=None,
        help=(
            "Model identifier for Layer-2 Evaluator (auto-resolves per backend: "
            f"'{DEFAULT_DECISIONS_MODEL}' for decisions, "
            f"'{DEFAULT_DIFFUSION_MODEL}' for diffusion-gemma, or "
            f"'{DEFAULT_GEMINI_EVALUATOR_MODEL}' for gemini when omitted)."
        ),
    )
    parser.add_argument(
        "--evaluator-endpoint-id",
        default=None,
        help="Optional dedicated Vertex AI Endpoint ID for DiffusionGemma.",
    )
    parser.add_argument(
        "--decisions-auth",
        choices=["vertex", "api-key"],
        default="vertex",
        help=(
            "Authentication mode for --evaluator-backend decisions: 'vertex' (default) or "
            "'api-key' (Gemini API v1beta via $GEMINI_API_KEY / $GOOGLE_API_KEY)."
        ),
    )
    parser.add_argument(
        "--decisions-violation-threshold",
        type=float,
        default=0.70,
        help=(
            "Probability threshold in (0.0, 1.0) on NoulQuestion P(true) to trigger a "
            "Layer-1 assumption-violation interrupt (default: 0.70)."
        ),
    )
    parser.add_argument(
        "--opponent",
        choices=["baseline", "bluechip", "tenace"],
        default="baseline",
        help=(
            "Opponent team engine: 'baseline', external BlueChip v18 bot ('bluechip'), "
            "or head-to-head Tenace model comparison ('tenace')."
        ),
    )
    parser.add_argument(
        "--bluechip-cmd",
        default=None,
        help=(
            "Command template with '{port}' placeholder for --opponent bluechip, e.g. "
            "'wine /google/data/ro/users/lo/locked/wbridge5/Wbridge5.exe Autoconnect {port}'."
        ),
    )
    parser.add_argument(
        "--bluechip-timeout",
        type=float,
        default=60.0,
        help="Socket/handshake timeout in seconds for BlueChip subprocesses (default: 60.0).",
    )
    parser.add_argument(
        "--opponent-strategist-location",
        default=None,
        help="Override Layer-1 Strategist Vertex AI region for Team B (--opponent tenace).",
    )
    parser.add_argument(
        "--opponent-strategist-model",
        default=None,
        help="Override Layer-1 Strategist model name for Team B (--opponent tenace).",
    )
    parser.add_argument(
        "--opponent-evaluator-backend",
        choices=["decisions", "diffusion-gemma", "gemini"],
        default=None,
        help="Override Layer-2 Evaluator backend for Team B (--opponent tenace).",
    )
    parser.add_argument(
        "--opponent-evaluator-location",
        default=None,
        help="Override Layer-2 Evaluator Vertex AI region for Team B (--opponent tenace).",
    )
    parser.add_argument(
        "--opponent-evaluator-model",
        default=None,
        help="Override Layer-2 Evaluator model identifier for Team B (--opponent tenace).",
    )
    parser.add_argument(
        "--opponent-evaluator-endpoint-id",
        default=None,
        help="Override DiffusionGemma Vertex AI Endpoint ID for Team B (--opponent tenace).",
    )
    parser.add_argument(
        "--opponent-decisions-auth",
        choices=["vertex", "api-key"],
        default=None,
        help="Override Decisions API auth mode ('vertex' or 'api-key') for Team B (--opponent tenace).",
    )
    parser.add_argument(
        "--opponent-decisions-violation-threshold",
        type=float,
        default=None,
        help="Override Decisions API violation threshold in (0.0, 1.0) for Team B (--opponent tenace).",
    )
    parser.add_argument(
        "--boards",
        type=int,
        default=4,
        help="Number of duplicate boards to play (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic tournament seed (default: 42).",
    )
    parser.add_argument(
        "--dds-worlds",
        type=int,
        default=16,
        help="Number of constrained infostate DDS worlds per candidate rollouts (default: 16).",
    )
    parser.add_argument(
        "--double-dummy",
        action="store_true",
        default=False,
        help="Skip card play and score auction contracts directly via OpenSpiel double-dummy.",
    )
    return parser


def resolve_model_profiles(
    args: argparse.Namespace,
) -> tuple[ModelProfile, ModelProfile | None]:
    if args.opponent != "tenace":
        provided_opponent_flags = [
            cli_flag
            for attr_name, cli_flag in _OPPONENT_TENACE_FLAGS
            if getattr(args, attr_name, None) is not None
        ]
        if provided_opponent_flags:
            flags_str = ", ".join(provided_opponent_flags)
            raise ValueError(
                f"{flags_str} require --opponent tenace (got --opponent {args.opponent})"
            )

    _validate_violation_threshold(
        args.decisions_violation_threshold,
        flag_name="--decisions-violation-threshold",
    )
    if args.opponent_decisions_violation_threshold is not None:
        _validate_violation_threshold(
            args.opponent_decisions_violation_threshold,
            flag_name="--opponent-decisions-violation-threshold",
        )

    team_a = ModelProfile(
        mode=args.mode,
        project=args.project,
        strategist_location=args.strategist_location,
        strategist_model=args.strategist_model,
        evaluator_backend=args.evaluator_backend,
        evaluator_location=args.evaluator_location,
        evaluator_model=args.evaluator_model,
        evaluator_endpoint_id=args.evaluator_endpoint_id,
        decisions_auth=args.decisions_auth,
        decisions_violation_threshold=args.decisions_violation_threshold,
    )

    if args.opponent != "tenace":
        return team_a, None

    opp_backend = (
        args.opponent_evaluator_backend
        if args.opponent_evaluator_backend is not None
        else team_a.evaluator_backend
    )
    same_backend = opp_backend == team_a.evaluator_backend

    opp_eval_location = (
        args.opponent_evaluator_location
        if args.opponent_evaluator_location is not None
        else (team_a.evaluator_location if same_backend else None)
    )
    opp_eval_model = (
        args.opponent_evaluator_model
        if args.opponent_evaluator_model is not None
        else (team_a.evaluator_model if same_backend else None)
    )
    opp_eval_endpoint_id = (
        args.opponent_evaluator_endpoint_id
        if args.opponent_evaluator_endpoint_id is not None
        else (team_a.evaluator_endpoint_id if same_backend else None)
    )
    opp_decisions_auth = (
        args.opponent_decisions_auth
        if args.opponent_decisions_auth is not None
        else (team_a.decisions_auth if opp_backend == "decisions" else "vertex")
    )

    team_b = ModelProfile(
        mode=args.mode,
        project=args.project,
        strategist_location=(
            args.opponent_strategist_location
            if args.opponent_strategist_location is not None
            else team_a.strategist_location
        ),
        strategist_model=(
            args.opponent_strategist_model
            if args.opponent_strategist_model is not None
            else team_a.strategist_model
        ),
        evaluator_backend=opp_backend,
        evaluator_location=opp_eval_location,
        evaluator_model=opp_eval_model,
        evaluator_endpoint_id=opp_eval_endpoint_id,
        decisions_auth=opp_decisions_auth,
        decisions_violation_threshold=(
            args.opponent_decisions_violation_threshold
            if args.opponent_decisions_violation_threshold is not None
            else team_a.decisions_violation_threshold
        ),
    )
    return team_a, team_b


def build_clients_for_profile(
    profile: ModelProfile,
) -> tuple[StructuredModelClient, StructuredModelClient]:
    _validate_violation_threshold(profile.decisions_violation_threshold)
    if profile.evaluator_endpoint_id is not None and profile.evaluator_backend != "diffusion-gemma":
        raise ValueError(
            "evaluator_endpoint_id is only supported when "
            f"evaluator_backend='diffusion-gemma', got {profile.evaluator_backend!r}"
        )
    if profile.decisions_auth != "vertex" and profile.evaluator_backend != "decisions":
        raise ValueError(
            f"decisions_auth={profile.decisions_auth!r} is only supported when "
            f"evaluator_backend='decisions', got {profile.evaluator_backend!r}"
        )

    if profile.mode == "mock":
        return HeuristicMockModelClient(), HeuristicMockModelClient()

    strategist_client: StructuredModelClient = VertexGeminiClient(
        project=profile.project,
        location=profile.strategist_location,
        model=profile.strategist_model,
    )

    evaluator_client: StructuredModelClient
    if profile.evaluator_backend == "decisions":
        if profile.decisions_auth == "vertex":
            evaluator_client = GeminiDecisionsClient(
                project=profile.project,
                location=profile.evaluator_location,
                model=profile.evaluator_model,
                vertexai=True,
                violation_threshold=profile.decisions_violation_threshold,
            )
        else:
            evaluator_client = GeminiDecisionsClient(
                project=None,
                location=profile.evaluator_location,
                model=profile.evaluator_model,
                vertexai=False,
                violation_threshold=profile.decisions_violation_threshold,
            )
    elif profile.evaluator_backend == "diffusion-gemma":
        evaluator_client = DiffusionGemmaClient(
            project=profile.project,
            location=profile.evaluator_location,
            endpoint_id=profile.evaluator_endpoint_id,
            maas_model_name=profile.evaluator_model,
        )
    elif profile.evaluator_backend == "gemini":
        evaluator_client = VertexGeminiClient(
            project=profile.project,
            location=profile.evaluator_location,
            model=profile.evaluator_model,
            temperature=0.2,
        )
    else:
        raise ValueError(f"Unsupported evaluator_backend: {profile.evaluator_backend!r}")

    return strategist_client, evaluator_client


def _format_room_section(
    lines: list[str],
    room_title: str,
    transcript: RoomTranscript | None,
    ns_score: float,
) -> None:
    lines.append(f"  {room_title}")
    if transcript is None:
        lines.append(f"    N/S Score : {ns_score:+.0f}")
        lines.append("    Strategist Traces: (none)")
        return

    lines.append(f"    Contract  : {transcript.contract}  |  N/S Score: {transcript.ns_score:+.0f}")
    lines.append(f"    Auction   : {transcript.auction}")
    if transcript.tricks:
        lines.append(f"    Card Play : {len(transcript.tricks)} tricks")
        for trick_line in transcript.tricks:
            lines.append(f"      {trick_line}")
    else:
        lines.append("    Card Play : (no tricks played)")

    lines.append("    Strategist Traces:")
    if transcript.strategy_traces:
        for seat_name, traces in transcript.strategy_traces.items():
            for trace in traces:
                lines.append(f"      [{seat_name}] {trace}")
    else:
        lines.append("      (none)")


def format_match_report(
    summary: MatchSummary,
    team_a_name: str = "Tenace",
    team_b_name: str = "Opponent",
    team_a_profile: str | None = None,
    team_b_profile: str | None = None,
) -> str:
    ruler = "=" * 76
    sub_ruler = "-" * 76
    lines: list[str] = [
        ruler,
        "TENACE DUPLICATE BRIDGE MATCH REPORT",
        f"Matchup : {team_a_name} vs {team_b_name}",
    ]
    if team_a_profile is not None:
        lines.append(f"Team A  : {team_a_name} ({team_a_profile})")
    if team_b_profile is not None:
        lines.append(f"Team B  : {team_b_name} ({team_b_profile})")
    lines.extend(
        [
            f"Boards  : {len(summary.boards)}  |  Tournament Seed: {summary.tournament_seed}",
            ruler,
        ]
    )

    for board in summary.boards:
        lines.append(f"Board {board.board_number}")
        _format_room_section(
            lines,
            f"Open Room ({team_a_name} N/S vs {team_b_name} E/W)",
            board.open_room,
            board.open_room_ns_score,
        )
        _format_room_section(
            lines,
            f"Closed Room ({team_b_name} N/S vs {team_a_name} E/W)",
            board.closed_room,
            board.closed_room_ns_score,
        )
        swing_diff = board.open_room_ns_score - board.closed_room_ns_score
        lines.append(
            f"  Board {board.board_number} Swing: {swing_diff:+.0f} pts "
            f"-> {board.imps_team_a:+d} IMPs ({team_a_name})"
        )
        lines.append(sub_ruler)

    if summary.total_imps_team_a > 0:
        outcome = f"Winner: {team_a_name} (+{summary.total_imps_team_a} IMP margin)"
    elif summary.total_imps_team_a < 0:
        outcome = f"Winner: {team_b_name} (+{abs(summary.total_imps_team_a)} IMP margin)"
    elif summary.total_imps_team_b > 0:
        outcome = f"Winner: {team_b_name} (+{summary.total_imps_team_b} IMP margin)"
    else:
        outcome = "Result: Exact Tie"

    lines.extend(
        [
            "Match Totals",
            f"  {team_a_name}: {summary.total_imps_team_a:+d} IMPs  |  "
            f"{team_b_name}: {summary.total_imps_team_b:+d} IMPs",
            f"  {outcome}",
            sub_ruler,
            "Telemetry",
            f"  strategist_calls : {summary.telemetry.get('strategist_calls', 0)}",
            f"  evaluator_calls  : {summary.telemetry.get('evaluator_calls', 0)}",
            f"  forced_moves     : {summary.telemetry.get('forced_moves', 0)}",
        ]
    )
    team_b_activity = (
        summary.telemetry.get("team_b_strategist_calls", 0)
        + summary.telemetry.get("team_b_evaluator_calls", 0)
        + summary.telemetry.get("team_b_forced_moves", 0)
    )
    if "team_a_strategist_calls" in summary.telemetry and (
        team_b_profile is not None or team_b_activity > 0
    ):
        lines.append(
            f"  {team_a_name}: "
            f"strategist_calls={summary.telemetry.get('team_a_strategist_calls', 0)}, "
            f"evaluator_calls={summary.telemetry.get('team_a_evaluator_calls', 0)}, "
            f"forced_moves={summary.telemetry.get('team_a_forced_moves', 0)}"
        )
        lines.append(
            f"  {team_b_name}: "
            f"strategist_calls={summary.telemetry.get('team_b_strategist_calls', 0)}, "
            f"evaluator_calls={summary.telemetry.get('team_b_evaluator_calls', 0)}, "
            f"forced_moves={summary.telemetry.get('team_b_forced_moves', 0)}"
        )
    lines.append(ruler)
    return "\n".join(lines)


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.boards < 1:
        raise ValueError("--boards must be >= 1")
    if args.dds_worlds < 1:
        raise ValueError("--dds-worlds must be >= 1")
    if not math.isfinite(args.bluechip_timeout) or args.bluechip_timeout <= 0:
        raise ValueError("--bluechip-timeout must be a finite number > 0")

    if args.opponent == "bluechip":
        if not args.bluechip_cmd or not args.bluechip_cmd.strip():
            raise ValueError("--bluechip-cmd is required when --opponent is bluechip")
        if "{port}" not in args.bluechip_cmd:
            raise ValueError("--bluechip-cmd must contain '{port}' placeholder")
        if args.double_dummy:
            raise ValueError(
                "--double-dummy cannot be combined with --opponent bluechip "
                "(BlueChip protocol v18 requires full card play)"
            )

    team_a_profile, team_b_profile = resolve_model_profiles(args)
    strategist_client, evaluator_client = build_clients_for_profile(team_a_profile)

    def team_a_factory(seat: Seat) -> BridgeSeatBot:
        return TenaceAgent(
            seat=seat,
            strategist_client=strategist_client,
            evaluator_client=evaluator_client,
            dds_evaluator=RuleDDSEvaluator(
                num_worlds=args.dds_worlds,
                rng_seed=(args.seed * 31) + int(seat),
            ),
            convention_card=SAYC_CONVENTION_CARD,
        )

    team_a_name = "Tenace"
    team_b_profile_label: str | None = None

    if args.opponent == "tenace":
        assert team_b_profile is not None
        team_a_name = "Tenace-A"
        team_b_name = "Tenace-B"
        team_b_profile_label = team_b_profile.short_label()
        opp_strategist_client, opp_evaluator_client = build_clients_for_profile(team_b_profile)

        def team_b_factory(seat: Seat) -> BridgeSeatBot:
            return TenaceAgent(
                seat=seat,
                strategist_client=opp_strategist_client,
                evaluator_client=opp_evaluator_client,
                dds_evaluator=RuleDDSEvaluator(
                    num_worlds=args.dds_worlds,
                    rng_seed=(args.seed * 37) + int(seat) + 100,
                ),
                convention_card=SAYC_CONVENTION_CARD,
            )
    elif args.opponent == "bluechip":
        team_b_name = "BlueChip"
        cmd_template = args.bluechip_cmd
        timeout_secs = args.bluechip_timeout

        def team_b_factory(seat: Seat) -> BridgeSeatBot:
            is_ns = seat in (Seat.NORTH, Seat.SOUTH)
            return BlueChipSeatBot(
                seat=seat,
                controller_factory=lambda: SubprocessTcpBlueChipController(
                    cmd_template=cmd_template,
                    timeout_secs=timeout_secs,
                ),
                team_ns="BlueChip" if is_ns else "Tenace",
                team_ew="Tenace" if is_ns else "BlueChip",
            )
    else:
        team_b_name = "Baseline"
        team_b_factory = FirstLegalPassBaselineBot

    runner = DuplicateMatchRunner(
        tournament_seed=args.seed,
        use_double_dummy_result=args.double_dummy,
    )
    summary = runner.play_match(
        team_a_factory=team_a_factory,
        team_b_factory=team_b_factory,
        num_boards=args.boards,
    )
    print(
        format_match_report(
            summary,
            team_a_name=team_a_name,
            team_b_name=team_b_name,
            team_a_profile=team_a_profile.short_label(),
            team_b_profile=team_b_profile_label,
        )
    )
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
