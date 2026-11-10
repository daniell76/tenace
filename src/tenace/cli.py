from __future__ import annotations

import argparse
import os
import re
from typing import TypeVar

from pydantic import BaseModel

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
    StructuredModelClient,
    VertexGeminiClient,
)

T = TypeVar("T", bound=BaseModel)

DEFAULT_GCP_PROJECT = "your-gcp-project-id"
DEFAULT_STRATEGIST_MODEL = "gemini-2.5-pro"
DEFAULT_DIFFUSION_MODEL = "publishers/google/models/diffusiongemma-26b-a4b-it"
DEFAULT_GEMINI_EVALUATOR_MODEL = "gemini-2.5-flash"

_HCP_RE = re.compile(r"Own Hand\s*\((\d+)\s*HCP\)", re.IGNORECASE)
_PHASE_RE = re.compile(r"Phase:\s*(auction|play)\b", re.IGNORECASE)
_CANDIDATE_LINE_RE = re.compile(
    r"^-\s+(?P<label>[^:]+):\s*baseline_ev=(?P<ev>[-+]?\d+(?:\.\d+)?),\s*"
    r"make_prob=(?P<mp>N/A|[-+]?\d+(?:\.\d+)?)\s*$",
    re.MULTILINE,
)

__all__ = [
    "HeuristicMockModelClient",
    "build_arg_parser",
    "format_match_report",
    "main",
    "run_cli",
]


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
    default_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or DEFAULT_GCP_PROJECT

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
        choices=["diffusion-gemma", "gemini"],
        default="diffusion-gemma",
        help="Layer-2 Evaluator backend ('diffusion-gemma' or 'gemini').",
    )
    parser.add_argument(
        "--evaluator-location",
        default="us-central1",
        help="Vertex AI region for Layer-2 Evaluator (default: us-central1).",
    )
    parser.add_argument(
        "--evaluator-model",
        default=DEFAULT_DIFFUSION_MODEL,
        help=(
            "Model identifier for Layer-2 Evaluator (defaults to DiffusionGemma MaaS "
            f"'{DEFAULT_DIFFUSION_MODEL}', or '{DEFAULT_GEMINI_EVALUATOR_MODEL}' when "
            "--evaluator-backend=gemini is used without overriding --evaluator-model)."
        ),
    )
    parser.add_argument(
        "--evaluator-endpoint-id",
        default=None,
        help="Optional dedicated Vertex AI Endpoint ID for DiffusionGemma.",
    )
    parser.add_argument(
        "--opponent",
        choices=["baseline", "bluechip"],
        default="baseline",
        help="Opponent team engine: 'baseline' or external BlueChip v18 bot ('bluechip').",
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
) -> str:
    ruler = "=" * 76
    sub_ruler = "-" * 76
    lines: list[str] = [
        ruler,
        "TENACE DUPLICATE BRIDGE MATCH REPORT",
        f"Matchup : {team_a_name} vs {team_b_name}",
        f"Boards  : {len(summary.boards)}  |  Tournament Seed: {summary.tournament_seed}",
        ruler,
    ]

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

    if summary.total_imps_team_a > summary.total_imps_team_b:
        outcome = f"Winner: {team_a_name} (+{summary.total_imps_team_a - summary.total_imps_team_b} IMP margin)"
    elif summary.total_imps_team_b > summary.total_imps_team_a:
        outcome = f"Winner: {team_b_name} (+{summary.total_imps_team_b - summary.total_imps_team_a} IMP margin)"
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
            ruler,
        ]
    )
    return "\n".join(lines)


def _build_model_clients(
    args: argparse.Namespace,
) -> tuple[StructuredModelClient, StructuredModelClient]:
    if args.mode == "mock":
        return HeuristicMockModelClient(), HeuristicMockModelClient()

    strategist_client: StructuredModelClient = VertexGeminiClient(
        project=args.project,
        location=args.strategist_location,
        model=args.strategist_model,
    )

    if args.evaluator_backend == "diffusion-gemma":
        evaluator_client: StructuredModelClient = DiffusionGemmaClient(
            project=args.project,
            location=args.evaluator_location,
            endpoint_id=args.evaluator_endpoint_id,
            maas_model_name=args.evaluator_model,
        )
    else:
        resolved_eval_model = (
            DEFAULT_GEMINI_EVALUATOR_MODEL
            if args.evaluator_model == DEFAULT_DIFFUSION_MODEL
            else args.evaluator_model
        )
        evaluator_client = VertexGeminiClient(
            project=args.project,
            location=args.evaluator_location,
            model=resolved_eval_model,
            temperature=0.2,
        )

    return strategist_client, evaluator_client


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.boards < 1:
        raise ValueError("--boards must be >= 1")
    if args.dds_worlds < 1:
        raise ValueError("--dds-worlds must be >= 1")

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

    strategist_client, evaluator_client = _build_model_clients(args)

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

    if args.opponent == "bluechip":
        team_b_name = "BlueChip"
        cmd_template = args.bluechip_cmd
        timeout_secs = args.bluechip_timeout

        def team_b_factory(seat: Seat) -> BridgeSeatBot:
            return BlueChipSeatBot(
                seat=seat,
                controller_factory=lambda: SubprocessTcpBlueChipController(
                    cmd_template=cmd_template,
                    timeout_secs=timeout_secs,
                ),
                team_ns="Tenace",
                team_ew="BlueChip",
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
    print(format_match_report(summary, team_a_name="Tenace", team_b_name=team_b_name))
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
