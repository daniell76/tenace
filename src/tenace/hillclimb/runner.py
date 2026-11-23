from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
import time
from typing import Any, Literal

from tenace.agent import TenaceAgent
from tenace.cli import ModelProfile, build_clients_for_profile
from tenace.domain.types import Seat
from tenace.harness.bluechip import BlueChipSeatBot, SubprocessTcpBlueChipController
from tenace.harness.duplicate import (
    BridgeSeatBot,
    DuplicateMatchRunner,
    FirstLegalPassBaselineBot,
    _build_room_transcript,
)
from tenace.harness.scoring import wbf_continuous_vp
from tenace.hillclimb.diagnostics import build_board_record, extract_full_deal_hands
from tenace.hillclimb.models import (
    ABRetestRecord,
    BoardTelemetryRecord,
    IterationRecord,
    MatchMode,
    RoomSnapshot,
    RunKind,
    apply_patch_to_variables,
)
from tenace.hillclimb.optimizer import PromptProposer
from tenace.hillclimb.store import HillClimbStore
from tenace.layer1_strategist.conventions import ConventionCard
from tenace.layer3_rules import RuleDDSEvaluator

OpponentKind = Literal["bluechip", "baseline", "tenace"]

__all__ = ["HillClimbEngine", "OpponentKind"]


def _close_bots(bots: dict[Seat, BridgeSeatBot]) -> None:
    for bot in bots.values():
        close_fn = getattr(bot, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def _room_snapshot_from_transcript(tr: Any, elapsed_s: float) -> RoomSnapshot:
    return RoomSnapshot(
        contract=tr.contract,
        auction=tr.auction,
        tricks=list(tr.tricks),
        ns_score=float(tr.ns_score),
        strategy_traces=dict(tr.strategy_traces),
        elapsed_s=round(elapsed_s, 2),
    )


def _summarize_batch(
    *,
    batch_id: str,
    run_kind: RunKind,
    prompt_version: str,
    match_mode: MatchMode,
    seed: int,
    start_board: int,
    records: Sequence[BoardTelemetryRecord],
    elapsed_s: float,
) -> IterationRecord:
    n = len(records)
    net = sum(r.imps_challenger for r in records)
    won = sum(r.imps_challenger for r in records if r.imps_challenger > 0)
    lost = sum(-r.imps_challenger for r in records if r.imps_challenger < 0)
    wins = sum(1 for r in records if r.imps_challenger > 0)
    losses = sum(1 for r in records if r.imps_challenger < 0)
    pushes = sum(1 for r in records if r.imps_challenger == 0)
    vp_c, vp_r = wbf_continuous_vp(net, max(1, n))
    auc_match = sum(1 for r in records if r.auction_exact_match)
    con_match = sum(1 for r in records if r.contract_match)
    swing_counts = dict(Counter(r.swing_category for r in records))
    tel_totals = {
        "strategist_calls": sum(r.telemetry.get("strategist_calls", 0) for r in records),
        "evaluator_calls": sum(r.telemetry.get("evaluator_calls", 0) for r in records),
        "forced_moves": sum(r.telemetry.get("forced_moves", 0) for r in records),
    }
    return IterationRecord(
        batch_id=batch_id,
        run_kind=run_kind,
        prompt_version=prompt_version,
        match_mode=match_mode,
        tournament_seed=seed,
        start_board=start_board,
        num_boards=n,
        net_imps=net,
        gross_won_imps=won,
        gross_lost_imps=lost,
        imp_per_board=round(net / max(1, n), 3),
        vp_challenger=vp_c,
        vp_reference=vp_r,
        wins=wins,
        losses=losses,
        pushes=pushes,
        auction_match_rate=round(auc_match / max(1, n), 4),
        contract_match_rate=round(con_match / max(1, n), 4),
        swing_counts=swing_counts,
        telemetry=tel_totals,
        elapsed_s=round(elapsed_s, 2),
    )


class HillClimbEngine:
    """Executes versioned matches, matched-seed A/B retests, and automated hill-climb loops."""

    def __init__(
        self,
        *,
        store: HillClimbStore,
        challenger_profile: ModelProfile,
        opponent: OpponentKind = "baseline",
        bluechip_cmd: str | None = None,
        bluechip_timeout: float = 300.0,
        reference_profile: ModelProfile | None = None,
        reference_card: ConventionCard | None = None,
        dds_worlds: int = 16,
        double_dummy: bool = False,
        reference_cache_tag: str | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> None:
        if opponent == "bluechip":
            if not bluechip_cmd or "{port}" not in bluechip_cmd:
                raise ValueError(
                    "bluechip_cmd containing '{port}' is required when opponent='bluechip'"
                )
            if double_dummy:
                raise ValueError(
                    "double_dummy cannot be combined with opponent='bluechip'"
                )
        self.store = store
        self.challenger_profile = challenger_profile
        self.opponent = opponent
        self.bluechip_cmd = bluechip_cmd
        self.bluechip_timeout = bluechip_timeout
        self.reference_profile = reference_profile or challenger_profile
        self.reference_card = reference_card
        self.dds_worlds = dds_worlds
        self.double_dummy = double_dummy
        self.reference_cache_tag = reference_cache_tag or opponent
        self.progress_cb = progress_cb

    def _log(self, msg: str) -> None:
        if self.progress_cb is not None:
            self.progress_cb(msg)

    def _make_tenace_bot(
        self,
        seat: Seat,
        card: ConventionCard,
        strat_client: Any,
        eval_client: Any,
        seed: int,
        salt: int = 0,
    ) -> TenaceAgent:
        return TenaceAgent(
            seat=seat,
            strategist_client=strat_client,
            evaluator_client=eval_client,
            dds_evaluator=RuleDDSEvaluator(
                num_worlds=self.dds_worlds,
                rng_seed=(seed * 31) + int(seat) + salt,
            ),
            convention_card=card,
        )

    def _make_ref_bot(
        self,
        seat: Seat,
        *,
        seed: int,
        team_ns: str,
        team_ew: str,
        ref_clients: tuple[Any, Any] | None = None,
    ) -> BridgeSeatBot:
        if self.opponent == "baseline":
            return FirstLegalPassBaselineBot(seat)
        if self.opponent == "bluechip":
            assert self.bluechip_cmd is not None
            return BlueChipSeatBot(
                seat=seat,
                controller_factory=lambda s=seat: SubprocessTcpBlueChipController(
                    cmd_template=self.bluechip_cmd,
                    timeout_secs=self.bluechip_timeout,
                    seat=s,
                ),
                team_ns=team_ns,
                team_ew=team_ew,
            )
        assert ref_clients is not None
        from tenace.layer1_strategist import SAYC_CONVENTION_CARD

        card = self.reference_card or SAYC_CONVENTION_CARD
        return self._make_tenace_bot(
            seat, card, ref_clients[0], ref_clients[1], seed=seed, salt=100
        )

    def evaluate_boards(
        self,
        *,
        version: str,
        seed: int,
        board_numbers: Sequence[int],
        batch_id: str,
        run_kind: RunKind = "eval",
        match_mode: MatchMode | None = None,
        use_reference_cache: bool = True,
        record_iteration: bool = True,
    ) -> tuple[list[BoardTelemetryRecord], IterationRecord]:
        manifest = self.store.load_manifest()
        mode: MatchMode = match_mode or manifest.match_mode
        card = self.store.load_convention_card(version)
        runner = DuplicateMatchRunner(
            tournament_seed=seed,
            use_double_dummy_result=self.double_dummy,
        )
        t_start = time.perf_counter()
        chal_clients = build_clients_for_profile(self.challenger_profile)
        ref_clients = (
            build_clients_for_profile(self.reference_profile)
            if self.opponent == "tenace"
            else None
        )

        # Pre-check reference cache in controlled mode (`Table A = 4× reference`).
        cached_table_a: dict[int, RoomSnapshot] = {}
        deals_by_board: dict[int, Any] = {}
        for b in board_numbers:
            st_deal = runner._new_dealt_board_state(b)
            deals_by_board[b] = extract_full_deal_hands(
                st_deal, board_number=b, seed=seed
            )
            if mode == "controlled" and use_reference_cache:
                hit = self.store.load_reference_board(self.reference_cache_tag, seed, b)
                if hit is not None:
                    cached_table_a[b] = hit

        records: list[BoardTelemetryRecord] = []

        if mode == "controlled":
            uncached = [b for b in board_numbers if b not in cached_table_a]
            if uncached:
                ref_seats: dict[Seat, BridgeSeatBot] = {}
                try:
                    for s in Seat:
                        ref_seats[s] = self._make_ref_bot(
                            s,
                            seed=seed,
                            team_ns="Ref_NS",
                            team_ew="Ref_EW",
                            ref_clients=ref_clients,
                        )
                    for b in uncached:
                        bt0 = time.perf_counter()
                        st = runner._new_dealt_board_state(b)
                        ns_score = runner._play_room(st, ref_seats)
                        tr = _build_room_transcript(st, ref_seats, ns_score)
                        snap = _room_snapshot_from_transcript(tr, time.perf_counter() - bt0)
                        cached_table_a[b] = snap
                        if use_reference_cache:
                            self.store.save_reference_board(
                                self.reference_cache_tag, seed, b, snap
                            )
                        self._log(
                            f"[{batch_id}] Table A {seed}:{b} -> {snap.contract} (NS {ns_score:+.0f})"
                        )
                finally:
                    _close_bots(ref_seats)

            # Table B: Challenger N/S vs Reference E/W
            table_b_seats: dict[Seat, BridgeSeatBot] = {}
            try:
                table_b_seats[Seat.NORTH] = self._make_tenace_bot(
                    Seat.NORTH, card, chal_clients[0], chal_clients[1], seed=seed
                )
                table_b_seats[Seat.SOUTH] = self._make_tenace_bot(
                    Seat.SOUTH, card, chal_clients[0], chal_clients[1], seed=seed
                )
                table_b_seats[Seat.EAST] = self._make_ref_bot(
                    Seat.EAST,
                    seed=seed,
                    team_ns=manifest.system_id,
                    team_ew="Ref_EW",
                    ref_clients=ref_clients,
                )
                table_b_seats[Seat.WEST] = self._make_ref_bot(
                    Seat.WEST,
                    seed=seed,
                    team_ns=manifest.system_id,
                    team_ew="Ref_EW",
                    ref_clients=ref_clients,
                )
                for b in board_numbers:
                    prev_l1 = (
                        table_b_seats[Seat.NORTH].strategist_calls  # type: ignore[attr-defined]
                        + table_b_seats[Seat.SOUTH].strategist_calls  # type: ignore[attr-defined]
                    )
                    prev_l2 = (
                        table_b_seats[Seat.NORTH].evaluator_calls  # type: ignore[attr-defined]
                        + table_b_seats[Seat.SOUTH].evaluator_calls  # type: ignore[attr-defined]
                    )
                    prev_fm = (
                        table_b_seats[Seat.NORTH].forced_moves  # type: ignore[attr-defined]
                        + table_b_seats[Seat.SOUTH].forced_moves  # type: ignore[attr-defined]
                    )
                    bt0 = time.perf_counter()
                    st = runner._new_dealt_board_state(b)
                    ns_score = runner._play_room(st, table_b_seats)
                    tr = _build_room_transcript(st, table_b_seats, ns_score)
                    snap_b = _room_snapshot_from_transcript(tr, time.perf_counter() - bt0)
                    delta_tel = {
                        "strategist_calls": int(
                            table_b_seats[Seat.NORTH].strategist_calls  # type: ignore[attr-defined]
                            + table_b_seats[Seat.SOUTH].strategist_calls  # type: ignore[attr-defined]
                            - prev_l1
                        ),
                        "evaluator_calls": int(
                            table_b_seats[Seat.NORTH].evaluator_calls  # type: ignore[attr-defined]
                            + table_b_seats[Seat.SOUTH].evaluator_calls  # type: ignore[attr-defined]
                            - prev_l2
                        ),
                        "forced_moves": int(
                            table_b_seats[Seat.NORTH].forced_moves  # type: ignore[attr-defined]
                            + table_b_seats[Seat.SOUTH].forced_moves  # type: ignore[attr-defined]
                            - prev_fm
                        ),
                    }
                    rec = build_board_record(
                        run_kind=run_kind,
                        batch_id=batch_id,
                        prompt_version=version,
                        match_mode="controlled",
                        tournament_seed=seed,
                        board_number=b,
                        deal=deals_by_board[b],
                        table_a=cached_table_a[b],
                        table_b=snap_b,
                        reference_cache_hit=(b not in uncached),
                        telemetry=delta_tel,
                    )
                    self.store.append_board(rec)
                    records.append(rec)
                    self._log(
                        f"[{batch_id}] {version} {seed}:{b} | {snap_b.contract} "
                        f"vs {cached_table_a[b].contract} -> {rec.imps_challenger:+d} IMP"
                    )
            finally:
                _close_bots(table_b_seats)
        else:
            # Full symmetric 2-room duplicate match:
            # table_a (Open): Challenger N/S vs Reference E/W
            # table_b (Closed): Reference N/S vs Challenger E/W
            open_seats: dict[Seat, BridgeSeatBot] = {}
            closed_seats: dict[Seat, BridgeSeatBot] = {}
            try:
                open_seats[Seat.NORTH] = self._make_tenace_bot(
                    Seat.NORTH, card, chal_clients[0], chal_clients[1], seed=seed
                )
                open_seats[Seat.SOUTH] = self._make_tenace_bot(
                    Seat.SOUTH, card, chal_clients[0], chal_clients[1], seed=seed
                )
                open_seats[Seat.EAST] = self._make_ref_bot(
                    Seat.EAST, seed=seed, team_ns=manifest.system_id, team_ew="Ref", ref_clients=ref_clients
                )
                open_seats[Seat.WEST] = self._make_ref_bot(
                    Seat.WEST, seed=seed, team_ns=manifest.system_id, team_ew="Ref", ref_clients=ref_clients
                )
                closed_seats[Seat.NORTH] = self._make_ref_bot(
                    Seat.NORTH, seed=seed, team_ns="Ref", team_ew=manifest.system_id, ref_clients=ref_clients
                )
                closed_seats[Seat.SOUTH] = self._make_ref_bot(
                    Seat.SOUTH, seed=seed, team_ns="Ref", team_ew=manifest.system_id, ref_clients=ref_clients
                )
                closed_seats[Seat.EAST] = self._make_tenace_bot(
                    Seat.EAST, card, chal_clients[0], chal_clients[1], seed=seed, salt=17
                )
                closed_seats[Seat.WEST] = self._make_tenace_bot(
                    Seat.WEST, card, chal_clients[0], chal_clients[1], seed=seed, salt=17
                )
                chal_bots = [
                    open_seats[Seat.NORTH],
                    open_seats[Seat.SOUTH],
                    closed_seats[Seat.EAST],
                    closed_seats[Seat.WEST],
                ]
                for b in board_numbers:
                    p_l1 = sum(int(getattr(x, "strategist_calls", 0)) for x in chal_bots)
                    p_l2 = sum(int(getattr(x, "evaluator_calls", 0)) for x in chal_bots)
                    p_fm = sum(int(getattr(x, "forced_moves", 0)) for x in chal_bots)

                    t0 = time.perf_counter()
                    so = runner._new_dealt_board_state(b)
                    open_ns = runner._play_room(so, open_seats)
                    snap_open = _room_snapshot_from_transcript(
                        _build_room_transcript(so, open_seats, open_ns),
                        time.perf_counter() - t0,
                    )
                    t1 = time.perf_counter()
                    sc = runner._new_dealt_board_state(b)
                    closed_ns = runner._play_room(sc, closed_seats)
                    snap_closed = _room_snapshot_from_transcript(
                        _build_room_transcript(sc, closed_seats, closed_ns),
                        time.perf_counter() - t1,
                    )
                    delta_tel = {
                        "strategist_calls": sum(int(getattr(x, "strategist_calls", 0)) for x in chal_bots) - p_l1,
                        "evaluator_calls": sum(int(getattr(x, "evaluator_calls", 0)) for x in chal_bots) - p_l2,
                        "forced_moves": sum(int(getattr(x, "forced_moves", 0)) for x in chal_bots) - p_fm,
                    }
                    rec = build_board_record(
                        run_kind=run_kind,
                        batch_id=batch_id,
                        prompt_version=version,
                        match_mode="duplicate",
                        tournament_seed=seed,
                        board_number=b,
                        deal=deals_by_board[b],
                        table_a=snap_open,
                        table_b=snap_closed,
                        reference_cache_hit=False,
                        telemetry=delta_tel,
                    )
                    self.store.append_board(rec)
                    records.append(rec)
            finally:
                _close_bots(open_seats)
                _close_bots(closed_seats)

        iter_rec = _summarize_batch(
            batch_id=batch_id,
            run_kind=run_kind,
            prompt_version=version,
            match_mode=mode,
            seed=seed,
            start_board=min(board_numbers) if board_numbers else 1,
            records=records,
            elapsed_s=time.perf_counter() - t_start,
        )
        if record_iteration:
            self.store.append_iteration(iter_rec)
        return records, iter_rec

    def retest_candidate(
        self,
        *,
        candidate_version: str,
        board_pairs: Sequence[tuple[int, int]],
        min_delta_imps: int = 1,
        batch_id: str | None = None,
    ) -> ABRetestRecord:
        cand_meta = self.store.load_version_meta(candidate_version)
        parent_version = cand_meta.parent_version or "v000"
        all_boards = self.store.read_boards()

        # Index latest baseline IMP score for each board under parent_version.
        parent_lookup: dict[tuple[int, int], BoardTelemetryRecord] = {}
        for r in all_boards:
            if r.prompt_version == parent_version:
                parent_lookup[(r.tournament_seed, r.board_number)] = r

        grouped: dict[int, list[int]] = {}
        for s, b in board_pairs:
            grouped.setdefault(s, []).append(b)

        before_map: dict[tuple[int, int], int] = {}
        after_map: dict[tuple[int, int], int] = {}
        tag = batch_id or f"ab_{candidate_version}"

        for seed, bnums in sorted(grouped.items()):
            missing_parent = [b for b in bnums if (seed, b) not in parent_lookup]
            if missing_parent:
                parent_recs, _ = self.evaluate_boards(
                    version=parent_version,
                    seed=seed,
                    board_numbers=missing_parent,
                    batch_id=f"{tag}_base",
                    run_kind="ab_retest",
                    record_iteration=False,
                )
                for pr in parent_recs:
                    parent_lookup[(seed, pr.board_number)] = pr
            for b in bnums:
                before_map[(seed, b)] = parent_lookup[(seed, b)].imps_challenger

            after_recs, _ = self.evaluate_boards(
                version=candidate_version,
                seed=seed,
                board_numbers=bnums,
                batch_id=tag,
                run_kind="ab_retest",
                record_iteration=False,
            )
            for ar in after_recs:
                after_map[(seed, ar.board_number)] = ar.imps_challenger

        before_total = sum(before_map[(s, b)] for s, b in board_pairs)
        after_total = sum(after_map[(s, b)] for s, b in board_pairs)
        delta = after_total - before_total
        accepted = delta >= min_delta_imps
        verdict: Literal["accepted", "rejected"] = "accepted" if accepted else "rejected"
        per_board = [
            {
                "key": f"{s}:{b}",
                "before": before_map[(s, b)],
                "after": after_map[(s, b)],
                "delta": after_map[(s, b)] - before_map[(s, b)],
            }
            for s, b in board_pairs
        ]
        retest_rec = ABRetestRecord(
            candidate_version=candidate_version,
            parent_version=parent_version,
            board_keys=[f"{s}:{b}" for s, b in board_pairs],
            before_imps=before_total,
            after_imps=after_total,
            delta_imps=delta,
            verdict=verdict,
            min_delta_required=min_delta_imps,
            per_board=per_board,
        )
        self.store.append_ab_retest(retest_rec)
        self.store.set_version_status(
            candidate_version,
            verdict,
            promote_best=accepted,
            decision_note=(
                f"A/B retest on {len(board_pairs)} board(s): "
                f"{before_total:+d} -> {after_total:+d} ({delta:+d} IMP, gate >= {min_delta_imps:+d})"
            ),
        )
        return retest_rec

    def run_loop(
        self,
        *,
        seeds: Sequence[int],
        boards_per_batch: int = 10,
        start_board: int = 1,
        proposer: PromptProposer | None = None,
        retest_top_k: int = 4,
        min_delta_imps: int = 1,
    ) -> dict[str, Any]:
        """Run an iterative hill-climbing loop over `seeds`: eval -> propose -> A/B retest -> report."""
        total_eval = 0
        iterations_done: list[str] = []
        retests_done: list[ABRetestRecord] = []

        for idx, seed in enumerate(seeds, start=1):
            active_ver = self.store.load_manifest().active_version
            batch_id = f"iter{idx:02d}_s{seed}"
            bnums = list(range(start_board, start_board + boards_per_batch))
            batch_recs, iter_rec = self.evaluate_boards(
                version=active_ver,
                seed=seed,
                board_numbers=bnums,
                batch_id=batch_id,
                run_kind="eval",
                record_iteration=True,
            )
            total_eval += len(batch_recs)
            iterations_done.append(batch_id)

            if proposer is not None and idx < len(seeds):
                losing = sorted(
                    [r for r in batch_recs if r.imps_challenger < 0],
                    key=lambda r: r.imps_challenger,
                )
                candidates_for_opt = losing if losing else list(batch_recs)
                current_vars = self.store.load_variables(active_ver)
                history = [
                    self.store.load_version_meta(v) for v in self.store.list_versions()
                ]
                patch = proposer.propose(
                    system_id=self.store.load_manifest().system_id,
                    current_version=active_ver,
                    current_variables=current_vars,
                    losing_boards=candidates_for_opt,
                    version_history=history,
                )
                if patch is not None:
                    try:
                        next_vars = apply_patch_to_variables(current_vars, patch)
                    except ValueError:
                        next_vars = None
                    if next_vars is not None:
                        cand_meta = self.store.create_version(
                            parent_version=active_ver,
                            variables=next_vars,
                            hypothesis=patch.hypothesis,
                            rationale=patch.rationale,
                            target_boards=patch.target_boards,
                            proposer=proposer.name,
                        )
                        retest_rows = candidates_for_opt[: max(1, retest_top_k)]
                        pairs = [(r.tournament_seed, r.board_number) for r in retest_rows]
                        rt = self.retest_candidate(
                            candidate_version=cand_meta.version,
                            board_pairs=pairs,
                            min_delta_imps=min_delta_imps,
                        )
                        retests_done.append(rt)

            self.store.generate_reports()

        reports = self.store.generate_reports()
        manifest = self.store.load_manifest()
        return {
            "system_id": manifest.system_id,
            "best_version": manifest.best_version,
            "total_eval_boards": total_eval,
            "batches": iterations_done,
            "retests": len(retests_done),
            "accepted_retests": sum(1 for r in retests_done if r.verdict == "accepted"),
            "report_paths": {k: str(v) for k, v in reports.items()},
        }
