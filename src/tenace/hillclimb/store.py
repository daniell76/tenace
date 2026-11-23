from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path
import re
import threading

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

from tenace.harness.scoring import wbf_continuous_vp
from tenace.hillclimb.models import (
    ABRetestRecord,
    BoardTelemetryRecord,
    IterationRecord,
    MatchMode,
    PromptVariables,
    RoomSnapshot,
    RunManifest,
    SkillVersionMeta,
    VersionStatus,
    changed_variable_slots,
    extract_prompt_variables,
    render_prompt_snapshots,
    unified_diff_variables,
    utc_now_iso,
)
from tenace.skills.registry import BiddingSkill, render_skill_markdown

_VERSION_RE = re.compile(r"^v(\d{3,})$")
_SAFE_TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")

__all__ = ["HillClimbStore"]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _locked_append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(line + "\n")
            fh.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _safe_tag(raw: str) -> str:
    cleaned = _SAFE_TAG_RE.sub("_", raw.strip())
    return cleaned or "default"


class HillClimbStore:
    """Crash-safe, append-only artifact ledger preserving every prompt version and raw board."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self._lock = threading.Lock()

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def versions_root(self) -> Path:
        return self.run_dir / "versions"

    @property
    def raw_root(self) -> Path:
        return self.run_dir / "raw"

    @property
    def analysis_root(self) -> Path:
        return self.run_dir / "analysis"

    @property
    def boards_jsonl_path(self) -> Path:
        return self.raw_root / "boards.jsonl"

    @property
    def iterations_jsonl_path(self) -> Path:
        return self.raw_root / "iterations.jsonl"

    @property
    def ab_retests_jsonl_path(self) -> Path:
        return self.raw_root / "ab_retests.jsonl"

    def version_dir(self, version: str) -> Path:
        if not _VERSION_RE.match(version):
            raise ValueError(f"Invalid version identifier {version!r}; expected e.g. 'v000'")
        return self.versions_root / version

    @classmethod
    def init_run(
        cls,
        run_dir: Path | str,
        *,
        base_skill: BiddingSkill,
        match_mode: MatchMode = "controlled",
        reference_label: str = "reference",
        challenger_profile_label: str = "",
        notes: str = "",
        overwrite: bool = False,
    ) -> HillClimbStore:
        store = cls(run_dir)
        if store.manifest_path.exists() and not overwrite:
            raise FileExistsError(
                f"Hill-climb run already initialized at {store.run_dir} "
                "(pass overwrite=True to re-initialize)"
            )
        store.versions_root.mkdir(parents=True, exist_ok=True)
        store.raw_root.mkdir(parents=True, exist_ok=True)
        store.analysis_root.mkdir(parents=True, exist_ok=True)
        (store.run_dir / "best").mkdir(parents=True, exist_ok=True)

        base_vars = extract_prompt_variables(base_skill)
        manifest = RunManifest(
            system_id=base_skill.meta.id,
            system_name=base_skill.meta.name,
            match_mode=match_mode,
            reference_label=reference_label,
            challenger_profile_label=challenger_profile_label,
            active_version="v000",
            best_version="v000",
            base_skill_meta=base_skill.meta,
            notes=notes,
        )
        store._write_version_files(
            version="v000",
            variables=base_vars,
            skill_meta=base_skill.meta,
            meta_record=SkillVersionMeta(
                version="v000",
                parent_version=None,
                status="baseline",
                hypothesis=f"Baseline built-in/initial SKILL.md for '{base_skill.meta.id}'",
                proposer="baseline",
                changed_slots=[],
                combined_hash=base_vars.combined_hash(),
                slot_hashes=base_vars.slot_hashes(),
                slot_char_counts=base_vars.slot_char_counts(),
                skill_meta=base_skill.meta,
            ),
            diff_text=None,
        )
        store._save_manifest(manifest)
        store._materialize_best("v000")
        return store

    def _save_manifest(self, manifest: RunManifest) -> None:
        updated = manifest.model_copy(update={"updated_at": utc_now_iso()})
        _atomic_write_text(self.manifest_path, updated.model_dump_json(indent=2) + "\n")

    def load_manifest(self) -> RunManifest:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"No manifest.json found in {self.run_dir}; run 'init' first"
            )
        return RunManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )

    def _write_version_files(
        self,
        *,
        version: str,
        variables: PromptVariables,
        skill_meta,
        meta_record: SkillVersionMeta,
        diff_text: str | None,
    ) -> None:
        vdir = self.version_dir(version)
        vdir.mkdir(parents=True, exist_ok=True)
        skill_md = render_skill_markdown(
            skill_meta,
            summary_rules=variables.summary_rules,
            detailed_rules=variables.detailed_rules,
            calibration_notes=variables.calibration_notes,
        )
        _atomic_write_text(vdir / "SKILL.md", skill_md)
        _atomic_write_text(
            vdir / "variables.json", variables.model_dump_json(indent=2) + "\n"
        )
        rendered = render_prompt_snapshots(skill_meta, variables)
        _atomic_write_text(
            vdir / "rendered_prompts.json",
            json.dumps(rendered, indent=2, ensure_ascii=False) + "\n",
        )
        _atomic_write_text(
            vdir / "metadata.json", meta_record.model_dump_json(indent=2) + "\n"
        )
        if diff_text is not None:
            _atomic_write_text(vdir / "diff_from_parent.patch", diff_text + "\n")

    def _materialize_best(self, version: str) -> None:
        src = self.version_dir(version) / "SKILL.md"
        dst = self.run_dir / "best" / "SKILL.md"
        _atomic_write_text(dst, src.read_text(encoding="utf-8"))
        vars_src = self.version_dir(version) / "variables.json"
        _atomic_write_text(
            self.run_dir / "best" / "variables.json",
            vars_src.read_text(encoding="utf-8"),
        )
        _atomic_write_text(
            self.run_dir / "best" / "VERSION",
            f"{version}\n",
        )

    def list_versions(self) -> list[str]:
        if not self.versions_root.is_dir():
            return []
        found = [
            p.name
            for p in self.versions_root.iterdir()
            if p.is_dir() and _VERSION_RE.match(p.name)
        ]
        return sorted(found, key=lambda name: int(name[1:]))

    def next_version_id(self) -> str:
        existing = self.list_versions()
        if not existing:
            return "v000"
        highest = max(int(v[1:]) for v in existing)
        return f"v{highest + 1:03d}"

    def load_variables(self, version: str) -> PromptVariables:
        path = self.version_dir(version) / "variables.json"
        return PromptVariables.model_validate_json(path.read_text(encoding="utf-8"))

    def load_version_meta(self, version: str) -> SkillVersionMeta:
        path = self.version_dir(version) / "metadata.json"
        return SkillVersionMeta.model_validate_json(path.read_text(encoding="utf-8"))

    def load_convention_card(self, version: str):
        meta = self.load_version_meta(version)
        variables = self.load_variables(version)
        return variables.to_convention_card(meta.skill_meta)

    def create_version(
        self,
        *,
        parent_version: str,
        variables: PromptVariables,
        hypothesis: str,
        rationale: str = "",
        target_boards: list[str] | None = None,
        proposer: str = "manual",
        status: VersionStatus = "candidate",
    ) -> SkillVersionMeta:
        with self._lock:
            manifest = self.load_manifest()
            parent_vars = self.load_variables(parent_version)
            changed = changed_variable_slots(parent_vars, variables)
            if not changed:
                raise ValueError(
                    f"Candidate prompt variables match parent {parent_version!r} identically"
                )
            version = self.next_version_id()
            skill_meta = manifest.base_skill_meta.model_copy(
                update={"version": f"{manifest.base_skill_meta.version}-{version}"}
            )
            meta_record = SkillVersionMeta(
                version=version,
                parent_version=parent_version,
                status=status,
                hypothesis=hypothesis.strip(),
                rationale=rationale.strip(),
                proposer=proposer,
                target_boards=list(target_boards or []),
                changed_slots=changed,
                combined_hash=variables.combined_hash(),
                slot_hashes=variables.slot_hashes(),
                slot_char_counts=variables.slot_char_counts(),
                skill_meta=skill_meta,
            )
            diff_text = unified_diff_variables(
                parent_version, parent_vars, version, variables
            )
            self._write_version_files(
                version=version,
                variables=variables,
                skill_meta=skill_meta,
                meta_record=meta_record,
                diff_text=diff_text,
            )
            return meta_record

    def set_version_status(
        self,
        version: str,
        status: VersionStatus,
        *,
        promote_best: bool = False,
        decision_note: str = "",
    ) -> SkillVersionMeta:
        with self._lock:
            meta = self.load_version_meta(version)
            updated = meta.model_copy(
                update={"status": status, "decision_note": decision_note.strip()}
            )
            _atomic_write_text(
                self.version_dir(version) / "metadata.json",
                updated.model_dump_json(indent=2) + "\n",
            )
            if promote_best:
                manifest = self.load_manifest()
                manifest = manifest.model_copy(
                    update={"active_version": version, "best_version": version}
                )
                self._save_manifest(manifest)
                self._materialize_best(version)
            return updated

    def diff_versions(self, from_version: str, to_version: str) -> str:
        return unified_diff_variables(
            from_version,
            self.load_variables(from_version),
            to_version,
            self.load_variables(to_version),
        )

    # --- Append-only raw JSONL ledgers ---

    def append_board(self, record: BoardTelemetryRecord) -> None:
        with self._lock:
            _locked_append_line(self.boards_jsonl_path, record.model_dump_json())

    def read_boards(
        self,
        *,
        prompt_version: str | None = None,
        run_kind: str | None = None,
        batch_id: str | None = None,
    ) -> list[BoardTelemetryRecord]:
        if not self.boards_jsonl_path.is_file():
            return []
        rows: list[BoardTelemetryRecord] = []
        for line in self.boards_jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = BoardTelemetryRecord.model_validate_json(line)
            if prompt_version is not None and rec.prompt_version != prompt_version:
                continue
            if run_kind is not None and rec.run_kind != run_kind:
                continue
            if batch_id is not None and rec.batch_id != batch_id:
                continue
            rows.append(rec)
        return rows

    def append_iteration(self, record: IterationRecord) -> None:
        with self._lock:
            _locked_append_line(self.iterations_jsonl_path, record.model_dump_json())

    def read_iterations(self) -> list[IterationRecord]:
        if not self.iterations_jsonl_path.is_file():
            return []
        rows: list[IterationRecord] = []
        for line in self.iterations_jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(IterationRecord.model_validate_json(line))
        return rows

    def append_ab_retest(self, record: ABRetestRecord) -> None:
        with self._lock:
            _locked_append_line(self.ab_retests_jsonl_path, record.model_dump_json())

    def read_ab_retests(self) -> list[ABRetestRecord]:
        if not self.ab_retests_jsonl_path.is_file():
            return []
        rows: list[ABRetestRecord] = []
        for line in self.ab_retests_jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(ABRetestRecord.model_validate_json(line))
        return rows

    # --- Reusable Table A reference cache ---

    def _ref_cache_path(self, engine_tag: str, seed: int, board_number: int) -> Path:
        return (
            self.raw_root
            / "reference_cache"
            / _safe_tag(engine_tag)
            / f"seed_{seed}_bd_{board_number}.json"
        )

    def save_reference_board(
        self,
        engine_tag: str,
        seed: int,
        board_number: int,
        snapshot: RoomSnapshot,
    ) -> Path:
        path = self._ref_cache_path(engine_tag, seed, board_number)
        _atomic_write_text(path, snapshot.model_dump_json(indent=2) + "\n")
        return path

    def load_reference_board(
        self,
        engine_tag: str,
        seed: int,
        board_number: int,
    ) -> RoomSnapshot | None:
        path = self._ref_cache_path(engine_tag, seed, board_number)
        if not path.is_file():
            return None
        return RoomSnapshot.model_validate_json(path.read_text(encoding="utf-8"))

    # --- Human- & Pandas-friendly report / CSV / changelog generation ---

    def generate_reports(self) -> dict[str, Path]:
        manifest = self.load_manifest()
        versions = [self.load_version_meta(v) for v in self.list_versions()]
        boards = self.read_boards()
        iterations = self.read_iterations()
        retests = self.read_ab_retests()

        self.analysis_root.mkdir(parents=True, exist_ok=True)
        boards_csv = self.analysis_root / "boards.csv"
        versions_csv = self.analysis_root / "versions.csv"
        changelog_md = self.analysis_root / "changelog.md"
        report_md = self.analysis_root / "report.md"

        # 1. Flat boards CSV
        with boards_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "recorded_at",
                    "run_kind",
                    "batch_id",
                    "prompt_version",
                    "match_mode",
                    "seed",
                    "board_number",
                    "dealer",
                    "vul",
                    "table_a_contract",
                    "table_a_ns_score",
                    "table_b_contract",
                    "table_b_ns_score",
                    "score_diff",
                    "imps_challenger",
                    "auction_exact_match",
                    "contract_match",
                    "first_div_idx",
                    "first_div_seat",
                    "first_div_ref_call",
                    "first_div_chal_call",
                    "swing_category",
                    "ref_cache_hit",
                    "l1_calls",
                    "l2_calls",
                    "forced_moves",
                    "table_a_elapsed_s",
                    "table_b_elapsed_s",
                    "north_hcp",
                    "east_hcp",
                    "south_hcp",
                    "west_hcp",
                ],
            )
            writer.writeheader()
            for r in boards:
                div = r.first_divergence
                writer.writerow(
                    {
                        "recorded_at": r.recorded_at,
                        "run_kind": r.run_kind,
                        "batch_id": r.batch_id,
                        "prompt_version": r.prompt_version,
                        "match_mode": r.match_mode,
                        "seed": r.tournament_seed,
                        "board_number": r.board_number,
                        "dealer": r.dealer,
                        "vul": r.vul,
                        "table_a_contract": r.table_a.contract,
                        "table_a_ns_score": r.table_a.ns_score,
                        "table_b_contract": r.table_b.contract,
                        "table_b_ns_score": r.table_b.ns_score,
                        "score_diff": r.score_diff,
                        "imps_challenger": r.imps_challenger,
                        "auction_exact_match": r.auction_exact_match,
                        "contract_match": r.contract_match,
                        "first_div_idx": "" if div is None else div.call_index,
                        "first_div_seat": "" if div is None else div.seat,
                        "first_div_ref_call": "" if div is None else div.reference_call,
                        "first_div_chal_call": "" if div is None else div.challenger_call,
                        "swing_category": r.swing_category,
                        "ref_cache_hit": r.reference_cache_hit,
                        "l1_calls": r.telemetry.get("strategist_calls", 0),
                        "l2_calls": r.telemetry.get("evaluator_calls", 0),
                        "forced_moves": r.telemetry.get("forced_moves", 0),
                        "table_a_elapsed_s": r.table_a.elapsed_s,
                        "table_b_elapsed_s": r.table_b.elapsed_s,
                        "north_hcp": r.deal.hand_hcp.get("NORTH", ""),
                        "east_hcp": r.deal.hand_hcp.get("EAST", ""),
                        "south_hcp": r.deal.hand_hcp.get("SOUTH", ""),
                        "west_hcp": r.deal.hand_hcp.get("WEST", ""),
                    }
                )

        # 2. Versions progression CSV
        eval_boards_by_ver: dict[str, list[BoardTelemetryRecord]] = {}
        for r in boards:
            if r.run_kind == "eval":
                eval_boards_by_ver.setdefault(r.prompt_version, []).append(r)
        retest_by_ver = {rt.candidate_version: rt for rt in retests}

        with versions_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "version",
                    "parent_version",
                    "status",
                    "proposer",
                    "created_at",
                    "combined_hash",
                    "changed_slots",
                    "summary_chars",
                    "detailed_chars",
                    "calibration_chars",
                    "eval_boards",
                    "eval_net_imps",
                    "eval_imp_per_board",
                    "ab_retest_before",
                    "ab_retest_after",
                    "ab_retest_delta",
                    "hypothesis",
                ],
            )
            writer.writeheader()
            for vm in versions:
                vb = eval_boards_by_ver.get(vm.version, [])
                net = sum(b.imps_challenger for b in vb)
                ipb = round(net / len(vb), 3) if vb else ""
                rt = retest_by_ver.get(vm.version)
                writer.writerow(
                    {
                        "version": vm.version,
                        "parent_version": vm.parent_version or "",
                        "status": vm.status,
                        "proposer": vm.proposer,
                        "created_at": vm.created_at,
                        "combined_hash": vm.combined_hash,
                        "changed_slots": ",".join(vm.changed_slots),
                        "summary_chars": vm.slot_char_counts.get("summary_rules", 0),
                        "detailed_chars": vm.slot_char_counts.get("detailed_rules", 0),
                        "calibration_chars": vm.slot_char_counts.get("calibration_notes", 0),
                        "eval_boards": len(vb),
                        "eval_net_imps": net if vb else "",
                        "eval_imp_per_board": ipb,
                        "ab_retest_before": "" if rt is None else rt.before_imps,
                        "ab_retest_after": "" if rt is None else rt.after_imps,
                        "ab_retest_delta": "" if rt is None else rt.delta_imps,
                        "hypothesis": vm.hypothesis,
                    }
                )

        # 3. Changelog with full unified diffs (including rejected candidates!)
        cl_lines = [
            f"# Prompt Version History & Diff Changelog — `{manifest.system_id}`",
            "",
            f"- **System:** {manifest.system_name}",
            f"- **Best Promoted Version:** `{manifest.best_version}` (copied to `best/SKILL.md`)",
            f"- **Total Versions Archived:** `{len(versions)}`",
            "",
        ]
        for vm in versions:
            badge = vm.status.upper()
            parent = f" (parent: `{vm.parent_version}`)" if vm.parent_version else " (root)"
            cl_lines.append(f"## `{vm.version}` — [{badge}]{parent}")
            cl_lines.append(f"- **Created:** `{vm.created_at}` | **Proposer:** `{vm.proposer}` | **Hash:** `{vm.combined_hash}`")
            if vm.hypothesis:
                cl_lines.append(f"- **Hypothesis:** {vm.hypothesis}")
            if vm.rationale:
                cl_lines.append(f"- **Rationale:** {vm.rationale}")
            if vm.target_boards:
                cl_lines.append(f"- **Target Boards:** `{', '.join(vm.target_boards)}`")
            if vm.changed_slots:
                cl_lines.append(f"- **Changed Slots:** `{', '.join(vm.changed_slots)}`")
            rt = retest_by_ver.get(vm.version)
            if rt is not None:
                cl_lines.append(
                    f"- **A/B Retest (`{len(rt.board_keys)}` boards):** "
                    f"`{rt.before_imps:+d}` → **`{rt.after_imps:+d}` IMPs** "
                    f"(**`{rt.delta_imps:+d}` Δ IMP**, verdict: `{rt.verdict.upper()}`)"
                )
            if vm.decision_note:
                cl_lines.append(f"- **Decision Note:** {vm.decision_note}")
            patch_file = self.version_dir(vm.version) / "diff_from_parent.patch"
            if patch_file.is_file():
                patch_text = patch_file.read_text(encoding="utf-8").strip()
                if patch_text:
                    cl_lines.extend(["", "```diff", patch_text, "```"])
            cl_lines.append("")
        _atomic_write_text(changelog_md, "\n".join(cl_lines).rstrip() + "\n")

        # 4. Executive Analysis Report
        eval_only = [b for b in boards if b.run_kind == "eval"]
        total_n = len(eval_only)
        total_net = sum(b.imps_challenger for b in eval_only)
        total_won = sum(b.imps_challenger for b in eval_only if b.imps_challenger > 0)
        total_lost = sum(-b.imps_challenger for b in eval_only if b.imps_challenger < 0)
        wins = sum(1 for b in eval_only if b.imps_challenger > 0)
        losses = sum(1 for b in eval_only if b.imps_challenger < 0)
        pushes = sum(1 for b in eval_only if b.imps_challenger == 0)
        exact_auc = sum(1 for b in eval_only if b.auction_exact_match)
        exact_con = sum(1 for b in eval_only if b.contract_match)
        vp_str = (
            f"{wbf_continuous_vp(total_net, total_n)[0]:.2f} – {wbf_continuous_vp(total_net, total_n)[1]:.2f}"
            if total_n > 0
            else "—"
        )

        rp = [
            f"# Hill-Climbing Calibration Report — `{manifest.system_id}`",
            "",
            f"- **System:** {manifest.system_name} (`{manifest.system_id}`)",
            f"- **Match Mode:** `{manifest.match_mode}` vs `{manifest.reference_label}`",
            f"- **Best Version:** `{manifest.best_version}` (`best/SKILL.md`)",
            f"- **Fresh Eval Boards:** `{total_n}` | **Net IMPs:** **`{total_net:+d}`** (`{total_won}` won / `{total_lost}` lost, `{wins}W-{losses}L-{pushes}T`)",
            f"- **IMP / Board:** **`{(total_net / total_n):+.2f}`**" if total_n else "- **IMP / Board:** —",
            f"- **Aggregate WBF 20–0 VP:** `{vp_str}`",
            (
                f"- **Auction Exact Match:** `{exact_auc}/{total_n}` ({100 * exact_auc / total_n:.1f}%) | "
                f"**Contract Match:** `{exact_con}/{total_n}` ({100 * exact_con / total_n:.1f}%)"
                if total_n
                else "- **Match Rates:** —"
            ),
            "",
        ]

        if iterations:
            rp.extend(
                [
                    "## Batch / Match Progression",
                    "",
                    "| Batch | Ver | Seed | Bds | Net IMP | Gross (+/−) | IMP/bd | WBF VP (20–0) | W-L-T | Auction Match | Contract Match |",
                    "|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|",
                ]
            )
            for it in iterations:
                rp.append(
                    f"| `{it.batch_id}` | `{it.prompt_version}` | {it.tournament_seed} | {it.num_boards} | "
                    f"**`{it.net_imps:+d}`** | `+{it.gross_won_imps}/-{it.gross_lost_imps}` | "
                    f"{it.imp_per_board:+.2f} | {it.vp_challenger:.2f}–{it.vp_reference:.2f} | "
                    f"{it.wins}-{it.losses}-{it.pushes} | {100 * it.auction_match_rate:.0f}% | "
                    f"{100 * it.contract_match_rate:.0f}% |"
                )
            rp.append("")

        if eval_only:
            cat_counts = Counter(b.swing_category for b in eval_only)
            cat_imps: dict[str, int] = {}
            for b in eval_only:
                cat_imps[b.swing_category] = cat_imps.get(b.swing_category, 0) + b.imps_challenger
            rp.extend(
                [
                    "## Swing Taxonomy Breakdown (Fresh Eval Boards)",
                    "",
                    "| Category | Boards | Net IMPs | Avg IMP/Board |",
                    "|---|---:|---:|---:|",
                ]
            )
            for cat, cnt in sorted(cat_counts.items(), key=lambda kv: cat_imps[kv[0]]):
                net_c = cat_imps[cat]
                rp.append(f"| `{cat}` | {cnt} | `{net_c:+d}` | {net_c / cnt:+.2f} |")
            rp.append("")

            big_swings = sorted(
                [b for b in eval_only if b.imps_challenger != 0],
                key=lambda b: abs(b.imps_challenger),
                reverse=True,
            )[:15]
            if big_swings:
                rp.extend(
                    [
                        "## Largest Swings (Top 15 by |IMP|)",
                        "",
                        "| Board | Ver | Vul | Ref Contract | Chal Contract | Score Δ | IMP | Category | First Divergence |",
                        "|---|---|---|---|---|---:|---:|---|---|",
                    ]
                )
                for b in big_swings:
                    ref_r = b.table_a if b.match_mode == "controlled" else b.table_b
                    chal_r = b.table_b if b.match_mode == "controlled" else b.table_a
                    div_txt = (
                        f"#{b.first_divergence.call_index + 1} {b.first_divergence.seat}: "
                        f"`{b.first_divergence.reference_call}` vs `{b.first_divergence.challenger_call}`"
                        if b.first_divergence is not None
                        else "*(identical auction)*"
                    )
                    rp.append(
                        f"| `{b.tournament_seed}:{b.board_number}` | `{b.prompt_version}` | {b.vul} | "
                        f"`{ref_r.contract}` ({ref_r.ns_score:+.0f}) | `{chal_r.contract}` ({chal_r.ns_score:+.0f}) | "
                        f"{b.score_diff:+.0f} | **`{b.imps_challenger:+d}`** | `{b.swing_category}` | {div_txt} |"
                    )
                rp.append("")

        if retests:
            rp.extend(
                [
                    "## Matched-Seed A/B Retests",
                    "",
                    "| Candidate | Parent | Boards | Before | After | Δ IMP | Verdict |",
                    "|---|---|---:|---:|---:|---:|---|",
                ]
            )
            for rt in retests:
                rp.append(
                    f"| `{rt.candidate_version}` | `{rt.parent_version}` | {len(rt.board_keys)} | "
                    f"{rt.before_imps:+d} | {rt.after_imps:+d} | **`{rt.delta_imps:+d}`** | `{rt.verdict.upper()}` |"
                )
            rp.append("")

        _atomic_write_text(report_md, "\n".join(rp).rstrip() + "\n")
        return {
            "report_md": report_md,
            "changelog_md": changelog_md,
            "boards_csv": boards_csv,
            "versions_csv": versions_csv,
        }
