from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tenace.cli import (
    DEFAULT_DECISIONS_MODEL,
    DEFAULT_STRATEGIST_MODEL,
    ModelProfile,
)
from tenace.hillclimb.models import (
    ProposedPromptPatch,
    apply_patch_to_variables,
)
from tenace.hillclimb.optimizer import GeminiSkillOptimizer
from tenace.hillclimb.runner import HillClimbEngine
from tenace.hillclimb.store import HillClimbStore
from tenace.models.clients import VertexGeminiClient
from tenace.skills import default_skill_registry, parse_skill_markdown

__all__ = ["build_arg_parser", "main", "parse_boards_spec", "run_cli"]


def parse_boards_spec(spec: str) -> list[tuple[int, int]]:
    """Parse `'3000:1,2,5;3100:4'` into `[(3000, 1), (3000, 2), (3000, 5), (3100, 4)]`."""
    pairs: list[tuple[int, int]] = []
    for group in spec.split(";"):
        grp = group.strip()
        if not grp:
            continue
        if ":" not in grp:
            raise ValueError(f"Expected 'SEED:B1,B2' in boards-spec, got {grp!r}")
        seed_raw, boards_raw = grp.split(":", 1)
        seed = int(seed_raw.strip())
        for tok in boards_raw.split(","):
            t = tok.strip()
            if "-" in t:
                lo, hi = t.split("-", 1)
                for b in range(int(lo), int(hi) + 1):
                    pairs.append((seed, b))
            elif t:
                pairs.append((seed, int(t)))
    if not pairs:
        raise ValueError("boards-spec resolved to zero boards")
    return pairs


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["vertex", "mock"], default="vertex")
    p.add_argument("--project", default=None)
    p.add_argument("--strategist-location", default="global")
    p.add_argument("--strategist-model", default=DEFAULT_STRATEGIST_MODEL)
    p.add_argument(
        "--evaluator-backend",
        choices=["decisions", "diffusion-gemma", "gemini"],
        default="decisions",
    )
    p.add_argument("--evaluator-location", default="global")
    p.add_argument("--evaluator-model", default=DEFAULT_DECISIONS_MODEL)
    p.add_argument("--decisions-auth", choices=["vertex", "api-key"], default="vertex")
    p.add_argument("--dds-worlds", type=int, default=16)
    p.add_argument(
        "--opponent",
        choices=["bluechip", "baseline", "tenace"],
        default="bluechip",
    )
    p.add_argument("--bluechip-cmd", default=None)
    p.add_argument("--bluechip-timeout", type=float, default=300.0)
    p.add_argument("--double-dummy", action="store_true", default=False)
    p.add_argument("--ref-cache-tag", default=None)


def _profile_from_args(args: argparse.Namespace) -> ModelProfile:
    return ModelProfile(
        mode=args.mode,
        project=args.project,
        strategist_location=args.strategist_location,
        strategist_model=args.strategist_model,
        evaluator_backend=args.evaluator_backend,
        evaluator_location=args.evaluator_location,
        evaluator_model=args.evaluator_model,
        decisions_auth=args.decisions_auth,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tenace-hillclimb",
        description=(
            "Versioned hill-climbing calibration harness for Tenace SKILL.md bidding systems."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize a new versioned run directory (v000).")
    p_init.add_argument("--system", required=True, help="Skill ID or path to SKILL.md.")
    p_init.add_argument("--skills-dir", action="append", default=None)
    p_init.add_argument("--out-dir", required=True, help="Output directory for artifacts.")
    p_init.add_argument(
        "--match-mode", choices=["controlled", "duplicate"], default="controlled"
    )
    p_init.add_argument("--reference-label", default="BEN")
    p_init.add_argument("--notes", default="")
    p_init.add_argument("--overwrite", action="store_true", default=False)

    p_show = sub.add_parser(
        "show-prompts",
        help="Print the exact exposed variable prompt slots for a skill version.",
    )
    p_show.add_argument("--out-dir", required=True)
    p_show.add_argument("--version", default=None)
    p_show.add_argument("--rendered", action="store_true", default=False)

    p_diff = sub.add_parser("diff", help="Show unified diff between two archived versions.")
    p_diff.add_argument("--out-dir", required=True)
    p_diff.add_argument("--from", dest="from_ver", required=True)
    p_diff.add_argument("--to", dest="to_ver", required=True)

    p_eval = sub.add_parser("eval", help="Evaluate a skill version on a batch of boards.")
    p_eval.add_argument("--out-dir", required=True)
    p_eval.add_argument("--version", default=None)
    p_eval.add_argument("--match-mode", choices=["controlled", "duplicate"], default=None)
    p_eval.add_argument("--seed", type=int, required=True)
    p_eval.add_argument("--start-board", type=int, default=1)
    p_eval.add_argument("--boards", type=int, default=12)
    p_eval.add_argument("--batch-id", default=None)
    p_eval.add_argument("--no-ref-cache", action="store_true", default=False)
    _add_model_args(p_eval)

    p_prop = sub.add_parser(
        "propose", help="Create candidate prompt version vNNN with full diff & hypothesis."
    )
    p_prop.add_argument("--out-dir", required=True)
    p_prop.add_argument("--parent", default=None, help="Parent version (default: best/active).")
    p_prop.add_argument("--hypothesis", default="Targeted prompt calibration")
    p_prop.add_argument("--rationale", default="")
    p_prop.add_argument("--target-boards", default="", help="Comma-separated e.g. 3000:2,3000:5")
    p_prop.add_argument("--from-skill-file", default=None)
    p_prop.add_argument("--append-calibration", default=None)
    p_prop.add_argument("--replace-calibration", default=None)
    p_prop.add_argument("--append-reference", default=None)
    p_prop.add_argument("--replace-summary", default=None)
    p_prop.add_argument("--l1-directive", default=None)
    p_prop.add_argument("--auto-mutate", action="store_true", default=False)
    p_prop.add_argument("--project", default=None)
    p_prop.add_argument("--optimizer-model", default=DEFAULT_STRATEGIST_MODEL)
    p_prop.add_argument("--batch-id", default=None)

    p_ret = sub.add_parser(
        "retest",
        help="Run matched-seed A/B retest gating acceptance/rejection of a candidate version.",
    )
    p_ret.add_argument("--out-dir", required=True)
    p_ret.add_argument("--version", required=True)
    p_ret.add_argument(
        "--boards-spec", required=True, help="e.g. '3000:1,2,5;3100:4' or '3000:1-12'"
    )
    p_ret.add_argument("--min-delta-imps", type=int, default=1)
    _add_model_args(p_ret)

    p_run = sub.add_parser(
        "run", help="Run an end-to-end iterative hill-climbing loop over multiple batches."
    )
    p_run.add_argument("--out-dir", required=True)
    p_run.add_argument("--system", required=True)
    p_run.add_argument("--skills-dir", action="append", default=None)
    p_run.add_argument(
        "--match-mode", choices=["controlled", "duplicate"], default="controlled"
    )
    p_run.add_argument("--reference-label", default="BEN")
    p_run.add_argument("--seeds", required=True, help="Comma-separated seeds e.g. 3000,3100,3200")
    p_run.add_argument("--boards-per-batch", type=int, default=10)
    p_run.add_argument("--start-board", type=int, default=1)
    p_run.add_argument("--auto-mutate", action="store_true", default=False)
    p_run.add_argument("--optimizer-model", default=DEFAULT_STRATEGIST_MODEL)
    p_run.add_argument("--retest-top-k", type=int, default=4)
    p_run.add_argument("--min-delta-imps", type=int, default=1)
    _add_model_args(p_run)

    p_rep = sub.add_parser(
        "report", help="Regenerate analysis/report.md, changelog.md, boards.csv, versions.csv."
    )
    p_rep.add_argument("--out-dir", required=True)
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cmd = args.command

    if cmd == "init":
        reg = default_skill_registry(extra_dirs=args.skills_dir)
        skill = reg.load(args.system, allow_file_path=True)
        store = HillClimbStore.init_run(
            args.out_dir,
            base_skill=skill,
            match_mode=args.match_mode,
            reference_label=args.reference_label,
            notes=args.notes,
            overwrite=args.overwrite,
        )
        print(f"Initialized hill-climb run '{skill.meta.id}' at {store.run_dir} (v000)")
        return 0

    if cmd == "show-prompts":
        store = HillClimbStore(args.out_dir)
        ver = args.version or store.load_manifest().active_version
        target = "rendered_prompts.json" if args.rendered else "variables.json"
        vdir = store.version_dir(ver)
        payload = json.loads((vdir / target).read_text(encoding="utf-8"))
        print(f"=== {store.load_manifest().system_name} [{ver}] ({target}) ===")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if cmd == "diff":
        store = HillClimbStore(args.out_dir)
        diff_txt = store.diff_versions(args.from_ver, args.to_ver)
        print(diff_txt if diff_txt else f"(No differences between {args.from_ver} and {args.to_ver})")
        return 0

    if cmd == "report":
        store = HillClimbStore(args.out_dir)
        paths = store.generate_reports()
        for k, v in paths.items():
            print(f"{k}: {v}")
        return 0

    if cmd == "eval":
        store = HillClimbStore(args.out_dir)
        ver = args.version or store.load_manifest().active_version
        batch_id = args.batch_id or f"{ver}_s{args.seed}_b{args.start_board}"
        engine = HillClimbEngine(
            store=store,
            challenger_profile=_profile_from_args(args),
            opponent=args.opponent,
            bluechip_cmd=args.bluechip_cmd,
            bluechip_timeout=args.bluechip_timeout,
            dds_worlds=args.dds_worlds,
            double_dummy=args.double_dummy,
            reference_cache_tag=args.ref_cache_tag,
            progress_cb=lambda m: print(m, flush=True),
        )
        bnums = list(range(args.start_board, args.start_board + args.boards))
        _recs, it = engine.evaluate_boards(
            version=ver,
            seed=args.seed,
            board_numbers=bnums,
            batch_id=batch_id,
            match_mode=args.match_mode,
            use_reference_cache=not args.no_ref_cache,
        )
        store.generate_reports()
        print(
            f"Batch {batch_id} ({ver}): {it.net_imps:+d} IMPs "
            f"(+{it.gross_won_imps}/-{it.gross_lost_imps}, VP {it.vp_challenger:.2f}-{it.vp_reference:.2f})"
        )
        return 0

    if cmd == "propose":
        store = HillClimbStore(args.out_dir)
        manifest = store.load_manifest()
        parent = args.parent or manifest.active_version
        base_vars = store.load_variables(parent)
        if args.auto_mutate:
            boards = store.read_boards(prompt_version=parent, run_kind="eval", batch_id=args.batch_id)
            losing = [b for b in boards if b.imps_challenger < 0]
            if not losing:
                print(f"No negative-IMP eval boards found for {parent}; nothing to mutate.")
                return 0
            opt_client = VertexGeminiClient(
                project=args.project, location="global", model=args.optimizer_model
            )
            optimizer = GeminiSkillOptimizer(opt_client)
            patch = optimizer.propose(
                system_id=manifest.system_id,
                current_version=parent,
                current_variables=base_vars,
                losing_boards=losing,
                version_history=[store.load_version_meta(v) for v in store.list_versions()],
            )
            if patch is None:
                print("Optimizer returned no patch.")
                return 0
            next_vars = apply_patch_to_variables(base_vars, patch)
            meta = store.create_version(
                parent_version=parent,
                variables=next_vars,
                hypothesis=patch.hypothesis,
                rationale=patch.rationale,
                target_boards=patch.target_boards,
                proposer=optimizer.name,
            )
        elif args.from_skill_file:
            skill = parse_skill_markdown(
                Path(args.from_skill_file).read_text(encoding="utf-8")
            )
            next_vars = base_vars.model_copy(
                update={
                    "summary_rules": skill.summary_rules,
                    "detailed_rules": skill.detailed_rules,
                    "calibration_notes": skill.calibration_notes,
                }
            )
            targets = [t.strip() for t in args.target_boards.split(",") if t.strip()]
            meta = store.create_version(
                parent_version=parent,
                variables=next_vars,
                hypothesis=args.hypothesis,
                rationale=args.rationale,
                target_boards=targets,
                proposer="manual-file",
            )
        else:
            targets = [t.strip() for t in args.target_boards.split(",") if t.strip()]
            patch = ProposedPromptPatch(
                hypothesis=args.hypothesis,
                rationale=args.rationale,
                target_boards=targets,
                summary_rules_replacement=args.replace_summary,
                detailed_rules_append=args.append_reference,
                calibration_notes_append=args.append_calibration,
                calibration_notes_replacement=args.replace_calibration,
                l1_auction_directive_replacement=args.l1_directive,
            )
            next_vars = apply_patch_to_variables(base_vars, patch)
            meta = store.create_version(
                parent_version=parent,
                variables=next_vars,
                hypothesis=args.hypothesis,
                rationale=args.rationale,
                target_boards=targets,
                proposer="manual",
            )
        store.generate_reports()
        print(
            f"Created candidate {meta.version} (parent={meta.parent_version}, "
            f"changed={meta.changed_slots})"
        )
        return 0

    if cmd == "retest":
        store = HillClimbStore(args.out_dir)
        pairs = parse_boards_spec(args.boards_spec)
        engine = HillClimbEngine(
            store=store,
            challenger_profile=_profile_from_args(args),
            opponent=args.opponent,
            bluechip_cmd=args.bluechip_cmd,
            bluechip_timeout=args.bluechip_timeout,
            dds_worlds=args.dds_worlds,
            double_dummy=args.double_dummy,
            reference_cache_tag=args.ref_cache_tag,
            progress_cb=lambda m: print(m, flush=True),
        )
        rt = engine.retest_candidate(
            candidate_version=args.version,
            board_pairs=pairs,
            min_delta_imps=args.min_delta_imps,
        )
        store.generate_reports()
        print(
            f"Retest {rt.candidate_version} vs {rt.parent_version}: "
            f"{rt.before_imps:+d} -> {rt.after_imps:+d} ({rt.delta_imps:+d} IMP) => {rt.verdict.upper()}"
        )
        return 0

    if cmd == "run":
        reg = default_skill_registry(extra_dirs=args.skills_dir)
        skill = reg.load(args.system, allow_file_path=True)
        out_path = Path(args.out_dir)
        if (out_path / "manifest.json").is_file():
            store = HillClimbStore(out_path)
        else:
            store = HillClimbStore.init_run(
                out_path,
                base_skill=skill,
                match_mode=args.match_mode,
                reference_label=args.reference_label,
                challenger_profile_label=_profile_from_args(args).short_label(),
            )
        proposer = None
        if args.auto_mutate:
            opt_client = VertexGeminiClient(
                project=args.project, location="global", model=args.optimizer_model
            )
            proposer = GeminiSkillOptimizer(opt_client)
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        engine = HillClimbEngine(
            store=store,
            challenger_profile=_profile_from_args(args),
            opponent=args.opponent,
            bluechip_cmd=args.bluechip_cmd,
            bluechip_timeout=args.bluechip_timeout,
            dds_worlds=args.dds_worlds,
            double_dummy=args.double_dummy,
            reference_cache_tag=args.ref_cache_tag,
            progress_cb=lambda m: print(m, flush=True),
        )
        summary = engine.run_loop(
            seeds=seeds,
            boards_per_batch=args.boards_per_batch,
            start_board=args.start_board,
            proposer=proposer,
            retest_top_k=args.retest_top_k,
            min_delta_imps=args.min_delta_imps,
        )
        print(json.dumps(summary, indent=2))
        return 0

    raise ValueError(f"Unknown command: {cmd!r}")


def main() -> None:
    try:
        raise SystemExit(run_cli())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
