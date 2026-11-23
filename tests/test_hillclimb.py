from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
import pytest

from tenace.domain.types import (
    FIRST_CONTRACT_BID_ACTION,
    PASS_ACTION,
    CandidateOption,
    Seat,
    StrategyState,
    Suit,
    TacticalDecision,
    card_id,
)
from tenace.harness.duplicate import DuplicateMatchRunner
from tenace.harness.scoring import wbf_continuous_vp
from tenace.layer1_strategist import (
    SAYC_CONVENTION_CARD,
    ConventionCard,
    StrategistEngine,
)
from tenace.layer2_evaluator import TacticalOptionEvaluator
from tenace.layer3_rules.tracker import BoardFactSnapshot
from tenace.skills import (
    load_bidding_skill,
    parse_skill_markdown,
    render_skill_markdown,
)

T = TypeVar("T", bound=BaseModel)


class _RecordingClient:
    def __init__(self, response: BaseModel) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "schema": schema,
            }
        )
        assert isinstance(self._response, schema)
        return self._response


def _sample_facts(*, partner_acted: bool = False) -> BoardFactSnapshot:
    calls = (
        [(Seat.NORTH, FIRST_CONTRACT_BID_ACTION + 2)] if partner_acted else []
    )
    return BoardFactSnapshot(
        observer=Seat.SOUTH,
        phase="auction",
        own_cards=[
            card_id(Suit.SPADES, 12),
            card_id(Suit.HEARTS, 11),
            card_id(Suit.CLUBS, 12),
        ],
        own_hcp=11,
        dummy_seat=None,
        dummy_cards=[],
        played_cards=[],
        unseen_cards=[],
        auction_calls=calls,
        declarer=None,
        trump_denom=None,
        contract_level=0,
        known_voids={s: set() for s in Seat},
        surprise_events=[],
    )


def test_wbf_continuous_vp_scale_accuracy_and_bounds() -> None:
    assert wbf_continuous_vp(0, 12) == (10.0, 10.0)
    assert wbf_continuous_vp(4, 12) == (11.38, 8.62)
    assert wbf_continuous_vp(-4, 12) == (8.62, 11.38)
    for imp in (-60, -25, -10, -1, 0, 1, 10, 25, 60):
        a, b = wbf_continuous_vp(imp, 12)
        assert round(a + b, 2) == 20.0
        assert 0.0 <= a <= 20.0
        assert 0.0 <= b <= 20.0
    assert wbf_continuous_vp(100, 12) == (20.0, 0.0)
    assert wbf_continuous_vp(-100, 12) == (0.0, 20.0)
    with pytest.raises(ValueError):
        wbf_continuous_vp(0, 0)
    with pytest.raises(ValueError):
        wbf_continuous_vp(True, 12)  # type: ignore[arg-type]


def test_skill_markdown_calibration_notes_and_prompt_overrides_roundtrip() -> None:
    ccba = load_bidding_skill("ccba")
    md = render_skill_markdown(
        ccba.meta,
        summary_rules=ccba.summary_rules,
        detailed_rules=ccba.detailed_rules,
        calibration_notes="- Avoid reopening 2-level doubles without tolerance for all unbid suits.",
    )
    reparsed = parse_skill_markdown(md)
    assert reparsed.meta.id == "ccba"
    assert reparsed.summary_rules == ccba.summary_rules
    assert "Key Differences vs. SAYC" in reparsed.detailed_rules
    assert "Avoid reopening 2-level doubles" in reparsed.calibration_notes

    card = reparsed.to_convention_card()
    assert card.calibration_notes == reparsed.calibration_notes
    assert SAYC_CONVENTION_CARD.calibration_notes == ""

    # Tunable overrides flow verbatim into Layer 1 & Layer 2 rendered prompts
    tuned = card.model_copy(
        update={
            "l1_auction_directive": "CUSTOM_L1_DIRECTIVE_TOKEN",
            "l2_discipline_overrides": {
                "partner_active": "CUSTOM_L2_PARTNER_TOKEN",
                "unopened": "CUSTOM_L2_UNOPENED_TOKEN",
            },
        }
    )
    l1_client = _RecordingClient(StrategyState(phase="auction", plan_summary="Pass"))
    StrategistEngine(l1_client, convention_card=tuned).update_strategy(
        _sample_facts(), None, "init"
    )
    l1_sys = l1_client.calls[0]["system_prompt"]
    assert "Avoid reopening 2-level doubles" in l1_sys
    assert "CUSTOM_L1_DIRECTIVE_TOKEN" in l1_sys

    l2_unopened = _RecordingClient(TacticalDecision(option_scores={"Pass": 0.9}))
    TacticalOptionEvaluator(l2_unopened, convention_card=tuned).evaluate(
        _sample_facts(partner_acted=False),
        StrategyState(phase="auction", plan_summary="Pass"),
        [CandidateOption(action=PASS_ACTION, label="Pass")],
    )
    l2_prompt_u = l2_unopened.calls[0]["user_prompt"]
    assert "Avoid reopening 2-level doubles" in l2_prompt_u
    assert "CUSTOM_L2_UNOPENED_TOKEN" in l2_prompt_u

    l2_partner = _RecordingClient(TacticalDecision(option_scores={"Pass": 0.9}))
    TacticalOptionEvaluator(l2_partner, convention_card=tuned).evaluate(
        _sample_facts(partner_acted=True),
        StrategyState(phase="auction", plan_summary="Bid 2H"),
        [CandidateOption(action=PASS_ACTION, label="Pass")],
    )
    assert "CUSTOM_L2_PARTNER_TOKEN" in l2_partner.calls[0]["user_prompt"]

    # Reject unknown override slot keys strictly
    with pytest.raises(ValidationError):
        ConventionCard(
            name="Bad",
            summary_rules="Rules",
            l2_discipline_overrides={"nonexistent_slot": "nope"},
        )


def test_diagnostics_deal_extraction_divergence_and_swing_taxonomy() -> None:
    from tenace.hillclimb.diagnostics import (
        classify_swing,
        extract_full_deal_hands,
        find_first_auction_divergence,
    )

    runner = DuplicateMatchRunner(tournament_seed=42, use_double_dummy_result=False)
    state = runner._new_dealt_board_state(1)
    deal = extract_full_deal_hands(state, board_number=1, seed=42)
    assert set(deal.hands.keys()) == {"NORTH", "EAST", "SOUTH", "WEST"}
    assert sum(deal.hand_hcp.values()) == 40
    for seat_name in ("NORTH", "EAST", "SOUTH", "WEST"):
        assert len(deal.cards_by_seat[seat_name]) == 13

    identical = find_first_auction_divergence(
        "NORTH:1H -> EAST:Pass -> SOUTH:2H -> WEST:Pass -> NORTH:Pass -> EAST:Pass",
        "NORTH:1H -> EAST:Pass -> SOUTH:2H -> WEST:Pass -> NORTH:Pass -> EAST:Pass",
    )
    assert identical is None

    div = find_first_auction_divergence(
        "EAST:1D -> SOUTH:1H -> WEST:Pass -> NORTH:2H",
        "EAST:1D -> SOUTH:Pass -> WEST:Pass -> NORTH:1S",
    )
    assert div is not None
    assert div.call_index == 1
    assert div.seat == "SOUTH"
    assert div.reference_call == "1H"
    assert div.challenger_call == "Pass"

    assert classify_swing("3D by NORTH", "3D by NORTH", 0, 110, 110) == "push"
    assert classify_swing("5C by WEST", "5C by WEST", 3, 100, 200) == "card_play_gain"
    assert classify_swing("3NT by WEST", "3NT by WEST", -12, 100, -520) == "card_play_loss"
    assert classify_swing("6S by NORTH", "4S by NORTH", -13, 1430, 680) == "auction_slam_swing"
    assert classify_swing("4SX by SOUTH", "4H by WEST", 5, -300, -100) == "auction_doubled_penalty"
    assert classify_swing("4H by SOUTH", "3H by SOUTH", -10, 620, 170) == "auction_game_swing"
    assert classify_swing("Passed Out", "2H by SOUTH", 3, 0, 110) == "auction_passed_out_swing"
    assert (
        classify_swing("3H by SOUTH", "4D by EAST", -7, 140, -130)
        == "auction_competitive_partscore"
    )


def test_hillclimb_store_preserves_full_version_history_diffs_and_reports(
    tmp_path: Path,
) -> None:
    from tenace.hillclimb.models import (
        ABRetestRecord,
        BoardDealSnapshot,
        BoardTelemetryRecord,
        PromptVariables,
        RoomSnapshot,
    )
    from tenace.hillclimb.store import HillClimbStore

    base_skill = load_bidding_skill("ccba")
    store = HillClimbStore.init_run(
        tmp_path / "ccba_run",
        base_skill=base_skill,
        match_mode="controlled",
        reference_label="BEN_test",
        challenger_profile_label="mock",
    )
    manifest = store.load_manifest()
    assert manifest.system_id == "ccba"
    assert manifest.active_version == "v000"
    assert manifest.best_version == "v000"
    assert (store.run_dir / "best" / "SKILL.md").is_file()
    assert (store.version_dir("v000") / "variables.json").is_file()
    assert (store.version_dir("v000") / "rendered_prompts.json").is_file()

    # Load v000 rendered prompts and verify all variable slots are exposed verbatim
    rendered = json.loads(
        (store.version_dir("v000") / "rendered_prompts.json").read_text(encoding="utf-8")
    )
    assert "l1_system_prompt" in rendered
    assert "l2_system_prompt" in rendered
    assert "l2_discipline_partner_active" in rendered
    assert "CCBA Natural" in rendered["l1_system_prompt"]

    # Create candidate v001 with targeted calibration note & promote it
    v0_vars = store.load_variables("v000")
    v1_vars = PromptVariables(
        **{
            **v0_vars.model_dump(),
            "calibration_notes": "- Over opponent's 1D, bid 1H with good 5-card heart suit (8+ HCP).",
        }
    )
    v1_meta = store.create_version(
        parent_version="v000",
        variables=v1_vars,
        hypothesis="Compete more accurately with 5-card major overcalls over 1D.",
        target_boards=["3000:2"],
        proposer="manual",
    )
    assert v1_meta.version == "v001"
    assert "calibration_notes" in v1_meta.changed_slots
    patch_text = (store.version_dir("v001") / "diff_from_parent.patch").read_text(
        encoding="utf-8"
    )
    assert "Over opponent's 1D, bid 1H" in patch_text
    assert store.diff_versions("v000", "v001").strip()
    store.set_version_status("v001", "accepted", promote_best=True)
    assert store.load_manifest().best_version == "v001"
    assert "Over opponent's 1D" in (store.run_dir / "best" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    # Create candidate v002 that fails retest -> rejected! History MUST survive untouched.
    v2_vars = PromptVariables(
        **{
            **v1_vars.model_dump(),
            "calibration_notes": v1_vars.calibration_notes + "\n- Bad speculative rule.",
        }
    )
    store.create_version(
        parent_version="v001",
        variables=v2_vars,
        hypothesis="Speculative aggression test.",
        proposer="manual",
    )
    store.set_version_status("v002", "rejected", promote_best=False)
    assert store.load_manifest().best_version == "v001"
    assert store.load_manifest().active_version == "v001"
    assert (store.version_dir("v002") / "SKILL.md").is_file()
    assert store.load_version_meta("v002").status == "rejected"

    # Append a board record, cached reference, and A/B retest, then generate reports
    rec = BoardTelemetryRecord(
        run_kind="eval",
        batch_id="b01",
        prompt_version="v000",
        match_mode="controlled",
        tournament_seed=3000,
        board_number=2,
        dealer="EAST",
        vul="N/S",
        deal=BoardDealSnapshot(
            hands={"NORTH": "S: A", "EAST": "D: A", "SOUTH": "H: A", "WEST": "C: A"},
            cards_by_seat={"NORTH": ["S:A"], "EAST": ["D:A"], "SOUTH": ["H:A"], "WEST": ["C:A"]},
            hand_hcp={"NORTH": 10, "EAST": 10, "SOUTH": 10, "WEST": 10},
            shapes={"NORTH": "4-3-3-3", "EAST": "4-3-3-3", "SOUTH": "4-3-3-3", "WEST": "4-3-3-3"},
        ),
        table_a=RoomSnapshot(
            contract="3H by SOUTH",
            auction="EAST:1D -> SOUTH:1H",
            ns_score=140.0,
            elapsed_s=10.0,
        ),
        table_b=RoomSnapshot(
            contract="4D by EAST",
            auction="EAST:1D -> SOUTH:Pass",
            ns_score=-130.0,
            elapsed_s=20.0,
        ),
        score_diff=-270.0,
        imps_challenger=-7,
        auction_exact_match=False,
        contract_match=False,
        swing_category="auction_competitive_partscore",
        telemetry={"strategist_calls": 4, "evaluator_calls": 9, "forced_moves": 3},
    )
    store.append_board(rec)
    store.save_reference_board("ben_ref", 3000, 2, rec.table_a)
    cached = store.load_reference_board("ben_ref", 3000, 2)
    assert cached is not None and cached.ns_score == 140.0

    store.append_ab_retest(
        ABRetestRecord(
            candidate_version="v001",
            parent_version="v000",
            board_keys=["3000:2"],
            before_imps=-7,
            after_imps=0,
            delta_imps=7,
            verdict="accepted",
            per_board=[{"key": "3000:2", "before": -7, "after": 0, "delta": 7}],
        )
    )
    out = store.generate_reports()
    assert out["report_md"].is_file()
    assert out["changelog_md"].is_file()
    assert out["boards_csv"].is_file()
    assert out["versions_csv"].is_file()
    report_txt = out["report_md"].read_text(encoding="utf-8")
    changelog_txt = out["changelog_md"].read_text(encoding="utf-8")
    assert "ccba" in report_txt
    assert "v001" in changelog_txt and "v002" in changelog_txt
    assert "REJECTED" in changelog_txt.upper()
    with out["boards_csv"].open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1 and rows[0]["swing_category"] == "auction_competitive_partscore"


def test_gemini_skill_optimizer_and_offline_hillclimb_loop(tmp_path: Path) -> None:
    from tenace.cli import ModelProfile
    from tenace.hillclimb.models import ProposedPromptPatch
    from tenace.hillclimb.optimizer import GeminiSkillOptimizer
    from tenace.hillclimb.runner import HillClimbEngine
    from tenace.hillclimb.store import HillClimbStore

    # 1. GeminiSkillOptimizer with fake StructuredModelClient
    fake_patch = ProposedPromptPatch(
        hypothesis="Strengthen calibration notes for semi-forcing 1NT follow-ups.",
        rationale="Boards showed passive pass over 1D with 5-card major.",
        target_boards=["42:1"],
        calibration_notes_append="- Over enemy 1D, overcall a good 5-card major aggressively at the 1-level.",
    )
    opt_client = _RecordingClient(fake_patch)
    opt = GeminiSkillOptimizer(opt_client)
    store = HillClimbStore.init_run(
        tmp_path / "loop_run",
        base_skill=load_bidding_skill("ccba"),
        match_mode="controlled",
        reference_label="baseline",
        challenger_profile_label="mock",
    )
    engine = HillClimbEngine(
        store=store,
        challenger_profile=ModelProfile(mode="mock"),
        opponent="baseline",
        dds_worlds=1,
        double_dummy=True,
        reference_cache_tag="baseline_dd",
    )
    # Run 2 iterations of 2 boards each (`controlled` mode, mock baseline ref, cached Table A)
    loop_summary = engine.run_loop(
        seeds=[42, 43],
        boards_per_batch=2,
        proposer=opt,
        retest_top_k=1,
        min_delta_imps=0,  # deterministic acceptance check in mock DD mode
    )
    assert loop_summary["total_eval_boards"] == 4
    assert len(store.list_versions()) >= 2
    assert (store.run_dir / "raw" / "boards.jsonl").is_file()
    assert (store.run_dir / "raw" / "iterations.jsonl").is_file()
    assert (store.run_dir / "raw" / "ab_retests.jsonl").is_file()
    # Confirm reference cache populated for seed=42 bd=1..2 and seed=43 bd=1..2
    assert store.load_reference_board("baseline_dd", 42, 1) is not None
    assert store.load_reference_board("baseline_dd", 43, 2) is not None
    assert (store.run_dir / "analysis" / "report.md").is_file()
    assert (store.run_dir / "analysis" / "changelog.md").is_file()


def test_hillclimb_cli_subcommands_offline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tenace.hillclimb.cli import run_cli

    run_dir = tmp_path / "cli_run"
    assert run_cli(["init", "--system", "precision", "--out-dir", str(run_dir)]) == 0
    assert (run_dir / "manifest.json").is_file()

    assert run_cli(["show-prompts", "--out-dir", str(run_dir), "--version", "v000"]) == 0
    prompt_out = capsys.readouterr().out
    assert "Precision Club" in prompt_out
    assert "calibration_notes" in prompt_out

    assert (
        run_cli(
            [
                "eval",
                "--out-dir",
                str(run_dir),
                "--mode",
                "mock",
                "--match-mode",
                "controlled",
                "--opponent",
                "baseline",
                "--seed",
                "77",
                "--boards",
                "1",
                "--dds-worlds",
                "1",
                "--double-dummy",
            ]
        )
        == 0
    )
    assert (run_dir / "raw" / "boards.jsonl").is_file()

    assert (
        run_cli(
            [
                "propose",
                "--out-dir",
                str(run_dir),
                "--append-calibration",
                "- Test strong 1C precision overcall rule.",
                "--hypothesis",
                "Verify manual CLI patch recording",
            ]
        )
        == 0
    )
    assert (run_dir / "versions" / "v001" / "SKILL.md").is_file()

    assert run_cli(["diff", "--out-dir", str(run_dir), "--from", "v000", "--to", "v001"]) == 0
    diff_out = capsys.readouterr().out
    assert "Test strong 1C precision overcall rule" in diff_out

    assert (
        run_cli(
            [
                "retest",
                "--out-dir",
                str(run_dir),
                "--version",
                "v001",
                "--boards-spec",
                "77:1",
                "--mode",
                "mock",
                "--opponent",
                "baseline",
                "--dds-worlds",
                "1",
                "--double-dummy",
                "--min-delta-imps",
                "0",
            ]
        )
        == 0
    )
    assert run_cli(["report", "--out-dir", str(run_dir)]) == 0
    assert (run_dir / "analysis" / "report.md").is_file()

    # Verify symmetric 2-room duplicate mode via CLI
    dup_dir = tmp_path / "dup_run"
    assert (
        run_cli(
            [
                "init",
                "--system",
                "21gf",
                "--match-mode",
                "duplicate",
                "--out-dir",
                str(dup_dir),
            ]
        )
        == 0
    )
    assert (
        run_cli(
            [
                "eval",
                "--out-dir",
                str(dup_dir),
                "--mode",
                "mock",
                "--opponent",
                "baseline",
                "--seed",
                "55",
                "--boards",
                "1",
                "--dds-worlds",
                "1",
                "--double-dummy",
            ]
        )
        == 0
    )
    assert "duplicate" in (dup_dir / "analysis" / "report.md").read_text(encoding="utf-8")
