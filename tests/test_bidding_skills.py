from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

from fastapi.testclient import TestClient
import pytest
from pydantic import BaseModel

from tenace.cli import ModelProfile, main as _cli_main  # noqa: F401
from tenace.domain.types import (
    FIRST_CONTRACT_BID_ACTION,
    PASS_ACTION,
    CandidateOption,
    Seat,
    StrategyState,
    TacticalDecision,
    card_id,
    Suit,
)
from tenace.layer1_strategist import (
    SAYC_CONVENTION_CARD,
    ConventionCard,
    StrategistEngine,
)
from tenace.layer2_evaluator import TacticalOptionEvaluator
from tenace.layer3_rules.tracker import BoardFactSnapshot
from tenace.skills import (
    BUILTIN_BIDDING_SKILLS_DIR,
    BiddingSkill,
    BiddingSkillMeta,
    SkillRegistry,
    default_skill_registry,
    load_bidding_skill,
    parse_skill_markdown,
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


def _sample_auction_facts(
    *,
    auction_calls: list[tuple[Seat, int]] | None = None,
    phase: str = "auction",
) -> BoardFactSnapshot:
    return BoardFactSnapshot(
        observer=Seat.SOUTH,
        phase=phase,  # type: ignore[arg-type]
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
        auction_calls=auction_calls or [],
        declarer=Seat.SOUTH if phase == "play" else None,
        trump_denom=4 if phase == "play" else None,
        contract_level=3 if phase == "play" else 0,
        known_voids={s: set() for s in Seat},
        surprise_events=[],
    )


def test_builtin_bidding_skills_discovery_and_schema() -> None:
    registry = SkillRegistry()
    metas = registry.list_skills()
    ids = {m.id for m in metas}
    assert ids == {"21gf", "acol", "ccba", "polish_club", "precision", "sayc"}
    for meta in metas:
        assert isinstance(meta, BiddingSkillMeta)
        assert meta.is_builtin is True
        assert meta.name.strip()
        assert meta.description.strip()
        assert meta.version.strip()

        skill = registry.load(meta.id)
        assert isinstance(skill, BiddingSkill)
        assert skill.meta.id == meta.id
        assert skill.summary_rules.strip()
        assert skill.detailed_rules.strip()
        assert skill.source_path == BUILTIN_BIDDING_SKILLS_DIR / meta.id / "SKILL.md"
        card = skill.to_convention_card()
        assert isinstance(card, ConventionCard)
        assert card.id == meta.id
        assert card.name == meta.name
        assert card.summary_rules == skill.summary_rules
        assert card.detailed_rules == skill.detailed_rules

    # Default SAYC_CONVENTION_CARD comes straight from the built-in sayc/SKILL.md
    sayc_card = load_bidding_skill("sayc").to_convention_card()
    assert SAYC_CONVENTION_CARD == sayc_card


def test_parse_skill_markdown_rejects_malformed_frontmatter_and_missing_body() -> None:
    with pytest.raises(ValueError, match="frontmatter"):
        parse_skill_markdown("# Missing frontmatter\n## Summary Rules\nhello")

    with pytest.raises(ValueError, match="Duplicate"):
        parse_skill_markdown(
            "---\nid: custom\nid: dup\nname: Custom\ndescription: Desc\n---\n\n## Summary Rules\nRules\n"
        )

    with pytest.raises(ValueError, match="id"):
        parse_skill_markdown(
            "---\nid: ../evil\nname: Bad\ndescription: Bad slug\n---\n\n## Summary Rules\nRules\n"
        )

    with pytest.raises(ValueError, match="Summary Rules"):
        parse_skill_markdown(
            "---\nid: empty_body\nname: Empty\ndescription: Missing rules\n---\n\n   \n"
        )

    # Single body without explicit headings is still accepted as summary_rules (clean minimalist SKILL.md)
    simple = parse_skill_markdown(
        '---\nid: mini_card\nname: "Mini Card"\nversion: "0.2"\ndescription: Short card\n---\n\nPlay 4-card majors and weak NT.\n'
    )
    assert simple.meta.id == "mini_card"
    assert simple.meta.name == "Mini Card"
    assert simple.meta.version == "0.2"
    assert simple.summary_rules == "Play 4-card majors and weak NT."
    assert simple.detailed_rules == ""


def test_external_skill_directory_discovery_override_and_hot_reload(tmp_path: Path) -> None:
    custom_dir = tmp_path / "moscito"
    custom_dir.mkdir()
    custom_file = custom_dir / "SKILL.md"
    custom_file.write_text(
        "---\n"
        "id: moscito\n"
        "name: MOSCITO (Strong 1C / 4-card Majors)\n"
        "version: '1.0'\n"
        "description: Major-oriented strong club.\n"
        "---\n\n"
        "## Summary Rules\n"
        "1C is 15+ artificial forcing. 1D shows 4+ hearts.\n\n"
        "## Reference Guide\n"
        "- Transfer openings with 4-card majors.\n",
        encoding="utf-8",
    )

    # Override an existing built-in skill ('sayc') from the same external root
    override_dir = tmp_path / "sayc"
    override_dir.mkdir()
    override_file = override_dir / "SKILL.md"
    override_file.write_text(
        "---\n"
        "id: sayc\n"
        "name: House SAYC v2\n"
        "version: '2.0'\n"
        "description: Customized house override of SAYC.\n"
        "---\n\n"
        "## Summary Rules\n"
        "House rule v1: play 14-16 1NT.\n",
        encoding="utf-8",
    )

    registry = SkillRegistry(extra_dirs=[tmp_path])
    metas = {m.id: m for m in registry.list_skills()}
    assert "moscito" in metas
    assert metas["moscito"].is_builtin is False
    assert metas["sayc"].name == "House SAYC v2"
    assert metas["sayc"].is_builtin is False

    loaded_v1 = registry.load("sayc")
    assert "14-16 1NT" in loaded_v1.summary_rules

    # Modify the external SKILL.md on disk and bump mtime_ns -> hot reload automatically!
    override_file.write_text(
        "---\n"
        "id: sayc\n"
        "name: House SAYC v2\n"
        "version: '2.1'\n"
        "description: Customized house override of SAYC.\n"
        "---\n\n"
        "## Summary Rules\n"
        "House rule v2: play 15-17 1NT with Puppet Stayman.\n",
        encoding="utf-8",
    )
    stat = override_file.stat()
    os.utime(override_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

    loaded_v2 = registry.load("sayc")
    assert "Puppet Stayman" in loaded_v2.summary_rules
    assert loaded_v2.meta.version == "2.1"


def test_direct_file_path_loading_vs_security_boundary(tmp_path: Path) -> None:
    standalone = tmp_path / "club_card.md"
    standalone.write_text(
        "---\n"
        "id: club_card\n"
        "name: Local Club Card\n"
        "description: Ad-hoc file skill.\n"
        "---\n\n"
        "## Summary Rules\n"
        "Open 1NT with 14-16 balanced.\n",
        encoding="utf-8",
    )
    registry = SkillRegistry()

    # Allowed explicitly in local/CLI mode:
    loaded = registry.load(str(standalone), allow_file_path=True)
    assert loaded.meta.id == "club_card"
    assert "14-16 balanced" in loaded.summary_rules

    # Rejected cleanly in remote/HTTP mode (allow_file_path=False):
    for dangerous in (str(standalone), "../sayc/SKILL.md", "/etc/passwd", "sayc/../../secret"):
        with pytest.raises(ValueError):
            registry.load(dangerous, allow_file_path=False)


def test_default_skill_registry_reads_env_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ext = tmp_path / "fantunes"
    ext.mkdir()
    (ext / "SKILL.md").write_text(
        "---\nid: fantunes\nname: Fantunes\ndescription: Forcing 1-level openings.\n---\n\n"
        "## Summary Rules\nAll 1-level openings are 14+ HCP forcing 1 round.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TENACE_SKILLS_DIR", str(tmp_path))
    reg = default_skill_registry()
    assert "fantunes" in {m.id for m in reg.list_skills()}
    assert reg.load("fantunes").meta.name == "Fantunes"


def test_non_sayc_skill_prompts_have_zero_sayc_contradiction_and_progressive_disclosure() -> None:
    precision_card = load_bidding_skill("precision").to_convention_card()
    ccba_card = load_bidding_skill("ccba").to_convention_card()

    # Layer 1 Strategist: includes detailed Reference Guide during AUCTION, omits it during PLAY
    strat_client_auc = _RecordingClient(StrategyState(phase="auction", plan_summary="Bid 1C"))
    strategist_prec = StrategistEngine(strat_client_auc, convention_card=precision_card)
    strategist_prec.update_strategy(_sample_auction_facts(phase="auction"), None, "init")
    sys_auc = strat_client_auc.calls[0]["system_prompt"]
    assert "Precision Club" in sys_auc
    assert "Upper Limit Advantage" in sys_auc  # detailed_rules loaded in auction
    assert "5C=0/4" not in sys_auc  # zero SAYC Blackwood leakage!

    strat_client_play = _RecordingClient(StrategyState(phase="play", plan_summary="Lead D:4"))
    StrategistEngine(strat_client_play, convention_card=precision_card).update_strategy(
        _sample_auction_facts(phase="play"), None, "play"
    )
    assert "Upper Limit Advantage" not in strat_client_play.calls[0]["system_prompt"]

    # Layer 2 Evaluator: carries compact summary_rules ONLY (no detailed_rules bloat, no straight-Blackwood clash!)
    eval_client = _RecordingClient(TacticalDecision(option_scores={"Pass": 0.2, "4NT": 0.9}))
    evaluator_ccba = TacticalOptionEvaluator(eval_client, convention_card=ccba_card)
    partner_acted = _sample_auction_facts(
        auction_calls=[(Seat.NORTH, FIRST_CONTRACT_BID_ACTION + 2)]  # 1H
    )
    evaluator_ccba.evaluate(
        partner_acted,
        StrategyState(phase="auction", plan_summary="Bid 4NT"),
        [
            CandidateOption(action=PASS_ACTION, label="Pass"),
            CandidateOption(action=FIRST_CONTRACT_BID_ACTION + 19, label="4NT"),
        ],
    )
    l2_prompt = eval_client.calls[0]["user_prompt"]
    assert "CCBA Natural" in l2_prompt
    assert "RKCB 1430" in l2_prompt
    assert "5C=0/4" not in l2_prompt  # must NOT inject straight Blackwood into RKCB 1430 card!
    assert "5S=3" not in l2_prompt
    assert "Key Differences vs. SAYC" not in l2_prompt  # detailed_rules stay out of L2


def test_server_skills_endpoints_and_dynamic_system_selection(tmp_path: Path) -> None:
    from tenace.server import create_app

    ext = tmp_path / "mini_club"
    ext.mkdir()
    (ext / "SKILL.md").write_text(
        "---\nid: mini_club\nname: Mini Club\nversion: '1.2'\ndescription: Simple mini club.\n---\n\n"
        "## Summary Rules\nOpen 1C with 17+ HCP.\n",
        encoding="utf-8",
    )
    reg = SkillRegistry(extra_dirs=[tmp_path])
    app = create_app(
        default_profile=ModelProfile(mode="mock"),
        default_dds_worlds=1,
        skill_registry=reg,
        default_system="ccba",
    )
    client = TestClient(app)

    # 1. GET /v1/skills returns lightweight progressive metadata
    resp_list = client.get("/v1/skills")
    assert resp_list.status_code == 200
    payload = resp_list.json()
    assert payload["default_system"] == "ccba"
    listed_ids = {s["id"] for s in payload["skills"]}
    assert {"sayc", "21gf", "ccba", "precision", "acol", "polish_club", "mini_club"}.issubset(listed_ids)
    # Full markdown body not duplicated in listing
    assert "detailed_rules" not in payload["skills"][0]

    # 2. GET /v1/skills/{id} returns full parsed skill (summary + detailed rules)
    resp_detail = client.get("/v1/skills/precision")
    assert resp_detail.status_code == 200
    detail = resp_detail.json()
    assert detail["id"] == "precision"
    assert "16+ HCP" in detail["summary_rules"]
    assert detail["detailed_rules"]

    # 3. Unknown / traversal skill ID returns clean 404/422 without leaking disk paths
    assert client.get("/v1/skills/does_not_exist").status_code == 404
    assert client.get("/v1/skills/bad.slug").status_code == 422

    # 4. POST /v1/step accepts bidding_system OR inline convention_card (not both!)
    r_step = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "tournament_seed": 42,
            "bidding_system": "precision",
            "dds_worlds": 1,
        },
    )
    assert r_step.status_code == 200

    r_custom = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "tournament_seed": 42,
            "convention_card": {
                "id": "inline_test",
                "name": "Inline Custom",
                "summary_rules": "1C = 18+ forcing artificial.",
            },
            "dds_worlds": 1,
        },
    )
    assert r_custom.status_code == 200

    r_both = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "tournament_seed": 42,
            "bidding_system": "sayc",
            "convention_card": {"name": "Both", "summary_rules": "Oops"},
        },
    )
    assert r_both.status_code == 422

    # Reject unknown skill ID gracefully
    r_bad = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "tournament_seed": 42,
            "bidding_system": "../precision",
        },
    )
    assert r_bad.status_code == 422

    r_missing = client.post(
        "/v1/step",
        json={
            "seat": "NORTH",
            "board_number": 1,
            "tournament_seed": 42,
            "bidding_system": "unknown_system_xyz",
        },
    )
    assert r_missing.status_code == 404

    # 5. POST /v1/match supports system vs system (e.g. ccba vs precision!)
    r_match = client.post(
        "/v1/match",
        json={
            "boards": 1,
            "seed": 42,
            "dds_worlds": 1,
            "double_dummy": True,
            "opponent": "tenace",
            "team_a_system": "ccba",
            "team_b_system": "precision",
        },
    )
    assert r_match.status_code == 200
    assert "CCBA" in r_match.json()["report"]
    assert "Precision" in r_match.json()["report"]

    # Reject team_b_system when opponent='baseline'
    r_bad_match = client.post(
        "/v1/match",
        json={
            "boards": 1,
            "opponent": "baseline",
            "team_b_system": "precision",
        },
    )
    assert r_bad_match.status_code == 422


def test_cli_list_systems_and_custom_system_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from tenace.cli import run_cli

    code = run_cli(["--list-systems"])
    assert code == 0
    out = capsys.readouterr().out
    for sid in ("sayc", "21gf", "ccba", "precision", "acol", "polish_club"):
        assert sid in out

    # Direct local SKILL.md path via --system
    custom_file = tmp_path / "SKILL.md"
    custom_file.write_text(
        "---\nid: blue_club\nname: Blue Team Club\ndescription: Historic Italian strong club.\n---\n\n"
        "## Summary Rules\n1C = 17+ forcing, canape majors.\n",
        encoding="utf-8",
    )
    code_match = run_cli(
        [
            "--mode",
            "mock",
            "--boards",
            "1",
            "--dds-worlds",
            "1",
            "--double-dummy",
            "--opponent",
            "tenace",
            "--system",
            str(custom_file),
            "--opponent-system",
            "ccba",
        ]
    )
    assert code_match == 0
    report = capsys.readouterr().out
    assert "Blue Team Club" in report
    assert "CCBA Natural" in report

    # --skills-dir surfaces custom skills in --list-systems
    ext_root = tmp_path / "ext_dir"
    (ext_root / "blue_club").mkdir(parents=True)
    (ext_root / "blue_club" / "SKILL.md").write_text(
        custom_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert run_cli(["--skills-dir", str(ext_root), "--list-systems"]) == 0
    list_out = capsys.readouterr().out
    assert "blue_club" in list_out
    assert "external" in list_out

    # --opponent-system must be rejected unless --opponent tenace
    with pytest.raises(ValueError, match="--opponent-system"):
        run_cli(["--mode", "mock", "--opponent", "baseline", "--opponent-system", "ccba"])
