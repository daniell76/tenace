from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import (
    _api_client,
    decisions as genai_decisions,
    models as genai_models,
    types as genai_types,
)
from pydantic import BaseModel, ValidationError

import tenace.models as models
from tenace.domain.types import (
    CandidateOption,
    Seat,
    StrategyState,
    Suit,
    TacticalDecision,
)
from tenace.models import (
    DiffusionGemmaClient,
    DiffusionSamplingConfig,
    GeminiDecisionsClient,
    StructuredModelClient,
    VertexGeminiClient,
    extract_json_object,
)
from tenace.models.clients import __all__ as clients_all


def test_models_package_reexports():
    expected = {
        "StructuredModelClient",
        "VertexGeminiClient",
        "GeminiDecisionsClient",
        "DiffusionSamplingConfig",
        "DiffusionGemmaClient",
        "extract_json_object",
    }
    assert expected.issubset(set(models.__all__))
    assert expected.issubset(set(clients_all))
    for name in expected:
        assert getattr(models, name) is not None


def test_extract_json_object_handles_markdown_fences_and_thinking_prefix():
    raw = (
        "<thought>\nLet me analyze trick 4 {carefully}...\n</thought>\n"
        "Here is the tactical evaluation:\n"
        "```json\n"
        '{\n  "assumption_violated": true,\n'
        '  "violation_reason": "East showed out of hearts {unexpected}",\n'
        '  "option_scores": {"S:A": 0.85, "H:K": 0.2},\n'
        '  "rationale": "Cash spade winner first"\n}\n'
        "```\n"
        "Hope this helps {player}!"
    )
    decision = extract_json_object(raw, TacticalDecision)
    assert decision.assumption_violated is True
    assert decision.violation_reason == "East showed out of hearts {unexpected}"
    assert decision.option_scores == {"S:A": 0.85, "H:K": 0.2}
    assert decision.rationale == "Cash spade winner first"


def test_extract_json_object_handles_unfenced_json_with_trailing_braces_in_prose():
    raw = (
        "Analysis complete for {Pass, 1NT}:\n"
        '{"phase": "play", "plan_summary": "Draw trumps {3 rounds}", '
        '"assumptions": [{"claim": "Spades 3-2", "confidence": 0.68}]}'
        "\nNote: fallback set is {Pass, 1NT} if revised."
    )
    state = extract_json_object(raw, StrategyState)
    assert state.phase == "play"
    assert state.plan_summary == "Draw trumps {3 rounds}"
    assert len(state.assumptions) == 1
    assert state.assumptions[0].claim == "Spades 3-2"


def test_extract_json_object_prefers_json_fence_over_earlier_code_fence():
    raw = (
        "```text\n{not valid json}\n```\n"
        '```json\n{"assumption_violated": true, "rationale": "pick second fence"}\n```'
    )
    decision = extract_json_object(raw, TacticalDecision)
    assert decision.assumption_violated is True
    assert decision.rationale == "pick second fence"


def test_extract_json_object_skips_earlier_non_matching_json_dicts():
    class StrictActionScore(BaseModel):
        chosen_action: str
        score: float

    raw = (
        'Previous context: {"foo": 1, "bar": "illustrative"}\n'
        'Final answer: {"chosen_action": "S:A", "score": 0.92}\n'
    )
    parsed = extract_json_object(raw, StrictActionScore)
    assert parsed.chosen_action == "S:A"
    assert parsed.score == pytest.approx(0.92)

    with pytest.raises(ValidationError):
        extract_json_object('First: {"foo": 1}\nSecond: {"bar": 2}', StrictActionScore)


def test_extract_json_object_strategy_state_numeric_and_named_enum_keys():
    numeric_raw = json.dumps(
        {
            "phase": "auction",
            "plan_summary": "Numeric enum keys",
            "seat_constraints": {
                "1": {
                    "seat": 1,
                    "min_hcp": 12,
                    "max_hcp": 16,
                    "min_suit_lengths": {"3": 5, "0": 2},
                }
            },
        }
    )
    numeric_state = extract_json_object(numeric_raw, StrategyState)
    east_num = numeric_state.seat_constraints[Seat.EAST]
    assert east_num.seat == Seat.EAST
    assert east_num.min_suit_lengths[Suit.SPADES] == 5
    assert east_num.min_suit_lengths[Suit.CLUBS] == 2

    named_raw = json.dumps(
        {
            "phase": "play",
            "plan_summary": "Named enum keys",
            "seat_constraints": {
                "EAST": {
                    "seat": "east",
                    "min_hcp": 10,
                    "max_hcp": 15,
                    "min_suit_lengths": {"SPADES": 5, "c": 3},
                    "max_suit_lengths": {"Hearts": 2, "D": 4},
                },
                "w": {
                    "seat": "W",
                    "min_suit_lengths": {"h": 4},
                },
                "North": {
                    "seat": "N",
                },
                "s": {
                    "seat": "south",
                },
            },
        }
    )
    named_state = extract_json_object(named_raw, StrategyState)
    east_named = named_state.seat_constraints[Seat.EAST]
    assert east_named.seat == Seat.EAST
    assert east_named.min_suit_lengths == {Suit.SPADES: 5, Suit.CLUBS: 3}
    assert east_named.max_suit_lengths == {Suit.HEARTS: 2, Suit.DIAMONDS: 4}
    assert named_state.seat_constraints[Seat.WEST].min_suit_lengths == {Suit.HEARTS: 4}
    assert named_state.seat_constraints[Seat.NORTH].seat == Seat.NORTH
    assert named_state.seat_constraints[Seat.SOUTH].seat == Seat.SOUTH

    # Omitted inner "seat" field is inferred from outer dict key; mismatched key/seat raises
    inferred_seat = extract_json_object(
        json.dumps(
            {
                "phase": "auction",
                "plan_summary": "Inferred seat",
                "seat_constraints": {"WEST": {"min_hcp": 8}},
            }
        ),
        StrategyState,
    )
    assert inferred_seat.seat_constraints[Seat.WEST].seat == Seat.WEST
    assert inferred_seat.seat_constraints[Seat.WEST].min_hcp == 8

    with pytest.raises(ValidationError):
        extract_json_object(
            json.dumps(
                {
                    "phase": "auction",
                    "plan_summary": "Mismatched seat",
                    "seat_constraints": {"EAST": {"seat": "WEST", "min_hcp": 10}},
                }
            ),
            StrategyState,
        )


def test_extract_json_object_error_cases():
    with pytest.raises(ValueError, match="Empty"):
        extract_json_object("", TacticalDecision)

    with pytest.raises(ValueError, match="Empty"):
        extract_json_object("   \n\t  ", TacticalDecision)

    with pytest.raises(ValueError, match="Empty"):
        extract_json_object(None, TacticalDecision)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        extract_json_object("<thought>only thinking</thought>   ", TacticalDecision)

    with pytest.raises(ValueError):
        extract_json_object("No JSON object anywhere here", TacticalDecision)

    with pytest.raises(ValueError):
        extract_json_object('```json\n{"assumption_violated": tru}\n```', TacticalDecision)

    with pytest.raises(ValidationError):
        extract_json_object('{"phase": "invalid_phase", "plan_summary": "x"}', StrategyState)


def test_vertex_gemini_client_parses_structured_response():
    captured: dict[str, Any] = {}

    class FakeModels:
        def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            payload = {
                "phase": "auction",
                "plan_summary": "Investigate 4-4 heart fit",
                "target_or_threat": "4H",
                "assumptions": [{"claim": "Partner has 12+ HCP", "confidence": 0.75}],
                "revision": 1,
            }
            return SimpleNamespace(text=json.dumps(payload))

    fake_genai = SimpleNamespace(models=FakeModels())
    client = VertexGeminiClient(
        temperature=0.15,
        genai_client=fake_genai,
    )
    assert isinstance(client, StructuredModelClient)
    assert client.model == "gemini-4-argon"

    result = client.generate_structured(
        system_prompt="You are a Bridge strategist.",
        user_prompt="North opened 1C, East passed.",
        schema=StrategyState,
    )

    assert result.phase == "auction"
    assert result.plan_summary == "Investigate 4-4 heart fit"
    assert result.revision == 1
    assert captured["model"] == "gemini-4-argon"
    assert captured["contents"].startswith("North opened 1C, East passed.")
    assert "Respond ONLY with a single valid JSON object matching this JSON Schema:" in captured["contents"]
    assert captured["config"].system_instruction == "You are a Bridge strategist."
    assert captured["config"].temperature == pytest.approx(0.15)
    assert captured["config"].response_mime_type == "application/json"
    assert captured["config"].response_schema is None
    assert captured["config"].response_json_schema is None

    # Regression test: real google-genai Vertex serializer must accept prompt-schema + JSON mime
    # without triggering Vertex constrained-decoding open-dict failures on StrategyState/TacticalDecision.
    api_client = _api_client.BaseApiClient(
        vertexai=True,
        project="test-bridge-proj",
        location="global",
    )
    params = genai_types._GenerateContentParameters(
        model=captured["model"],
        contents=captured["contents"],
        config=captured["config"],
    )
    vertex_payload = genai_models._GenerateContentParameters_to_vertex(api_client, params)
    assert vertex_payload["generationConfig"]["responseMimeType"] == "application/json"
    assert "responseJsonSchema" not in vertex_payload["generationConfig"]

    # Layer-2 thinking configuration (thinking_level="LOW" on gemini-3.8-flash / gemini-4-argon)
    low_client = VertexGeminiClient(
        model="gemini-3.8-flash",
        thinking_level="LOW",
        genai_client=fake_genai,
    )
    low_client.generate_structured("sys", "usr", StrategyState)
    assert captured["config"].thinking_config is not None
    assert str(captured["config"].thinking_config.thinking_level).endswith("LOW")

    zero_budget_client = VertexGeminiClient(
        model="gemini-4-argon",
        thinking_budget=0,
        genai_client=fake_genai,
    )
    zero_budget_client.generate_structured("sys", "usr", StrategyState)
    assert captured["config"].thinking_config is not None
    assert captured["config"].thinking_config.thinking_budget == 0

    with pytest.raises(ValueError, match="thinking"):
        VertexGeminiClient(
            thinking_level="LOW",
            thinking_budget=0,
            genai_client=fake_genai,
        )


def test_vertex_gemini_client_validation_and_empty_response():
    fake_genai = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: SimpleNamespace(text=None))
    )
    with pytest.raises(ValueError, match="temperature"):
        VertexGeminiClient(temperature=-0.1, genai_client=fake_genai)

    with pytest.raises(ValueError, match="temperature"):
        VertexGeminiClient(temperature=2.5, genai_client=fake_genai)

    client = VertexGeminiClient(genai_client=fake_genai)
    with pytest.raises(ValueError, match="Empty response from Vertex Gemini model"):
        client.generate_structured("sys", "user", TacticalDecision)

    fake_blank = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: SimpleNamespace(text="   "))
    )
    client_blank = VertexGeminiClient(genai_client=fake_blank)
    with pytest.raises(ValueError, match="Empty response from Vertex Gemini model"):
        client_blank.generate_structured("sys", "user", TacticalDecision)


def test_diffusion_sampling_config_bounds():
    cfg = DiffusionSamplingConfig()
    assert cfg.max_denoising_steps == 48
    assert cfg.max_tokens == 256
    assert cfg.temperature == pytest.approx(0.4)
    assert cfg.entropy_threshold == pytest.approx(0.005)
    assert cfg.entropy_bound == pytest.approx(0.1)

    with pytest.raises(ValidationError):
        DiffusionSamplingConfig(max_denoising_steps=0)
    with pytest.raises(ValidationError):
        DiffusionSamplingConfig(max_denoising_steps=513)
    with pytest.raises(ValidationError):
        DiffusionSamplingConfig(max_tokens=0)
    with pytest.raises(ValidationError):
        DiffusionSamplingConfig(max_tokens=8193)
    with pytest.raises(ValidationError):
        DiffusionSamplingConfig(temperature=-0.01)
    with pytest.raises(ValidationError):
        DiffusionSamplingConfig(temperature=2.01)
    with pytest.raises(ValidationError):
        DiffusionSamplingConfig(entropy_threshold=-0.01)
    with pytest.raises(ValidationError):
        DiffusionSamplingConfig(entropy_threshold=1.01)
    with pytest.raises(ValidationError):
        DiffusionSamplingConfig(entropy_bound=-0.01)
    with pytest.raises(ValidationError):
        DiffusionSamplingConfig(entropy_bound=1.01)


def test_diffusion_gemma_client_sends_denoising_parameters_to_endpoint():
    captured: dict[str, Any] = {}

    class FakeEndpoint:
        def predict(self, *, instances: list[dict[str, Any]], parameters: dict[str, Any]) -> Any:
            captured["instances"] = instances
            captured["parameters"] = parameters
            return SimpleNamespace(
                predictions=[
                    {
                        "output": (
                            "<thought>Denoising tactical paths...</thought>\n"
                            '{"assumption_violated": false, '
                            '"option_scores": {"C:2": 0.4, "S:A": 0.9}, '
                            '"rationale": "Lead top spade"}'
                        )
                    }
                ]
            )

    sampling = DiffusionSamplingConfig(
        max_denoising_steps=32,
        max_tokens=384,
        temperature=0.35,
        entropy_threshold=0.01,
        entropy_bound=0.12,
    )
    client = DiffusionGemmaClient(endpoint=FakeEndpoint(), sampling=sampling)
    assert isinstance(client, StructuredModelClient)

    decision = client.generate_structured(
        system_prompt="You are Tenace Layer 2 tactician.",
        user_prompt="Evaluate options C:2 and S:A.",
        schema=TacticalDecision,
    )

    assert decision.assumption_violated is False
    assert decision.option_scores == {"C:2": 0.4, "S:A": 0.9}
    assert decision.rationale == "Lead top spade"

    assert len(captured["instances"]) == 1
    prompt_sent = captured["instances"][0]["prompt"]
    assert "You are Tenace Layer 2 tactician." in prompt_sent
    assert "Evaluate options C:2 and S:A." in prompt_sent
    assert "assumption_violated" in prompt_sent
    assert captured["parameters"] == {
        "max_tokens": 384,
        "temperature": 0.35,
        "max_denoising_steps": 32,
        "entropy_threshold": 0.01,
        "entropy_bound": 0.12,
    }


def test_diffusion_gemma_endpoint_text_string_and_empty_predictions():
    class TextKeyEndpoint:
        def predict(self, **_: Any) -> Any:
            return SimpleNamespace(
                predictions=[{"text": '{"assumption_violated": true, "violation_reason": "void"}'}]
            )

    client_text = DiffusionGemmaClient(endpoint=TextKeyEndpoint())
    res_text = client_text.generate_structured("sys", "usr", TacticalDecision)
    assert res_text.assumption_violated is True
    assert res_text.violation_reason == "void"

    class RawStrEndpoint:
        def predict(self, **_: Any) -> Any:
            return SimpleNamespace(predictions=['{"assumption_violated": false}'])

    client_str = DiffusionGemmaClient(endpoint=RawStrEndpoint())
    res_str = client_str.generate_structured("sys", "usr", TacticalDecision)
    assert res_str.assumption_violated is False

    class EmptyEndpoint:
        def predict(self, **_: Any) -> Any:
            return SimpleNamespace(predictions=[])

    client_empty = DiffusionGemmaClient(endpoint=EmptyEndpoint())
    with pytest.raises(ValueError, match="Empty"):
        client_empty.generate_structured("sys", "usr", TacticalDecision)


def test_diffusion_gemma_maas_fallback_and_errors():
    captured: dict[str, Any] = {}

    class FakeMaaSModels:
        def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return SimpleNamespace(
                text='```json\n{"assumption_violated": false, "option_scores": {"Pass": 0.7}}\n```'
            )

    fake_maas = SimpleNamespace(models=FakeMaaSModels())
    client = DiffusionGemmaClient(genai_maas_client=fake_maas)
    decision = client.generate_structured("System rules", "User query", TacticalDecision)

    assert decision.option_scores == {"Pass": 0.7}
    assert captured["model"] == "publishers/google/models/diffusiongemma-26b-a4b-it"
    assert "User query" in captured["contents"]
    assert "assumption_violated" in captured["contents"]
    assert captured["config"].response_mime_type == "application/json"

    fake_empty_maas = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: SimpleNamespace(text=""))
    )
    client_empty = DiffusionGemmaClient(genai_maas_client=fake_empty_maas)
    with pytest.raises(ValueError, match="Empty"):
        client_empty.generate_structured("sys", "usr", TacticalDecision)


def test_diffusion_gemma_rejects_ambiguous_client_configuration(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError):
        DiffusionGemmaClient(endpoint=object(), genai_maas_client=object())

    with pytest.raises(ValueError):
        DiffusionGemmaClient(endpoint_id="12345", genai_maas_client=object())

    with pytest.raises(ValueError):
        DiffusionGemmaClient(endpoint=object(), endpoint_id="12345")

    with pytest.raises(ValueError):
        DiffusionGemmaClient(endpoint_id="   ")

    with pytest.raises(ValueError):
        VertexGeminiClient(temperature=True, genai_client=object())  # type: ignore[arg-type]

    created_endpoints: list[dict[str, Any]] = []

    def fake_endpoint_ctor(*, endpoint_name: str, project: str | None, location: str) -> Any:
        created_endpoints.append(
            {"endpoint_name": endpoint_name, "project": project, "location": location}
        )
        return SimpleNamespace(
            predict=lambda **_: SimpleNamespace(
                predictions=[{"generated_text": '{"assumption_violated": false}'}]
            )
        )

    monkeypatch.setattr("tenace.models.clients.aiplatform.Endpoint", fake_endpoint_ctor)
    ep_client = DiffusionGemmaClient(
        project="bridge-proj",
        location="us-central1",
        endpoint_id="987654321",
    )
    res = ep_client.generate_structured("sys", "usr", TacticalDecision)
    assert res.assumption_violated is False
    assert created_endpoints == [
        {"endpoint_name": "987654321", "project": "bridge-proj", "location": "us-central1"}
    ]


def test_gemini_decisions_client_evaluates_options_with_real_sdk_types_and_normalizes_scores():
    calls: list[dict[str, Any]] = []

    def fake_create(*, model: str, input: Any, questions: dict[str, Any]) -> Any:
        # Validate outgoing payload against real google.genai.decisions Pydantic request schema
        validated_req = genai_decisions.CreateDecisionRequest.model_validate(
            {"model": model, "input": input, "questions": questions}
        )
        calls.append(
            {
                "model": model,
                "input": input,
                "questions": questions,
                "validated_req": validated_req,
            }
        )
        answers: dict[str, Any] = {
            "cand_0": genai_decisions.ScoreAnswer(
                score=3.6,
                confidence=0.82,
                legend={"0": "bad", "4": "best"},
                probabilities={"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.4, "4": 0.6},
            ),
            "cand_1": genai_decisions.ScoreAnswer(
                score=4.2,  # Clamped to 1.0 if slightly above 4.0
                confidence=0.91,
                legend={"0": "bad", "4": "best"},
                probabilities={"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0},
            ),
            "cand_2": genai_decisions.ScoreAnswer(
                score=1.0,
                confidence=0.70,
                legend={"0": "bad", "4": "best"},
                probabilities={"0": 0.2, "1": 0.6, "2": 0.2, "3": 0.0, "4": 0.0},
            ),
        }
        if "assumption_violated" in questions:
            answers["assumption_violated"] = genai_decisions.NoulAnswer(
                noul=0.18,
                confidence=0.82,
            )
        return genai_decisions.CreateDecisionResponse(answers=answers)

    fake_genai = SimpleNamespace(decisions=SimpleNamespace(create=fake_create))
    client = GeminiDecisionsClient(genai_client=fake_genai)
    assert client.model == "gemini-decision-1"
    assert client.violation_threshold == pytest.approx(0.70)
    assert isinstance(client, StructuredModelClient)

    candidates = [
        CandidateOption(action=51, label="S:A", baseline_ev=420.0, make_prob=0.85),
        CandidateOption(action=59, label="1NT", baseline_ev=120.0, make_prob=0.62),
        CandidateOption(action=20, label="C:7", baseline_ev=-50.0, make_prob=None),
    ]

    # Case 1: has_assumptions=True -> includes "assumption_violated" NoulQuestion
    res_with_assumptions = client.evaluate_options(
        context_text="=== Board Facts ===\nObserver: SOUTH",
        candidates=candidates,
        has_assumptions=True,
    )
    assert len(calls) == 1
    q_map = calls[0]["questions"]
    assert set(q_map.keys()) == {"cand_0", "cand_1", "cand_2", "assumption_violated"}
    for q_id in ("cand_0", "cand_1", "cand_2"):
        assert q_map[q_id]["type"] == "score"
        assert len(q_map[q_id]["criteria"]) == 5
    assert "S:A" in q_map["cand_0"]["instructions"]
    assert "baseline_ev=420.0" in q_map["cand_0"]["instructions"]
    assert "make_prob=0.85" in q_map["cand_0"]["instructions"]
    assert "make_prob=N/A" in q_map["cand_2"]["instructions"]
    assert q_map["assumption_violated"]["type"] == "noul"
    assert "true_description" in q_map["assumption_violated"]["criteria"]
    assert "false_description" in q_map["assumption_violated"]["criteria"]

    assert res_with_assumptions.assumption_violated is False
    assert res_with_assumptions.violation_reason == ""
    assert res_with_assumptions.option_scores == pytest.approx(
        {"S:A": 0.90, "1NT": 1.0, "C:7": 0.25}
    )
    assert "1NT" in res_with_assumptions.rationale
    assert "0.18" in res_with_assumptions.rationale

    # Case 2: has_assumptions=False -> omits "assumption_violated"
    res_no_assumptions = client.evaluate_options(
        context_text="=== Board Facts ===\nObserver: SOUTH",
        candidates=candidates[:2],
        has_assumptions=False,
    )
    assert set(calls[1]["questions"].keys()) == {"cand_0", "cand_1"}
    assert res_no_assumptions.assumption_violated is False
    assert res_no_assumptions.violation_reason == ""
    assert res_no_assumptions.option_scores == pytest.approx({"S:A": 0.90, "1NT": 1.0})


def test_gemini_decisions_client_triggers_violation_interrupt_and_accepts_dict_answers():
    def fake_create(*, model: str, input: Any, questions: dict[str, Any]) -> Any:
        return SimpleNamespace(
            answers={
                "cand_0": {"type": "score", "score": 3.2, "confidence": 0.88},
                "cand_1": {"type": "score", "score": -0.2, "confidence": 0.95},
                "assumption_violated": {"type": "noul", "noul": 0.78, "confidence": 0.78},
            }
        )

    client = GeminiDecisionsClient(
        violation_threshold=0.70,
        genai_client=SimpleNamespace(decisions=SimpleNamespace(create=fake_create)),
    )
    cands = [
        CandidateOption(action=51, label="S:A", baseline_ev=420.0, make_prob=0.80),
        CandidateOption(action=0, label="C:2", baseline_ev=-100.0, make_prob=0.20),
    ]
    decision = client.evaluate_options("Board context", cands, has_assumptions=True)
    assert decision.assumption_violated is True
    assert "0.78" in decision.violation_reason
    assert "0.70" in decision.violation_reason
    assert decision.option_scores == pytest.approx({"S:A": 0.80, "C:2": 0.0})
    assert "S:A" in decision.rationale
    assert "0.78" in decision.rationale

    # Higher threshold (0.85) does not trigger violation for p=0.78
    high_thresh_client = GeminiDecisionsClient(
        violation_threshold=0.85,
        genai_client=SimpleNamespace(decisions=SimpleNamespace(create=fake_create)),
    )
    decision_below = high_thresh_client.evaluate_options(
        "Board context", cands, has_assumptions=True
    )
    assert decision_below.assumption_violated is False
    assert decision_below.violation_reason == ""


def test_gemini_decisions_client_satisfies_structured_model_client_protocol():
    captured_calls: list[dict[str, Any]] = []

    def fake_create(*, model: str, input: Any, questions: dict[str, Any]) -> Any:
        captured_calls.append({"model": model, "input": input, "questions": questions})
        ans: dict[str, Any] = {
            "cand_0": {"score": 3.6, "confidence": 0.85},
            "cand_1": {"score": 1.2, "confidence": 0.75},
        }
        if "assumption_violated" in questions:
            ans["assumption_violated"] = {"noul": 0.12, "confidence": 0.88}
        return SimpleNamespace(answers=ans)

    client = GeminiDecisionsClient(
        genai_client=SimpleNamespace(decisions=SimpleNamespace(create=fake_create))
    )
    assert isinstance(client, StructuredModelClient)

    # Unsupported open-ended schema -> clear ValueError
    with pytest.raises(ValueError, match="closed-set TacticalDecision"):
        client.generate_structured("sys", "usr", StrategyState)

    # Prompt with no candidate lines -> ValueError
    with pytest.raises(ValueError, match="candidate"):
        client.generate_structured("sys", "=== Board Facts ===", TacticalDecision)

    # Prompt with assumptions present
    user_prompt_with_assumptions = "\n".join(
        [
            "=== Board Facts ===",
            "Observer: SOUTH | Phase: play",
            "=== Active Strategy ===",
            "Assumptions: Spades split 3-2 (conf=0.68)",
            "=== Candidate Options ===",
            "- S:A: baseline_ev=420.0, make_prob=0.81",
            "- C:7: baseline_ev=-100.0, make_prob=N/A",
        ]
    )
    res1 = client.generate_structured(
        "You are Tenace Layer 2 Tactical Option Evaluator.",
        user_prompt_with_assumptions,
        TacticalDecision,
    )
    assert res1.option_scores == pytest.approx({"S:A": 0.90, "C:7": 0.30})
    assert "assumption_violated" in captured_calls[0]["questions"]
    assert "You are Tenace Layer 2 Tactical Option Evaluator." in captured_calls[0]["input"]

    # Prompt with Assumptions: (none) -> omits assumption_violated
    user_prompt_no_assumptions = user_prompt_with_assumptions.replace(
        "Spades split 3-2 (conf=0.68)", "(none)"
    )
    res2 = client.generate_structured("Sys", user_prompt_no_assumptions, TacticalDecision)
    assert res2.option_scores == pytest.approx({"S:A": 0.90, "C:7": 0.30})
    assert "assumption_violated" not in captured_calls[1]["questions"]

    # Substring "Assumptions: (none)" inside another field must NOT suppress assumption_violated
    user_prompt_substring_trap = "\n".join(
        [
            "=== Board Facts ===",
            "Observer: SOUTH | Phase: play",
            "=== Active Strategy ===",
            "Plan Summary: Avoid line where Assumptions: (none) would hold",
            "Assumptions: Spades split 3-2 (conf=0.68)",
            "=== Candidate Options ===",
            "- S:A: baseline_ev=420.0, make_prob=0.81",
            "- C:7: baseline_ev=-100.0, make_prob=N/A",
        ]
    )
    client.generate_structured("Sys", user_prompt_substring_trap, TacticalDecision)
    assert "assumption_violated" in captured_calls[2]["questions"]


def test_gemini_decisions_client_constructor_auth_modes_and_error_handling(
    monkeypatch: pytest.MonkeyPatch,
):
    dummy = SimpleNamespace(decisions=SimpleNamespace(create=lambda **_: None))

    for bad_model in ("", "   ", None, 123):
        with pytest.raises(ValueError, match="model"):
            GeminiDecisionsClient(model=bad_model, genai_client=dummy)  # type: ignore[arg-type]

    for bad_thresh in (0.0, 1.0, -0.2, 1.5, math.nan, math.inf, -math.inf, True, "0.7"):
        with pytest.raises(ValueError, match="violation_threshold"):
            GeminiDecisionsClient(
                violation_threshold=bad_thresh,  # type: ignore[arg-type]
                genai_client=dummy,
            )

    with pytest.raises(ValueError, match="vertexai"):
        GeminiDecisionsClient(vertexai="true", genai_client=dummy)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="vertexai=False"):
        GeminiDecisionsClient(vertexai=True, api_key="secret-key", genai_client=dummy)

    with pytest.raises(ValueError, match="vertexai=True"):
        GeminiDecisionsClient(vertexai=False, project="my-gcp-proj", genai_client=dummy)

    with pytest.raises(ValueError, match="api_key"):
        GeminiDecisionsClient(vertexai=False, api_key="   ", genai_client=dummy)

    # Regression test against real genai.Client when GOOGLE_GENAI_USE_VERTEXAI=true is set in env
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    real_dev_client = GeminiDecisionsClient(vertexai=False, api_key="test-key")
    assert real_dev_client._client._api_client.vertexai is False

    # Verify genai.Client construction for Vertex AI vs Gemini Developer API v1beta
    client_inits: list[dict[str, Any]] = []

    def fake_genai_client(**kwargs: Any) -> Any:
        client_inits.append(kwargs)
        return dummy

    monkeypatch.setattr("tenace.models.clients.genai.Client", fake_genai_client)
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")

    v_client = GeminiDecisionsClient(project="my-gcp-proj", location="us-central1")
    assert v_client.project == "my-gcp-proj"
    assert v_client.location == "us-central1"
    assert client_inits[-1] == {
        "vertexai": True,
        "project": "my-gcp-proj",
        "location": "us-central1",
    }

    GeminiDecisionsClient(vertexai=False)
    assert client_inits[-1] == {
        "vertexai": False,
        "api_key": "env-gemini-key",
        "http_options": {"api_version": "v1beta"},
    }

    GeminiDecisionsClient(vertexai=False, api_key="  explicit-key  ")
    assert client_inits[-1] == {
        "vertexai": False,
        "api_key": "explicit-key",
        "http_options": {"api_version": "v1beta"},
    }

    # Verify runtime error handling in evaluate_options
    cands = [CandidateOption(action=52, label="Pass", baseline_ev=0.0, make_prob=0.5)]
    with pytest.raises(ValueError, match="context_text"):
        v_client.evaluate_options("   ", cands)
    with pytest.raises(ValueError, match="candidates"):
        v_client.evaluate_options("Board context", [])

    empty_client = GeminiDecisionsClient(
        genai_client=SimpleNamespace(
            decisions=SimpleNamespace(create=lambda **_: SimpleNamespace(answers={}))
        )
    )
    with pytest.raises(ValueError, match="Empty answers"):
        empty_client.evaluate_options("Board context", cands)

    blocked_resp = genai_decisions.CreateDecisionResponse(
        answers=None,
        prompt_feedback=genai_decisions.PromptFeedback(
            block_reason="safety",
            block_reason_message="Prompt blocked due to safety policy",
        ),
    )
    blocked_client = GeminiDecisionsClient(
        genai_client=SimpleNamespace(
            decisions=SimpleNamespace(create=lambda **_: blocked_resp)
        )
    )
    with pytest.raises(
        ValueError,
        match="Gemini Decisions API blocked request: safety: Prompt blocked due to safety policy",
    ):
        blocked_client.evaluate_options("Board context", cands)

    missing_cand_client = GeminiDecisionsClient(
        genai_client=SimpleNamespace(
            decisions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(answers={"other": {"score": 2.0}})
            )
        )
    )
    with pytest.raises(ValueError, match="Missing decision answer for candidate Pass"):
        missing_cand_client.evaluate_options("Board context", cands, has_assumptions=False)

    for bad_score in (math.nan, True):
        bad_score_client = GeminiDecisionsClient(
            genai_client=SimpleNamespace(
                decisions=SimpleNamespace(
                    create=lambda _s=bad_score, **_: SimpleNamespace(
                        answers={"cand_0": {"score": _s}}
                    )
                )
            )
        )
        with pytest.raises(ValueError, match="score"):
            bad_score_client.evaluate_options("Board context", cands, has_assumptions=False)

    missing_viol_client = GeminiDecisionsClient(
        genai_client=SimpleNamespace(
            decisions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(answers={"cand_0": {"score": 2.0}})
            )
        )
    )
    with pytest.raises(ValueError, match="assumption_violated"):
        missing_viol_client.evaluate_options("Board context", cands, has_assumptions=True)

    for bad_noul in (True, -0.1, 1.5, math.nan, math.inf):
        bad_noul_client = GeminiDecisionsClient(
            genai_client=SimpleNamespace(
                decisions=SimpleNamespace(
                    create=lambda _n=bad_noul, **_: SimpleNamespace(
                        answers={
                            "cand_0": {"score": 2.0},
                            "assumption_violated": {"noul": _n},
                        }
                    )
                )
            )
        )
        with pytest.raises(ValueError, match="noul"):
            bad_noul_client.evaluate_options("Board context", cands, has_assumptions=True)


def test_model_clients_strip_and_reject_blank_strings_and_env_api_key_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    dummy_genai = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: None),
        decisions=SimpleNamespace(create=lambda **_: None),
    )

    # 1. VertexGeminiClient strips and rejects blank model/location/project
    vg = VertexGeminiClient(
        project="  proj-1  ",
        location="  global  ",
        model="  gemini-4-argon  ",
        genai_client=dummy_genai,
    )
    assert vg.project == "proj-1"
    assert vg.location == "global"
    assert vg.model == "gemini-4-argon"
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="model"):
            VertexGeminiClient(model=blank, genai_client=dummy_genai)
        with pytest.raises(ValueError, match="location"):
            VertexGeminiClient(location=blank, genai_client=dummy_genai)
        with pytest.raises(ValueError, match="project"):
            VertexGeminiClient(project=blank, genai_client=dummy_genai)

    # 2. DiffusionGemmaClient strips and rejects blank maas_model_name/location/project/endpoint_id
    dg = DiffusionGemmaClient(
        project="  proj-2  ",
        location="  us-central1  ",
        maas_model_name="  publishers/google/models/diffusiongemma-26b-a4b-it  ",
        genai_maas_client=dummy_genai,
    )
    assert dg.project == "proj-2"
    assert dg.location == "us-central1"
    assert dg.maas_model_name == "publishers/google/models/diffusiongemma-26b-a4b-it"
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="maas_model_name"):
            DiffusionGemmaClient(maas_model_name=blank, genai_maas_client=dummy_genai)
        with pytest.raises(ValueError, match="location"):
            DiffusionGemmaClient(location=blank, genai_maas_client=dummy_genai)
        with pytest.raises(ValueError, match="project"):
            DiffusionGemmaClient(project=blank, genai_maas_client=dummy_genai)
        with pytest.raises(ValueError, match="endpoint_id"):
            DiffusionGemmaClient(endpoint_id=blank)

    # 3. GeminiDecisionsClient strips and rejects blank location/project/api_key + env fallback
    gd = GeminiDecisionsClient(
        project="  proj-3  ",
        location="  global  ",
        model="  gemini-decision-1  ",
        genai_client=dummy_genai,
    )
    assert gd.project == "proj-3"
    assert gd.location == "global"
    assert gd.model == "gemini-decision-1"
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="location"):
            GeminiDecisionsClient(location=blank, genai_client=dummy_genai)
        with pytest.raises(ValueError, match="project"):
            GeminiDecisionsClient(project=blank, genai_client=dummy_genai)
        with pytest.raises(ValueError, match="api_key"):
            GeminiDecisionsClient(vertexai=False, api_key=blank, genai_client=dummy_genai)

    # Env API key whitespace fallback: GEMINI_API_KEY="   " falls back to stripped GOOGLE_API_KEY
    client_inits: list[dict[str, Any]] = []

    def spy_genai_client(**kwargs: Any) -> Any:
        client_inits.append(kwargs)
        return dummy_genai

    monkeypatch.setattr("tenace.models.clients.genai.Client", spy_genai_client)
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.setenv("GOOGLE_API_KEY", "  google-fallback-key  ")

    GeminiDecisionsClient(vertexai=False)
    assert client_inits[-1] == {
        "vertexai": False,
        "api_key": "google-fallback-key",
        "http_options": {"api_version": "v1beta"},
    }

    # Explicit blank api_key still raises even when valid env key exists
    with pytest.raises(ValueError, match="api_key"):
        GeminiDecisionsClient(vertexai=False, api_key="")
    with pytest.raises(ValueError, match="api_key"):
        GeminiDecisionsClient(vertexai=False, api_key="   ")

    # Missing / whitespace-only env keys with api_key=None raises ValueError
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.setenv("GOOGLE_API_KEY", "   ")
    with pytest.raises(ValueError, match="api_key|API key"):
        GeminiDecisionsClient(vertexai=False)

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key|API key"):
        GeminiDecisionsClient(vertexai=False)


def test_gemini_decisions_client_rejects_duplicate_candidate_labels_before_api_call():
    api_calls = 0

    def spy_create(**_: Any) -> Any:
        nonlocal api_calls
        api_calls += 1
        return SimpleNamespace(answers={})

    client = GeminiDecisionsClient(
        genai_client=SimpleNamespace(decisions=SimpleNamespace(create=spy_create))
    )
    dup_cands = [
        CandidateOption(action=52, label="Pass", baseline_ev=0.0, make_prob=0.5),
        CandidateOption(action=53, label="Pass", baseline_ev=-100.0, make_prob=0.3),
    ]
    with pytest.raises(ValueError, match="[Dd]uplicate candidate label"):
        client.evaluate_options("Board context", dup_cands)
    assert api_calls == 0

    dup_prompt = "\n".join(
        [
            "=== Board Facts ===",
            "Observer: NORTH | Phase: auction",
            "=== Candidate Options ===",
            "- 1NT: baseline_ev=120.0, make_prob=0.65",
            "- 1NT: baseline_ev=90.0, make_prob=0.55",
        ]
    )
    with pytest.raises(ValueError, match="[Dd]uplicate candidate label"):
        client.generate_structured("Sys", dup_prompt, TacticalDecision)
    assert api_calls == 0



