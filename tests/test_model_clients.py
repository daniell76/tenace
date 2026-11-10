from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import tenace.models as models
from tenace.domain.types import StrategyState, TacticalDecision
from tenace.models import (
    DiffusionGemmaClient,
    DiffusionSamplingConfig,
    StructuredModelClient,
    VertexGeminiClient,
    extract_json_object,
)
from tenace.models.clients import __all__ as clients_all


def test_models_package_reexports():
    expected = {
        "StructuredModelClient",
        "VertexGeminiClient",
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
        model="gemini-2.5-pro",
        temperature=0.15,
        genai_client=fake_genai,
    )
    assert isinstance(client, StructuredModelClient)

    result = client.generate_structured(
        system_prompt="You are a Bridge strategist.",
        user_prompt="North opened 1C, East passed.",
        schema=StrategyState,
    )

    assert result.phase == "auction"
    assert result.plan_summary == "Investigate 4-4 heart fit"
    assert result.revision == 1
    assert captured["model"] == "gemini-2.5-pro"
    assert captured["contents"] == "North opened 1C, East passed."
    assert captured["config"].system_instruction == "You are a Bridge strategist."
    assert captured["config"].temperature == pytest.approx(0.15)
    assert captured["config"].response_mime_type == "application/json"
    assert captured["config"].response_schema is StrategyState


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

