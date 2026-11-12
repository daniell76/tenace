from __future__ import annotations

from collections.abc import Iterator, Mapping
import json
import math
import os
import re
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from google import genai
from google.cloud import aiplatform
from google.genai import types as genai_types
from pydantic import BaseModel, Field, ValidationError

from tenace.domain.types import CandidateOption, TacticalDecision

T = TypeVar("T", bound=BaseModel)

_THOUGHT_BLOCK_RE = re.compile(r"<thought>.*?</thought>", flags=re.DOTALL | re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```json\b\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```", flags=re.DOTALL)
_CANDIDATE_LINE_RE = re.compile(
    r"^\s*-\s*(?P<label>.+?):\s*baseline_ev=(?P<ev>[-+]?\d+(?:\.\d+)?),"
    r"\s*make_prob=(?P<mp>N/A|[-+]?\d+(?:\.\d+)?)\s*$",
    flags=re.MULTILINE,
)

_DECISION_SCORE_CRITERIA: tuple[str, ...] = (
    "0 - Severe tactical blunder that sacrifices expected value or contract make probability.",
    "1 - Suboptimal play or bid with inferior risk-reward or misaligned with strategy.",
    "2 - Reasonable neutral alternative with acceptable expected value.",
    "3 - Strong tactical choice aligned with both DDS expected value and strategic plan.",
    "4 - Optimal tactical play or bid maximizing strategic execution and contract safety.",
)

__all__ = [
    "DiffusionGemmaClient",
    "DiffusionSamplingConfig",
    "GeminiDecisionsClient",
    "StructuredModelClient",
    "VertexGeminiClient",
    "extract_json_object",
]


@runtime_checkable
class StructuredModelClient(Protocol):
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T: ...


def _iter_json_object_slices(text: str) -> Iterator[str]:
    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        brace_idx = text.find("{", search_from)
        if brace_idx == -1:
            return
        try:
            obj, end_idx = decoder.raw_decode(text, brace_idx)
        except json.JSONDecodeError:
            search_from = brace_idx + 1
            continue
        if isinstance(obj, dict):
            yield text[brace_idx:end_idx]
            search_from = end_idx
        else:
            search_from = brace_idx + 1


def extract_json_object(raw_text: str, schema: type[T]) -> T:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Empty model response text")

    cleaned = _THOUGHT_BLOCK_RE.sub("", raw_text).strip()
    if not cleaned:
        raise ValueError("Empty model response after stripping <thought> blocks")

    candidates: list[str] = [m.group(1).strip() for m in _JSON_FENCE_RE.finditer(cleaned)]
    candidates.extend(m.group(1).strip() for m in _ANY_FENCE_RE.finditer(cleaned))
    candidates.append(cleaned)

    last_validation_error: ValidationError | None = None
    for candidate in candidates:
        for json_slice in _iter_json_object_slices(candidate):
            try:
                return schema.model_validate_json(json_slice)
            except ValidationError as exc:
                last_validation_error = exc

    if last_validation_error is not None:
        raise last_validation_error

    raise ValueError("No valid JSON object found in model response")


class VertexGeminiClient:
    def __init__(
        self,
        project: str | None = None,
        location: str = "global",
        model: str = "gemini-4-argon",
        temperature: float = 0.2,
        genai_client: Any | None = None,
    ) -> None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not (0.0 <= temperature <= 2.0)
        ):
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature!r}")

        self.project = project
        self.location = location
        self.model = model
        self.temperature = float(temperature)
        self._client = (
            genai_client
            if genai_client is not None
            else genai.Client(vertexai=True, project=project, location=location)
        )

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
            ),
        )
        text = getattr(resp, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty response from Vertex Gemini model")
        return extract_json_object(text, schema)


class DiffusionSamplingConfig(BaseModel):
    max_denoising_steps: int = Field(default=48, ge=1, le=512)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    entropy_threshold: float = Field(default=0.005, ge=0.0, le=1.0)
    entropy_bound: float = Field(default=0.1, ge=0.0, le=1.0)


def _format_schema_instruction(schema: type[BaseModel]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return (
        "Respond ONLY with a single valid JSON object matching this JSON Schema:\n"
        f"{schema_json}"
    )


class DiffusionGemmaClient:
    def __init__(
        self,
        project: str | None = None,
        location: str = "us-central1",
        endpoint_id: str | None = None,
        maas_model_name: str = "publishers/google/models/diffusiongemma-26b-a4b-it",
        sampling: DiffusionSamplingConfig | None = None,
        endpoint: Any | None = None,
        genai_maas_client: Any | None = None,
    ) -> None:
        if endpoint_id is not None and not endpoint_id.strip():
            raise ValueError("endpoint_id must not be blank when provided")
        if endpoint is not None and endpoint_id is not None:
            raise ValueError("Pass at most one of endpoint or endpoint_id")
        if genai_maas_client is not None and (endpoint is not None or endpoint_id is not None):
            raise ValueError(
                "Cannot configure both Vertex Endpoint (endpoint/endpoint_id) and genai_maas_client"
            )

        self.project = project
        self.location = location
        self.endpoint_id = endpoint_id.strip() if endpoint_id is not None else None
        self.maas_model_name = maas_model_name
        self.sampling = sampling if sampling is not None else DiffusionSamplingConfig()

        if endpoint is not None:
            self._endpoint: Any | None = endpoint
            self._genai_maas: Any | None = None
        elif self.endpoint_id is not None:
            self._endpoint = aiplatform.Endpoint(
                endpoint_name=self.endpoint_id,
                project=project,
                location=location,
            )
            self._genai_maas = None
        elif genai_maas_client is not None:
            self._endpoint = None
            self._genai_maas = genai_maas_client
        else:
            self._endpoint = None
            self._genai_maas = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        schema_instruction = _format_schema_instruction(schema)

        if self._endpoint is not None:
            combined_prompt = (
                f"{system_prompt.strip()}\n\n"
                f"{user_prompt.strip()}\n\n"
                f"{schema_instruction}"
            )
            parameters = {
                "max_tokens": self.sampling.max_tokens,
                "temperature": self.sampling.temperature,
                "max_denoising_steps": self.sampling.max_denoising_steps,
                "entropy_threshold": self.sampling.entropy_threshold,
                "entropy_bound": self.sampling.entropy_bound,
            }
            prediction = self._endpoint.predict(
                instances=[{"prompt": combined_prompt}],
                parameters=parameters,
            )
            predictions = getattr(prediction, "predictions", None)
            if not predictions:
                raise ValueError("Empty predictions from DiffusionGemma endpoint")

            first = predictions[0]
            if isinstance(first, dict):
                raw_text: str | None = None
                for key in ("output", "text", "generated_text", "content"):
                    if key in first:
                        val = first[key]
                        raw_text = "" if val is None else str(val)
                        break
                if raw_text is None:
                    raw_text = json.dumps(first)
            else:
                raw_text = "" if first is None else str(first)

            if not raw_text.strip():
                raise ValueError("Empty response from DiffusionGemma endpoint")
            return extract_json_object(raw_text, schema)

        if self._genai_maas is not None:
            prompt_with_schema = f"{user_prompt.strip()}\n\n{schema_instruction}"
            resp = self._genai_maas.models.generate_content(
                model=self.maas_model_name,
                contents=prompt_with_schema,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.sampling.temperature,
                    max_output_tokens=self.sampling.max_tokens,
                    response_mime_type="application/json",
                ),
            )
            text = getattr(resp, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty response from DiffusionGemma MaaS model")
            return extract_json_object(text, schema)

        raise RuntimeError("DiffusionGemmaClient has neither endpoint nor MaaS client configured")


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


class GeminiDecisionsClient:
    """Zero-output-token closed-set Layer-2 evaluator backed by Gemini Decisions API."""

    def __init__(
        self,
        project: str | None = None,
        location: str = "global",
        model: str = "gemini-decision-1",
        api_key: str | None = None,
        vertexai: bool = True,
        violation_threshold: float = 0.70,
        genai_client: Any | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-blank string")
        if not isinstance(vertexai, bool):
            raise ValueError("vertexai must be a bool")
        if not _is_finite_number(violation_threshold) or not (0.0 < violation_threshold < 1.0):
            raise ValueError(
                "violation_threshold must be strictly between 0.0 and 1.0, "
                f"got {violation_threshold!r}"
            )
        if vertexai and api_key is not None:
            raise ValueError("Set vertexai=False when authenticating with api_key")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("api_key must not be blank when provided")

        self.project = project
        self.location = location
        self.model = model.strip()
        self.vertexai = vertexai
        self.violation_threshold = float(violation_threshold)

        if genai_client is not None:
            self._client = genai_client
        elif vertexai:
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        else:
            resolved_key = (
                api_key.strip()
                if api_key is not None
                else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
            )
            self._client = genai.Client(
                api_key=resolved_key,
                http_options={"api_version": "v1beta"},
            )

    def evaluate_options(
        self,
        context_text: str,
        candidates: list[CandidateOption],
        has_assumptions: bool = True,
    ) -> TacticalDecision:
        if not isinstance(context_text, str) or not context_text.strip():
            raise ValueError("context_text must be a non-blank string")
        if not candidates:
            raise ValueError("candidates must be non-empty")

        questions: dict[str, dict[str, Any]] = {}
        for idx, cand in enumerate(candidates):
            make_prob_str = (
                f"{cand.make_prob:.2f}" if cand.make_prob is not None else "N/A"
            )
            questions[f"cand_{idx}"] = {
                "type": "score",
                "instructions": (
                    f"Rate tactical quality of candidate action '{cand.label}' "
                    f"(DDS baseline_ev={cand.baseline_ev:.1f}, make_prob={make_prob_str}) "
                    "for achieving the Active Strategy while avoiding blunders."
                ),
                "criteria": list(_DECISION_SCORE_CRITERIA),
            }

        if has_assumptions:
            questions["assumption_violated"] = {
                "type": "noul",
                "instructions": (
                    "Do the observed Board Facts (such as unexpected suit show-outs, voids, "
                    "or auction calls) directly contradict a critical assumption in the "
                    "Active Strategy?"
                ),
                "criteria": {
                    "true_description": (
                        "Observed board facts directly contradict a critical Active Strategy "
                        "assumption, requiring immediate strategic replanning."
                    ),
                    "false_description": (
                        "Observed board facts remain consistent with the Active Strategy "
                        "assumptions."
                    ),
                },
            }

        resp = self._client.decisions.create(
            model=self.model,
            input=context_text,
            questions=questions,
        )
        answers = getattr(resp, "answers", None)
        if not isinstance(answers, Mapping) or not answers:
            raise ValueError("Empty answers from Gemini Decisions API")

        option_scores: dict[str, float] = {}
        option_confidences: dict[str, float] = {}
        for idx, cand in enumerate(candidates):
            q_id = f"cand_{idx}"
            ans = answers.get(q_id)
            if ans is None:
                raise ValueError(
                    f"Missing decision answer for candidate {cand.label} ({q_id})"
                )
            raw_score = (
                ans.get("score") if isinstance(ans, Mapping) else getattr(ans, "score", None)
            )
            if not _is_finite_number(raw_score):
                raise ValueError(
                    f"Invalid score for candidate {cand.label} ({q_id}): {raw_score!r}"
                )
            norm_score = min(1.0, max(0.0, float(raw_score) / 4.0))
            option_scores[cand.label] = norm_score

            raw_conf = (
                ans.get("confidence")
                if isinstance(ans, Mapping)
                else getattr(ans, "confidence", None)
            )
            if _is_finite_number(raw_conf):
                option_confidences[cand.label] = float(raw_conf)

        violated = False
        violation_reason = ""
        noul_prob_val: float | None = None
        noul_conf_val: float | None = None
        if has_assumptions:
            v_ans = answers.get("assumption_violated")
            if v_ans is None:
                raise ValueError("Missing decision answer for assumption_violated")
            raw_noul = (
                v_ans.get("noul")
                if isinstance(v_ans, Mapping)
                else getattr(v_ans, "noul", None)
            )
            if not _is_finite_number(raw_noul):
                raise ValueError(
                    f"Invalid noul probability for assumption_violated: {raw_noul!r}"
                )
            noul_prob_val = float(raw_noul)
            raw_v_conf = (
                v_ans.get("confidence")
                if isinstance(v_ans, Mapping)
                else getattr(v_ans, "confidence", None)
            )
            if _is_finite_number(raw_v_conf):
                noul_conf_val = float(raw_v_conf)

            violated = noul_prob_val >= self.violation_threshold
            if violated:
                violation_reason = (
                    "Decisions API flagged assumption contradiction "
                    f"(p={noul_prob_val:.2f} >= {self.violation_threshold:.2f})"
                )

        best_label, best_score = max(option_scores.items(), key=lambda kv: kv[1])
        best_conf = option_confidences.get(best_label)
        top_part = (
            f"Top candidate {best_label}={best_score:.2f} (conf={best_conf:.2f})"
            if best_conf is not None
            else f"Top candidate {best_label}={best_score:.2f}"
        )
        if noul_prob_val is not None:
            viol_part = (
                f"p_violation={noul_prob_val:.2f} (conf={noul_conf_val:.2f})"
                if noul_conf_val is not None
                else f"p_violation={noul_prob_val:.2f}"
            )
            rationale = f"Decisions API ({self.model}): {top_part}; {viol_part}."
        else:
            rationale = f"Decisions API ({self.model}): {top_part}."

        return TacticalDecision(
            assumption_violated=violated,
            violation_reason=violation_reason,
            option_scores=option_scores,
            rationale=rationale,
        )

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        if schema is not TacticalDecision:
            raise ValueError(
                "GeminiDecisionsClient only supports closed-set TacticalDecision evaluation, "
                "not open-ended generation"
            )

        parsed_candidates: list[CandidateOption] = []
        for match in _CANDIDATE_LINE_RE.finditer(user_prompt):
            label = match.group("label").strip()
            ev = float(match.group("ev"))
            mp_raw = match.group("mp")
            mp = None if mp_raw == "N/A" else min(1.0, max(0.0, float(mp_raw)))
            parsed_candidates.append(
                CandidateOption(
                    action=0,
                    label=label,
                    baseline_ev=ev,
                    make_prob=mp,
                )
            )

        if not parsed_candidates:
            raise ValueError("No candidate options found in user_prompt")

        has_assumptions = "Assumptions: (none)" not in user_prompt
        combined_context = (
            f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
            if system_prompt.strip()
            else user_prompt.strip()
        )
        decision = self.evaluate_options(
            context_text=combined_context,
            candidates=parsed_candidates,
            has_assumptions=has_assumptions,
        )
        return cast(T, decision)

