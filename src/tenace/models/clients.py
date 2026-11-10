from __future__ import annotations

import json
import math
import re
from typing import Any, Protocol, TypeVar, runtime_checkable

from google import genai
from google.cloud import aiplatform
from google.genai import types as genai_types
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

_THOUGHT_BLOCK_RE = re.compile(r"<thought>.*?</thought>", flags=re.DOTALL | re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```json\b\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```", flags=re.DOTALL)

__all__ = [
    "DiffusionGemmaClient",
    "DiffusionSamplingConfig",
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


def _find_first_json_object_slice(text: str) -> str | None:
    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        brace_idx = text.find("{", search_from)
        if brace_idx == -1:
            return None
        try:
            obj, end_idx = decoder.raw_decode(text, brace_idx)
        except json.JSONDecodeError:
            search_from = brace_idx + 1
            continue
        if isinstance(obj, dict):
            return text[brace_idx:end_idx]
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

    for candidate in candidates:
        json_slice = _find_first_json_object_slice(candidate)
        if json_slice is not None:
            return schema.model_validate_json(json_slice)

    raise ValueError("No valid JSON object found in model response")


class VertexGeminiClient:
    def __init__(
        self,
        project: str | None = None,
        location: str = "global",
        model: str = "gemini-2.5-pro",
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
                response_schema=schema,
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
