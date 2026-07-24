from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from .diagnostics import DiagnosticsRecorder


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class _VertexCompactChordDraft(BaseModel):
    """Vertex-facing equivalent without nested array-length constraints."""

    symbol: str = "X"
    startSeconds: float = 0.0
    endSeconds: float = 0.0
    confidence: float = 0.5
    alternatives: list[str] = Field(default_factory=list)


class _VertexCompactSpecialistDraft(BaseModel):
    """Request schema accepted by Vertex structured output conversion.

    The canonical CompactSpecialistDraft retains deterministic limits after the
    response is received. Sending nested maxItems constraints caused every
    specialist request to fail with HTTP 400 INVALID_ARGUMENT on Vertex AI.
    """

    chords: list[_VertexCompactChordDraft] = Field(default_factory=list)
    repeatedPattern: list[str] = Field(default_factory=list)


_REQUEST_SCHEMA_OVERRIDES: dict[str, type[BaseModel]] = {
    "CompactSpecialistDraft": _VertexCompactSpecialistDraft,
}

_TRANSIENT_MARKERS = (
    "429",
    "resource_exhausted",
    "resource exhausted",
    "503",
    "service_unavailable",
    "service unavailable",
    "500",
    "internal",
    "deadline_exceeded",
    "deadline exceeded",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "read timeout",
    "timed out",
)
_TRUNCATION_MARKERS = (
    "eof while parsing",
    "json_invalid",
    "unterminated string",
    "unexpected end",
    "empty response",
    "空の応答",
)


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(0.0, value)


class ModelGateway:
    def __init__(
        self,
        *,
        client: genai.Client,
        model: str,
        diagnostics: DiagnosticsRecorder,
        max_parallel_calls: int = 4,
    ) -> None:
        self.client = client
        self.model = model
        self.diagnostics = diagnostics
        self._semaphore = threading.BoundedSemaphore(max(1, max_parallel_calls))

    @staticmethod
    def _usage(response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return 0, 0
        input_tokens = int(
            getattr(usage, "prompt_token_count", 0)
            or getattr(usage, "input_token_count", 0)
            or 0
        )
        output_tokens = int(
            getattr(usage, "candidates_token_count", 0)
            or getattr(usage, "output_token_count", 0)
            or 0
        )
        return input_tokens, output_tokens

    @staticmethod
    def _parse_response(response: Any, schema: type[SchemaT]) -> SchemaT:
        parsed = response.parsed
        if isinstance(parsed, schema):
            return parsed
        if isinstance(parsed, BaseModel):
            parsed = parsed.model_dump()
        if parsed is not None:
            return schema.model_validate(parsed)
        if not response.text:
            raise RuntimeError("Geminiから空の応答が返されました。")
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError:
            return schema.model_validate_json(response.text)
        return schema.model_validate(payload)

    @staticmethod
    def _request_schema(schema: type[SchemaT]) -> type[BaseModel]:
        return _REQUEST_SCHEMA_OVERRIDES.get(schema.__name__, schema)

    @staticmethod
    def _effective_max_output_tokens(
        schema: type[BaseModel],
        requested: int,
    ) -> int:
        # Gemini 3.5 may consume part of max_output_tokens before emitting the
        # structured JSON. Small visible responses can therefore still end at
        # EOF when the budget is close to the visible JSON size.
        minimums = {
            "CompactSpecialistDraft": 4_096,
            "CompactResolutionDraft": 4_096,
            "ResolutionDraft": 4_096,
            "BatchResolutionDraft": 8_192,
            "StructureRefinementDraft": 6_144,
            "SemanticStructureRefinementDraft": 8_192,
            "FinalExplanationDraft": 8_192,
        }
        return max(requested, minimums.get(schema.__name__, requested))

    @staticmethod
    def _error_text(exc: BaseException) -> str:
        return f"{type(exc).__name__}: {exc}".lower()

    @classmethod
    def _is_transient_error(cls, exc: BaseException) -> bool:
        text = cls._error_text(exc)
        return any(marker in text for marker in _TRANSIENT_MARKERS)

    @classmethod
    def _is_truncated_output(cls, exc: BaseException) -> bool:
        text = cls._error_text(exc)
        return any(marker in text for marker in _TRUNCATION_MARKERS)

    @classmethod
    def _retry_delay(cls, attempt: int, exc: BaseException) -> float:
        base = _env_float("MODEL_RETRY_BASE_SECONDS", 1.25)
        maximum = max(base, _env_float("MODEL_RETRY_MAX_SECONDS", 12.0))
        if cls._is_transient_error(exc):
            multiplier = 2 ** attempt
        else:
            multiplier = attempt + 1
        return min(maximum, base * multiplier)

    def generate_typed(
        self,
        *,
        contents: list[Any],
        schema: type[SchemaT],
        system_instruction: str,
        temperature: float = 0.0,
        max_output_tokens: int = 16_384,
        retries: int = 3,
        diagnostic_label: str | None = None,
    ) -> SchemaT:
        last_error: Exception | None = None
        base_contents = list(contents)
        request_schema = self._request_schema(schema)
        effective_max_tokens = self._effective_max_output_tokens(
            schema,
            max_output_tokens,
        )
        attempts = max(1, retries)
        attempt = 0
        while attempt < attempts:
            request_contents = list(base_contents)
            if attempt:
                request_contents.append(
                    "前回の出力はJSONスキーマ検証に失敗、途中終了、または一時的なAPI障害になりました。"
                    "説明文を含めず、文字列を短くし、必須フィールドだけを含む完全に閉じたJSONを返してください。"
                    "数値は有限値、配列は配列、不明値は空配列または既定値にしてください。"
                )
            try:
                with self._semaphore:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=request_contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            response_schema=request_schema,
                            temperature=temperature,
                            max_output_tokens=effective_max_tokens,
                        ),
                    )
                input_tokens, output_tokens = self._usage(response)
                self.diagnostics.record_model_call(
                    self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                return self._parse_response(response, schema)
            except Exception as exc:  # noqa: BLE001 - retry boundary
                last_error = exc
                self.diagnostics.record_error(
                    diagnostic_label or schema.__name__,
                    exc,
                )
                if self._is_transient_error(exc):
                    attempts = max(attempts, 5)
                elif self._is_truncated_output(exc):
                    attempts = max(attempts, 4)
                if attempt + 1 < attempts:
                    time.sleep(self._retry_delay(attempt, exc))
                attempt += 1
        assert last_error is not None
        raise last_error

    def generate_text(
        self,
        *,
        contents: Any,
        system_instruction: str,
        temperature: float = 0.0,
        max_output_tokens: int = 3_072,
        tools: list[types.Tool] | None = None,
        retries: int = 4,
        diagnostic_label: str = "text-generation",
    ) -> Any:
        last_error: Exception | None = None
        attempts = max(1, retries)
        attempt = 0
        while attempt < attempts:
            try:
                with self._semaphore:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            tools=tools,
                        ),
                    )
                input_tokens, output_tokens = self._usage(response)
                self.diagnostics.record_model_call(
                    self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                return response
            except Exception as exc:  # noqa: BLE001 - retry boundary
                last_error = exc
                self.diagnostics.record_error(diagnostic_label, exc)
                if self._is_transient_error(exc):
                    attempts = max(attempts, 5)
                if attempt + 1 < attempts:
                    time.sleep(self._retry_delay(attempt, exc))
                attempt += 1
        assert last_error is not None
        raise last_error
