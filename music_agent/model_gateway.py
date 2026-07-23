from __future__ import annotations

import json
import threading
import time
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from .diagnostics import DiagnosticsRecorder


SchemaT = TypeVar("SchemaT", bound=BaseModel)


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
        if parsed is not None:
            return schema.model_validate(parsed)
        if not response.text:
            raise RuntimeError("Geminiから空の応答が返されました。")
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError:
            return schema.model_validate_json(response.text)
        return schema.model_validate(payload)

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
        for attempt in range(max(1, retries)):
            request_contents = list(base_contents)
            if attempt:
                request_contents.append(
                    "前回の出力はJSONスキーマ検証に失敗しました。"
                    "必須フィールドの型を守り、数値は有限値、配列は配列、"
                    "不明値は推測せず既定値または空配列で返してください。"
                )
            try:
                with self._semaphore:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=request_contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            response_schema=schema,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
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
                if attempt + 1 < max(1, retries):
                    time.sleep(0.6 * (attempt + 1))
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
    ) -> Any:
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
