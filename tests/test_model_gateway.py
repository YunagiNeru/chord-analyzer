from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from music_agent.diagnostics import DiagnosticsRecorder
from music_agent.model_gateway import ModelGateway
from music_agent.schemas import (
    CompactResolutionDraft,
    CompactSpecialistDraft,
    FinalExplanationDraft,
)


class _FakeModels:
    def __init__(self) -> None:
        self.request_schema = None
        self.max_output_tokens = None

    def generate_content(self, *, model, contents, config):
        del model, contents
        self.request_schema = config.response_schema
        self.max_output_tokens = config.max_output_tokens
        parsed = self.request_schema.model_validate(
            {
                "chords": [
                    {
                        "symbol": "C",
                        "startSeconds": 0.0,
                        "endSeconds": 2.0,
                        "confidence": 0.9,
                        "alternatives": ["Cmaj7"],
                    }
                ],
                "repeatedPattern": ["C"],
            }
        )
        return SimpleNamespace(
            parsed=parsed,
            text=None,
            usage_metadata=None,
        )


class _FakeClient:
    def __init__(self) -> None:
        self.models = _FakeModels()


class ModelGatewayTests(unittest.TestCase):
    def test_compact_specialist_uses_unbounded_vertex_request_schema(self) -> None:
        client = _FakeClient()
        gateway = ModelGateway(
            client=client,
            model="test-model",
            diagnostics=DiagnosticsRecorder(),
        )

        result = gateway.generate_typed(
            contents=["audio", "prompt"],
            schema=CompactSpecialistDraft,
            system_instruction="test",
            max_output_tokens=512,
            retries=1,
        )

        self.assertIsInstance(result, CompactSpecialistDraft)
        self.assertEqual(result.chords[0].symbol, "C")
        self.assertEqual(result.chords[0].alternatives, ["Cmaj7"])
        request_schema_json = json.dumps(
            client.models.request_schema.model_json_schema(),
            ensure_ascii=False,
        )
        self.assertNotIn("maxItems", request_schema_json)
        self.assertGreaterEqual(client.models.max_output_tokens, 4_096)

    def test_small_structured_contracts_receive_safe_token_floors(self) -> None:
        self.assertEqual(
            ModelGateway._effective_max_output_tokens(
                CompactResolutionDraft,
                512,
            ),
            4_096,
        )
        self.assertEqual(
            ModelGateway._effective_max_output_tokens(
                FinalExplanationDraft,
                3_072,
            ),
            8_192,
        )


if __name__ == "__main__":
    unittest.main()
