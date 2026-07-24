from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from music_agent.accuracy_v2_finalization_v2 import BatchResolutionDraft
from music_agent.model_gateway import ModelGateway
from music_agent.schemas import (
    CompactResolutionDraft,
    CompactSpecialistDraft,
    FinalExplanationDraft,
)


def _contains_schema_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(
            _contains_schema_key(item, target)
            for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_schema_key(item, target) for item in value)
    return False


class ModelGatewayTests(unittest.TestCase):
    def test_compact_specialist_uses_unbounded_vertex_request_schema(self) -> None:
        request_schema = ModelGateway._request_schema(CompactSpecialistDraft)

        self.assertIsNot(request_schema, CompactSpecialistDraft)
        self.assertFalse(
            _contains_schema_key(
                request_schema.model_json_schema(),
                "maxItems",
            )
        )

        parsed_for_vertex = request_schema.model_validate(
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
        result = ModelGateway._parse_response(
            SimpleNamespace(
                parsed=parsed_for_vertex,
                text=None,
            ),
            CompactSpecialistDraft,
        )

        self.assertIsInstance(result, CompactSpecialistDraft)
        self.assertEqual(result.chords[0].symbol, "C")
        self.assertEqual(result.chords[0].alternatives, ["Cmaj7"])

    def test_small_structured_contracts_receive_safe_token_floors(self) -> None:
        self.assertEqual(
            ModelGateway._effective_max_output_tokens(
                CompactSpecialistDraft,
                512,
            ),
            4_096,
        )
        self.assertEqual(
            ModelGateway._effective_max_output_tokens(
                CompactResolutionDraft,
                512,
            ),
            4_096,
        )
        self.assertEqual(
            ModelGateway._effective_max_output_tokens(
                BatchResolutionDraft,
                2_048,
            ),
            8_192,
        )
        self.assertEqual(
            ModelGateway._effective_max_output_tokens(
                FinalExplanationDraft,
                3_072,
            ),
            8_192,
        )

    def test_resource_exhausted_is_treated_as_transient(self) -> None:
        self.assertTrue(
            ModelGateway._is_transient_error(
                RuntimeError("429 RESOURCE_EXHAUSTED")
            )
        )

    def test_eof_json_is_treated_as_truncated_output(self) -> None:
        self.assertTrue(
            ModelGateway._is_truncated_output(
                RuntimeError("Invalid JSON: EOF while parsing a value")
            )
        )


if __name__ == "__main__":
    unittest.main()
