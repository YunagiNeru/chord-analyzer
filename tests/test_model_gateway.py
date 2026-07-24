from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from music_agent.model_gateway import ModelGateway
from music_agent.schemas import (
    CompactResolutionDraft,
    CompactSpecialistDraft,
    FinalExplanationDraft,
)


class ModelGatewayTests(unittest.TestCase):
    def test_compact_specialist_uses_unbounded_vertex_request_schema(self) -> None:
        request_schema = ModelGateway._request_schema(CompactSpecialistDraft)

        self.assertIsNot(request_schema, CompactSpecialistDraft)
        request_schema_json = json.dumps(
            request_schema.model_json_schema(),
            ensure_ascii=False,
        )
        self.assertNotIn("maxItems", request_schema_json)

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
                FinalExplanationDraft,
                3_072,
            ),
            8_192,
        )


if __name__ == "__main__":
    unittest.main()
