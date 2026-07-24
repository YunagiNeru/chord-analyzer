from __future__ import annotations

import unittest

from music_agent.schemas import AnalysisResult, SectionResult, TrackResult


class ApiCompatibilityTests(unittest.TestCase):
    def test_existing_required_fields_remain(self) -> None:
        result = AnalysisResult(
            track=TrackResult(durationSeconds=10.0),
            sections=[
                SectionResult(
                    id="full",
                    name="全体",
                    type="other",
                    startSeconds=0.0,
                    endSeconds=10.0,
                )
            ],
            sourceType="youtube",
            sourceLabel="https://youtu.be/example",
            analysisMethod="ai_only",
        )
        payload = result.model_dump()
        for key in (
            "track",
            "sections",
            "musicalSummary",
            "warnings",
            "analysisVersion",
            "sourceType",
            "sourceLabel",
            "analysisMethod",
            "waveform",
            "limitations",
        ):
            self.assertIn(key, payload)
        self.assertIn("diagnostics", payload)
        self.assertIn("uncertainRanges", payload)


if __name__ == "__main__":
    unittest.main()
