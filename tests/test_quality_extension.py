from __future__ import annotations

import unittest

from music_agent.quality_extension import validate_quality_extended
from music_agent.schemas import (
    AnalysisResult,
    ChordEvent,
    Measure,
    SectionResult,
    TrackResult,
)


class QualityExtensionTests(unittest.TestCase):
    def _result(self, *, agreement: float) -> AnalysisResult:
        chord = ChordEvent(
            symbol="Bmaj7",
            startSeconds=0.0,
            endSeconds=16.0,
            confidence=0.9,
            agreement=agreement,
            source="ai",
        )
        return AnalysisResult(
            track=TrackResult(
                durationSeconds=16.0,
                bpm=120.0,
                timeSignature="4/4",
            ),
            sections=[
                SectionResult(
                    id="verse",
                    name="Aメロ",
                    type="verse",
                    startSeconds=0.0,
                    endSeconds=16.0,
                    agreement=agreement,
                    measures=[
                        Measure(
                            bar=1,
                            startSeconds=0.0,
                            endSeconds=16.0,
                            chords=[chord],
                        )
                    ],
                )
            ],
            sourceType="youtube",
            sourceLabel="fixture",
            analysisMethod="ai_only",
        )

    def test_low_agreement_eight_bar_single_chord_is_rejected(self) -> None:
        errors = validate_quality_extended(self._result(agreement=0.42))
        self.assertTrue(
            any(item.startswith("implausible_static_section:verse") for item in errors)
        )

    def test_high_agreement_drone_is_not_automatically_rejected(self) -> None:
        errors = validate_quality_extended(self._result(agreement=0.9))
        self.assertFalse(
            any(item.startswith("implausible_static_section:verse") for item in errors)
        )


if __name__ == "__main__":
    unittest.main()
