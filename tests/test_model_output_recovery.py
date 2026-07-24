from __future__ import annotations

import unittest

from music_agent.schemas import (
    AnalysisResult,
    ChordEvent,
    Measure,
    SectionResult,
    SectionStructureDraft,
    SpecialistChordDraft,
    SpecialistSectionDraft,
    TrackResult,
    UncertainRange,
)
from music_agent.validators import (
    normalise_sections,
    normalise_specialist_result,
    validate_quality,
)


class ModelOutputRecoveryTests(unittest.TestCase):
    def test_structure_draft_maps_japanese_type_and_discards_invalid_range(self) -> None:
        sections = normalise_sections(
            [
                SectionStructureDraft(
                    id="a",
                    name="Aメロ1",
                    type="Aメロ",
                    startSeconds=0.0,
                    endSeconds=32.0,
                    confidence=1.4,
                ),
                SectionStructureDraft(
                    id="bad",
                    name="broken",
                    type="chorus",
                    startSeconds=40.0,
                    endSeconds=20.0,
                ),
                SectionStructureDraft(
                    id="b",
                    name="サビ1",
                    type="サビ",
                    startSeconds=32.0,
                    endSeconds=64.0,
                ),
            ],
            duration=64.0,
        )
        self.assertEqual([item.type for item in sections], ["verse", "chorus"])
        self.assertEqual(sections[0].confidence, 1.0)
        self.assertEqual(sections[-1].endSeconds, 64.0)

    def test_specialist_clip_relative_timestamps_are_converted_to_track_time(self) -> None:
        section = SectionStructureDraft(
            id="chorus-1",
            name="サビ1",
            type="chorus",
            startSeconds=60.0,
            endSeconds=68.0,
        )
        result = normalise_specialist_result(
            SpecialistSectionDraft(
                sectionId="wrong",
                role="日本語の役割",
                chords=[
                    SpecialistChordDraft(
                        symbol="dbm",
                        startSeconds=1.0,
                        endSeconds=5.0,
                        confidence=1.2,
                    )
                ],
            ),
            section=section,
            role="root_quality",
            clip_start=59.0,
            clip_end=69.0,
            beat_duration=0.5,
        )
        self.assertEqual(result.sectionId, "chorus-1")
        self.assertEqual(result.role, "root_quality")
        self.assertEqual(result.chords[0].symbol, "C#m")
        self.assertEqual(result.chords[0].startSeconds, 60.0)
        self.assertEqual(result.chords[0].endSeconds, 64.0)
        self.assertEqual(result.chords[0].confidence, 1.0)

    def test_full_track_unknown_result_is_rejected(self) -> None:
        result = AnalysisResult(
            track=TrackResult(durationSeconds=188.0, bpm=180.0),
            sections=[
                SectionResult(
                    id="full-track",
                    name="全体",
                    type="other",
                    startSeconds=0.0,
                    endSeconds=188.0,
                    summary="構造分析が失敗したためフォールバックしました。",
                    measures=[
                        Measure(
                            bar=1,
                            startSeconds=0.0,
                            endSeconds=1.0,
                            chords=[
                                ChordEvent(
                                    symbol="X",
                                    startSeconds=0.0,
                                    endSeconds=188.0,
                                    source="ai",
                                )
                            ],
                        )
                    ],
                )
            ],
            sourceType="youtube",
            sourceLabel="https://youtu.be/9QLT1Aw_45s",
            analysisMethod="ai_only",
            uncertainRanges=[
                UncertainRange(
                    sectionId="full-track",
                    startSeconds=0.0,
                    endSeconds=188.0,
                    reason="unknown",
                    candidates=["X"],
                    resolved=True,
                )
            ],
        )
        errors = validate_quality(result)
        self.assertIn("no_known_chords", errors)
        self.assertIn("structure_fallback_present", errors)
        self.assertTrue(any(item.startswith("too_few_sections:") for item in errors))
        self.assertTrue(any(item.startswith("too_few_chords:") for item in errors))


if __name__ == "__main__":
    unittest.main()
