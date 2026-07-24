from __future__ import annotations

import unittest

from music_agent.beat_grid import beats_per_bar, build_beat_grid
from music_agent.schemas import (
    RhythmDraft,
    SectionStructureDraft,
    TempoCandidate,
    TempoSegment,
)


class BeatGridTests(unittest.TestCase):
    def test_time_signature(self) -> None:
        self.assertEqual(beats_per_bar("4/4"), 4)
        self.assertEqual(beats_per_bar("6/8"), 6)
        self.assertEqual(beats_per_bar(None), 4)

    def test_build_regular_grid(self) -> None:
        sections = [
            SectionStructureDraft(
                id="a",
                name="Aメロ",
                type="verse",
                startSeconds=0.0,
                endSeconds=16.0,
            ),
            SectionStructureDraft(
                id="b",
                name="サビ",
                type="chorus",
                startSeconds=16.0,
                endSeconds=32.0,
            ),
        ]
        rhythm = RhythmDraft(
            durationSeconds=32.0,
            bpmCandidates=[TempoCandidate(bpm=120.0, confidence=0.9)],
            selectedBpm=120.0,
            timeSignature="4/4",
            downbeatOffsetSeconds=0.0,
            confidence=0.9,
        )
        grid = build_beat_grid(rhythm=rhythm, sections=sections, duration=32.0)
        self.assertAlmostEqual(grid.bpm, 120.0)
        self.assertAlmostEqual(grid.beat_duration, 0.5)
        self.assertEqual(grid.bar_for_time(0.1), 1)
        self.assertEqual(grid.bar_for_time(2.1), 2)
        self.assertGreater(grid.score, 0.5)

    def test_rounded_section_seconds_do_not_force_false_double_time(self) -> None:
        sections = [
            SectionStructureDraft(
                id=f"section-{index}",
                name=f"Section {index}",
                type="verse" if index % 2 else "chorus",
                startSeconds=float(start),
                endSeconds=float(end),
            )
            for index, (start, end) in enumerate(
                ((0, 11), (11, 27), (27, 50), (50, 61), (61, 84)),
                start=1,
            )
        ]
        rhythm = RhythmDraft(
            durationSeconds=84.0,
            bpmCandidates=[
                TempoCandidate(bpm=120.0, confidence=0.95, interpretation="model"),
                TempoCandidate(
                    bpm=170.0,
                    confidence=0.35,
                    interpretation="grounded-reference",
                ),
            ],
            selectedBpm=120.0,
            timeSignature="4/4",
            tempoSegments=[
                TempoSegment(
                    startSeconds=0.0,
                    endSeconds=84.0,
                    bpm=120.0,
                    confidence=0.95,
                )
            ],
            confidence=0.95,
        )

        grid = build_beat_grid(rhythm=rhythm, sections=sections, duration=84.0)

        self.assertEqual(grid.bpm, 170.0)
        self.assertEqual(len(grid.tempo_segments), 1)
        self.assertEqual(grid.tempo_segments[0].bpm, 170.0)


if __name__ == "__main__":
    unittest.main()
