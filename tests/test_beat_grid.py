from __future__ import annotations

import unittest

from music_agent.beat_grid import beats_per_bar, build_beat_grid
from music_agent.schemas import RhythmDraft, SectionStructureDraft, TempoCandidate


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


if __name__ == "__main__":
    unittest.main()
