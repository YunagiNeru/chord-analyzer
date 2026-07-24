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

    def test_review_snapshot_uses_corroborated_120_not_ungrounded_240(self) -> None:
        sections = [
            SectionStructureDraft(
                id=section_id,
                name=section_id,
                type=section_type,
                startSeconds=start,
                endSeconds=end,
            )
            for section_id, section_type, start, end in (
                ("intro1", "intro", 0.0, 11.0),
                ("intro2", "intro", 11.0, 27.0),
                ("verse1", "verse", 27.0, 50.0),
                ("prechorus1", "pre_chorus", 50.0, 61.0),
                ("chorus1", "chorus", 61.0, 84.0),
                ("interlude1", "interlude", 84.0, 95.0),
                ("verse2", "verse", 95.0, 107.0),
                ("prechorus2", "pre_chorus", 107.0, 129.0),
                ("interlude2", "interlude", 129.0, 140.0),
                ("bridge", "bridge", 140.0, 160.0),
                ("chorus3", "chorus", 160.0, 181.0),
                ("outro", "outro", 181.0, 190.0),
            )
        ]
        rhythm = RhythmDraft(
            durationSeconds=190.0,
            selectedBpm=120.0,
            bpmCandidates=[
                TempoCandidate(bpm=120.0, confidence=0.95, interpretation="model"),
                TempoCandidate(bpm=240.0, confidence=0.95, interpretation="double-time"),
            ],
            tempoSegments=[
                TempoSegment(
                    startSeconds=0.0,
                    endSeconds=190.0,
                    bpm=120.0,
                    confidence=0.95,
                )
            ],
            timeSignature="4/4",
            confidence=0.95,
        )

        grid = build_beat_grid(rhythm=rhythm, sections=sections, duration=190.0)

        self.assertEqual(grid.bpm, 120.0)
        self.assertEqual(grid.tempo_segments[0].bpm, 120.0)
        self.assertEqual(grid.bar_for_time(61.0), 31)


if __name__ == "__main__":
    unittest.main()
