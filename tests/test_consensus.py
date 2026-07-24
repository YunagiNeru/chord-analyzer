from __future__ import annotations

import unittest

from music_agent.beat_grid import BeatGrid
from music_agent.consensus import consensus_section
from music_agent.schemas import (
    SectionStructureDraft,
    SpecialistChordDraft,
    SpecialistSectionDraft,
)


class ConsensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = BeatGrid(
            bpm=120.0,
            time_signature="4/4",
            beats_per_bar=4,
            beat_duration=0.5,
            downbeat_offset=0.0,
            duration=2.0,
            beat_times=(0.0, 0.5, 1.0, 1.5, 2.0),
            bar_starts=(0.0, 2.0),
            score=0.9,
        )
        self.section = SectionStructureDraft(
            id="chorus-1",
            name="サビ1",
            type="chorus",
            startSeconds=0.0,
            endSeconds=2.0,
        )

    @staticmethod
    def _specialist(
        role: str,
        symbol: str,
        confidence: float,
    ) -> SpecialistSectionDraft:
        return SpecialistSectionDraft(
            sectionId="chorus-1",
            role=role,
            chords=[
                SpecialistChordDraft(
                    symbol=symbol,
                    startSeconds=0.0,
                    endSeconds=2.0,
                    confidence=confidence,
                )
            ],
        )

    def test_majority_root_quality_wins(self) -> None:
        specialists = [
            self._specialist("root_quality", "C", 0.95),
            self._specialist("bass_extension", "Cmaj7", 0.8),
            self._specialist("rhythm_pattern", "G", 0.25),
        ]
        output = consensus_section(
            section=self.section,
            specialists=specialists,
            grid=self.grid,
        )
        self.assertTrue(output.chords)
        self.assertIn(output.chords[0].symbol, {"C", "Cmaj7"})
        self.assertGreater(output.agreement, 0.8)
        self.assertEqual(output.uncertain_ranges, [])

    def test_extension_disagreement_is_not_critical_uncertainty(self) -> None:
        specialists = [
            self._specialist("root_quality", "Am", 0.95),
            self._specialist("bass_extension", "Am7", 0.9),
            self._specialist("rhythm_pattern", "Am/E", 0.85),
        ]
        output = consensus_section(
            section=self.section,
            specialists=specialists,
            grid=self.grid,
        )
        self.assertEqual(output.uncertain_ranges, [])
        self.assertGreaterEqual(output.agreement, 0.99)
        self.assertIn(output.chords[0].symbol, {"Am", "Am7", "Am/E"})

    def test_root_quality_disagreement_remains_uncertain(self) -> None:
        specialists = [
            self._specialist("root_quality", "C", 0.8),
            self._specialist("bass_extension", "Cm7", 0.8),
            self._specialist("rhythm_pattern", "G", 0.8),
        ]
        output = consensus_section(
            section=self.section,
            specialists=specialists,
            grid=self.grid,
        )
        self.assertTrue(output.uncertain_ranges)
        self.assertIn(
            "基本コード品質",
            output.uncertain_ranges[0].reason,
        )

    def test_single_primary_contributor_is_uncertain(self) -> None:
        output = consensus_section(
            section=self.section,
            specialists=[self._specialist("root_quality", "C", 0.95)],
            grid=self.grid,
        )
        self.assertTrue(output.uncertain_ranges)
        self.assertIn("独立分析", output.uncertain_ranges[0].reason)


if __name__ == "__main__":
    unittest.main()
