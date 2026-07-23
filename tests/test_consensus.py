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
    def test_majority_root_quality_wins(self) -> None:
        grid = BeatGrid(
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
        section = SectionStructureDraft(
            id="chorus-1",
            name="サビ1",
            type="chorus",
            startSeconds=0.0,
            endSeconds=2.0,
        )
        specialists = [
            SpecialistSectionDraft(
                sectionId=section.id,
                role="root_quality",
                chords=[
                    SpecialistChordDraft(
                        symbol="C",
                        startSeconds=0.0,
                        endSeconds=2.0,
                        confidence=0.95,
                    )
                ],
            ),
            SpecialistSectionDraft(
                sectionId=section.id,
                role="bass_extension",
                chords=[
                    SpecialistChordDraft(
                        symbol="Cmaj7",
                        startSeconds=0.0,
                        endSeconds=2.0,
                        confidence=0.8,
                    )
                ],
            ),
            SpecialistSectionDraft(
                sectionId=section.id,
                role="rhythm_pattern",
                chords=[
                    SpecialistChordDraft(
                        symbol="G",
                        startSeconds=0.0,
                        endSeconds=2.0,
                        confidence=0.25,
                    )
                ],
            ),
        ]
        output = consensus_section(section=section, specialists=specialists, grid=grid)
        self.assertTrue(output.chords)
        self.assertIn(output.chords[0].symbol, {"C", "Cmaj7"})
        self.assertGreater(output.agreement, 0.4)


if __name__ == "__main__":
    unittest.main()
