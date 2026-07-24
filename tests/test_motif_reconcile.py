from __future__ import annotations

import unittest

from music_agent.consensus import ConsensusOutput
from music_agent.motif_reconcile import reconcile_repeated_sections_enhanced
from music_agent.schemas import ChordEvent, SectionStructureDraft, UncertainRange


class MotifReconcileTests(unittest.TestCase):
    @staticmethod
    def _pair(
        section_id: str,
        start: float,
        symbol: str,
        confidence: float,
    ) -> tuple[SectionStructureDraft, ConsensusOutput]:
        section = SectionStructureDraft(
            id=section_id,
            name="Aメロ1",
            type="verse",
            startSeconds=0.0,
            endSeconds=24.0,
            confidence=0.9,
        )
        chord = ChordEvent(
            symbol=symbol,
            startSeconds=start,
            endSeconds=start + 1.0,
            confidence=confidence,
            agreement=confidence,
            alternatives=["E", "Emaj7", "G#m"],
            source="ai",
        )
        target = UncertainRange(
            sectionId=section_id,
            startSeconds=start,
            endSeconds=start + 1.0,
            reason="ルートまたは基本コード品質の一致度が閾値未満です。",
            candidates=["E", "Emaj7", "G#m"],
        )
        output = ConsensusOutput(
            chords=[chord],
            agreement=confidence,
            uncertain_ranges=[target],
            slot_alternatives=[["E", "Emaj7", "G#m"]],
            slot_scores=[{"E": 1.0}],
        )
        return section, output

    def test_three_repeated_positions_share_confident_winner(self) -> None:
        pairs = [
            self._pair("verse-a", 29.0, "E", 0.82),
            self._pair("verse-b", 37.0, "Emaj7", 0.79),
            self._pair("verse-c", 45.0, "G#m", 0.40),
        ]

        reconcile_repeated_sections_enhanced(pairs)

        self.assertEqual(pairs[2][1].chords[0].symbol, "E")
        self.assertGreaterEqual(pairs[2][1].chords[0].confidence, 0.62)
        self.assertTrue(all(not output.uncertain_ranges for _, output in pairs))

    def test_two_occurrences_are_not_enough_to_force_resolution(self) -> None:
        pairs = [
            self._pair("verse-a", 29.0, "E", 0.82),
            self._pair("verse-b", 37.0, "G#m", 0.40),
        ]

        reconcile_repeated_sections_enhanced(pairs)

        self.assertTrue(pairs[0][1].uncertain_ranges)
        self.assertTrue(pairs[1][1].uncertain_ranges)


if __name__ == "__main__":
    unittest.main()
