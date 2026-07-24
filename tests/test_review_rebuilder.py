from __future__ import annotations

import json
import unittest
from pathlib import Path

from music_agent.review_rebuilder import (
    choose_canonical_bpm,
    rebuild_review_snapshot,
)


FIXTURE = Path(__file__).with_name("fixtures") / "phony_v2_review_regression.json"


class ReviewRebuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_full_duration_tempo_segment_repairs_false_double_time(self) -> None:
        self.assertEqual(choose_canonical_bpm(self.snapshot), 120.0)

        rebuilt = rebuild_review_snapshot(self.snapshot)

        self.assertEqual(rebuilt["track"]["bpm"], 120.0)
        self.assertEqual(rebuilt["tempoSegments"][0]["bpm"], 120.0)
        self.assertTrue(rebuilt["rebuildDiagnostics"]["bpmChanged"])
        self.assertEqual(rebuilt["rebuildDiagnostics"]["originalBpm"], 240.0)

    def test_bar_numbers_are_rebuilt_from_corrected_grid(self) -> None:
        rebuilt = rebuild_review_snapshot(self.snapshot)
        chorus = next(
            section
            for section in rebuilt["sections"]
            if section["id"] == "chorus1"
        )
        first_chord = chorus["chords"][0]

        self.assertEqual(first_chord["startSeconds"], 61.0)
        self.assertEqual(first_chord["bar"], 31)
        self.assertEqual(first_chord["beat"], 3.0)
        self.assertLess(
            rebuilt["rebuildDiagnostics"]["maxBarAfter"],
            rebuilt["rebuildDiagnostics"]["maxBarBefore"],
        )

    def test_semantic_merges_are_reported_not_guessed_apart(self) -> None:
        rebuilt = rebuild_review_snapshot(self.snapshot)
        diagnostics = rebuilt["rebuildDiagnostics"]

        self.assertTrue(diagnostics["semanticMergeDetected"])
        self.assertIn("prechorus2+prechorus3", diagnostics["mergedSectionIds"])
        self.assertIn("bridge+chorus2_quiet", diagnostics["mergedSectionIds"])
        self.assertFalse(diagnostics["sectionStructureRebuilt"])


if __name__ == "__main__":
    unittest.main()
