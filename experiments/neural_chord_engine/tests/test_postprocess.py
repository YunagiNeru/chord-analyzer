from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from models import ChordSegment, NoChordRange, canonicalize_symbol
from postprocess import (
    apply_no_chord_ranges,
    merge_adjacent,
    remove_short_isolated,
    snap_to_beats,
)


class CanonicalizeSymbolTests(unittest.TestCase):
    def test_omnizart_major_and_minor_labels(self) -> None:
        self.assertEqual(canonicalize_symbol("C:maj"), "C")
        self.assertEqual(canonicalize_symbol("F#:min"), "F#m")
        self.assertEqual(canonicalize_symbol("Bb:maj"), "A#")
        self.assertEqual(canonicalize_symbol("N"), "N")
        self.assertEqual(canonicalize_symbol("X"), "X")


class PostprocessTests(unittest.TestCase):
    def test_merge_adjacent_equal_chords(self) -> None:
        result = merge_adjacent(
            [
                ChordSegment(0.0, 1.0, "C", "omnizart"),
                ChordSegment(1.0, 2.0, "C", "omnizart"),
            ]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start_seconds, 0.0)
        self.assertEqual(result[0].end_seconds, 2.0)

    def test_remove_short_segment_between_equal_neighbors(self) -> None:
        result = remove_short_isolated(
            [
                ChordSegment(0.0, 1.0, "C", "omnizart"),
                ChordSegment(1.0, 1.2, "G", "omnizart"),
                ChordSegment(1.2, 2.0, "C", "omnizart"),
            ]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].symbol, "C")
        self.assertEqual(result[0].end_seconds, 2.0)

    def test_snap_only_when_boundary_is_near_beat(self) -> None:
        result = snap_to_beats(
            [
                ChordSegment(0.0, 1.08, "C", "omnizart"),
                ChordSegment(1.08, 2.0, "G", "omnizart"),
            ],
            [0.0, 0.5, 1.0, 1.5, 2.0],
        )

        self.assertEqual(result[0].end_seconds, 1.0)
        self.assertTrue(result[0].was_beat_snapped)
        self.assertTrue(result[1].was_beat_snapped)

    def test_silence_overrides_model_chord(self) -> None:
        result = apply_no_chord_ranges(
            [
                ChordSegment(0.0, 3.0, "C", "omnizart"),
            ],
            [
                NoChordRange(1.0, 2.0),
            ],
        )

        self.assertEqual(
            [item.symbol for item in result],
            ["C", "N", "C"],
        )
        self.assertEqual(result[1].source, "no_chord")


if __name__ == "__main__":
    unittest.main()
