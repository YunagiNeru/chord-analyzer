from __future__ import annotations

import unittest

from music_agent.chord_symbol import (
    canonicalize_symbol,
    chord_distance,
    parse_chord,
    transpose_symbol,
)


class ChordSymbolTests(unittest.TestCase):
    def test_common_symbols(self) -> None:
        self.assertEqual(canonicalize_symbol("Dbmaj7/F"), "C#maj7/F")
        self.assertEqual(canonicalize_symbol("F#min7b5"), "F#m7-5")
        self.assertEqual(canonicalize_symbol("N.C."), "N")
        self.assertEqual(canonicalize_symbol("Csus"), "Csus4")

    def test_components(self) -> None:
        parsed = parse_chord("Am7/E")
        self.assertEqual(parsed.root, "A")
        self.assertEqual(parsed.quality, "minor")
        self.assertEqual(parsed.seventh, "minor7")
        self.assertEqual(parsed.bass, "E")

    def test_transpose(self) -> None:
        self.assertEqual(transpose_symbol("Cmaj7/E", 2), "Dmaj7/F#")

    def test_distance(self) -> None:
        self.assertEqual(chord_distance("C", "C"), 0.0)
        self.assertLess(chord_distance("C", "Cm"), chord_distance("C", "F#m"))


if __name__ == "__main__":
    unittest.main()
