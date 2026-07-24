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

    def test_compound_extension_cleanup(self) -> None:
        self.assertEqual(canonicalize_symbol("Emaj79"), "Emaj9")
        self.assertEqual(parse_chord("Emaj79").extensions, ("9",))
        self.assertEqual(canonicalize_symbol("D#79b9"), "D#7b9")
        self.assertEqual(canonicalize_symbol("C713"), "C13")

    def test_simplify_keeps_only_root_and_basic_quality(self) -> None:
        self.assertEqual(canonicalize_symbol("Emaj7", simplify=True), "E")
        self.assertEqual(canonicalize_symbol("Am7/E", simplify=True), "Am")
        self.assertEqual(canonicalize_symbol("F#m7-5", simplify=True), "F#dim")
        self.assertEqual(canonicalize_symbol("D#7b9/G", simplify=True), "D#")

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
