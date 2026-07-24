from __future__ import annotations

import unittest

from music_agent.sequence_optimizer import optimize_sequence


class SequenceOptimizerTests(unittest.TestCase):
    def test_removes_one_slot_excursion(self) -> None:
        sequence = optimize_sequence(
            [
                {"C": 0.9, "G": 0.1},
                {"C": 0.46, "G": 0.54},
                {"C": 0.9, "G": 0.1},
            ]
        )
        self.assertEqual(sequence, ["C", "C", "C"])

    def test_keeps_supported_change(self) -> None:
        sequence = optimize_sequence(
            [
                {"C": 0.95, "G": 0.05},
                {"C": 0.05, "G": 0.95},
                {"C": 0.05, "G": 0.95},
            ]
        )
        self.assertEqual(sequence[-1], "G")


if __name__ == "__main__":
    unittest.main()
