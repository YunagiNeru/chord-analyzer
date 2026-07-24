from __future__ import annotations

import unittest

from music_agent.beat_grid import BeatGrid
from music_agent.schemas import ChordEvent, SectionStructureDraft
from music_agent.validators import build_section_result, normalise_sections, roman_numeral


class ValidatorTests(unittest.TestCase):
    def test_sections_cover_duration(self) -> None:
        sections = normalise_sections(
            [
                SectionStructureDraft(
                    id="a",
                    name="A",
                    type="verse",
                    startSeconds=1.0,
                    endSeconds=5.0,
                ),
                SectionStructureDraft(
                    id="b",
                    name="B",
                    type="chorus",
                    startSeconds=4.5,
                    endSeconds=9.0,
                ),
            ],
            duration=10.0,
        )
        self.assertEqual(sections[0].startSeconds, 0.0)
        self.assertEqual(sections[-1].endSeconds, 10.0)
        self.assertTrue(
            all(
                left.endSeconds <= right.startSeconds + 1e-6
                for left, right in zip(sections, sections[1:])
            )
        )

    def test_distinct_sections_are_not_merged_by_static_twelve_cap(self) -> None:
        raw = []
        for index in range(14):
            section_type = (
                "bridge"
                if index == 11
                else "chorus"
                if index == 12
                else "verse"
                if index % 2
                else "interlude"
            )
            raw.append(
                SectionStructureDraft(
                    id=f"s{index}",
                    name=f"S{index}",
                    type=section_type,
                    startSeconds=index * 10.0,
                    endSeconds=(index + 1) * 10.0,
                )
            )

        sections = normalise_sections(
            raw,
            duration=140.0,
            max_sections=12,
        )

        self.assertEqual(len(sections), 14)
        self.assertEqual(sections[11].type, "bridge")
        self.assertEqual(sections[12].type, "chorus")
        self.assertNotIn("セクション数上限", " ".join(item.notes for item in sections))

    def test_measure_assignment_uses_downbeat(self) -> None:
        grid = BeatGrid(
            bpm=120.0,
            time_signature="4/4",
            beats_per_bar=4,
            beat_duration=0.5,
            downbeat_offset=0.5,
            duration=4.5,
            beat_times=tuple(index * 0.5 for index in range(10)),
            bar_starts=(0.5, 2.5, 4.5),
            score=0.9,
        )
        section = SectionStructureDraft(
            id="a",
            name="A",
            type="verse",
            startSeconds=0.5,
            endSeconds=4.5,
            key="C",
            mode="major",
        )
        result = build_section_result(
            section=section,
            chords=[
                ChordEvent(
                    symbol="C",
                    startSeconds=0.5,
                    endSeconds=2.5,
                    confidence=0.9,
                    source="ai",
                ),
                ChordEvent(
                    symbol="G",
                    startSeconds=2.5,
                    endSeconds=4.5,
                    confidence=0.9,
                    source="ai",
                ),
            ],
            grid=grid,
            agreement=0.9,
        )
        self.assertEqual([measure.bar for measure in result.measures], [1, 2])
        self.assertEqual(result.measures[0].chords[0].beat, 1.0)
        self.assertTrue(
            all(
                chord.startSeconds >= measure.startSeconds
                and chord.endSeconds <= measure.endSeconds
                for measure in result.measures
                for chord in measure.chords
            )
        )

    def test_long_chord_is_split_at_measure_boundaries(self) -> None:
        grid = BeatGrid(
            bpm=120.0,
            time_signature="4/4",
            beats_per_bar=4,
            beat_duration=0.5,
            downbeat_offset=0.0,
            duration=4.0,
            beat_times=tuple(index * 0.5 for index in range(9)),
            bar_starts=(0.0, 2.0, 4.0),
            score=0.9,
        )
        section = SectionStructureDraft(
            id="long",
            name="Long",
            type="intro",
            startSeconds=0.0,
            endSeconds=4.0,
            key="C",
            mode="major",
        )
        result = build_section_result(
            section=section,
            chords=[
                ChordEvent(
                    symbol="Cmaj7",
                    startSeconds=0.0,
                    endSeconds=4.0,
                    confidence=0.9,
                    source="ai",
                )
            ],
            grid=grid,
            agreement=0.9,
        )
        self.assertEqual([measure.bar for measure in result.measures], [1, 2])
        self.assertEqual(
            [chord.symbol for measure in result.measures for chord in measure.chords],
            ["Cmaj7", "Cmaj7"],
        )

    def test_roman(self) -> None:
        self.assertEqual(roman_numeral("G", "C", "major"), "V")
        self.assertEqual(roman_numeral("Am", "C", "major"), "vi")


if __name__ == "__main__":
    unittest.main()
