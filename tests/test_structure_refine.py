from __future__ import annotations

import unittest

from music_agent.beat_grid import BeatGrid
from music_agent.consensus import consensus_section
from music_agent.schemas import SectionStructureDraft, StructureRefinementDraft
from music_agent.structure_refine import (
    align_sections_to_grid,
    build_analysis_slices,
    maximum_section_bars,
    refine_sections,
    section_bar_count,
)


class StructureRefineTests(unittest.TestCase):
    def setUp(self) -> None:
        bpm = 170.0
        beat_duration = 60.0 / bpm
        bar_duration = beat_duration * 4
        duration = 190.0
        offset = 0.75
        beat_times = []
        current = offset
        while current <= duration:
            beat_times.append(round(current, 6))
            current += beat_duration
        bar_starts = []
        current = offset
        while current <= duration:
            bar_starts.append(round(current, 6))
            current += bar_duration
        self.grid = BeatGrid(
            bpm=bpm,
            time_signature="4/4",
            beats_per_bar=4,
            beat_duration=beat_duration,
            downbeat_offset=offset,
            duration=duration,
            beat_times=tuple(beat_times),
            bar_starts=tuple(bar_starts),
            score=0.93,
        )

    def test_boundaries_snap_to_shared_beat_grid(self) -> None:
        sections = [
            SectionStructureDraft(
                id="intro",
                name="イントロ",
                type="intro",
                startSeconds=0.0,
                endSeconds=11.0,
            ),
            SectionStructureDraft(
                id="verse",
                name="Aメロ",
                type="verse",
                startSeconds=11.0,
                endSeconds=27.0,
            ),
            SectionStructureDraft(
                id="chorus",
                name="サビ",
                type="chorus",
                startSeconds=27.0,
                endSeconds=50.0,
            ),
        ]

        aligned, adjustments = align_sections_to_grid(
            sections,
            grid=self.grid,
            duration=50.0,
        )

        self.assertGreater(adjustments, 0)
        self.assertEqual(aligned[0].startSeconds, 0.0)
        self.assertEqual(aligned[-1].endSeconds, 50.0)
        self.assertEqual(aligned[0].endSeconds, aligned[1].startSeconds)
        self.assertEqual(aligned[1].endSeconds, aligned[2].startSeconds)
        for section in aligned[1:]:
            distance = min(
                abs(section.startSeconds - value)
                for value in self.grid.beat_times
            )
            self.assertLess(distance, 1e-7)

    def test_grid_aligned_boundary_does_not_create_zero_length_unknown_slot(self) -> None:
        sections = [
            SectionStructureDraft(
                id="left",
                name="左",
                type="verse",
                startSeconds=0.0,
                endSeconds=50.0,
            ),
            SectionStructureDraft(
                id="right",
                name="右",
                type="chorus",
                startSeconds=50.0,
                endSeconds=70.0,
            ),
        ]
        aligned, _ = align_sections_to_grid(
            sections,
            grid=self.grid,
            duration=70.0,
        )

        for section in aligned:
            output = consensus_section(
                section=section,
                specialists=[],
                grid=self.grid,
            )
            self.assertTrue(output.chords)
            self.assertTrue(
                all(chord.endSeconds > chord.startSeconds for chord in output.chords)
            )
            self.assertTrue(
                all(item.endSeconds > item.startSeconds for item in output.uncertain_ranges)
            )

    def test_oversized_chorus_is_deterministically_split(self) -> None:
        sections = [
            SectionStructureDraft(
                id="intro1",
                name="イントロ1",
                type="intro",
                startSeconds=0.0,
                endSeconds=25.0,
            ),
            SectionStructureDraft(
                id="intro2",
                name="イントロ2",
                type="intro",
                startSeconds=25.0,
                endSeconds=50.0,
            ),
            SectionStructureDraft(
                id="chorus1a",
                name="サビ1a",
                type="chorus",
                startSeconds=50.0,
                endSeconds=101.0,
                key="F#",
                mode="minor",
                confidence=0.9,
            ),
        ]

        refined, _ = refine_sections(
            sections,
            grid=self.grid,
            duration=101.0,
            model_refinements={"chorus1a": StructureRefinementDraft()},
        )

        chorus_children = [item for item in refined if item.id.startswith("chorus1a")]
        self.assertGreater(len(chorus_children), 1)
        self.assertEqual(refined[0].startSeconds, 0.0)
        self.assertEqual(refined[-1].endSeconds, 101.0)
        self.assertTrue(
            all(
                section_bar_count(section, self.grid)
                <= maximum_section_bars(section.type) + 0.25
                for section in refined
            )
        )
        self.assertEqual(len({section.id for section in refined}), len(refined))

    def test_analysis_slices_never_exceed_sixteen_bars(self) -> None:
        section = SectionStructureDraft(
            id="long-interlude",
            name="長い間奏",
            type="interlude",
            startSeconds=0.0,
            endSeconds=60.0,
        )
        slices = build_analysis_slices(
            [section],
            grid=self.grid,
            max_bars=16,
        )
        self.assertGreater(len(slices), 1)
        self.assertTrue(
            all(
                item.section.endSeconds - item.section.startSeconds
                <= self.grid.beat_duration * self.grid.beats_per_bar * 16 + 0.01
                for item in slices
            )
        )


if __name__ == "__main__":
    unittest.main()
