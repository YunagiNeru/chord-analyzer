from __future__ import annotations

import unittest

from music_agent.accuracy_v2_finalization import ResolverWork
from music_agent.accuracy_v2_finalization_v2 import (
    build_resolver_batches,
    deterministic_semantic_fallback,
    finalize_result_before_quality,
)
from music_agent.beat_grid import BeatGrid
from music_agent.consensus import ConsensusOutput
from music_agent.schemas import (
    AnalysisDiagnostics,
    AnalysisResult,
    ChordEvent,
    Measure,
    ReferenceSource,
    SectionResult,
    SectionStructureDraft,
    TrackResult,
    UncertainRange,
)
from music_agent.structure_refine import maximum_section_bars, section_bar_count


class AccuracyV2FinalizationV2Tests(unittest.TestCase):
    def _grid(self) -> BeatGrid:
        bpm = 170.0
        beat = 60.0 / bpm
        duration = 190.0
        return BeatGrid(
            bpm=bpm,
            time_signature="4/4",
            beats_per_bar=4,
            beat_duration=beat,
            downbeat_offset=0.0,
            duration=duration,
            beat_times=tuple(round(index * beat, 6) for index in range(540)),
            bar_starts=tuple(round(index * beat * 4, 6) for index in range(136)),
            score=0.94,
        )

    def test_94_states_fit_in_12_batches(self) -> None:
        section = SectionStructureDraft(
            id="whole",
            name="whole",
            type="other",
            startSeconds=0.0,
            endSeconds=94.0,
        )
        chords = [
            ChordEvent(
                symbol="Am" if index % 2 == 0 else "F",
                startSeconds=float(index),
                endSeconds=float(index + 1),
                confidence=0.45,
                agreement=0.4,
                alternatives=["C"],
                source="ai",
            )
            for index in range(94)
        ]
        output = ConsensusOutput(
            chords=chords,
            agreement=0.4,
            uncertain_ranges=[],
            slot_alternatives=[],
            slot_scores=[],
        )
        work = [
            ResolverWork(
                section=section,
                output=output,
                chord_index=index,
                target=UncertainRange(
                    sectionId=section.id,
                    startSeconds=float(index),
                    endSeconds=float(index + 1),
                    reason="low agreement",
                    candidates=[chord.symbol, "C"],
                ),
            )
            for index, chord in enumerate(chords)
        ]

        batches = build_resolver_batches(work, batch_size=8, max_batches=12)

        self.assertEqual(len(batches), 12)
        self.assertEqual(sum(len(batch) for batch in batches), 94)
        self.assertTrue(all(len(batch) <= 8 for batch in batches))

    def test_oversized_chorus_gets_deterministic_fallback(self) -> None:
        grid = self._grid()
        parent = SectionStructureDraft(
            id="chorus3",
            name="サビ3",
            type="chorus",
            startSeconds=157.720592,
            endSeconds=190.0,
            confidence=0.9,
        )

        children = deterministic_semantic_fallback(parent, grid=grid)

        self.assertGreaterEqual(len(children), 2)
        self.assertEqual(children[0].startSeconds, parent.startSeconds)
        self.assertEqual(children[-1].endSeconds, parent.endSeconds)
        self.assertTrue(
            all(
                section_bar_count(child, grid)
                <= maximum_section_bars(child.type) + 0.25
                for child in children
            )
        )

    def test_harmony_is_finalized_before_quality_gate(self) -> None:
        chords = [
            ChordEvent(
                symbol=symbol,
                startSeconds=float(index),
                endSeconds=float(index + 1),
                confidence=0.9,
                agreement=0.9,
                source="ai",
            )
            for index, symbol in enumerate(["F#m", "D", "A", "E"])
        ]
        result = AnalysisResult(
            track=TrackResult(
                durationSeconds=4.0,
                bpm=170.0,
                timeSignature="4/4",
                globalKey="C#",
                globalMode="minor",
            ),
            sections=[
                SectionResult(
                    id="chorus",
                    name="サビ",
                    type="chorus",
                    startSeconds=0.0,
                    endSeconds=4.0,
                    key="C#",
                    mode="minor",
                    measures=[
                        Measure(
                            bar=1,
                            startSeconds=0.0,
                            endSeconds=4.0,
                            chords=chords,
                        )
                    ],
                )
            ],
            sourceType="youtube",
            sourceLabel="fixture",
            analysisMethod="ai_only",
            referenceSources=[
                ReferenceSource(
                    title="fixture",
                    url="https://example.com",
                    facts=["KEY=F# minor"],
                )
            ],
            diagnostics=AnalysisDiagnostics(),
        )

        finalize_result_before_quality(result)

        self.assertEqual((result.track.globalKey, result.track.globalMode), ("F#", "minor"))
        self.assertEqual((result.sections[0].key, result.sections[0].mode), ("F#", "minor"))
        self.assertTrue(all(chord.roman for chord in chords))


if __name__ == "__main__":
    unittest.main()
