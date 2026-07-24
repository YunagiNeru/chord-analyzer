from __future__ import annotations

import json
import unittest
from pathlib import Path

from music_agent.accuracy_v2_finalization import (
    SemanticRefinedSectionDraft,
    SemanticStructureRefinementDraft,
    _apply_state_resolution,
    _normalise_semantic_refinement,
    _split_uncertain_ranges,
    _state_count,
)
from music_agent.beat_grid import BeatGrid
from music_agent.consensus import ConsensusOutput
from music_agent.harmonic_reconcile import (
    normalize_tempo_segments,
    reconcile_result_harmony,
    validate_tempo_coverage,
)
from music_agent.schemas import (
    AnalysisResult,
    ChordEvent,
    CompactResolutionDraft,
    Measure,
    SectionResult,
    SectionStructureDraft,
    TempoSegment,
    TrackResult,
    UncertainRange,
)


FIXTURE = Path(__file__).parent / "fixtures" / "phony_v2_final_regression.json"


class AccuracyV2FinalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def _grid(self) -> BeatGrid:
        bpm = 170.0
        beat = 60.0 / bpm
        duration = 190.0
        beat_times = tuple(round(index * beat, 6) for index in range(540))
        bar_times = tuple(round(index * beat * 4, 6) for index in range(136))
        return BeatGrid(
            bpm=bpm,
            time_signature="4/4",
            beats_per_bar=4,
            beat_duration=beat,
            downbeat_offset=0.0,
            duration=duration,
            beat_times=beat_times,
            bar_starts=bar_times,
            score=0.94,
        )

    def test_uncertain_range_is_split_and_one_decision_changes_one_state(self) -> None:
        case = self.fixture["resolverCollapseCase"]
        section = SectionStructureDraft(
            id=case["sectionId"],
            name="Aメロ1",
            type="verse",
            startSeconds=case["startSeconds"],
            endSeconds=case["endSeconds"],
        )
        chords = [
            ChordEvent(
                symbol=symbol,
                startSeconds=float(index),
                endSeconds=float(index + 1),
                confidence=0.45,
                agreement=0.4,
                alternatives=["Bmaj7"],
                source="ai",
            )
            for index, symbol in enumerate(case["symbols"])
        ]
        output = ConsensusOutput(
            chords=chords,
            agreement=0.4,
            uncertain_ranges=[
                UncertainRange(
                    sectionId=section.id,
                    startSeconds=section.startSeconds,
                    endSeconds=section.endSeconds,
                    reason="low agreement",
                    candidates=case["candidates"],
                )
            ],
            slot_alternatives=[],
            slot_scores=[],
        )

        ranges, work = _split_uncertain_ranges(section, output)
        self.assertEqual(len(ranges), len(chords))
        self.assertEqual(len(work), len(chords))

        _apply_state_resolution(
            work[0],
            CompactResolutionDraft(chosenSymbol="Bmaj7", confidence=0.9),
        )
        self.assertEqual(output.chords[0].symbol, "Bmaj7")
        self.assertEqual(
            [item.symbol for item in output.chords[1:]],
            case["symbols"][1:],
        )
        self.assertEqual(_state_count([output]), len(chords))

    def test_tempo_segment_is_normalized_to_selected_grid_and_duration(self) -> None:
        track = self.fixture["track"]
        source_segment = self.fixture["tempoSegments"][0]
        result = AnalysisResult(
            track=TrackResult(**track),
            sections=[],
            sourceType="youtube",
            sourceLabel="fixture",
            analysisMethod="ai_only",
            tempoSegments=[TempoSegment(**source_segment)],
        )
        normalize_tempo_segments(result)
        self.assertEqual(len(result.tempoSegments), 1)
        self.assertEqual(result.tempoSegments[0].bpm, 170.0)
        self.assertEqual(result.tempoSegments[0].startSeconds, 0.0)
        self.assertEqual(result.tempoSegments[0].endSeconds, 190.0)
        self.assertEqual(validate_tempo_coverage(result), [])

    def test_harmony_reconciliation_detects_front_and_back_minor_centres(self) -> None:
        sections = []
        cursor = 0.0
        for fixture_section in self.fixture["keySections"]:
            chords = []
            for symbol in fixture_section["symbols"]:
                chords.append(
                    ChordEvent(
                        symbol=symbol,
                        startSeconds=cursor,
                        endSeconds=cursor + 1.0,
                        confidence=0.9,
                        agreement=0.9,
                        source="ai",
                    )
                )
                cursor += 1.0
            section_start = chords[0].startSeconds
            section_end = chords[-1].endSeconds
            sections.append(
                SectionResult(
                    id=fixture_section["id"],
                    name=fixture_section["id"],
                    type="chorus",
                    startSeconds=section_start,
                    endSeconds=section_end,
                    key=f"{fixture_section['key']} {fixture_section['mode']}",
                    mode=fixture_section["mode"],
                    measures=[
                        Measure(
                            bar=len(sections) + 1,
                            startSeconds=section_start,
                            endSeconds=section_end,
                            chords=chords,
                        )
                    ],
                )
            )

        result = AnalysisResult(
            track=TrackResult(
                durationSeconds=cursor,
                bpm=170.0,
                timeSignature="4/4",
                globalKey="Ab minor",
                globalMode="minor",
            ),
            sections=sections,
            sourceType="youtube",
            sourceLabel="fixture",
            analysisMethod="ai_only",
        )
        reconcile_result_harmony(result)
        self.assertEqual((result.sections[0].key, result.sections[0].mode), ("F#", "minor"))
        self.assertEqual((result.sections[1].key, result.sections[1].mode), ("G#", "minor"))
        self.assertNotEqual(result.track.globalKey, "Ab")
        self.assertTrue(all(chord.roman for section in result.sections for measure in section.measures for chord in measure.chords))

    def test_generic_equal_split_is_not_accepted_as_semantic_structure(self) -> None:
        parent = SectionStructureDraft(
            id="chorus_1a",
            name="サビ1-A",
            type="chorus",
            startSeconds=0.0,
            endSeconds=45.0,
            confidence=0.9,
        )
        draft = SemanticStructureRefinementDraft(
            sections=[
                SemanticRefinedSectionDraft(
                    name="サビ1-A1",
                    type="chorus",
                    startSeconds=0.0,
                    endSeconds=22.5,
                    confidence=0.9,
                    summary="同じ要約",
                ),
                SemanticRefinedSectionDraft(
                    name="サビ1-A2",
                    type="chorus",
                    startSeconds=22.5,
                    endSeconds=45.0,
                    confidence=0.9,
                    summary="同じ要約",
                ),
            ]
        )
        self.assertEqual(
            _normalise_semantic_refinement(parent, draft, grid=self._grid()),
            [],
        )


if __name__ == "__main__":
    unittest.main()
