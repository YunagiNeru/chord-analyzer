from __future__ import annotations

import unittest

from music_agent.accuracy_v2_finalization import _split_uncertain_ranges
from music_agent.accuracy_v2_release_recovery import (
    _group_resolver_work,
    _prune_ambiguous_source_facts,
    _stable_reconcile_result_harmony,
    _validate_quality_with_semantic_limit,
)
from music_agent.consensus import ConsensusOutput
from music_agent.reference_research import ReferenceResearchResult
from music_agent.schemas import (
    AnalysisResult,
    ChordEvent,
    Measure,
    ReferenceSource,
    SectionResult,
    SectionStructureDraft,
    TrackResult,
    UncertainRange,
)


class AccuracyV2ReleaseRecoveryTests(unittest.TestCase):
    def _resolver_case(self, section_id: str, offset: float):
        symbols = ["D", "E", "C#m", "F#m"]
        chords = [
            ChordEvent(
                symbol=symbol,
                startSeconds=offset + index,
                endSeconds=offset + index + 1.0,
                confidence=0.45,
                agreement=0.4,
                alternatives=["C#", "C#m"],
                source="ai",
            )
            for index, symbol in enumerate(symbols)
        ]
        section = SectionStructureDraft(
            id=section_id,
            name=section_id,
            type="chorus",
            startSeconds=offset,
            endSeconds=offset + 4.0,
        )
        output = ConsensusOutput(
            chords=chords,
            agreement=0.4,
            uncertain_ranges=[
                UncertainRange(
                    sectionId=section_id,
                    startSeconds=offset + 2.0,
                    endSeconds=offset + 3.0,
                    reason="low agreement",
                    candidates=["C#m", "C#"],
                )
            ],
            slot_alternatives=[],
            slot_scores=[],
        )
        return section, output

    def test_repeated_state_contexts_share_one_resolver_group(self) -> None:
        work_items = []
        for section_id, offset in (("chorus-a", 0.0), ("chorus-b", 8.0)):
            section, output = self._resolver_case(section_id, offset)
            _, work = _split_uncertain_ranges(section, output)
            work_items.extend(work)

        groups = _group_resolver_work(work_items, beat_duration=0.5)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].members), 2)
        self.assertEqual(groups[0].coverage_seconds, 2.0)

    @staticmethod
    def _section(section_id: str, start: float, bars: float) -> SectionResult:
        bpm = 170.0
        bar_duration = (60.0 / bpm) * 4.0
        end = start + bars * bar_duration
        symbols = ["D", "E", "C#m", "F#m"] * 2
        span = (end - start) / len(symbols)
        chords = [
            ChordEvent(
                symbol=symbol,
                startSeconds=start + index * span,
                endSeconds=start + (index + 1) * span,
                confidence=0.9,
                agreement=0.9,
                source="ai",
            )
            for index, symbol in enumerate(symbols)
        ]
        return SectionResult(
            id=section_id,
            name=section_id,
            type="chorus",
            startSeconds=start,
            endSeconds=end,
            confidence=0.9,
            agreement=0.9,
            measures=[
                Measure(
                    bar=1,
                    startSeconds=start,
                    endSeconds=end,
                    chords=chords,
                )
            ],
        )

    def test_user_facing_section_may_exceed_internal_slice_limit(self) -> None:
        section = self._section("long-chorus", 0.0, 24.0)
        result = AnalysisResult(
            track=TrackResult(
                durationSeconds=section.endSeconds,
                bpm=170.0,
                timeSignature="4/4",
            ),
            sections=[section],
            sourceType="youtube",
            sourceLabel="fixture",
            analysisMethod="ai_only",
        )
        errors = _validate_quality_with_semantic_limit(result)
        self.assertFalse(any(item.startswith("oversized_section:") for item in errors))

        too_long = self._section("too-long-chorus", 0.0, 33.0)
        result.sections = [too_long]
        result.track.durationSeconds = too_long.endSeconds
        errors = _validate_quality_with_semantic_limit(result)
        self.assertTrue(any(item.startswith("oversized_section:") for item in errors))

    def test_global_key_prior_prevents_every_phrase_becoming_c_sharp_minor(self) -> None:
        sections = []
        cursor = 0.0
        for index in range(4):
            symbols = ["D", "E", "C#m", "F#m"]
            chords = []
            for symbol in symbols:
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
            sections.append(
                SectionResult(
                    id=f"section-{index}",
                    name=f"section-{index}",
                    type="chorus",
                    startSeconds=chords[0].startSeconds,
                    endSeconds=chords[-1].endSeconds,
                    key="C#",
                    mode="minor",
                    confidence=0.9,
                    agreement=0.9,
                    measures=[
                        Measure(
                            bar=index + 1,
                            startSeconds=chords[0].startSeconds,
                            endSeconds=chords[-1].endSeconds,
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
                globalKey="F#",
                globalMode="minor",
            ),
            sections=sections,
            sourceType="youtube",
            sourceLabel="fixture",
            analysisMethod="ai_only",
        )
        _stable_reconcile_result_harmony(result)
        self.assertEqual(
            (result.track.globalKey, result.track.globalMode),
            ("F#", "minor"),
        )
        self.assertTrue(
            all((section.key, section.mode) == ("F#", "minor") for section in result.sections)
        )

    def test_ambiguous_aggregate_key_facts_are_removed_per_source(self) -> None:
        result = ReferenceResearchResult(
            sources=[
                ReferenceSource(
                    title="ambiguous",
                    url="https://example.test/ambiguous",
                    sourceType="google-search",
                    facts=[
                        "BPM=170",
                        "KEY=A minor",
                        "KEY=E minor",
                        "KEY=D minor",
                        "KEY=C# minor",
                    ],
                ),
                ReferenceSource(
                    title="specific",
                    url="https://example.test/specific",
                    sourceType="google-search",
                    facts=["BPM=170", "KEY=E minor"],
                ),
            ]
        )
        _prune_ambiguous_source_facts(result)
        self.assertEqual(result.sources[0].facts, ["BPM=170"])
        self.assertEqual(result.sources[1].facts, ["BPM=170", "KEY=E minor"])
        self.assertEqual(result.bpm_candidates, [170.0])
        self.assertEqual(result.bpm_support, {170.0: 2})
        self.assertEqual(result.key_candidates, ["E minor"])


if __name__ == "__main__":
    unittest.main()
