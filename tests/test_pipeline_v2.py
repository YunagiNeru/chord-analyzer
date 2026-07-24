from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from music_agent.pipeline_v2 import AccuracyPipelineV2
from music_agent.schemas import (
    CompactChordDraft,
    CompactSpecialistDraft,
    FinalExplanationDraft,
    RhythmDraft,
    SectionStructureDraft,
    StructureDraft,
    TempoCandidate,
    TrackStructureDraft,
)
from music_agent.youtube_metadata import YouTubeMetadata


class FakeGateway:
    def __init__(self, *, diagnostics, model, **_: object) -> None:
        self.diagnostics = diagnostics
        self.model = model

    def generate_typed(self, *, contents, schema, **_: object):
        self.diagnostics.record_model_call(self.model)
        if schema is StructureDraft:
            return StructureDraft(
                track=TrackStructureDraft(
                    title="Model title",
                    artist="Model artist",
                    durationSeconds=8.0,
                    bpm=120.0,
                    timeSignature="4/4",
                    globalKey="C",
                    globalMode="major",
                    confidence=0.9,
                ),
                sections=[
                    SectionStructureDraft(
                        id="section-1",
                        name="サビ1",
                        type="chorus",
                        startSeconds=0.0,
                        endSeconds=8.0,
                        key="C",
                        mode="major",
                        confidence=0.9,
                    )
                ],
            )
        if schema is RhythmDraft:
            return RhythmDraft(
                durationSeconds=8.0,
                bpmCandidates=[TempoCandidate(bpm=120.0, confidence=0.95)],
                selectedBpm=120.0,
                timeSignature="4/4",
                downbeatOffsetSeconds=0.0,
                globalKey="C",
                globalMode="major",
                confidence=0.95,
            )
        if schema is CompactSpecialistDraft:
            prompt = json.loads(str(contents[-1]).split("\n", 1)[1])
            self.assert_prompt_is_compact(prompt)
            return CompactSpecialistDraft(
                chords=[
                    CompactChordDraft(
                        symbol="C",
                        startSeconds=0.0,
                        endSeconds=2.0,
                        confidence=0.95,
                    ),
                    CompactChordDraft(
                        symbol="G",
                        startSeconds=2.0,
                        endSeconds=4.0,
                        confidence=0.95,
                    ),
                    CompactChordDraft(
                        symbol="Am",
                        startSeconds=4.0,
                        endSeconds=6.0,
                        confidence=0.95,
                    ),
                    CompactChordDraft(
                        symbol="F",
                        startSeconds=6.0,
                        endSeconds=8.0,
                        confidence=0.95,
                    ),
                ],
                repeatedPattern=["C", "G", "Am", "F"],
            )
        if schema is FinalExplanationDraft:
            return FinalExplanationDraft(
                musicalSummary="Deterministic test result.",
                sectionSummaries={"section-1": "Test section."},
            )
        raise AssertionError(f"unexpected schema: {schema}")

    @staticmethod
    def assert_prompt_is_compact(prompt: dict[str, object]) -> None:
        output = prompt.get("output")
        assert isinstance(output, dict)
        assert output["noProse"] is True
        assert "rhythm" not in prompt

    def generate_text(self, **_: object):
        raise AssertionError("reference research must be disabled in this test")


class PipelineV2Tests(unittest.TestCase):
    @patch("music_agent.pipeline_v2.resolve_youtube_metadata")
    @patch("music_agent.pipeline_v2.ModelGateway", FakeGateway)
    def test_complete_youtube_pipeline(self, metadata_mock) -> None:
        metadata_mock.return_value = YouTubeMetadata(
            video_id="9QLT1Aw_45s",
            canonical_url="https://www.youtube.com/watch?v=9QLT1Aw_45s",
            title="Official title",
            channel_title="Official channel",
            duration_seconds=8.0,
            metadata_source="test",
        )
        with patch.dict(
            os.environ,
            {
                "ENABLE_REFERENCE_RESEARCH": "0",
                "MAX_RESOLVER_CALLS": "0",
            },
            clear=False,
        ):
            pipeline = AccuracyPipelineV2(
                client=MagicMock(),
                model="test-model",
                max_parallel_calls=4,
            )
            result = pipeline.analyze_youtube(
                url="https://youtu.be/9QLT1Aw_45s"
            )

        self.assertEqual(result.analysisVersion, "2.0")
        self.assertEqual(result.track.title, "Official title")
        self.assertEqual(result.track.artist, "Official channel")
        self.assertEqual(result.track.durationSeconds, 8.0)
        self.assertEqual(result.track.bpm, 120.0)
        self.assertEqual(result.diagnostics.invariantErrors, [])
        self.assertEqual(result.diagnostics.requiredModelFailures, [])
        self.assertEqual(result.diagnostics.analysisSliceCount, 1)
        self.assertEqual(result.diagnostics.chordStateCount, 4)
        self.assertEqual(result.diagnostics.displayChordEventCount, 4)
        chords = [
            chord.symbol
            for section in result.sections
            for measure in section.measures
            for chord in measure.chords
        ]
        self.assertEqual(chords, ["C", "G", "Am", "F"])


if __name__ == "__main__":
    unittest.main()
