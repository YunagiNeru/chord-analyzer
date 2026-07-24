from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from music_agent.pipeline_v2 import AccuracyPipelineV2
from music_agent.schemas import (
    CompactChordDraft,
    CompactResolutionDraft,
    CompactSpecialistDraft,
    FinalExplanationDraft,
    RhythmDraft,
    SectionStructureDraft,
    StructureDraft,
    TempoCandidate,
    TrackStructureDraft,
)
from music_agent.youtube_metadata import YouTubeMetadata


class _BaseGateway:
    def __init__(self, *, diagnostics, model, **_: object) -> None:
        self.diagnostics = diagnostics
        self.model = model

    def _global(self, schema):
        if schema is StructureDraft:
            return StructureDraft(
                track=TrackStructureDraft(
                    title="Test",
                    artist="Test",
                    durationSeconds=8.0,
                    globalKey="C",
                    globalMode="major",
                    confidence=0.9,
                ),
                sections=[
                    SectionStructureDraft(
                        id="section-1",
                        name="Section",
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
        if schema is FinalExplanationDraft:
            return FinalExplanationDraft(musicalSummary="Test")
        return None

    def generate_text(self, **_: object):
        raise AssertionError("reference research must be disabled")


class SpecialistFailureGateway(_BaseGateway):
    def generate_typed(self, *, schema, **_: object):
        self.diagnostics.record_model_call(self.model)
        global_value = self._global(schema)
        if global_value is not None:
            return global_value
        if schema is CompactSpecialistDraft:
            raise RuntimeError("EOF while parsing specialist JSON")
        raise AssertionError(f"unexpected schema: {schema}")


class ResolverFailureGateway(_BaseGateway):
    call_index = 0

    def generate_typed(self, *, schema, **_: object):
        self.diagnostics.record_model_call(self.model)
        global_value = self._global(schema)
        if global_value is not None:
            return global_value
        if schema is CompactSpecialistDraft:
            symbols = ("C", "Cm", "G")
            symbol = symbols[ResolverFailureGateway.call_index % len(symbols)]
            ResolverFailureGateway.call_index += 1
            return CompactSpecialistDraft(
                chords=[
                    CompactChordDraft(
                        symbol=symbol,
                        startSeconds=0.0,
                        endSeconds=8.0,
                        confidence=0.9,
                        alternatives=["C", "Cm", "G"],
                    )
                ]
            )
        if schema is CompactResolutionDraft:
            raise RuntimeError("EOF while parsing resolver JSON")
        raise AssertionError(f"unexpected schema: {schema}")


class RequiredModelFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        ResolverFailureGateway.call_index = 0
        self.metadata = YouTubeMetadata(
            video_id="test",
            canonical_url="https://www.youtube.com/watch?v=test",
            title="Test",
            channel_title="Test",
            duration_seconds=8.0,
            metadata_source="test",
        )

    def _pipeline(self) -> AccuracyPipelineV2:
        return AccuracyPipelineV2(
            client=MagicMock(),
            model="test-model",
            max_parallel_calls=1,
        )

    @patch("music_agent.pipeline_v2.resolve_youtube_metadata")
    def test_required_specialist_failure_rejects_result(self, metadata_mock) -> None:
        metadata_mock.return_value = self.metadata
        with patch("music_agent.pipeline_v2.ModelGateway", SpecialistFailureGateway), patch.dict(
            os.environ,
            {"ENABLE_REFERENCE_RESEARCH": "0", "MAX_RESOLVER_CALLS": "0"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "required_model_failure:specialist"):
                self._pipeline().analyze_youtube(url="https://youtu.be/test")

    @patch("music_agent.pipeline_v2.resolve_youtube_metadata")
    def test_required_resolver_failure_rejects_result(self, metadata_mock) -> None:
        metadata_mock.return_value = self.metadata
        with patch("music_agent.pipeline_v2.ModelGateway", ResolverFailureGateway), patch.dict(
            os.environ,
            {"ENABLE_REFERENCE_RESEARCH": "0", "MAX_RESOLVER_CALLS": "1"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "required_model_failure:resolver"):
                self._pipeline().analyze_youtube(url="https://youtu.be/test")


if __name__ == "__main__":
    unittest.main()
