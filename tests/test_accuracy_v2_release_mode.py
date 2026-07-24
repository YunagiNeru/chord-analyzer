from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from music_agent.accuracy_v2_release_mode import (
    _append_degraded_warnings,
    _is_fatal_quality_error,
    _max_degraded_unresolved_ratio,
    _strict_quality_gate,
)
from music_agent.schemas import (
    AnalysisDiagnostics,
    AnalysisResult,
    ChordEvent,
    Measure,
    SectionResult,
    TrackResult,
    UncertainRange,
)


class AccuracyV2ReleaseModeTests(unittest.TestCase):
    def _result(self) -> AnalysisResult:
        chord = ChordEvent(
            symbol="C",
            startSeconds=0.0,
            endSeconds=4.0,
            confidence=0.8,
            agreement=0.8,
            source="ai",
        )
        return AnalysisResult(
            track=TrackResult(
                durationSeconds=4.0,
                bpm=120.0,
                timeSignature="4/4",
                globalKey="C",
                globalMode="major",
            ),
            sections=[
                SectionResult(
                    id="section",
                    name="Section",
                    type="chorus",
                    startSeconds=0.0,
                    endSeconds=4.0,
                    key="C",
                    mode="major",
                    measures=[
                        Measure(
                            bar=1,
                            startSeconds=0.0,
                            endSeconds=4.0,
                            chords=[chord],
                        )
                    ],
                )
            ],
            sourceType="youtube",
            sourceLabel="fixture",
            analysisMethod="ai_only",
            uncertainRanges=[
                UncertainRange(
                    sectionId="section",
                    startSeconds=0.0,
                    endSeconds=2.0,
                    reason="low agreement",
                    candidates=["C", "Am"],
                )
            ],
            diagnostics=AnalysisDiagnostics(),
        )

    def test_quality_gate_remains_strict_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRICT_QUALITY_GATE", None)
            self.assertTrue(_strict_quality_gate())

    def test_degraded_uncertainty_limit_is_bounded(self) -> None:
        with patch.dict(
            os.environ,
            {"MAX_DEGRADED_UNRESOLVED_RATIO": "0.40"},
            clear=False,
        ):
            self.assertEqual(_max_degraded_unresolved_ratio(), 0.40)
        with patch.dict(
            os.environ,
            {"MAX_DEGRADED_UNRESOLVED_RATIO": "9"},
            clear=False,
        ):
            self.assertEqual(_max_degraded_unresolved_ratio(), 0.60)

    def test_only_resolver_degradation_is_fail_soft(self) -> None:
        self.assertFalse(
            _is_fatal_quality_error("excessive_unresolved_coverage:0.352")
        )
        self.assertFalse(
            _is_fatal_quality_error(
                "required_model_failure:resolver_recovery_incomplete"
            )
        )
        self.assertTrue(
            _is_fatal_quality_error(
                "required_model_failure:resolver_state_collapse"
            )
        )
        self.assertTrue(_is_fatal_quality_error("no_known_chords"))

    def test_degraded_result_explains_uncertain_coverage(self) -> None:
        result = self._result()
        _append_degraded_warnings(
            result,
            ["excessive_unresolved_coverage:0.500"],
        )
        self.assertTrue(any("50.0%" in item for item in result.warnings))
        self.assertTrue(any("自動採譜の下書き" in item for item in result.warnings))
        self.assertTrue(result.limitations)


if __name__ == "__main__":
    unittest.main()
