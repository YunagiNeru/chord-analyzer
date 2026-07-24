from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from music_agent.accuracy_v2_finalization_v2 import (
    BatchResolutionChoice,
    BatchResolutionDraft,
)
from music_agent.accuracy_v2_resolver_recovery import _patched_run_resolvers
from music_agent.beat_grid import BeatGrid
from music_agent.consensus import ConsensusOutput
from music_agent.diagnostics import DiagnosticsRecorder
from music_agent.schemas import ChordEvent, SectionStructureDraft, UncertainRange


class _Media:
    def part(self, start: float = 0.0, end: float | None = None):
        return (start, end)


class _FailLargeThenSucceedGateway:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def generate_typed(self, *, contents, diagnostic_label: str, **kwargs):
        del kwargs
        self.labels.append(diagnostic_label)
        prompt = contents[1]
        targets = json.loads(prompt.split("targets=", 1)[1])
        if "recovery" not in diagnostic_label:
            raise ValueError("truncated batch JSON")
        return BatchResolutionDraft(
            choices=[
                BatchResolutionChoice(
                    targetIndex=item["targetIndex"],
                    chosenSymbol=item["currentSymbol"],
                    confidence=0.9,
                )
                for item in targets
            ]
        )


class _AlwaysFailGateway:
    def generate_typed(self, **kwargs):
        del kwargs
        raise ValueError("truncated batch JSON")


class AccuracyV2ResolverRecoveryTests(unittest.TestCase):
    def _grid(self) -> BeatGrid:
        beat = 0.5
        return BeatGrid(
            bpm=120.0,
            time_signature="4/4",
            beats_per_bar=4,
            beat_duration=beat,
            downbeat_offset=0.0,
            duration=4.0,
            beat_times=tuple(index * beat for index in range(9)),
            bar_starts=(0.0, 2.0, 4.0),
            score=0.9,
        )

    def _pair(self):
        section = SectionStructureDraft(
            id="verse1",
            name="Aメロ1",
            type="verse",
            startSeconds=0.0,
            endSeconds=4.0,
            key="A",
            mode="minor",
        )
        chords = [
            ChordEvent(
                symbol=symbol,
                startSeconds=float(index),
                endSeconds=float(index + 1),
                confidence=0.45,
                agreement=0.4,
                alternatives=["C"],
                source="ai",
            )
            for index, symbol in enumerate(["Am", "F", "C", "G"])
        ]
        output = ConsensusOutput(
            chords=chords,
            agreement=0.4,
            uncertain_ranges=[
                UncertainRange(
                    sectionId=section.id,
                    startSeconds=0.0,
                    endSeconds=4.0,
                    reason="low agreement",
                    candidates=["Am", "F", "C", "G"],
                )
            ],
            slot_alternatives=[],
            slot_scores=[],
        )
        return section, output

    def test_failed_large_batch_is_retried_as_smaller_batches(self) -> None:
        section, output = self._pair()
        diagnostics = DiagnosticsRecorder()
        gateway = _FailLargeThenSucceedGateway()
        warnings: list[str] = []

        with patch.dict(
            "os.environ",
            {
                "RESOLVER_BATCH_SIZE": "8",
                "MAX_RESOLVER_RECOVERY_CALLS": "4",
            },
        ):
            ranges = _patched_run_resolvers(
                SimpleNamespace(max_resolver_calls=1, max_parallel_calls=1),
                media=_Media(),
                resolver_gateway=gateway,
                consensus_pairs=[(section, output)],
                grid=self._grid(),
                duration=4.0,
                diagnostics=diagnostics,
                global_warnings=warnings,
            )

        self.assertEqual(diagnostics.resolver_calls, 3)
        self.assertEqual(len(gateway.labels), 3)
        self.assertTrue(all(item.resolved for item in ranges))
        self.assertEqual(diagnostics.required_failures, [])

    def test_exhausted_recovery_is_left_to_quality_gate(self) -> None:
        section, output = self._pair()
        diagnostics = DiagnosticsRecorder()
        warnings: list[str] = []

        with patch.dict(
            "os.environ",
            {
                "RESOLVER_BATCH_SIZE": "8",
                "MAX_RESOLVER_RECOVERY_CALLS": "1",
            },
        ):
            ranges = _patched_run_resolvers(
                SimpleNamespace(max_resolver_calls=1, max_parallel_calls=1),
                media=_Media(),
                resolver_gateway=_AlwaysFailGateway(),
                consensus_pairs=[(section, output)],
                grid=self._grid(),
                duration=4.0,
                diagnostics=diagnostics,
                global_warnings=warnings,
            )

        self.assertTrue(any(not item.resolved for item in ranges))
        self.assertEqual(diagnostics.required_failures, [])
        self.assertTrue(diagnostics.build([]).modelErrors)
        self.assertTrue(any("recovery budget exhausted" in item for item in warnings))


if __name__ == "__main__":
    unittest.main()
