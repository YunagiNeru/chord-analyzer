from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from .schemas import AnalysisDiagnostics, ModelUsage, StageTiming


@dataclass(slots=True)
class DiagnosticsRecorder:
    pipeline: str = "accuracy-v2"
    _timings: list[StageTiming] = field(default_factory=list)
    _usage: dict[str, ModelUsage] = field(default_factory=dict)
    _errors: list[str] = field(default_factory=list)
    _required_failures: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    section_specialist_calls: int = 0
    resolver_calls: int = 0
    structure_refinement_calls: int = 0
    analysis_slice_count: int = 0
    chord_state_count: int = 0
    display_chord_event_count: int = 0
    boundary_adjustments: int = 0
    grid_score: float | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self._timings.append(StageTiming(stage=name, seconds=round(elapsed, 4)))

    def record_model_call(
        self,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        with self._lock:
            usage = self._usage.get(model)
            if usage is None:
                usage = ModelUsage(model=model)
                self._usage[model] = usage
            usage.calls += 1
            usage.inputTokens += max(0, int(input_tokens))
            usage.outputTokens += max(0, int(output_tokens))

    def record_error(self, label: str, exc: BaseException) -> None:
        detail = str(exc).replace("\n", " ").strip()
        if len(detail) > 600:
            detail = detail[:597] + "..."
        rendered = f"{label}: {type(exc).__name__}"
        if detail:
            rendered += f": {detail}"
        with self._lock:
            if rendered not in self._errors:
                self._errors.append(rendered)

    def record_required_failure(self, label: str, exc: BaseException | None = None) -> None:
        rendered = label
        if exc is not None:
            detail = str(exc).replace("\n", " ").strip()
            rendered += f": {type(exc).__name__}"
            if detail:
                rendered += f": {detail[:240]}"
        with self._lock:
            if rendered not in self._required_failures:
                self._required_failures.append(rendered)

    @property
    def required_failures(self) -> list[str]:
        with self._lock:
            return list(self._required_failures)

    def build(self, invariant_errors: list[str]) -> AnalysisDiagnostics:
        return AnalysisDiagnostics(
            pipeline=self.pipeline,
            stageTimings=list(self._timings),
            modelUsage=sorted(self._usage.values(), key=lambda item: item.model),
            invariantErrors=list(invariant_errors),
            modelErrors=list(self._errors),
            requiredModelFailures=list(self._required_failures),
            sectionSpecialistCalls=self.section_specialist_calls,
            resolverCalls=self.resolver_calls,
            structureRefinementCalls=self.structure_refinement_calls,
            analysisSliceCount=self.analysis_slice_count,
            chordStateCount=self.chord_state_count,
            displayChordEventCount=self.display_chord_event_count,
            boundaryAdjustments=self.boundary_adjustments,
            gridScore=self.grid_score,
        )
