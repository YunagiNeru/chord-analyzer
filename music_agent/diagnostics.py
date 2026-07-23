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
    _lock: threading.Lock = field(default_factory=threading.Lock)
    section_specialist_calls: int = 0
    resolver_calls: int = 0
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

    def build(self, invariant_errors: list[str]) -> AnalysisDiagnostics:
        return AnalysisDiagnostics(
            pipeline=self.pipeline,
            stageTimings=list(self._timings),
            modelUsage=sorted(self._usage.values(), key=lambda item: item.model),
            invariantErrors=list(invariant_errors),
            sectionSpecialistCalls=self.section_specialist_calls,
            resolverCalls=self.resolver_calls,
            gridScore=self.grid_score,
        )
