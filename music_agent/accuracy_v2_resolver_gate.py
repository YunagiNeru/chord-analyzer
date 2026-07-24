from __future__ import annotations

from . import accuracy_v2_resolver_recovery as resolver_recovery
from . import pipeline_v2 as pipeline_module


def _patched_run_resolvers(self, **kwargs):
    diagnostics = kwargs["diagnostics"]
    duration = float(kwargs["duration"])
    ranges = resolver_recovery._patched_run_resolvers(self, **kwargs)

    unresolved_coverage = sum(
        item.endSeconds - item.startSeconds
        for item in ranges
        if not item.resolved
    )
    unresolved_ratio = unresolved_coverage / duration if duration > 0.0 else 0.0
    model_errors = diagnostics.build([]).modelErrors
    resolver_errors = [
        item
        for item in model_errors
        if item.startswith("resolver-")
    ]

    if resolver_errors and unresolved_ratio > 0.30:
        diagnostics.record_required_failure(
            f"resolver_recovery_incomplete:{unresolved_ratio:.3f}"
        )
    return ranges


def install_accuracy_v2_resolver_gate() -> None:
    if getattr(pipeline_module, "_accuracy_v2_resolver_gate_installed", False):
        return
    pipeline_module._accuracy_v2_resolver_gate_installed = True
    pipeline_module.AccuracyPipelineV2._run_resolvers = _patched_run_resolvers
