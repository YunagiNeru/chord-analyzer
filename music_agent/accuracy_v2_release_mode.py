from __future__ import annotations

import os

from . import accuracy_v2_finalization_v2 as finalization_v2
from . import pipeline_v2 as pipeline_module
from .harmonic_reconcile import validate_tempo_coverage


_FATAL_QUALITY_PREFIXES = (
    "no_known_chords",
    "too_few_chords",
    "insufficient_known_chord_coverage",
    "structure_fallback_present",
    "semantic_sections_merged_by_limit",
    "implausible_static_section",
    "oversized_section",
    "section_boundary_off_beat",
    "required_model_failure:specialist",
    "required_model_failure:structure",
    "required_model_failure:global",
    "required_model_failure:rhythm",
    "required_model_failure:resolver_state_collapse",
)
_DEGRADABLE_QUALITY_PREFIXES = (
    "excessive_unresolved_coverage",
    "required_model_failure:resolver-batch",
    "required_model_failure:resolver_group",
    "required_model_failure:resolver-group",
    "required_model_failure:resolver_recovery",
)


def _strict_quality_gate() -> bool:
    return os.environ.get("STRICT_QUALITY_GATE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _max_degraded_unresolved_ratio() -> float:
    try:
        value = float(os.environ.get("MAX_DEGRADED_UNRESOLVED_RATIO", "0.40"))
    except (TypeError, ValueError):
        value = 0.40
    return max(0.30, min(0.60, value))


def _is_fatal_quality_error(error: str) -> bool:
    if error.startswith(_FATAL_QUALITY_PREFIXES):
        return True
    if error.startswith(_DEGRADABLE_QUALITY_PREFIXES):
        return False
    return True


def _unresolved_ratio(result) -> float:
    duration = float(result.track.durationSeconds or 0.0)
    if duration <= 0.0:
        return 0.0
    seconds = sum(
        item.endSeconds - item.startSeconds
        for item in result.uncertainRanges
        if not item.resolved
    )
    return max(0.0, seconds / duration)


def _append_degraded_warnings(result, errors: list[str]) -> None:
    ratio = _unresolved_ratio(result)
    warnings = list(result.warnings)
    limitations = list(result.limitations)
    if ratio > 0.0:
        warnings.append(
            f"低信頼または未確定の区間が全体の{ratio * 100:.1f}%あります。"
            "該当コードは候補と信頼度を確認してください。"
        )
    if any(error.startswith("required_model_failure:resolver") for error in errors):
        warnings.append(
            "一部のコード再判定はVertex AIの一時的な応答不良により完了しませんでした。"
            "確定済みの区間と未確定候補を保持した結果を表示しています。"
        )
    warnings.append(
        "この結果は自動採譜の下書きです。低信頼区間は音源を再生して確認してください。"
    )
    limitations.append(
        "本番モードでは、時間軸と既知コードの品質を満たし、未確定率が許容上限以内の場合に限り、"
        "resolverの一時障害が残っても未確定区間を明示して結果を返します。"
    )
    result.warnings = list(dict.fromkeys(warnings))
    result.limitations = list(dict.fromkeys(limitations))


def _release_run(self, *args, **kwargs):
    finalization_v2._CONTEXT.defer_quality = True
    try:
        result = finalization_v2._ORIGINAL_RUN(self, *args, **kwargs)
    finally:
        finalization_v2._CONTEXT.defer_quality = False

    finalization_v2.finalize_result_before_quality(result)
    invariant_errors = pipeline_module.validate_invariants(result)
    invariant_errors.extend(validate_tempo_coverage(result))
    invariant_errors = sorted(set(invariant_errors))

    validator = finalization_v2._ORIGINAL_VALIDATE_QUALITY
    if validator is None:
        raise RuntimeError("quality validator was not installed")
    quality_errors = list(validator(result))
    required_failures = (
        result.diagnostics.requiredModelFailures
        if result.diagnostics is not None
        else []
    )
    quality_errors.extend(
        f"required_model_failure:{pipeline_module._compact_failure_label(item)}"
        for item in required_failures
    )
    quality_errors = sorted(set(quality_errors))
    combined_errors = invariant_errors + [
        f"quality:{item}" for item in quality_errors
    ]
    if result.diagnostics is not None:
        result.diagnostics.invariantErrors = combined_errors

    if invariant_errors:
        raise RuntimeError(
            "Accuracy v2 invariant violation: " + ", ".join(invariant_errors)
        )

    strict = _strict_quality_gate()
    fatal_errors = (
        quality_errors
        if strict
        else [error for error in quality_errors if _is_fatal_quality_error(error)]
    )
    if not strict and _unresolved_ratio(result) > _max_degraded_unresolved_ratio():
        fatal_errors.append(
            f"excessive_unresolved_coverage:{_unresolved_ratio(result):.3f}"
        )
    fatal_errors = sorted(set(fatal_errors))
    if fatal_errors:
        raise RuntimeError(
            "Accuracy v2 quality gate violation: " + ", ".join(fatal_errors)
        )
    if quality_errors:
        _append_degraded_warnings(result, quality_errors)
    return result


def install_accuracy_v2_release_mode() -> None:
    if getattr(pipeline_module, "_accuracy_v2_release_mode_installed", False):
        return
    pipeline_module._accuracy_v2_release_mode_installed = True
    pipeline_module.AccuracyPipelineV2._run = _release_run
