from __future__ import annotations

import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from . import pipeline_v2 as pipeline_module
from .accuracy_v2_finalization import (
    ResolverWork,
    SemanticStructureRefinementDraft,
    _apply_state_resolution,
    _count_result_states,
    _key_runs,
    _normalise_semantic_refinement,
    _reference_key_hints,
    _semantic_prompt,
    _split_uncertain_ranges,
    _state_count,
)
from .diagnostics import DiagnosticsRecorder
from .harmonic_reconcile import (
    normalize_tempo_segments,
    reconcile_result_harmony,
    validate_tempo_coverage,
)
from .reference_attribution import research_references_attributed
from .structure_refine import (
    align_sections_to_grid,
    deterministic_split,
    oversized_sections,
)


_CONTEXT = threading.local()
_ORIGINAL_RUN = pipeline_module.AccuracyPipelineV2._run
_ORIGINAL_REQUIRED_FAILURES = DiagnosticsRecorder.required_failures
_ORIGINAL_VALIDATE_QUALITY = None


class BatchResolutionChoice(BaseModel):
    targetIndex: int = Field(ge=0)
    chosenSymbol: str = "X"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class BatchResolutionDraft(BaseModel):
    choices: list[BatchResolutionChoice] = Field(default_factory=list)


def _resolver_batch_size() -> int:
    try:
        value = int(os.environ.get("RESOLVER_BATCH_SIZE", "8"))
    except ValueError:
        value = 8
    return max(1, min(8, value))


def build_resolver_batches(
    work: list[ResolverWork],
    *,
    batch_size: int,
    max_batches: int,
) -> list[list[ResolverWork]]:
    """Group chronological states into audio-coherent batches."""

    ordered = sorted(
        work,
        key=lambda item: (
            item.target.startSeconds,
            item.target.endSeconds,
            item.section.id,
            item.chord_index,
        ),
    )
    batches = [
        ordered[index : index + batch_size]
        for index in range(0, len(ordered), batch_size)
    ]
    if len(batches) <= max_batches:
        return batches

    def priority(batch: list[ResolverWork]) -> tuple[float, float, float]:
        coverage = sum(
            item.target.endSeconds - item.target.startSeconds
            for item in batch
        )
        agreements = [
            item.output.chords[item.chord_index].agreement or 0.0
            for item in batch
        ]
        return (
            -coverage,
            mean(agreements) if agreements else 0.0,
            batch[0].target.startSeconds,
        )

    return sorted(batches, key=priority)[:max_batches]


def _batch_prompt(batch: list[ResolverWork], grid) -> str:
    targets: list[dict[str, Any]] = []
    for target_index, work in enumerate(batch):
        chord = work.output.chords[work.chord_index]
        previous_symbol = (
            work.output.chords[work.chord_index - 1].symbol
            if work.chord_index > 0
            else None
        )
        next_symbol = (
            work.output.chords[work.chord_index + 1].symbol
            if work.chord_index + 1 < len(work.output.chords)
            else None
        )
        targets.append(
            {
                "targetIndex": target_index,
                "sectionId": work.section.id,
                "sectionName": work.section.name,
                "sectionType": work.section.type,
                "key": work.section.key,
                "mode": work.section.mode,
                "startSeconds": work.target.startSeconds,
                "endSeconds": work.target.endSeconds,
                "currentSymbol": chord.symbol,
                "candidates": list(
                    dict.fromkeys(
                        [
                            chord.symbol,
                            *chord.alternatives,
                            *work.target.candidates,
                            "N",
                            "X",
                        ]
                    )
                )[:10],
                "previousSymbol": previous_symbol,
                "nextSymbol": next_symbol,
                "reason": work.target.reason,
            }
        )
    return (
        "音声を聴き、targetsの各コード状態を個別に判定してください。"
        "各targetIndexについて必ず1件ずつchoicesへ返してください。"
        "chosenSymbolは対象targetのcandidatesに含まれる値だけを使用してください。"
        "判断不能な場合だけX、無音の場合だけNを選んでください。"
        "別targetの判断を流用してはいけません。\n"
        f"bpm={grid.bpm:.6f}\n"
        f"timeSignature={grid.time_signature}\n"
        f"downbeatOffsetSeconds={grid.downbeat_offset:.6f}\n"
        "targets="
        + json.dumps(targets, ensure_ascii=False, separators=(",", ":"))
    )


def _apply_batch(
    batch: list[ResolverWork],
    decision: BatchResolutionDraft,
) -> int:
    choices: dict[int, BatchResolutionChoice] = {}
    for choice in decision.choices:
        if choice.targetIndex < len(batch):
            choices.setdefault(choice.targetIndex, choice)

    applied = 0
    for index, work in enumerate(batch):
        choice = choices.get(index)
        if choice is not None and _apply_state_resolution(work, choice):
            applied += 1
    return applied


def _patched_run_resolvers(
    self,
    *,
    media,
    resolver_gateway,
    consensus_pairs,
    grid,
    duration,
    diagnostics,
    global_warnings,
):
    outputs = [output for _, output in consensus_pairs]
    before_count = _state_count(outputs)
    all_ranges = []
    all_work: list[ResolverWork] = []
    for section, output in consensus_pairs:
        ranges, work = _split_uncertain_ranges(section, output)
        all_ranges.extend(ranges)
        all_work.extend(work)

    batches = build_resolver_batches(
        all_work,
        batch_size=_resolver_batch_size(),
        max_batches=self.max_resolver_calls,
    )
    with diagnostics.stage("targeted-resolution"):
        with ThreadPoolExecutor(
            max_workers=min(self.max_parallel_calls, max(1, len(batches)))
        ) as executor:
            future_map: dict[
                Future[BatchResolutionDraft],
                tuple[list[ResolverWork], str],
            ] = {}
            bar_duration = grid.beat_duration * grid.beats_per_bar
            for batch_index, batch in enumerate(batches, start=1):
                clip_start = max(
                    0.0,
                    min(item.target.startSeconds for item in batch) - bar_duration,
                )
                clip_end = min(
                    duration,
                    max(item.target.endSeconds for item in batch) + bar_duration,
                )
                label = f"resolver-batch-{batch_index}"
                future = executor.submit(
                    resolver_gateway.generate_typed,
                    contents=[
                        media.part(clip_start, clip_end),
                        _batch_prompt(batch, grid),
                    ],
                    schema=BatchResolutionDraft,
                    system_instruction=(
                        "You are a deterministic chord-state resolver. "
                        "Return one independent allow-listed decision for every targetIndex."
                    ),
                    temperature=0.0,
                    max_output_tokens=2_048,
                    retries=3,
                    diagnostic_label=label,
                )
                future_map[future] = (batch, label)
                diagnostics.resolver_calls += 1

            for future in as_completed(future_map):
                batch, label = future_map[future]
                try:
                    decision = future.result()
                    applied = _apply_batch(batch, decision)
                    if len(decision.choices) < len(batch):
                        global_warnings.append(
                            f"{label} returned {len(decision.choices)}/{len(batch)} decisions"
                        )
                    if applied == 0 and batch:
                        global_warnings.append(f"{label} resolved no chord states")
                except Exception as exc:  # noqa: BLE001
                    diagnostics.record_error(label, exc)
                    diagnostics.record_required_failure(label, exc)
                    global_warnings.append(f"{label} failed: {type(exc).__name__}")

    after_count = _state_count(outputs)
    if before_count - after_count >= 4 and after_count < before_count * 0.70:
        diagnostics.record_required_failure(
            f"resolver_state_collapse:{before_count}->{after_count}"
        )
    diagnostics.chord_state_count = after_count
    return all_ranges


def deterministic_semantic_fallback(section, *, grid):
    children = deterministic_split(section, grid=grid)
    if len(children) <= 1:
        return children
    return [
        child.model_copy(
            update={
                "id": f"{section.id}-fallback-{index}",
                "name": f"{section.name} {index}",
                "notes": (
                    section.notes
                    or "意味的分割を確定できなかったため、小節境界で安全に分割しました。"
                ),
            }
        )
        for index, child in enumerate(children, start=1)
    ]


def _retry_semantic_prompt(section, grid, clip_start, clip_end) -> str:
    return (
        _semantic_prompt(section, grid, clip_start, clip_end)
        + "\n前回の出力は、区間数、固有名称、固有summary、境界、または最大小節数の"
        "いずれかを満たしませんでした。親区間を必ず2区間以上へ分割し、"
        "各区間を最大小節数以内にしてください。"
    )


def _patched_refine_structure(
    self,
    *,
    media,
    gateway,
    sections,
    grid,
    duration,
    diagnostics,
):
    aligned, adjustments = align_sections_to_grid(
        sections,
        grid=grid,
        duration=duration,
    )
    diagnostics.boundary_adjustments += adjustments
    targets = oversized_sections(aligned, grid=grid)
    if not targets:
        return aligned

    bar_duration = grid.beat_duration * grid.beats_per_bar
    first_drafts: dict[str, SemanticStructureRefinementDraft] = {}
    refinements: dict[str, list] = {}
    clips: dict[str, tuple[float, float, Any]] = {}

    with diagnostics.stage("structure-refinement"):
        with ThreadPoolExecutor(
            max_workers=min(self.max_parallel_calls, len(targets))
        ) as executor:
            future_map = {}
            for section in targets:
                clip_start = max(0.0, section.startSeconds - bar_duration)
                clip_end = min(duration, section.endSeconds + bar_duration)
                part = media.part(clip_start, clip_end)
                clips[section.id] = (clip_start, clip_end, part)
                future = executor.submit(
                    gateway.generate_typed,
                    contents=[
                        part,
                        _semantic_prompt(section, grid, clip_start, clip_end),
                    ],
                    schema=SemanticStructureRefinementDraft,
                    system_instruction=(
                        "You are a musical form analyst. Return only semantic song sections, "
                        "never arbitrary equal slices. Every child requires a unique summary."
                    ),
                    temperature=0.0,
                    max_output_tokens=6_144,
                    retries=3,
                    diagnostic_label=f"semantic-refinement-{section.id}",
                )
                future_map[future] = section
                diagnostics.structure_refinement_calls += 1

            for future in as_completed(future_map):
                section = future_map[future]
                try:
                    first_drafts[section.id] = future.result()
                except Exception as exc:  # noqa: BLE001
                    diagnostics.record_error(
                        f"semantic-refinement-{section.id}",
                        exc,
                    )

        retry_targets = []
        for section in targets:
            normalized = _normalise_semantic_refinement(
                section,
                first_drafts.get(
                    section.id,
                    SemanticStructureRefinementDraft(),
                ),
                grid=grid,
            )
            if normalized:
                refinements[section.id] = normalized
            else:
                retry_targets.append(section)

        if retry_targets:
            with ThreadPoolExecutor(
                max_workers=min(self.max_parallel_calls, len(retry_targets))
            ) as executor:
                retry_map = {}
                for section in retry_targets:
                    clip_start, clip_end, part = clips[section.id]
                    future = executor.submit(
                        gateway.generate_typed,
                        contents=[
                            part,
                            _retry_semantic_prompt(
                                section,
                                grid,
                                clip_start,
                                clip_end,
                            ),
                        ],
                        schema=SemanticStructureRefinementDraft,
                        system_instruction=(
                            "You are correcting an invalid musical-form result. "
                            "Return valid semantic sections with unique names and summaries."
                        ),
                        temperature=0.0,
                        max_output_tokens=6_144,
                        retries=2,
                        diagnostic_label=f"semantic-refinement-{section.id}-retry",
                    )
                    retry_map[future] = section
                    diagnostics.structure_refinement_calls += 1

                for future in as_completed(retry_map):
                    section = retry_map[future]
                    try:
                        normalized = _normalise_semantic_refinement(
                            section,
                            future.result(),
                            grid=grid,
                        )
                        if normalized:
                            refinements[section.id] = normalized
                    except Exception as exc:  # noqa: BLE001
                        diagnostics.record_error(
                            f"semantic-refinement-{section.id}-retry",
                            exc,
                        )

    expanded = []
    for section in aligned:
        if section not in targets:
            expanded.append(section)
        elif section.id in refinements:
            expanded.extend(refinements[section.id])
        else:
            expanded.extend(deterministic_semantic_fallback(section, grid=grid))

    final, final_adjustments = align_sections_to_grid(
        expanded,
        grid=grid,
        duration=duration,
    )
    diagnostics.boundary_adjustments += final_adjustments

    remaining = oversized_sections(final, grid=grid)
    if remaining:
        remaining_ids = {section.id for section in remaining}
        forced = []
        for section in final:
            if section.id in remaining_ids:
                forced.extend(deterministic_semantic_fallback(section, grid=grid))
            else:
                forced.append(section)
        final, forced_adjustments = align_sections_to_grid(
            forced,
            grid=grid,
            duration=duration,
        )
        diagnostics.boundary_adjustments += forced_adjustments

    for section in oversized_sections(final, grid=grid):
        diagnostics.record_required_failure(
            f"structure-oversized-after-fallback:{section.id}"
        )
    return final


def finalize_result_before_quality(result) -> None:
    normalize_tempo_segments(result)
    reconcile_result_harmony(
        result,
        reference_hints=_reference_key_hints(result),
    )
    if result.diagnostics is not None:
        result.diagnostics.chordStateCount = _count_result_states(result)
        result.diagnostics.displayChordEventCount = sum(
            len(measure.chords)
            for section in result.sections
            for measure in section.measures
        )

    stale_prefixes = ("全体キーは", "track BPM", "tempo segment")
    result.warnings = [
        warning
        for warning in result.warnings
        if not warning.startswith(stale_prefixes)
    ]
    runs = _key_runs(result)
    if runs:
        key_text = (
            f"中心調は{runs[0]}です。"
            if len(runs) == 1
            else "調性は" + " → ".join(runs) + "と推移します。"
        )
        result.musicalSummary = (
            f"BPM{result.track.bpm:g}の{result.track.timeSignature}。{key_text}"
            "コード進行は確定後の調性に基づいて度数を再計算しています。"
        )


def _required_failures_view(self: DiagnosticsRecorder) -> list[str]:
    values = _ORIGINAL_REQUIRED_FAILURES.fget(self)
    return [] if getattr(_CONTEXT, "defer_quality", False) else values


def _quality_view(result) -> list[str]:
    if getattr(_CONTEXT, "defer_quality", False):
        return []
    if _ORIGINAL_VALIDATE_QUALITY is None:
        raise RuntimeError("quality validator was not installed")
    return _ORIGINAL_VALIDATE_QUALITY(result)


def _patched_run(self, *args, **kwargs):
    _CONTEXT.defer_quality = True
    try:
        result = _ORIGINAL_RUN(self, *args, **kwargs)
    finally:
        _CONTEXT.defer_quality = False

    finalize_result_before_quality(result)
    invariant_errors = pipeline_module.validate_invariants(result)
    invariant_errors.extend(validate_tempo_coverage(result))
    invariant_errors = sorted(set(invariant_errors))

    if _ORIGINAL_VALIDATE_QUALITY is None:
        raise RuntimeError("quality validator was not installed")
    quality_errors = _ORIGINAL_VALIDATE_QUALITY(result)
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
    if quality_errors:
        raise RuntimeError(
            "Accuracy v2 quality gate violation: " + ", ".join(quality_errors)
        )
    return result


def install_accuracy_v2_finalization() -> None:
    global _ORIGINAL_VALIDATE_QUALITY
    if getattr(pipeline_module, "_accuracy_v2_finalization_v2_installed", False):
        return
    pipeline_module._accuracy_v2_finalization_v2_installed = True
    _ORIGINAL_VALIDATE_QUALITY = pipeline_module.validate_quality
    DiagnosticsRecorder.required_failures = property(_required_failures_view)
    pipeline_module.validate_quality = _quality_view
    pipeline_module.AccuracyPipelineV2._run_resolvers = _patched_run_resolvers
    pipeline_module.AccuracyPipelineV2._refine_structure = _patched_refine_structure
    pipeline_module.AccuracyPipelineV2._run = _patched_run
    pipeline_module.research_references = research_references_attributed
