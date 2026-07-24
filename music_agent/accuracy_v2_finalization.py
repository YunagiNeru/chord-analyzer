from __future__ import annotations

import math
import re
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from . import pipeline_v2 as pipeline_module
from .chord_symbol import canonicalize_symbol
from .consensus import ConsensusOutput
from .harmonic_reconcile import (
    normalize_tempo_segments,
    reconcile_result_harmony,
    validate_tempo_coverage,
)
from .prompts import RESOLUTION_SYSTEM_PROMPT
from .reference_attribution import research_references_attributed
from .schemas import (
    CompactResolutionDraft,
    SectionStructureDraft,
    UncertainRange,
)
from .structure_refine import (
    align_sections_to_grid,
    maximum_section_bars,
    nearest_bar,
    nearest_beat,
    oversized_sections,
    section_bar_count,
)


@dataclass(slots=True)
class ResolverWork:
    section: SectionStructureDraft
    output: ConsensusOutput
    chord_index: int
    target: UncertainRange


class SemanticRefinedSectionDraft(BaseModel):
    name: str = ""
    type: str = "other"
    startSeconds: float = 0.0
    endSeconds: float = 0.0
    confidence: float = 0.5
    summary: str = ""


class SemanticStructureRefinementDraft(BaseModel):
    sections: list[SemanticRefinedSectionDraft] = Field(default_factory=list)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _confidence(value: object, default: float = 0.5) -> float:
    return max(0.0, min(1.0, _finite(value, default)))


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _state_count(outputs: list[ConsensusOutput]) -> int:
    count = 0
    for output in outputs:
        previous_symbol: str | None = None
        previous_end = -1.0
        for chord in sorted(output.chords, key=lambda item: (item.startSeconds, item.endSeconds)):
            symbol = canonicalize_symbol(chord.symbol)
            if symbol != previous_symbol or chord.startSeconds > previous_end + 0.003:
                count += 1
            previous_symbol = symbol
            previous_end = max(previous_end, chord.endSeconds)
    return count


def _split_uncertain_ranges(
    section: SectionStructureDraft,
    output: ConsensusOutput,
) -> tuple[list[UncertainRange], list[ResolverWork]]:
    ranges: list[UncertainRange] = []
    work: list[ResolverWork] = []
    seen: set[tuple[float, float, str]] = set()

    for uncertain in output.uncertain_ranges:
        if uncertain.resolved:
            ranges.append(uncertain)
            continue
        matched = False
        for chord_index, chord in enumerate(output.chords):
            start = max(uncertain.startSeconds, chord.startSeconds)
            end = min(uncertain.endSeconds, chord.endSeconds)
            if end - start < 0.03:
                continue
            matched = True
            key = (round(start, 3), round(end, 3), uncertain.reason)
            if key in seen:
                continue
            seen.add(key)
            target = UncertainRange(
                sectionId=section.id,
                startSeconds=round(start, 3),
                endSeconds=round(end, 3),
                reason=uncertain.reason,
                candidates=list(
                    dict.fromkeys(
                        [
                            chord.symbol,
                            *chord.alternatives,
                            *uncertain.candidates,
                        ]
                    )
                )[:8],
                resolved=False,
            )
            ranges.append(target)
            work.append(
                ResolverWork(
                    section=section,
                    output=output,
                    chord_index=chord_index,
                    target=target,
                )
            )
        if not matched:
            ranges.append(uncertain)
    output.uncertain_ranges = ranges
    return ranges, work


def _apply_state_resolution(
    work: ResolverWork,
    decision: CompactResolutionDraft,
) -> bool:
    chosen = canonicalize_symbol(decision.chosenSymbol)
    chord = work.output.chords[work.chord_index]
    allowed = {
        canonicalize_symbol(item)
        for item in [
            chord.symbol,
            *chord.alternatives,
            *work.target.candidates,
            "N",
            "X",
        ]
    }
    if chosen not in allowed:
        raise ValueError(f"resolver returned a candidate outside the allow-list: {chosen}")
    if chosen == "X":
        return False

    previous = canonicalize_symbol(chord.symbol)
    chord.symbol = chosen
    chord.confidence = max(chord.confidence, _confidence(decision.confidence))
    chord.agreement = max(chord.agreement or 0.0, _confidence(decision.confidence))
    chord.alternatives = list(
        dict.fromkeys(
            [previous]
            + [
                canonicalize_symbol(item)
                for item in chord.alternatives
                if canonicalize_symbol(item) != chosen
            ]
        )
    )[:3]
    work.target.resolved = True
    return True


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
    all_ranges: list[UncertainRange] = []
    all_work: list[ResolverWork] = []
    for section, output in consensus_pairs:
        ranges, work = _split_uncertain_ranges(section, output)
        all_ranges.extend(ranges)
        all_work.extend(work)

    all_work.sort(
        key=lambda item: (
            item.output.chords[item.chord_index].agreement
            if item.output.chords[item.chord_index].agreement is not None
            else 0.0,
            -(item.target.endSeconds - item.target.startSeconds),
            item.target.startSeconds,
        )
    )
    selected = all_work[: self.max_resolver_calls]

    with diagnostics.stage("targeted-resolution"):
        with ThreadPoolExecutor(
            max_workers=min(self.max_parallel_calls, max(1, len(selected)))
        ) as executor:
            future_map: dict[Future[CompactResolutionDraft], tuple[ResolverWork, str]] = {}
            bar_duration = grid.beat_duration * grid.beats_per_bar
            for work in selected:
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
                clip_start = max(0.0, work.target.startSeconds - bar_duration)
                clip_end = min(duration, work.target.endSeconds + bar_duration)
                label = f"resolver-{work.section.id}-{work.target.startSeconds:.3f}"
                future = executor.submit(
                    resolver_gateway.generate_typed,
                    contents=[
                        media.part(clip_start, clip_end),
                        pipeline_module.build_resolution_prompt(
                            section=work.section,
                            start=work.target.startSeconds,
                            end=work.target.endSeconds,
                            candidates=list(
                                dict.fromkeys(
                                    [chord.symbol, *chord.alternatives, *work.target.candidates]
                                )
                            )[:8],
                            previous_symbol=previous_symbol,
                            next_symbol=next_symbol,
                            key=work.section.key,
                            grid=grid,
                        ),
                    ],
                    schema=CompactResolutionDraft,
                    system_instruction=RESOLUTION_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=512,
                    retries=3,
                    diagnostic_label=label,
                )
                future_map[future] = (work, label)
                diagnostics.resolver_calls += 1

            for future in as_completed(future_map):
                work, label = future_map[future]
                try:
                    decision = future.result()
                    _apply_state_resolution(work, decision)
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


def _semantic_prompt(section: SectionStructureDraft, grid, clip_start: float, clip_end: float) -> str:
    return (
        "次の長い楽曲区間を、利用者に見せる意味的な楽曲構造へ再分割してください。"
        "単なる等分割やA1/A2の機械的名称は禁止です。各区間に固有の歌詞・編曲・役割の要約をsummaryへ入れてください。"
        "内部分析用スライスではなく、verse、pre_chorus、chorus、bridge等の意味的境界だけを返してください。\n"
        f"parent={section.model_dump_json()}\n"
        f"clipStartSeconds={clip_start:.6f}\n"
        f"clipEndSeconds={clip_end:.6f}\n"
        f"bpm={grid.bpm:.6f}\n"
        f"timeSignature={grid.time_signature}\n"
        f"downbeatOffsetSeconds={grid.downbeat_offset:.6f}"
    )


def _generic_child_name(parent_name: str, child_name: str) -> bool:
    parent = re.sub(r"[\s_\-0-9A-Za-z]+", "", parent_name).lower()
    child = re.sub(r"[\s_\-0-9A-Za-z]+", "", child_name).lower()
    if not child:
        return True
    return child == parent


def _normalise_semantic_refinement(
    parent: SectionStructureDraft,
    draft: SemanticStructureRefinementDraft,
    *,
    grid,
) -> list[SectionStructureDraft]:
    raw = list(draft.sections)
    if len(raw) < 2:
        return []
    if any(not item.name.strip() or not item.summary.strip() for item in raw):
        return []
    if len({item.summary.strip() for item in raw}) < 2:
        return []
    if all(_generic_child_name(parent.name, item.name) for item in raw):
        return []

    parent_duration = parent.endSeconds - parent.startSeconds
    valid_values = [
        _finite(value, -1.0)
        for item in raw
        for value in (item.startSeconds, item.endSeconds)
        if _finite(value, -1.0) >= 0.0
    ]
    relative = (
        bool(valid_values)
        and max(valid_values) <= parent_duration + 1.0
        and parent.startSeconds > 0.0
    )
    offset = parent.startSeconds if relative else 0.0
    candidates: list[tuple[float, float, SemanticRefinedSectionDraft]] = []
    for item in raw:
        start = max(parent.startSeconds, min(parent.endSeconds, item.startSeconds + offset))
        end = max(parent.startSeconds, min(parent.endSeconds, item.endSeconds + offset))
        if end - start < grid.beat_duration:
            continue
        candidates.append((start, end, item))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < 2:
        return []

    boundaries = [parent.startSeconds]
    for index in range(len(candidates) - 1):
        midpoint = (candidates[index][1] + candidates[index + 1][0]) / 2.0
        boundary = nearest_bar(grid, midpoint)
        if abs(boundary - midpoint) > grid.beat_duration:
            boundary = nearest_beat(grid, midpoint)
        if boundary <= boundaries[-1] + grid.beat_duration:
            return []
        boundaries.append(round(boundary, 6))
    boundaries.append(parent.endSeconds)

    children: list[SectionStructureDraft] = []
    for index, ((_, _, item), start, end) in enumerate(
        zip(candidates, boundaries[:-1], boundaries[1:], strict=True),
        start=1,
    ):
        child = SectionStructureDraft(
            id=f"{parent.id}-semantic-{index}",
            name=item.name.strip(),
            type=pipeline_module.normalise_sections(
                [
                    SectionStructureDraft(
                        id="type-probe",
                        name="type-probe",
                        type=item.type,
                        startSeconds=0.0,
                        endSeconds=1.0,
                    )
                ],
                duration=1.0,
            )[0].type,
            startSeconds=round(start, 6),
            endSeconds=round(end, 6),
            key=parent.key,
            mode=parent.mode,
            confidence=_confidence(item.confidence, parent.confidence),
            notes=item.summary.strip(),
        )
        if section_bar_count(child, grid) > maximum_section_bars(child.type) + 0.25:
            return []
        children.append(child)
    return children


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

    refinements: dict[str, SemanticStructureRefinementDraft] = {}
    bar_duration = grid.beat_duration * grid.beats_per_bar
    with diagnostics.stage("structure-refinement"):
        with ThreadPoolExecutor(
            max_workers=min(self.max_parallel_calls, len(targets))
        ) as executor:
            future_map: dict[Future[SemanticStructureRefinementDraft], SectionStructureDraft] = {}
            for section in targets:
                clip_start = max(0.0, section.startSeconds - bar_duration)
                clip_end = min(duration, section.endSeconds + bar_duration)
                future = executor.submit(
                    gateway.generate_typed,
                    contents=[
                        media.part(clip_start, clip_end),
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
                    refinements[section.id] = future.result()
                except Exception as exc:  # noqa: BLE001
                    diagnostics.record_error(f"semantic-refinement-{section.id}", exc)
                    diagnostics.record_required_failure(
                        f"semantic-refinement-{section.id}", exc
                    )

    expanded: list[SectionStructureDraft] = []
    for section in aligned:
        if section not in targets:
            expanded.append(section)
            continue
        refined = _normalise_semantic_refinement(
            section,
            refinements.get(section.id, SemanticStructureRefinementDraft()),
            grid=grid,
        )
        if not refined:
            diagnostics.record_required_failure(
                f"semantic-refinement-{section.id}-invalid"
            )
            expanded.append(section)
        else:
            expanded.extend(refined)
    final, final_adjustments = align_sections_to_grid(
        expanded,
        grid=grid,
        duration=duration,
    )
    diagnostics.boundary_adjustments += final_adjustments
    return final


def _count_result_states(result) -> int:
    count = 0
    for section in result.sections:
        previous_symbol: str | None = None
        previous_end = -1.0
        events = sorted(
            [chord for measure in section.measures for chord in measure.chords],
            key=lambda item: (item.startSeconds, item.endSeconds),
        )
        for chord in events:
            symbol = canonicalize_symbol(chord.symbol)
            if symbol != previous_symbol or chord.startSeconds > previous_end + 0.003:
                count += 1
            previous_symbol = symbol
            previous_end = max(previous_end, chord.endSeconds)
    return count


def _reference_key_hints(result) -> list[str]:
    hints: list[str] = []
    for source in result.referenceSources:
        for fact in source.facts:
            if fact.startswith("KEY="):
                hints.append(fact.split("=", 1)[1].strip())
    return hints


def _key_runs(result) -> list[str]:
    runs: list[str] = []
    for section in result.sections:
        label = " ".join(value for value in (section.key, section.mode) if value)
        if label and (not runs or runs[-1] != label):
            runs.append(label)
    return runs


def _patched_run(self, *args, **kwargs):
    result = _ORIGINAL_RUN(self, *args, **kwargs)
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

    stale_prefixes = (
        "全体キーは",
        "track BPM",
        "tempo segment",
    )
    result.warnings = [
        warning
        for warning in result.warnings
        if not warning.startswith(stale_prefixes)
    ]
    runs = _key_runs(result)
    if runs:
        if len(runs) == 1:
            key_text = f"中心調は{runs[0]}です。"
        else:
            key_text = "調性は" + " → ".join(runs) + "と推移します。"
        result.musicalSummary = (
            f"BPM{result.track.bpm:g}の{result.track.timeSignature}。{key_text}"
            "コード進行は確定後の調性に基づいて度数を再計算しています。"
        )

    post_errors = validate_tempo_coverage(result)
    if post_errors:
        if result.diagnostics is not None:
            result.diagnostics.invariantErrors.extend(post_errors)
        raise RuntimeError(
            "Accuracy v2 postprocess invariant violation: " + ", ".join(post_errors)
        )
    return result


def install_accuracy_v2_finalization() -> None:
    if getattr(pipeline_module, "_accuracy_v2_finalization_installed", False):
        return
    pipeline_module._accuracy_v2_finalization_installed = True
    pipeline_module.AccuracyPipelineV2._run_resolvers = _patched_run_resolvers
    pipeline_module.AccuracyPipelineV2._refine_structure = _patched_refine_structure
    pipeline_module.AccuracyPipelineV2._run = _patched_run
    pipeline_module.research_references = research_references_attributed


_ORIGINAL_RUN = pipeline_module.AccuracyPipelineV2._run
