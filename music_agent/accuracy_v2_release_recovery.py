from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

from . import accuracy_v2_finalization as finalization
from . import pipeline_v2 as pipeline_module
from .chord_symbol import canonicalize_symbol, parse_chord
from .harmonic_reconcile import (
    _canonical_hint,
    _key_mode_label,
    _merge_display_events,
    _score_candidate,
    estimate_key,
)
from .prompts import RESOLUTION_SYSTEM_PROMPT
from .quality_extension import validate_quality_extended
from .reference_attribution import _collect_candidates, research_references_attributed
from .reference_research import ReferenceResearchResult
from .schemas import CompactResolutionDraft, SectionStructureDraft, UncertainRange
from .structure_refine import (
    align_sections_to_grid,
    nearest_bar,
    nearest_beat,
    oversized_sections,
    section_bar_count,
)
from .validators import roman_numeral


SOFT_SEMANTIC_SECTION_LIMIT_BARS = 32.0


@dataclass(slots=True)
class ResolverGroup:
    signature: tuple[object, ...]
    members: list[finalization.ResolverWork]

    @property
    def coverage_seconds(self) -> float:
        return sum(
            member.target.endSeconds - member.target.startSeconds
            for member in self.members
        )

    @property
    def minimum_agreement(self) -> float:
        values = [
            member.output.chords[member.chord_index].agreement
            for member in self.members
        ]
        finite = [float(value) for value in values if value is not None]
        return min(finite, default=0.0)

    @property
    def representative(self) -> finalization.ResolverWork:
        return min(
            self.members,
            key=lambda member: (
                member.output.chords[member.chord_index].agreement
                if member.output.chords[member.chord_index].agreement is not None
                else 0.0,
                -(member.target.endSeconds - member.target.startSeconds),
                member.target.startSeconds,
            ),
        )


def _simple(symbol: str | None) -> str:
    return canonicalize_symbol(symbol, simplify=True)


def _resolver_signature(
    work: finalization.ResolverWork,
    *,
    beat_duration: float,
) -> tuple[object, ...]:
    chords = work.output.chords
    chord = chords[work.chord_index]
    previous_symbol = (
        chords[work.chord_index - 1].symbol
        if work.chord_index > 0
        else None
    )
    next_symbol = (
        chords[work.chord_index + 1].symbol
        if work.chord_index + 1 < len(chords)
        else None
    )
    duration_beats = (
        work.target.endSeconds - work.target.startSeconds
    ) / max(0.001, beat_duration)
    duration_bucket = round(duration_beats * 2.0) / 2.0
    return (
        work.section.type,
        _simple(chord.symbol),
        _simple(previous_symbol),
        _simple(next_symbol),
        duration_bucket,
    )


def _group_resolver_work(
    work_items: list[finalization.ResolverWork],
    *,
    beat_duration: float,
) -> list[ResolverGroup]:
    grouped: dict[tuple[object, ...], list[finalization.ResolverWork]] = defaultdict(list)
    for work in work_items:
        grouped[_resolver_signature(work, beat_duration=beat_duration)].append(work)
    groups = [
        ResolverGroup(signature=signature, members=members)
        for signature, members in grouped.items()
    ]
    groups.sort(
        key=lambda group: (
            -group.coverage_seconds,
            group.minimum_agreement,
            group.representative.target.startSeconds,
        )
    )
    return groups


def _allowed_symbols(work: finalization.ResolverWork) -> set[str]:
    chord = work.output.chords[work.chord_index]
    return {
        canonicalize_symbol(item)
        for item in [
            chord.symbol,
            *chord.alternatives,
            *work.target.candidates,
            "N",
            "X",
        ]
    }


def _run_grouped_resolvers(
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
    before_count = finalization._state_count(outputs)
    all_ranges: list[UncertainRange] = []
    all_work: list[finalization.ResolverWork] = []
    for section, output in consensus_pairs:
        ranges, work = finalization._split_uncertain_ranges(section, output)
        all_ranges.extend(ranges)
        all_work.extend(work)

    groups = _group_resolver_work(
        all_work,
        beat_duration=grid.beat_duration,
    )[: self.max_resolver_calls]

    with diagnostics.stage("targeted-resolution"):
        with ThreadPoolExecutor(
            max_workers=min(self.max_parallel_calls, max(1, len(groups)))
        ) as executor:
            future_map: dict[
                Future[CompactResolutionDraft],
                tuple[ResolverGroup, str],
            ] = {}
            bar_duration = grid.beat_duration * grid.beats_per_bar
            for group in groups:
                work = group.representative
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
                label = (
                    f"resolver-group-{work.section.id}-"
                    f"{work.target.startSeconds:.3f}-n{len(group.members)}"
                )
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
                                    [
                                        chord.symbol,
                                        *chord.alternatives,
                                        *work.target.candidates,
                                    ]
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
                future_map[future] = (group, label)
                diagnostics.resolver_calls += 1

            for future in as_completed(future_map):
                group, label = future_map[future]
                try:
                    decision = future.result()
                    chosen = canonicalize_symbol(decision.chosenSymbol)
                    for member in group.members:
                        if chosen not in _allowed_symbols(member):
                            continue
                        finalization._apply_state_resolution(member, decision)
                except Exception as exc:  # noqa: BLE001
                    diagnostics.record_error(label, exc)
                    diagnostics.record_required_failure(label, exc)
                    global_warnings.append(f"{label} failed: {type(exc).__name__}")

    after_count = finalization._state_count(outputs)
    if before_count - after_count >= 4 and after_count < before_count * 0.70:
        diagnostics.record_required_failure(
            f"resolver_state_collapse:{before_count}->{after_count}"
        )
    diagnostics.chord_state_count = after_count
    return all_ranges


def _refine_structure_with_soft_semantic_limit(
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

    refinements: dict[str, finalization.SemanticStructureRefinementDraft] = {}
    failures: dict[str, Exception] = {}
    bar_duration = grid.beat_duration * grid.beats_per_bar
    with diagnostics.stage("structure-refinement"):
        with ThreadPoolExecutor(
            max_workers=min(self.max_parallel_calls, len(targets))
        ) as executor:
            future_map: dict[
                Future[finalization.SemanticStructureRefinementDraft],
                SectionStructureDraft,
            ] = {}
            for section in targets:
                clip_start = max(0.0, section.startSeconds - bar_duration)
                clip_end = min(duration, section.endSeconds + bar_duration)
                future = executor.submit(
                    gateway.generate_typed,
                    contents=[
                        media.part(clip_start, clip_end),
                        finalization._semantic_prompt(
                            section,
                            grid,
                            clip_start,
                            clip_end,
                        ),
                    ],
                    schema=finalization.SemanticStructureRefinementDraft,
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
                    failures[section.id] = exc

    expanded: list[SectionStructureDraft] = []
    for section in aligned:
        if section not in targets:
            expanded.append(section)
            continue
        refined = finalization._normalise_semantic_refinement(
            section,
            refinements.get(
                section.id,
                finalization.SemanticStructureRefinementDraft(),
            ),
            grid=grid,
        )
        if refined:
            expanded.extend(refined)
            continue

        bars = section_bar_count(section, grid)
        if bars > SOFT_SEMANTIC_SECTION_LIMIT_BARS + 0.25:
            failure = failures.get(section.id)
            diagnostics.record_required_failure(
                f"semantic-refinement-{section.id}-invalid",
                failure,
            )
        expanded.append(section)

    final, final_adjustments = align_sections_to_grid(
        expanded,
        grid=grid,
        duration=duration,
    )
    diagnostics.boundary_adjustments += final_adjustments
    return final


def _result_state_count(result) -> int:
    count = 0
    for section in result.sections:
        previous_symbol: str | None = None
        previous_end = -1.0
        events = sorted(
            [chord for measure in section.measures for chord in measure.chords],
            key=lambda item: (item.startSeconds, item.endSeconds, item.symbol),
        )
        for chord in events:
            symbol = canonicalize_symbol(chord.symbol)
            if symbol != previous_symbol or chord.startSeconds > previous_end + 0.003:
                count += 1
            previous_symbol = symbol
            previous_end = max(previous_end, chord.endSeconds)
    return count


def _cadential_support(events, tonic: int) -> bool:
    roots = [
        chord.root_pc
        for event in events
        if (chord := parse_chord(event.symbol)).root_pc is not None
    ]
    distinct = set(roots)
    if len(distinct) < 3 or tonic not in distinct:
        return False
    degrees = {(root - tonic) % 12 for root in distinct}
    boundary_roots = roots[:2] + roots[-2:]
    return (
        0 in {(root - tonic) % 12 for root in boundary_roots}
        and bool(degrees & {5, 7})
    )


def _stable_reconcile_result_harmony(
    result,
    *,
    reference_hints: Iterable[str] = (),
) -> None:
    section_events = {
        section.id: _merge_display_events(section)
        for section in result.sections
    }
    all_events = [
        event
        for section in result.sections
        for event in section_events[section.id]
    ]
    original_global_hint = _key_mode_label(
        result.track.globalKey,
        result.track.globalMode,
    )
    global_estimate = estimate_key(
        all_events,
        hints=[original_global_hint, *reference_hints],
    )
    result.track.globalKey = global_estimate.key
    result.track.globalMode = global_estimate.mode
    global_identity = _canonical_hint(
        f"{global_estimate.key} {global_estimate.mode}"
    )
    assert global_identity is not None

    proposals: list[tuple[str, str] | None] = []
    for section in result.sections:
        events = section_events[section.id]
        local = estimate_key(
            events,
            hints=[
                _key_mode_label(section.key, section.mode),
                original_global_hint,
                *reference_hints,
            ],
        )
        local_identity = _canonical_hint(f"{local.key} {local.mode}")
        global_score = _score_candidate(
            events,
            tonic=global_identity[0],
            mode=global_estimate.mode,
        )
        use_local = (
            local_identity is not None
            and local_identity != global_identity
            and local.score >= global_score + 0.16
            and local.margin >= 0.05
            and _cadential_support(events, local_identity[0])
        )
        proposals.append((local.key, local.mode) if use_local else None)

    accepted: list[tuple[str, str] | None] = [None] * len(result.sections)
    index = 0
    while index < len(result.sections):
        proposal = proposals[index]
        if proposal is None:
            index += 1
            continue
        end = index + 1
        while end < len(result.sections) and proposals[end] == proposal:
            end += 1
        run_events = [
            event
            for section in result.sections[index:end]
            for event in section_events[section.id]
        ]
        run_estimate = estimate_key(
            run_events,
            hints=[f"{proposal[0]} {proposal[1]}", *reference_hints],
        )
        run_identity = _canonical_hint(
            f"{run_estimate.key} {run_estimate.mode}"
        )
        run_global_score = _score_candidate(
            run_events,
            tonic=global_identity[0],
            mode=global_estimate.mode,
        )
        run_seconds = sum(
            section.endSeconds - section.startSeconds
            for section in result.sections[index:end]
        )
        accept_run = (
            run_identity == _canonical_hint(f"{proposal[0]} {proposal[1]}")
            and run_estimate.score >= run_global_score + 0.16
            and run_estimate.margin >= 0.05
            and (end - index >= 2 or run_seconds >= 20.0)
        )
        if accept_run:
            for member_index in range(index, end):
                accepted[member_index] = proposal
        index = end

    for index, section in enumerate(result.sections):
        chosen = accepted[index] or (
            global_estimate.key,
            global_estimate.mode,
        )
        section.key, section.mode = chosen
        for measure in section.measures:
            for chord in measure.chords:
                chord.roman = roman_numeral(
                    chord.symbol,
                    section.key,
                    section.mode,
                )


def _prune_ambiguous_source_facts(result: ReferenceResearchResult) -> ReferenceResearchResult:
    for source in result.sources:
        bpm_facts = list(
            dict.fromkeys(fact for fact in source.facts if fact.startswith("BPM="))
        )
        key_facts = list(
            dict.fromkeys(fact for fact in source.facts if fact.startswith("KEY="))
        )
        other = [
            fact
            for fact in source.facts
            if not fact.startswith(("BPM=", "KEY="))
        ]
        if len(bpm_facts) > 2:
            bpm_facts = []
        if len(key_facts) > 2:
            key_facts = []
        source.facts = [*bpm_facts, *key_facts, *other]
    bpms, support, keys = _collect_candidates(result.sources)
    result.bpm_candidates = bpms
    result.bpm_support = support
    result.key_candidates = keys
    return result


def _research_references_pruned(gateway, metadata) -> ReferenceResearchResult:
    return _prune_ambiguous_source_facts(
        research_references_attributed(gateway, metadata)
    )


def _validate_quality_with_semantic_limit(result) -> list[str]:
    errors = validate_quality_extended(result)
    retained: list[str] = []
    for error in errors:
        if not error.startswith("oversized_section:"):
            retained.append(error)
            continue
        parts = error.split(":")
        section_id = parts[1] if len(parts) > 1 else ""
        section = next(
            (item for item in result.sections if item.id == section_id),
            None,
        )
        if section is None:
            retained.append(error)
            continue
        bpm = result.track.bpm or 120.0
        try:
            beats_per_bar = max(
                1,
                int((result.track.timeSignature or "4/4").split("/", 1)[0]),
            )
        except (TypeError, ValueError, IndexError):
            beats_per_bar = 4
        bars = (
            section.endSeconds - section.startSeconds
        ) / max(0.001, (60.0 / bpm) * beats_per_bar)
        if bars > SOFT_SEMANTIC_SECTION_LIMIT_BARS + 0.25:
            retained.append(error)
    return sorted(set(retained))


def install_accuracy_v2_release_recovery() -> None:
    if getattr(pipeline_module, "_accuracy_v2_release_recovery_installed", False):
        return
    pipeline_module._accuracy_v2_release_recovery_installed = True
    pipeline_module.AccuracyPipelineV2._run_resolvers = _run_grouped_resolvers
    pipeline_module.AccuracyPipelineV2._refine_structure = (
        _refine_structure_with_soft_semantic_limit
    )
    pipeline_module.validate_quality = _validate_quality_with_semantic_limit
    pipeline_module.research_references = _research_references_pruned
    finalization.reconcile_result_harmony = _stable_reconcile_result_harmony
    finalization._count_result_states = _result_state_count
