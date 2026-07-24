from __future__ import annotations

import math
from dataclasses import dataclass

from .beat_grid import BeatGrid
from .schemas import SectionStructureDraft, StructureRefinementDraft
from .validators import normalise_section_type


@dataclass(frozen=True, slots=True)
class AnalysisSlice:
    parent_id: str
    index: int
    count: int
    section: SectionStructureDraft


_MAJOR_SECTION_TYPES = {"verse", "pre_chorus", "chorus", "post_chorus"}


def _finite(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp(value: float, start: float, end: float) -> float:
    return max(start, min(end, value))


def _nearest(values: tuple[float, ...], value: float) -> float:
    if not values:
        return value
    return min(values, key=lambda item: (abs(item - value), item))


def nearest_beat(grid: BeatGrid, value: float) -> float:
    return _nearest(grid.beat_times, _clamp(value, 0.0, grid.duration))


def nearest_bar(grid: BeatGrid, value: float) -> float:
    return _nearest(grid.bar_starts, _clamp(value, 0.0, grid.duration))


def boundary_distance_beats(grid: BeatGrid, value: float) -> float:
    if grid.beat_duration <= 0:
        return 0.0
    return abs(nearest_beat(grid, value) - value) / grid.beat_duration


def align_sections_to_grid(
    sections: list[SectionStructureDraft],
    *,
    grid: BeatGrid,
    duration: float,
) -> tuple[list[SectionStructureDraft], int]:
    """Give every adjacent section one shared beat-aware boundary.

    Major section boundaries prefer a bar line when the source boundary is at
    most one beat away. Otherwise the nearest beat is used. The first and last
    boundaries remain exactly 0 and duration so coverage is never lost.
    """

    if not sections:
        return [], 0
    ordered = sorted(sections, key=lambda item: (item.startSeconds, item.endSeconds, item.id))
    raw_boundaries = [0.0]
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        raw_boundaries.append((left.endSeconds + right.startSeconds) / 2.0)
    raw_boundaries.append(duration)

    aligned = [0.0]
    adjustments = 0
    minimum_span = max(0.25, grid.beat_duration)
    for index, raw in enumerate(raw_boundaries[1:-1], start=1):
        left = ordered[index - 1]
        right = ordered[index]
        beat = nearest_beat(grid, raw)
        bar = nearest_bar(grid, raw)
        prefer_bar = (
            normalise_section_type(left.type) in _MAJOR_SECTION_TYPES
            or normalise_section_type(right.type) in _MAJOR_SECTION_TYPES
        )
        candidate = beat
        if prefer_bar and abs(bar - raw) <= grid.beat_duration:
            candidate = bar
        lower = aligned[-1] + minimum_span
        remaining = len(raw_boundaries) - index - 1
        upper = duration - remaining * minimum_span
        candidate = _clamp(candidate, lower, upper)
        if abs(candidate - raw) > 1e-3:
            adjustments += 1
        aligned.append(round(candidate, 6))
    aligned.append(round(duration, 6))

    output: list[SectionStructureDraft] = []
    for index, section in enumerate(ordered):
        start = aligned[index]
        end = aligned[index + 1]
        if end - start < 0.05:
            continue
        output.append(
            section.model_copy(
                update={
                    "startSeconds": round(start, 3),
                    "endSeconds": round(end, 3),
                    "type": normalise_section_type(section.type),
                }
            )
        )
    return output, adjustments


def maximum_section_bars(section_type: str) -> int:
    resolved = normalise_section_type(section_type)
    return 16 if resolved in _MAJOR_SECTION_TYPES else 20


def section_bar_count(section: SectionStructureDraft, grid: BeatGrid) -> float:
    bar_duration = grid.beat_duration * grid.beats_per_bar
    return (section.endSeconds - section.startSeconds) / max(0.001, bar_duration)


def oversized_sections(
    sections: list[SectionStructureDraft],
    *,
    grid: BeatGrid,
) -> list[SectionStructureDraft]:
    return [
        section
        for section in sections
        if section_bar_count(section, grid) > maximum_section_bars(section.type) + 0.25
    ]


def _alpha_suffix(index: int) -> str:
    value = index
    output = ""
    while True:
        output = chr(ord("a") + (value % 26)) + output
        value = value // 26 - 1
        if value < 0:
            return output


def _make_children(
    parent: SectionStructureDraft,
    boundaries: list[float],
    *,
    names: list[str] | None = None,
    types: list[str] | None = None,
    confidences: list[float] | None = None,
) -> list[SectionStructureDraft]:
    output: list[SectionStructureDraft] = []
    count = len(boundaries) - 1
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        suffix = _alpha_suffix(index)
        name = names[index] if names and index < len(names) and names[index] else f"{parent.name}{suffix}"
        section_type = (
            normalise_section_type(types[index])
            if types and index < len(types)
            else normalise_section_type(parent.type)
        )
        confidence = (
            max(0.0, min(1.0, confidences[index]))
            if confidences and index < len(confidences)
            else parent.confidence
        )
        output.append(
            parent.model_copy(
                update={
                    "id": f"{parent.id}-{suffix}" if count > 1 else parent.id,
                    "name": name,
                    "type": section_type,
                    "startSeconds": round(start, 3),
                    "endSeconds": round(end, 3),
                    "confidence": confidence,
                    "notes": parent.notes,
                }
            )
        )
    return output


def deterministic_split(
    section: SectionStructureDraft,
    *,
    grid: BeatGrid,
) -> list[SectionStructureDraft]:
    max_bars = maximum_section_bars(section.type)
    bar_duration = grid.beat_duration * grid.beats_per_bar
    max_span = bar_duration * max_bars
    if section.endSeconds - section.startSeconds <= max_span + grid.beat_duration:
        return [section]

    boundaries = [section.startSeconds]
    cursor = section.startSeconds
    while section.endSeconds - cursor > max_span + grid.beat_duration:
        proposed = cursor + max_span
        boundary = nearest_bar(grid, proposed)
        if boundary <= cursor + bar_duration * 4 or boundary >= section.endSeconds - bar_duration * 4:
            boundary = proposed
        boundaries.append(round(boundary, 6))
        cursor = boundary
    boundaries.append(section.endSeconds)
    return _make_children(section, boundaries)


def normalise_refinement(
    parent: SectionStructureDraft,
    draft: StructureRefinementDraft,
    *,
    grid: BeatGrid,
) -> list[SectionStructureDraft]:
    raw = list(draft.sections)
    if len(raw) < 2:
        return []
    parent_duration = parent.endSeconds - parent.startSeconds
    valid_values = [
        _finite(value, -1.0)
        for item in raw
        for value in (item.startSeconds, item.endSeconds)
        if _finite(value, -1.0) >= 0.0
    ]
    relative = bool(valid_values) and max(valid_values) <= parent_duration + 1.0 and parent.startSeconds > 0.0
    offset = parent.startSeconds if relative else 0.0

    candidates = []
    for item in raw:
        start = _clamp(_finite(item.startSeconds, -1.0) + offset, parent.startSeconds, parent.endSeconds)
        end = _clamp(_finite(item.endSeconds, -1.0) + offset, parent.startSeconds, parent.endSeconds)
        if end - start < grid.beat_duration:
            continue
        candidates.append((start, end, item))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < 2:
        return []

    boundaries = [parent.startSeconds]
    names: list[str] = []
    types: list[str] = []
    confidences: list[float] = []
    for index, (_, _, item) in enumerate(candidates):
        names.append(item.name)
        types.append(item.type)
        confidences.append(max(0.0, min(1.0, _finite(item.confidence, parent.confidence))))
        if index < len(candidates) - 1:
            midpoint = (candidates[index][1] + candidates[index + 1][0]) / 2.0
            boundary = nearest_bar(grid, midpoint)
            if abs(boundary - midpoint) > grid.beat_duration:
                boundary = nearest_beat(grid, midpoint)
            if boundary <= boundaries[-1] + grid.beat_duration:
                return []
            boundaries.append(boundary)
    boundaries.append(parent.endSeconds)

    children = _make_children(
        parent,
        boundaries,
        names=names,
        types=types,
        confidences=confidences,
    )
    if any(
        section_bar_count(child, grid) > maximum_section_bars(child.type) + 0.25
        for child in children
    ):
        return []
    return children


def refine_sections(
    sections: list[SectionStructureDraft],
    *,
    grid: BeatGrid,
    duration: float,
    model_refinements: dict[str, StructureRefinementDraft] | None = None,
) -> tuple[list[SectionStructureDraft], int]:
    refinements = model_refinements or {}
    expanded: list[SectionStructureDraft] = []
    for section in sections:
        if section not in oversized_sections([section], grid=grid):
            expanded.append(section)
            continue
        refined = normalise_refinement(
            section,
            refinements.get(section.id, StructureRefinementDraft()),
            grid=grid,
        )
        expanded.extend(refined or deterministic_split(section, grid=grid))
    return align_sections_to_grid(expanded, grid=grid, duration=duration)


def build_analysis_slices(
    sections: list[SectionStructureDraft],
    *,
    grid: BeatGrid,
    max_bars: int = 16,
) -> list[AnalysisSlice]:
    output: list[AnalysisSlice] = []
    bar_duration = grid.beat_duration * grid.beats_per_bar
    max_span = max(4.0 * bar_duration, max_bars * bar_duration)
    for parent in sections:
        boundaries = [parent.startSeconds]
        cursor = parent.startSeconds
        while parent.endSeconds - cursor > max_span + grid.beat_duration:
            proposed = cursor + max_span
            boundary = nearest_bar(grid, proposed)
            if boundary <= cursor + bar_duration * 4:
                boundary = proposed
            boundaries.append(boundary)
            cursor = boundary
        boundaries.append(parent.endSeconds)
        count = len(boundaries) - 1
        for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
            slice_section = parent.model_copy(
                update={
                    "id": f"{parent.id}__slice{index + 1}",
                    "name": f"{parent.name} 分析区間{index + 1}/{count}",
                    "startSeconds": round(start, 3),
                    "endSeconds": round(end, 3),
                }
            )
            output.append(
                AnalysisSlice(
                    parent_id=parent.id,
                    index=index,
                    count=count,
                    section=slice_section,
                )
            )
    return output
