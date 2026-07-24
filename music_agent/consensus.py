from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from .accuracy_profile import AccuracyProfile, load_accuracy_profile
from .beat_grid import BeatGrid
from .chord_symbol import canonicalize_symbol, parse_chord, same_root_quality
from .schemas import (
    ChordEvent,
    DspChordRun,
    SectionStructureDraft,
    SpecialistChordDraft,
    SpecialistSectionDraft,
    UncertainRange,
)
from .sequence_optimizer import optimize_sequence


@dataclass(slots=True)
class ConsensusOutput:
    chords: list[ChordEvent]
    agreement: float
    uncertain_ranges: list[UncertainRange]
    slot_alternatives: list[list[str]]
    slot_scores: list[dict[str, float]]


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _best_event(
    events: list[SpecialistChordDraft],
    start: float,
    end: float,
) -> SpecialistChordDraft | None:
    candidates = [
        (_overlap(start, end, item.startSeconds, item.endSeconds), item)
        for item in events
    ]
    candidates = [item for item in candidates if item[0] > 0.0]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1].confidence))[1]


def _best_dsp_event(
    events: list[DspChordRun],
    start: float,
    end: float,
) -> DspChordRun | None:
    candidates = [
        (_overlap(start, end, item.startSeconds, item.endSeconds), item)
        for item in events
    ]
    candidates = [item for item in candidates if item[0] > 0.0]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1].confidence))[1]


def _symbol_vote(
    votes: dict[str, float],
    symbol: str,
    weight: float,
    *,
    simple_triad_reinforcement: float,
) -> None:
    canonical = canonicalize_symbol(symbol)
    votes[canonical] += max(0.001, weight)
    parsed = parse_chord(canonical)
    if parsed.root and parsed.quality not in {"unknown", "no_chord"}:
        simple = canonicalize_symbol(canonical, simplify=True)
        votes[simple] += max(0.001, weight * simple_triad_reinforcement)


def _primary_agreement(
    selected: str,
    evidence: list[tuple[str, float]],
) -> tuple[float, float, int]:
    """Return exact agreement, root/quality agreement, and usable contributors.

    Alternatives and synthetic triad reinforcement deliberately do not count as
    independent evidence. Counting them in the denominator was the cause of the
    original excessive unresolved coverage.
    """

    canonical_selected = canonicalize_symbol(selected)
    total = sum(max(0.0, weight) for _, weight in evidence)
    if total <= 0.0:
        return 0.0, 0.0, 0

    exact_support = 0.0
    component_support = 0.0
    contributors = 0
    for raw_symbol, weight in evidence:
        weight = max(0.0, weight)
        canonical = canonicalize_symbol(raw_symbol)
        parsed = parse_chord(canonical)
        if canonical not in {"X"}:
            contributors += 1
        if canonical == canonical_selected:
            exact_support += weight
        if canonical_selected == "N":
            if canonical == "N":
                component_support += weight
        elif canonical_selected != "X" and not parsed.unknown:
            if same_root_quality(canonical, canonical_selected):
                component_support += weight

    return (
        exact_support / total,
        component_support / total,
        contributors,
    )


def _uncertainty_reason(
    *,
    selected: str,
    component_agreement: float,
    contributors: int,
    profile: AccuracyProfile,
) -> str | None:
    if canonicalize_symbol(selected) == "X":
        return "コードを特定できませんでした。"
    if contributors < profile.minimum_primary_contributors:
        return "有効な独立分析が不足しています。"
    if component_agreement < profile.uncertainty_threshold:
        return "ルートまたは基本コード品質の一致度が閾値未満です。"
    return None


def consensus_section(
    *,
    section: SectionStructureDraft,
    specialists: list[SpecialistSectionDraft],
    grid: BeatGrid,
    dsp_runs: list[DspChordRun] | None = None,
    profile: AccuracyProfile | None = None,
) -> ConsensusOutput:
    profile = profile or load_accuracy_profile()
    boundaries = grid.slot_boundaries(section.startSeconds, section.endSeconds)
    if len(boundaries) < 2:
        boundaries = [section.startSeconds, section.endSeconds]

    slot_scores: list[dict[str, float]] = []
    slot_alternatives: list[list[str]] = []
    slot_primary_evidence: list[list[tuple[str, float]]] = []

    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        votes: dict[str, float] = defaultdict(float)
        primary_evidence: list[tuple[str, float]] = []
        for specialist in specialists:
            event = _best_event(specialist.chords, start, end)
            if event is None:
                continue
            role_weight = profile.role_weights.get(specialist.role, 0.75)
            primary_weight = role_weight * event.confidence
            primary_evidence.append((event.symbol, primary_weight))
            _symbol_vote(
                votes,
                event.symbol,
                primary_weight,
                simple_triad_reinforcement=profile.simple_triad_reinforcement,
            )
            for alternative in event.alternatives[:3]:
                _symbol_vote(
                    votes,
                    alternative,
                    primary_weight * profile.alternative_weight,
                    simple_triad_reinforcement=profile.simple_triad_reinforcement,
                )

        if dsp_runs:
            dsp_event = _best_dsp_event(dsp_runs, start, end)
            if dsp_event is not None:
                dsp_primary_weight = profile.dsp_weight * dsp_event.confidence
                primary_evidence.append((dsp_event.symbol, dsp_primary_weight))
                _symbol_vote(
                    votes,
                    dsp_event.symbol,
                    dsp_primary_weight,
                    simple_triad_reinforcement=profile.simple_triad_reinforcement,
                )
                for alternative in dsp_event.alternatives[:2]:
                    _symbol_vote(
                        votes,
                        alternative.symbol,
                        profile.dsp_weight
                        * profile.alternative_weight
                        * alternative.score,
                        simple_triad_reinforcement=profile.simple_triad_reinforcement,
                    )

        if not votes:
            votes["X"] = 1.0

        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        slot_scores.append(dict(ranked[:8]))
        slot_alternatives.append([symbol for symbol, _ in ranked[:4]])
        slot_primary_evidence.append(primary_evidence)

    selected = optimize_sequence(
        slot_scores,
        change_penalty=profile.change_penalty,
        isolated_penalty=profile.isolated_penalty,
    )

    slot_exact_agreements: list[float] = []
    slot_component_agreements: list[float] = []
    slot_contributors: list[int] = []
    slot_confidences: list[float] = []
    for symbol, evidence in zip(selected, slot_primary_evidence, strict=True):
        exact, component, contributors = _primary_agreement(symbol, evidence)
        confidence = (
            exact * profile.exact_agreement_weight
            + component * profile.component_agreement_weight
        )
        slot_exact_agreements.append(exact)
        slot_component_agreements.append(component)
        slot_contributors.append(contributors)
        slot_confidences.append(confidence)

    source = "hybrid" if dsp_runs else "ai"
    events: list[ChordEvent] = []
    for index, (start, end, symbol) in enumerate(
        zip(boundaries[:-1], boundaries[1:], selected, strict=True)
    ):
        alternatives = [item for item in slot_alternatives[index] if item != symbol][:3]
        confidence = min(0.99, max(0.05, slot_confidences[index]))
        event = ChordEvent(
            symbol=symbol,
            startSeconds=round(start, 3),
            endSeconds=round(end, 3),
            confidence=round(confidence, 4),
            source=source,
            alternatives=alternatives,
            agreement=round(slot_component_agreements[index], 4),
        )
        if events and events[-1].symbol == event.symbol:
            previous = events[-1]
            previous.endSeconds = event.endSeconds
            previous.confidence = round((previous.confidence + event.confidence) / 2.0, 4)
            values = [value for value in (previous.agreement, event.agreement) if value is not None]
            previous.agreement = round(mean(values), 4) if values else None
            previous.alternatives = list(
                dict.fromkeys(previous.alternatives + event.alternatives)
            )[:3]
        else:
            events.append(event)

    uncertain_ranges: list[UncertainRange] = []
    current_start: float | None = None
    current_end = 0.0
    current_candidates: list[str] = []
    current_reason: str | None = None

    for index, symbol in enumerate(selected):
        reason = _uncertainty_reason(
            selected=symbol,
            component_agreement=slot_component_agreements[index],
            contributors=slot_contributors[index],
            profile=profile,
        )
        if reason is not None:
            if current_start is None or current_reason != reason:
                if current_start is not None and current_reason is not None:
                    uncertain_ranges.append(
                        UncertainRange(
                            sectionId=section.id,
                            startSeconds=round(current_start, 3),
                            endSeconds=round(current_end, 3),
                            reason=current_reason,
                            candidates=list(dict.fromkeys(current_candidates))[:8],
                        )
                    )
                current_start = boundaries[index]
                current_candidates = []
                current_reason = reason
            current_end = boundaries[index + 1]
            current_candidates.extend(slot_alternatives[index])
        elif current_start is not None and current_reason is not None:
            uncertain_ranges.append(
                UncertainRange(
                    sectionId=section.id,
                    startSeconds=round(current_start, 3),
                    endSeconds=round(current_end, 3),
                    reason=current_reason,
                    candidates=list(dict.fromkeys(current_candidates))[:8],
                )
            )
            current_start = None
            current_candidates = []
            current_reason = None

    if current_start is not None and current_reason is not None:
        uncertain_ranges.append(
            UncertainRange(
                sectionId=section.id,
                startSeconds=round(current_start, 3),
                endSeconds=round(current_end, 3),
                reason=current_reason,
                candidates=list(dict.fromkeys(current_candidates))[:8],
            )
        )

    return ConsensusOutput(
        chords=events,
        agreement=(
            round(mean(slot_component_agreements), 4)
            if slot_component_agreements
            else 0.0
        ),
        uncertain_ranges=uncertain_ranges,
        slot_alternatives=slot_alternatives,
        slot_scores=slot_scores,
    )


def reconcile_repeated_sections(
    sections: list[tuple[SectionStructureDraft, ConsensusOutput]],
    *,
    profile: AccuracyProfile | None = None,
) -> None:
    profile = profile or load_accuracy_profile()
    groups: dict[str, list[tuple[SectionStructureDraft, ConsensusOutput]]] = defaultdict(list)
    for section, output in sections:
        if section.type in {"chorus", "verse", "pre_chorus"}:
            groups[section.type].append((section, output))

    for group in groups.values():
        if len(group) < 2:
            continue
        max_length = max(len(output.chords) for _, output in group)
        for index in range(max_length):
            confident_symbols = [
                output.chords[index].symbol
                for _, output in group
                if index < len(output.chords)
                and output.chords[index].confidence
                >= profile.repeated_section_winner_threshold
            ]
            if len(confident_symbols) < 2:
                continue
            winner = max(
                sorted(set(confident_symbols)),
                key=lambda value: confident_symbols.count(value),
            )
            if confident_symbols.count(winner) < 2:
                continue
            for _, output in group:
                if index >= len(output.chords):
                    continue
                chord = output.chords[index]
                if (
                    chord.confidence >= profile.repeated_section_confidence_threshold
                    or chord.symbol == winner
                    or winner not in chord.alternatives
                ):
                    continue
                chord.alternatives = [chord.symbol] + [
                    value
                    for value in chord.alternatives
                    if value not in {winner, chord.symbol}
                ]
                chord.symbol = winner
                chord.confidence = max(
                    chord.confidence,
                    profile.repeated_section_confidence_threshold + 0.04,
                )
                chord.agreement = max(
                    chord.agreement or 0.0,
                    profile.repeated_section_confidence_threshold + 0.04,
                )
