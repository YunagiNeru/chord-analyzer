from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from .accuracy_profile import AccuracyProfile, load_accuracy_profile
from .beat_grid import BeatGrid
from .chord_symbol import canonicalize_symbol, parse_chord
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
    slot_agreements: list[float] = []
    slot_alternatives: list[list[str]] = []

    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        votes: dict[str, float] = defaultdict(float)
        for specialist in specialists:
            event = _best_event(specialist.chords, start, end)
            if event is None:
                continue
            role_weight = profile.role_weights.get(specialist.role, 0.75)
            _symbol_vote(
                votes,
                event.symbol,
                role_weight * event.confidence,
                simple_triad_reinforcement=profile.simple_triad_reinforcement,
            )
            for alternative in event.alternatives[:3]:
                _symbol_vote(
                    votes,
                    alternative,
                    role_weight * event.confidence * profile.alternative_weight,
                    simple_triad_reinforcement=profile.simple_triad_reinforcement,
                )

        if dsp_runs:
            dsp_event = _best_dsp_event(dsp_runs, start, end)
            if dsp_event is not None:
                _symbol_vote(
                    votes,
                    dsp_event.symbol,
                    profile.dsp_weight * dsp_event.confidence,
                    simple_triad_reinforcement=profile.simple_triad_reinforcement,
                )
                for alternative in dsp_event.alternatives[:2]:
                    _symbol_vote(
                        votes,
                        alternative.symbol,
                        profile.dsp_weight * profile.alternative_weight * alternative.score,
                        simple_triad_reinforcement=profile.simple_triad_reinforcement,
                    )

        if not votes:
            votes["X"] = 1.0

        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        total = sum(score for _, score in ranked)
        agreement = ranked[0][1] / total if total > 0 else 0.0
        slot_scores.append(dict(ranked[:8]))
        slot_agreements.append(float(agreement))
        slot_alternatives.append([symbol for symbol, _ in ranked[:4]])

    selected = optimize_sequence(
        slot_scores,
        change_penalty=profile.change_penalty,
        isolated_penalty=profile.isolated_penalty,
    )
    source = "hybrid" if dsp_runs else "ai"
    events: list[ChordEvent] = []
    for index, (start, end, symbol) in enumerate(
        zip(boundaries[:-1], boundaries[1:], selected, strict=True)
    ):
        alternatives = [item for item in slot_alternatives[index] if item != symbol][:3]
        confidence = min(0.99, max(0.05, slot_agreements[index]))
        event = ChordEvent(
            symbol=symbol,
            startSeconds=round(start, 3),
            endSeconds=round(end, 3),
            confidence=round(confidence, 4),
            source=source,
            alternatives=alternatives,
            agreement=round(slot_agreements[index], 4),
        )
        if events and events[-1].symbol == event.symbol:
            previous = events[-1]
            previous.endSeconds = event.endSeconds
            previous.confidence = round((previous.confidence + event.confidence) / 2.0, 4)
            values = [value for value in (previous.agreement, event.agreement) if value is not None]
            previous.agreement = round(mean(values), 4) if values else None
            previous.alternatives = list(dict.fromkeys(previous.alternatives + event.alternatives))[:3]
        else:
            events.append(event)

    uncertain_ranges: list[UncertainRange] = []
    current_start: float | None = None
    current_end = 0.0
    current_candidates: list[str] = []
    for index, agreement in enumerate(slot_agreements):
        uncertain = agreement < profile.uncertainty_threshold or selected[index] == "X"
        if uncertain:
            if current_start is None:
                current_start = boundaries[index]
                current_candidates = []
            current_end = boundaries[index + 1]
            current_candidates.extend(slot_alternatives[index])
        elif current_start is not None:
            uncertain_ranges.append(
                UncertainRange(
                    sectionId=section.id,
                    startSeconds=round(current_start, 3),
                    endSeconds=round(current_end, 3),
                    reason="専門分析間の一致度が閾値未満です。",
                    candidates=list(dict.fromkeys(current_candidates))[:8],
                )
            )
            current_start = None
            current_candidates = []
    if current_start is not None:
        uncertain_ranges.append(
            UncertainRange(
                sectionId=section.id,
                startSeconds=round(current_start, 3),
                endSeconds=round(current_end, 3),
                reason="専門分析間の一致度が閾値未満です。",
                candidates=list(dict.fromkeys(current_candidates))[:8],
            )
        )

    return ConsensusOutput(
        chords=events,
        agreement=round(mean(slot_agreements), 4) if slot_agreements else 0.0,
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
