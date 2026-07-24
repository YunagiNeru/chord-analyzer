from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .accuracy_profile import AccuracyProfile, load_accuracy_profile
from .chord_symbol import canonicalize_symbol
from .consensus import ConsensusOutput, reconcile_repeated_sections
from .schemas import ChordEvent, SectionStructureDraft, UncertainRange


@dataclass(slots=True)
class _MotifItem:
    section: SectionStructureDraft
    output: ConsensusOutput
    target: UncertainRange
    chord: ChordEvent
    previous_symbol: str | None
    next_symbol: str | None


def _simple(symbol: str | None) -> str:
    if not symbol:
        return ""
    return canonicalize_symbol(symbol, simplify=True)


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _context(output: ConsensusOutput, target: UncertainRange) -> tuple[ChordEvent | None, str | None, str | None]:
    overlapping = [
        (
            _overlap(
                target.startSeconds,
                target.endSeconds,
                chord.startSeconds,
                chord.endSeconds,
            ),
            chord,
        )
        for chord in output.chords
    ]
    overlapping = [item for item in overlapping if item[0] > 0.0]
    chord = max(overlapping, key=lambda item: (item[0], item[1].confidence))[1] if overlapping else None

    previous_symbol = None
    next_symbol = None
    for candidate in output.chords:
        if candidate.endSeconds <= target.startSeconds + 1e-3:
            previous_symbol = candidate.symbol
        elif candidate.startSeconds >= target.endSeconds - 1e-3:
            next_symbol = candidate.symbol
            break
    return chord, previous_symbol, next_symbol


def _candidate_signature(target: UncertainRange) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                _simple(value)
                for value in target.candidates
                if _simple(value) not in {"", "X", "N"}
            }
        )
    )


def _group_key(item: _MotifItem) -> tuple[object, ...]:
    duration = item.target.endSeconds - item.target.startSeconds
    return (
        item.section.type,
        _candidate_signature(item.target),
        _simple(item.previous_symbol),
        _simple(item.next_symbol),
        round(duration * 2.0) / 2.0,
    )


def _preferred_exact_symbol(item: _MotifItem, winner_simple: str) -> str | None:
    ordered = [item.chord.symbol] + item.target.candidates + item.chord.alternatives
    for symbol in ordered:
        canonical = canonicalize_symbol(symbol)
        if canonical not in {"X", "N"} and _simple(canonical) == winner_simple:
            return canonical
    return None


def _apply_winner(
    item: _MotifItem,
    winner_simple: str,
    *,
    minimum_confidence: float,
) -> bool:
    chosen = _preferred_exact_symbol(item, winner_simple)
    if not chosen:
        return False
    old = item.chord.symbol
    item.chord.symbol = chosen
    item.chord.confidence = max(item.chord.confidence, minimum_confidence)
    item.chord.agreement = max(item.chord.agreement or 0.0, minimum_confidence)
    item.chord.alternatives = list(
        dict.fromkeys(
            [old]
            + [
                value
                for value in item.chord.alternatives
                if canonicalize_symbol(value) != chosen
            ]
        )
    )[:3]
    item.target.resolved = True
    return True


def _reconcile_motif_groups(
    sections: list[tuple[SectionStructureDraft, ConsensusOutput]],
    profile: AccuracyProfile,
) -> None:
    items: list[_MotifItem] = []
    for section, output in sections:
        for target in output.uncertain_ranges:
            if target.resolved or "一致度" not in target.reason:
                continue
            chord, previous_symbol, next_symbol = _context(output, target)
            if chord is None or canonicalize_symbol(chord.symbol) in {"X", "N"}:
                continue
            items.append(
                _MotifItem(
                    section=section,
                    output=output,
                    target=target,
                    chord=chord,
                    previous_symbol=previous_symbol,
                    next_symbol=next_symbol,
                )
            )

    groups: dict[tuple[object, ...], list[_MotifItem]] = defaultdict(list)
    for item in items:
        groups[_group_key(item)].append(item)

    for group in groups.values():
        if len(group) < 3:
            continue
        confident = [
            _simple(item.chord.symbol)
            for item in group
            if item.chord.confidence >= profile.repeated_section_confidence_threshold
            and _simple(item.chord.symbol) not in {"", "X", "N"}
        ]
        if len(confident) < 2:
            continue
        counts = Counter(confident)
        winner, support = counts.most_common(1)[0]
        if support < 2 or support / len(confident) < 0.66:
            continue

        applied = 0
        for item in group:
            if _apply_winner(
                item,
                winner,
                minimum_confidence=profile.repeated_section_confidence_threshold + 0.04,
            ):
                applied += 1
        if applied >= 2:
            for item in group:
                item.output.uncertain_ranges = [
                    target
                    for target in item.output.uncertain_ranges
                    if not target.resolved
                ]


def reconcile_repeated_sections_enhanced(
    sections: list[tuple[SectionStructureDraft, ConsensusOutput]],
    *,
    profile: AccuracyProfile | None = None,
) -> None:
    resolved_profile = profile or load_accuracy_profile()
    reconcile_repeated_sections(sections, profile=resolved_profile)
    _reconcile_motif_groups(sections, resolved_profile)
