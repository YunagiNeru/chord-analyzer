from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .chord_symbol import canonicalize_symbol, chord_distance


def optimize_sequence(
    slot_scores: Sequence[Mapping[str, float]],
    *,
    change_penalty: float = 0.28,
    isolated_penalty: float = 0.22,
) -> list[str]:
    """Select a globally consistent chord sequence with deterministic Viterbi DP."""
    if not slot_scores:
        return []

    states_per_slot: list[list[str]] = []
    emissions: list[dict[str, float]] = []
    for raw in slot_scores:
        cleaned = {
            canonicalize_symbol(symbol): max(1e-9, float(score))
            for symbol, score in raw.items()
        }
        if not cleaned:
            cleaned = {"X": 1.0}
        ranked = sorted(cleaned, key=lambda symbol: (-cleaned[symbol], symbol))[:8]
        states_per_slot.append(ranked)
        total = sum(cleaned[symbol] for symbol in ranked)
        emissions.append({symbol: math.log(cleaned[symbol] / total) for symbol in ranked})

    costs: list[dict[str, float]] = []
    backpointers: list[dict[str, str | None]] = []
    first_costs = {symbol: -emissions[0][symbol] for symbol in states_per_slot[0]}
    costs.append(first_costs)
    backpointers.append({symbol: None for symbol in states_per_slot[0]})

    for index in range(1, len(states_per_slot)):
        current_costs: dict[str, float] = {}
        current_back: dict[str, str] = {}
        for current in states_per_slot[index]:
            best_cost = float("inf")
            best_previous = states_per_slot[index - 1][0]
            for previous in states_per_slot[index - 1]:
                transition = 0.0
                if current != previous:
                    transition = change_penalty + 0.12 * chord_distance(previous, current)
                candidate = costs[index - 1][previous] + transition - emissions[index][current]
                if candidate < best_cost - 1e-12 or (
                    abs(candidate - best_cost) <= 1e-12 and previous < best_previous
                ):
                    best_cost = candidate
                    best_previous = previous
            current_costs[current] = best_cost
            current_back[current] = best_previous
        costs.append(current_costs)
        backpointers.append(current_back)

    final = min(costs[-1], key=lambda symbol: (costs[-1][symbol], symbol))
    sequence = [final]
    for index in range(len(states_per_slot) - 1, 0, -1):
        previous = backpointers[index][sequence[-1]]
        assert previous is not None
        sequence.append(previous)
    sequence.reverse()

    # Remove weak one-slot excursions when both neighbours agree and the local
    # emission does not strongly support the excursion.
    output = list(sequence)
    for index in range(1, len(output) - 1):
        if output[index - 1] != output[index + 1] or output[index] == output[index - 1]:
            continue
        stay = emissions[index].get(output[index - 1], math.log(1e-9))
        switch = emissions[index].get(output[index], math.log(1e-9))
        if switch - stay < isolated_penalty:
            output[index] = output[index - 1]
    return output
