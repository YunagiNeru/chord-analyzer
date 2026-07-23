from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from music_agent.accuracy_profile import AccuracyProfile, load_accuracy_profile
from music_agent.chord_symbol import parse_chord
from music_agent.sequence_optimizer import optimize_sequence


def _score(reference: list[str], predicted: list[str]) -> float:
    if len(reference) != len(predicted) or not reference:
        return 0.0
    root = 0.0
    quality = 0.0
    for expected, actual in zip(reference, predicted, strict=True):
        left = parse_chord(expected)
        right = parse_chord(actual)
        if left.no_chord and right.no_chord:
            root += 1.0
            quality += 1.0
        elif left.root_pc is not None and left.root_pc == right.root_pc:
            root += 1.0
            if left.quality == right.quality:
                quality += 1.0
    count = len(reference)
    return 0.6 * (root / count) + 0.4 * (quality / count)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset must contain a non-empty cases array")
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        slots = case.get("slotScores")
        reference = case.get("reference")
        if not isinstance(slots, list) or not isinstance(reference, list):
            raise ValueError(f"case {index} requires slotScores and reference")
        if len(slots) != len(reference):
            raise ValueError(f"case {index} slot/reference length mismatch")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--base-profile",
        type=Path,
        default=Path("music_agent/accuracy_profile.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("accuracy-artifacts/tuned-accuracy-profile.json"),
    )
    args = parser.parse_args()

    base = AccuracyProfile.from_dict(
        json.loads(args.base_profile.read_text(encoding="utf-8"))
    )
    cases = _load_cases(args.dataset)

    best_profile = base
    best_score = -1.0
    candidates = itertools.product(
        (0.16, 0.22, 0.28, 0.34, 0.40),
        (0.12, 0.18, 0.22, 0.28, 0.34),
        (0.50, 0.54, 0.58, 0.62, 0.66),
    )
    for change_penalty, isolated_penalty, uncertainty_threshold in candidates:
        scores: list[float] = []
        for case in cases:
            predicted = optimize_sequence(
                case["slotScores"],
                change_penalty=change_penalty,
                isolated_penalty=isolated_penalty,
            )
            scores.append(_score(case["reference"], predicted))
        average = sum(scores) / len(scores)
        if average > best_score:
            best_score = average
            best_profile = replace(
                base,
                change_penalty=change_penalty,
                isolated_penalty=isolated_penalty,
                uncertainty_threshold=uncertainty_threshold,
            )

    payload = best_profile.to_dict()
    payload["tuningScore"] = round(best_score, 6)
    payload["tuningCases"] = len(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
