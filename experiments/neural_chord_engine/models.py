from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


_ROOT_ALIASES = {
    "CB": "B",
    "DB": "C#",
    "EB": "D#",
    "FB": "E",
    "GB": "F#",
    "AB": "G#",
    "BB": "A#",
    "B#": "C",
    "E#": "F",
}


@dataclass(frozen=True)
class ChordSegment:
    start_seconds: float
    end_seconds: float
    symbol: str
    source: str
    confidence: float = 1.0
    was_beat_snapped: bool = False

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("start_seconds must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NoChordRange:
    start_seconds: float
    end_seconds: float
    reason: str = "silence"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("start_seconds must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonicalize_symbol(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return "X"

    compact = text.upper().replace(" ", "")
    if compact in {"N", "NC", "N.C.", "NO_CHORD", "NOCHORD"}:
        return "N"
    if compact in {"X", "UNK", "UNKNOWN"}:
        return "X"

    text = (
        text.replace("♯", "#")
        .replace("♭", "b")
        .replace(":major", ":maj")
        .replace(":minor", ":min")
    )
    match = re.match(r"^([A-Ga-g])([#b]?)(?::)?(.*)$", text)
    if not match:
        return "X"

    root = f"{match.group(1).upper()}{match.group(2)}"
    root = _ROOT_ALIASES.get(root.upper(), root)
    quality = match.group(3).strip().lower()

    if quality in {"", "maj", "major"}:
        return root
    if quality in {"min", "minor", "m"}:
        return f"{root}m"
    if quality.startswith("maj7"):
        return f"{root}maj7"
    if quality.startswith(("min7", "m7")):
        return f"{root}m7"
    if quality.startswith("7"):
        return f"{root}7"
    if quality.startswith("sus4") or quality == "sus":
        return f"{root}sus4"
    if quality.startswith("dim"):
        return f"{root}dim"
    return root
