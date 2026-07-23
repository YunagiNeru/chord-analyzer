from __future__ import annotations

import re
from dataclasses import dataclass


NOTE_TO_PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}
PC_TO_NOTE = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NO_CHORD = {"N", "NC", "N.C.", "NO CHORD", "SILENCE", "NONE", "REST"}
UNKNOWN = {"X", "?", "UNKNOWN", "UNSURE"}


@dataclass(frozen=True, slots=True)
class ParsedChord:
    root: str | None
    quality: str
    seventh: str = "none"
    extensions: tuple[str, ...] = ()
    alterations: tuple[str, ...] = ()
    bass: str | None = None
    no_chord: bool = False
    unknown: bool = False

    @property
    def root_pc(self) -> int | None:
        if self.root is None:
            return None
        return NOTE_TO_PC.get(self.root.upper())

    @property
    def bass_pc(self) -> int | None:
        if self.bass is None:
            return None
        return NOTE_TO_PC.get(self.bass.upper())


ROOT_RE = re.compile(r"^([A-Ga-g])([#b♯♭]?)(.*)$")


def canonicalize_note(note: str | None) -> str | None:
    if not note:
        return None
    value = note.strip().replace("♯", "#").replace("♭", "b")
    match = re.fullmatch(r"([A-Ga-g])([#b]?)", value)
    if not match:
        return None
    token = (match.group(1).upper() + match.group(2)).upper()
    pc = NOTE_TO_PC.get(token)
    return PC_TO_NOTE[pc] if pc is not None else None


def _normalise_suffix(raw: str) -> str:
    value = raw.strip().replace("−", "-").replace("♭", "b").replace("♯", "#")
    value = value.replace("Δ", "maj").replace("△", "maj").replace("ø", "m7-5")
    value = value.replace("°", "dim").replace("＋", "+")
    value = value.replace("minor", "m").replace("min", "m")
    value = value.replace("major", "maj")
    value = re.sub(r"\s+", "", value)
    return value


def parse_chord(symbol: str | None) -> ParsedChord:
    if symbol is None:
        return ParsedChord(root=None, quality="unknown", unknown=True)
    value = symbol.strip()
    upper = value.upper()
    if upper in NO_CHORD:
        return ParsedChord(root=None, quality="no_chord", no_chord=True)
    if upper in UNKNOWN or not value:
        return ParsedChord(root=None, quality="unknown", unknown=True)

    if "/" in value:
        body, bass_raw = value.rsplit("/", 1)
        bass = canonicalize_note(bass_raw)
    else:
        body, bass = value, None

    match = ROOT_RE.match(body)
    if not match:
        return ParsedChord(root=None, quality="unknown", unknown=True)

    root = canonicalize_note(match.group(1) + match.group(2))
    suffix = _normalise_suffix(match.group(3))
    lower = suffix.lower()

    quality = "major"
    seventh = "none"
    extensions: list[str] = []
    alterations: list[str] = []

    if lower.startswith(("m7-5", "m7b5", "halfdim")):
        quality = "diminished"
        seventh = "minor7"
        lower = lower.replace("m7-5", "", 1).replace("m7b5", "", 1).replace("halfdim", "", 1)
    elif lower.startswith(("dim7", "o7")):
        quality = "diminished"
        seventh = "diminished7"
        lower = lower.replace("dim7", "", 1).replace("o7", "", 1)
    elif lower.startswith(("dim", "o")):
        quality = "diminished"
        lower = lower.replace("dim", "", 1).lstrip("o")
    elif lower.startswith(("aug", "+")):
        quality = "augmented"
        lower = lower.replace("aug", "", 1).lstrip("+")
    elif lower.startswith("sus2"):
        quality = "sus2"
        lower = lower[4:]
    elif lower.startswith("sus4") or lower == "sus":
        quality = "sus4"
        lower = lower[4:] if lower.startswith("sus4") else ""
    elif lower.startswith("5"):
        quality = "power"
        lower = lower[1:]
    elif lower.startswith("maj7"):
        quality = "major"
        seventh = "major7"
        lower = lower[4:]
    elif lower.startswith("maj"):
        quality = "major"
        lower = lower[3:]
    elif lower.startswith("m"):
        quality = "minor"
        lower = lower[1:]

    if seventh == "none":
        if lower.startswith("maj7"):
            seventh = "major7"
            lower = lower[4:]
        elif lower.startswith("7"):
            seventh = "minor7"
            lower = lower[1:]
        elif lower.startswith("6"):
            extensions.append("6")
            lower = lower[1:]

    for token in ("13", "11", "9", "6"):
        if token in lower:
            extensions.append(token)
            lower = lower.replace(token, "")

    for token in ("b13", "#11", "b9", "#9", "b5", "#5"):
        if token in suffix.lower():
            alterations.append(token)

    if "add9" in suffix.lower() and "9" not in extensions:
        extensions.append("add9")
    if "add11" in suffix.lower() and "11" not in extensions:
        extensions.append("add11")

    return ParsedChord(
        root=root,
        quality=quality,
        seventh=seventh,
        extensions=tuple(dict.fromkeys(extensions)),
        alterations=tuple(dict.fromkeys(alterations)),
        bass=bass,
    )


def format_chord(chord: ParsedChord, *, simplify: bool = False) -> str:
    if chord.no_chord:
        return "N"
    if chord.unknown or chord.root is None:
        return "X"

    suffix = ""
    if chord.quality == "minor":
        suffix = "m"
    elif chord.quality == "diminished":
        suffix = "dim"
    elif chord.quality == "augmented":
        suffix = "aug"
    elif chord.quality == "sus2":
        suffix = "sus2"
    elif chord.quality == "sus4":
        suffix = "sus4"
    elif chord.quality == "power":
        suffix = "5"

    if chord.quality == "diminished" and chord.seventh == "minor7":
        suffix = "m7-5"
    elif chord.seventh == "major7":
        suffix += "maj7"
    elif chord.seventh == "minor7":
        suffix += "7"
    elif chord.seventh == "diminished7":
        suffix = "dim7"

    if not simplify:
        for extension in chord.extensions:
            if extension.startswith("add"):
                suffix += extension
            elif extension not in suffix:
                suffix += extension
        for alteration in chord.alterations:
            if alteration not in suffix:
                suffix += alteration

    bass = ""
    if chord.bass and chord.bass != chord.root:
        bass = f"/{chord.bass}"
    return f"{chord.root}{suffix}{bass}"


def canonicalize_symbol(symbol: str | None, *, simplify: bool = False) -> str:
    return format_chord(parse_chord(symbol), simplify=simplify)


def root_pc(symbol: str | None) -> int | None:
    return parse_chord(symbol).root_pc


def quality_family(symbol: str | None) -> str:
    chord = parse_chord(symbol)
    if chord.no_chord:
        return "no_chord"
    if chord.unknown:
        return "unknown"
    return chord.quality


def transpose_symbol(symbol: str, semitones: int) -> str:
    chord = parse_chord(symbol)
    if chord.root_pc is None:
        return format_chord(chord)
    bass_pc = chord.bass_pc
    moved = ParsedChord(
        root=PC_TO_NOTE[(chord.root_pc + semitones) % 12],
        quality=chord.quality,
        seventh=chord.seventh,
        extensions=chord.extensions,
        alterations=chord.alterations,
        bass=PC_TO_NOTE[(bass_pc + semitones) % 12] if bass_pc is not None else None,
    )
    return format_chord(moved)


def chord_distance(left: str, right: str) -> float:
    a = parse_chord(left)
    b = parse_chord(right)
    if format_chord(a) == format_chord(b):
        return 0.0
    if a.no_chord or b.no_chord:
        return 1.4
    if a.unknown or b.unknown:
        return 0.8
    distance = 0.0
    if a.root_pc != b.root_pc:
        if a.root_pc is None or b.root_pc is None:
            distance += 1.0
        else:
            interval = abs(a.root_pc - b.root_pc)
            interval = min(interval, 12 - interval)
            distance += 0.55 + 0.08 * interval
    if a.quality != b.quality:
        distance += 0.55
    if a.seventh != b.seventh:
        distance += 0.16
    if a.bass_pc != b.bass_pc:
        distance += 0.12
    return distance


def same_root_quality(left: str, right: str) -> bool:
    a = parse_chord(left)
    b = parse_chord(right)
    return a.root_pc == b.root_pc and a.quality == b.quality
