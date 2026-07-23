from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .schemas import (
    ChordAlternative,
    DspChordRun,
    DspSummary,
    KeyCandidate,
)

SAMPLE_RATE = 22_050
HOP_LENGTH = 512
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)

CHORD_QUALITIES: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "m7-5": (0, 3, 6, 10),
}


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.zeros_like(vector, dtype=float)
    return vector.astype(float) / norm


def _build_chord_templates() -> list[tuple[str, np.ndarray, int]]:
    templates: list[tuple[str, np.ndarray, int]] = []
    for root in range(12):
        for suffix, intervals in CHORD_QUALITIES.items():
            template = np.zeros(12, dtype=float)
            for interval in intervals:
                template[(root + interval) % 12] = 1.0
            # Root and fifth get a slight weight because they are generally more stable.
            template[root] += 0.22
            template[(root + 7) % 12] += 0.08
            symbol = f"{NOTE_NAMES[root]}{suffix}"
            templates.append((symbol, _normalise(template), len(intervals)))
    return templates


CHORD_TEMPLATES = _build_chord_templates()


def _estimate_keys(chroma: np.ndarray) -> list[KeyCandidate]:
    mean_chroma = np.mean(chroma, axis=1)
    mean_chroma = _normalise(mean_chroma)
    if not np.any(mean_chroma):
        return []

    candidates: list[tuple[str, float]] = []
    for root in range(12):
        major = _normalise(np.roll(MAJOR_PROFILE, root))
        minor = _normalise(np.roll(MINOR_PROFILE, root))
        candidates.append((f"{NOTE_NAMES[root]} major", float(np.dot(mean_chroma, major))))
        candidates.append((f"{NOTE_NAMES[root]} minor", float(np.dot(mean_chroma, minor))))

    candidates.sort(key=lambda item: item[1], reverse=True)
    raw = np.array([score for _, score in candidates], dtype=float)
    minimum = float(np.min(raw))
    maximum = float(np.max(raw))
    denominator = max(maximum - minimum, 1e-9)

    return [
        KeyCandidate(key=key, score=float(np.clip((score - minimum) / denominator, 0.0, 1.0)))
        for key, score in candidates[:5]
    ]


def _chord_candidates(vector: np.ndarray, top_k: int = 3) -> list[ChordAlternative]:
    vector = _normalise(vector)
    if not np.any(vector):
        return [ChordAlternative(symbol="N", score=0.0)]

    scored: list[tuple[str, float]] = []
    for symbol, template, note_count in CHORD_TEMPLATES:
        score = float(np.dot(vector, template))
        # Avoid preferring richer chords solely because they contain more matching notes.
        score -= max(0, note_count - 3) * 0.018
        scored.append((symbol, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    selected = scored[:top_k]
    values = np.array([score for _, score in selected], dtype=float)
    exp = np.exp((values - np.max(values)) * 7.0)
    probabilities = exp / max(float(np.sum(exp)), 1e-9)

    return [
        ChordAlternative(symbol=symbol, score=float(np.clip(probability, 0.0, 1.0)))
        for (symbol, _), probability in zip(selected, probabilities, strict=True)
    ]


def _beat_synced_chords(
    chroma: np.ndarray,
    rms: np.ndarray,
    beat_frames: np.ndarray,
    duration: float,
    sr: int,
) -> list[DspChordRun]:
    frame_count = chroma.shape[1]
    if frame_count == 0:
        return []

    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < frame_count)]

    if len(beat_frames) < 2:
        step_frames = max(1, int(round((sr / HOP_LENGTH) * 0.5)))
        beat_frames = np.arange(0, frame_count, step_frames, dtype=int)

    if len(beat_frames) == 0 or beat_frames[0] != 0:
        beat_frames = np.insert(beat_frames, 0, 0)
    if beat_frames[-1] != frame_count:
        beat_frames = np.append(beat_frames, frame_count)

    rms_values = rms.reshape(-1) if rms.size else np.zeros(frame_count, dtype=float)
    positive_rms = rms_values[rms_values > 0]
    silence_threshold = float(np.percentile(positive_rms, 8)) * 0.55 if positive_rms.size else 0.0

    raw_runs: list[DspChordRun] = []
    for start_frame, end_frame in zip(beat_frames[:-1], beat_frames[1:], strict=True):
        if end_frame <= start_frame:
            continue

        vector = np.mean(chroma[:, start_frame:end_frame], axis=1)
        local_window = rms_values[start_frame:min(end_frame, len(rms_values))]
        local_rms = float(np.mean(local_window)) if local_window.size else 0.0
        candidates = _chord_candidates(vector)
        best = candidates[0]

        symbol = best.symbol
        confidence = best.score
        if local_rms <= silence_threshold:
            symbol = "N"
            confidence = max(0.35, 1.0 - min(1.0, local_rms / max(silence_threshold, 1e-9)))

        start_seconds = float(librosa.frames_to_time(start_frame, sr=sr, hop_length=HOP_LENGTH))
        end_seconds = float(librosa.frames_to_time(end_frame, sr=sr, hop_length=HOP_LENGTH))
        end_seconds = min(duration, max(end_seconds, start_seconds + 0.05))

        alternatives = candidates[1:]
        run = DspChordRun(
            startSeconds=max(0.0, start_seconds),
            endSeconds=end_seconds,
            symbol=symbol,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            alternatives=alternatives,
        )

        if raw_runs and raw_runs[-1].symbol == run.symbol:
            previous = raw_runs[-1]
            previous.endSeconds = run.endSeconds
            previous.confidence = float((previous.confidence + run.confidence) / 2.0)
        else:
            raw_runs.append(run)

    return raw_runs[:600]


def _section_boundaries(y: np.ndarray, chroma: np.ndarray, sr: int, duration: float) -> list[float]:
    if duration < 12.0 or chroma.shape[1] < 8:
        return [0.0, round(duration, 3)]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8, hop_length=HOP_LENGTH)
    common_frames = min(chroma.shape[1], mfcc.shape[1])
    features = np.vstack(
        [
            librosa.util.normalize(chroma[:, :common_frames], axis=1),
            librosa.util.normalize(mfcc[1:7, :common_frames], axis=1),
        ]
    )
    novelty = np.linalg.norm(np.diff(features, axis=1), axis=0)
    novelty = gaussian_filter1d(novelty, sigma=max(1.0, (sr / HOP_LENGTH) * 0.8))

    frames_per_second = sr / HOP_LENGTH
    minimum_distance = max(1, int(frames_per_second * 8.0))
    prominence = max(float(np.std(novelty)) * 0.55, float(np.max(novelty)) * 0.05)
    peaks, properties = find_peaks(novelty, distance=minimum_distance, prominence=prominence)

    if not len(peaks):
        return [0.0, round(duration, 3)]

    prominences = properties.get("prominences", np.ones(len(peaks)))
    ranked = sorted(zip(peaks, prominences, strict=True), key=lambda item: item[1], reverse=True)
    max_internal = int(np.clip(round(duration / 24.0), 3, 12))
    selected_frames = sorted(frame for frame, _ in ranked[:max_internal])

    boundaries = [0.0]
    for frame in selected_frames:
        seconds = float(librosa.frames_to_time(frame + 1, sr=sr, hop_length=HOP_LENGTH))
        if seconds - boundaries[-1] >= 7.0 and duration - seconds >= 7.0:
            boundaries.append(round(seconds, 3))
    boundaries.append(round(duration, 3))
    return boundaries


def _waveform_peaks(y: np.ndarray, bins: int = 240) -> list[float]:
    if not len(y):
        return []
    bins = max(32, bins)
    edges = np.linspace(0, len(y), bins + 1, dtype=int)
    values: list[float] = []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        chunk = y[start:max(end, start + 1)]
        values.append(float(np.sqrt(np.mean(np.square(chunk))) if len(chunk) else 0.0))

    maximum = max(values, default=0.0)
    if maximum <= 1e-9:
        return [0.0 for _ in values]
    return [round(float(np.clip(value / maximum, 0.0, 1.0)), 4) for value in values]


def analyze_audio_file(path: str | Path) -> DspSummary:
    audio_path = Path(path)
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError("音声データが空です。")

    duration = float(librosa.get_duration(y=y, sr=sr))
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("音声の長さを取得できませんでした。")

    y = librosa.util.normalize(y)
    harmonic = librosa.effects.harmonic(y, margin=3.0)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH, units="frames")
    tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else None
    if tempo_value is not None and (not math.isfinite(tempo_value) or tempo_value <= 0):
        tempo_value = None

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    beat_times_list = [round(float(value), 3) for value in beat_times if 0.0 <= value <= duration]

    return DspSummary(
        durationSeconds=round(duration, 3),
        bpm=round(tempo_value, 2) if tempo_value else None,
        keyCandidates=_estimate_keys(chroma),
        beatTimes=beat_times_list[:2000],
        sectionBoundaryCandidates=_section_boundaries(y, chroma, sr, duration),
        chordRuns=_beat_synced_chords(chroma, rms, beat_frames, duration, sr),
        waveform=_waveform_peaks(y),
    )
