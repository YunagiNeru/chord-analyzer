from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .chord_symbol import canonicalize_symbol
from .schemas import ChordAlternative, DspChordRun, DspSummary, KeyCandidate
from .sequence_optimizer import optimize_sequence


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
            template[root] += 0.26
            template[(root + 7) % 12] += 0.08
            templates.append((f"{NOTE_NAMES[root]}{suffix}", _normalise(template), len(intervals)))
    return templates


CHORD_TEMPLATES = _build_chord_templates()


def _estimate_keys(chroma: np.ndarray) -> list[KeyCandidate]:
    mean_chroma = _normalise(np.mean(chroma, axis=1))
    if not np.any(mean_chroma):
        return []
    candidates: list[tuple[str, float]] = []
    for root in range(12):
        candidates.append(
            (f"{NOTE_NAMES[root]} major", float(np.dot(mean_chroma, _normalise(np.roll(MAJOR_PROFILE, root)))))
        )
        candidates.append(
            (f"{NOTE_NAMES[root]} minor", float(np.dot(mean_chroma, _normalise(np.roll(MINOR_PROFILE, root)))))
        )
    candidates.sort(key=lambda item: item[1], reverse=True)
    values = np.asarray([score for _, score in candidates], dtype=float)
    minimum = float(values.min())
    spread = max(float(values.max()) - minimum, 1e-9)
    return [
        KeyCandidate(key=key, score=float(np.clip((score - minimum) / spread, 0.0, 1.0)))
        for key, score in candidates[:5]
    ]


def _candidate_scores(vector: np.ndarray, bass_vector: np.ndarray, top_k: int = 5) -> dict[str, float]:
    vector = _normalise(vector)
    bass_vector = _normalise(bass_vector)
    if not np.any(vector):
        return {"N": 1.0}
    scored: list[tuple[str, float]] = []
    for symbol, template, note_count in CHORD_TEMPLATES:
        root = NOTE_NAMES.index(symbol[:2]) if len(symbol) > 1 and symbol[1] == "#" else NOTE_NAMES.index(symbol[0])
        score = float(np.dot(vector, template))
        score += float(bass_vector[root]) * 0.18
        score -= max(0, note_count - 3) * 0.018
        scored.append((canonicalize_symbol(symbol), score))
    scored.sort(key=lambda item: item[1], reverse=True)
    selected = scored[:top_k]
    values = np.asarray([score for _, score in selected], dtype=float)
    probabilities = np.exp((values - values.max()) * 6.0)
    probabilities /= max(float(probabilities.sum()), 1e-9)
    return {symbol: float(probability) for (symbol, _), probability in zip(selected, probabilities, strict=True)}


def _beat_synced_chords(
    chroma: np.ndarray,
    bass_chroma: np.ndarray,
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
    slot_scores: list[dict[str, float]] = []
    intervals: list[tuple[float, float]] = []

    for start_frame, end_frame in zip(beat_frames[:-1], beat_frames[1:], strict=True):
        if end_frame <= start_frame:
            continue
        full_vector = np.median(chroma[:, start_frame:end_frame], axis=1)
        bass_vector = np.median(bass_chroma[:, start_frame:end_frame], axis=1)
        local_rms_values = rms_values[start_frame:min(end_frame, len(rms_values))]
        local_rms = float(np.mean(local_rms_values)) if local_rms_values.size else 0.0
        scores = _candidate_scores(full_vector, bass_vector)
        if local_rms <= silence_threshold:
            scores = {"N": max(0.8, 1.0 - local_rms / max(silence_threshold, 1e-9)), **scores}
        start_seconds = float(librosa.frames_to_time(start_frame, sr=sr, hop_length=HOP_LENGTH))
        end_seconds = float(librosa.frames_to_time(end_frame, sr=sr, hop_length=HOP_LENGTH))
        intervals.append((max(0.0, start_seconds), min(duration, max(end_seconds, start_seconds + 0.05))))
        slot_scores.append(scores)

    selected = optimize_sequence(slot_scores, change_penalty=0.33, isolated_penalty=0.28)
    output: list[DspChordRun] = []
    for (start, end), symbol, scores in zip(intervals, selected, slot_scores, strict=True):
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        confidence = scores.get(symbol, ranked[0][1] if ranked else 0.0)
        alternatives = [
            ChordAlternative(symbol=item, score=float(score))
            for item, score in ranked
            if item != symbol
        ][:3]
        run = DspChordRun(
            startSeconds=round(start, 3),
            endSeconds=round(end, 3),
            symbol=symbol,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            alternatives=alternatives,
        )
        if output and output[-1].symbol == run.symbol:
            previous = output[-1]
            previous.endSeconds = run.endSeconds
            previous.confidence = round((previous.confidence + run.confidence) / 2.0, 4)
        else:
            output.append(run)
    return output[:600]


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
    edges = np.linspace(0, len(y), max(32, bins) + 1, dtype=int)
    values = [
        float(np.sqrt(np.mean(np.square(y[start:max(end, start + 1)]))))
        for start, end in zip(edges[:-1], edges[1:], strict=True)
    ]
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
    harmonic, percussive = librosa.effects.hpss(y, margin=(2.0, 3.0))
    chroma_cqt = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=HOP_LENGTH)
    chroma_cens = librosa.feature.chroma_cens(y=harmonic, sr=sr, hop_length=HOP_LENGTH)
    common = min(chroma_cqt.shape[1], chroma_cens.shape[1])
    chroma = 0.72 * chroma_cqt[:, :common] + 0.28 * chroma_cens[:, :common]
    bass_harmonic = librosa.effects.preemphasis(harmonic, coef=-0.85)
    bass_chroma = librosa.feature.chroma_cqt(y=bass_harmonic, sr=sr, hop_length=HOP_LENGTH)
    bass_chroma = bass_chroma[:, :common]
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[:, :common]

    tempo, beat_frames = librosa.beat.beat_track(
        y=percussive,
        sr=sr,
        hop_length=HOP_LENGTH,
        units="frames",
        sparse=True,
    )
    tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else None
    if tempo_value is not None and (not math.isfinite(tempo_value) or tempo_value <= 0):
        tempo_value = None
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_frames = beat_frames[beat_frames < common]
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    return DspSummary(
        durationSeconds=round(duration, 3),
        bpm=round(tempo_value, 2) if tempo_value else None,
        keyCandidates=_estimate_keys(chroma),
        beatTimes=[round(float(value), 3) for value in beat_times if 0.0 <= value <= duration][:2000],
        sectionBoundaryCandidates=_section_boundaries(y, chroma, sr, duration),
        chordRuns=_beat_synced_chords(chroma, bass_chroma, rms, beat_frames, duration, sr),
        waveform=_waveform_peaks(y),
    )
