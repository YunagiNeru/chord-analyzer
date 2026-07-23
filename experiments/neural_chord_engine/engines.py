from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from models import ChordSegment, NoChordRange, canonicalize_symbol


def run_legacy(
    audio_path: Path,
    repository_root: Path,
) -> tuple[list[ChordSegment], float]:
    sys.path.insert(0, str(repository_root))

    from music_agent.dsp import analyze_audio_file

    started_at = time.perf_counter()
    result = analyze_audio_file(audio_path)
    elapsed = time.perf_counter() - started_at

    segments = [
        ChordSegment(
            start_seconds=float(item.startSeconds),
            end_seconds=float(item.endSeconds),
            symbol=canonicalize_symbol(item.symbol),
            source="legacy",
            confidence=float(item.confidence),
        )
        for item in result.chordRuns
    ]

    return segments, elapsed


def _find_omnizart_csv(
    output_dir: Path,
    started_at: float,
) -> Path:
    candidates = [
        path
        for path in output_dir.rglob("*.csv")
        if path.stat().st_mtime >= started_at - 2.0
    ]

    if not candidates:
        raise RuntimeError(
            "OmnizartのCSV出力が見つかりません。"
            " omnizart chord transcribe の出力を確認してください。"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _column_index(
    header: list[str],
    candidates: tuple[str, ...],
) -> int | None:
    for index, name in enumerate(header):
        if any(candidate in name for candidate in candidates):
            return index

    return None


def parse_omnizart_csv(path: Path) -> list[ChordSegment]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise RuntimeError(f"Omnizart CSVが空です: {path}")

    header = [cell.strip().lower() for cell in rows[0]]
    has_header = any(
        any(token in cell for token in ("start", "end", "chord", "label"))
        for cell in header
    )

    if has_header:
        start_index = _column_index(
            header,
            ("start", "onset", "begin"),
        )
        end_index = _column_index(
            header,
            ("end", "offset", "stop"),
        )
        chord_index = _column_index(
            header,
            ("chord", "label", "name", "symbol"),
        )
        data_rows = rows[1:]
    else:
        start_index = 0
        end_index = 1
        chord_index = 2
        data_rows = rows

    if start_index is None or end_index is None or chord_index is None:
        raise RuntimeError(
            f"Omnizart CSVの列を認識できません: header={rows[0]}"
        )

    maximum_index = max(start_index, end_index, chord_index)
    segments: list[ChordSegment] = []

    for row in data_rows:
        if maximum_index >= len(row):
            continue

        try:
            start = float(row[start_index])
            end = float(row[end_index])
        except ValueError:
            continue

        if end <= start:
            continue

        segments.append(
            ChordSegment(
                start_seconds=start,
                end_seconds=end,
                symbol=canonicalize_symbol(row[chord_index]),
                source="omnizart",
            )
        )

    if not segments:
        raise RuntimeError(
            f"Omnizart CSVからコードを取得できませんでした: {path}"
        )

    return segments


def run_omnizart(
    audio_path: Path,
    output_dir: Path,
) -> tuple[list[ChordSegment], float, Path]:
    executable = shutil.which("omnizart")

    if not executable:
        raise RuntimeError(
            "omnizart コマンドが見つかりません。"
            " setup_cloud_shell.sh を実行してください。"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    performance_start = time.perf_counter()

    subprocess.run(
        [
            executable,
            "chord",
            "transcribe",
            str(audio_path),
            "--output",
            str(output_dir),
        ],
        check=True,
    )

    elapsed = time.perf_counter() - performance_start
    csv_path = _find_omnizart_csv(output_dir, started_at)

    return parse_omnizart_csv(csv_path), elapsed, csv_path


def run_beat_this(
    audio_path: Path,
    model_name: str,
) -> tuple[list[float], list[float], float]:
    from beat_this.inference import File2Beats

    started_at = time.perf_counter()
    tracker = File2Beats(
        checkpoint_path=model_name,
        device="cpu",
        dbn=False,
    )
    beats, downbeats = tracker(str(audio_path))
    elapsed = time.perf_counter() - started_at

    beat_values = sorted(
        float(value)
        for value in np.asarray(beats).reshape(-1)
    )
    downbeat_values = sorted(
        float(value)
        for value in np.asarray(downbeats).reshape(-1)
    )

    return beat_values, downbeat_values, elapsed


def detect_silence_ranges(
    audio_path: Path,
    *,
    duration_seconds: float,
    top_db: float = 38.0,
    minimum_gap_seconds: float = 0.22,
) -> list[NoChordRange]:
    signal, sample_rate = librosa.load(
        audio_path,
        sr=22_050,
        mono=True,
    )
    intervals = librosa.effects.split(
        signal,
        top_db=top_db,
        ref=np.max,
        frame_length=2_048,
        hop_length=512,
    )

    if intervals.size == 0:
        return [
            NoChordRange(
                start_seconds=0.0,
                end_seconds=duration_seconds,
            )
        ]

    ranges: list[NoChordRange] = []
    first_start = float(intervals[0][0] / sample_rate)

    if first_start >= minimum_gap_seconds:
        ranges.append(
            NoChordRange(
                start_seconds=0.0,
                end_seconds=first_start,
            )
        )

    for previous, following in zip(
        intervals[:-1],
        intervals[1:],
        strict=True,
    ):
        start = float(previous[1] / sample_rate)
        end = float(following[0] / sample_rate)

        if end - start >= minimum_gap_seconds:
            ranges.append(
                NoChordRange(
                    start_seconds=start,
                    end_seconds=end,
                )
            )

    last_end = float(intervals[-1][1] / sample_rate)

    if duration_seconds - last_end >= minimum_gap_seconds:
        ranges.append(
            NoChordRange(
                start_seconds=last_end,
                end_seconds=duration_seconds,
            )
        )

    return ranges


def engine_versions() -> dict[str, str]:
    import importlib.metadata

    versions: dict[str, str] = {}

    for package in (
        "omnizart",
        "beat-this",
        "mir_eval",
        "librosa",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"

    return versions
