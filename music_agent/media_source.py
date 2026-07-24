from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from google.genai import types


class AnalysisMediaSource(ABC):
    @abstractmethod
    def part(self, start: float | None = None, end: float | None = None) -> types.Part:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> "AnalysisMediaSource":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class YouTubeMediaSource(AnalysisMediaSource):
    def __init__(self, url: str) -> None:
        self.url = url

    @staticmethod
    def _duration(value: float) -> str:
        return f"{max(0.0, value):.3f}s"

    def part(self, start: float | None = None, end: float | None = None) -> types.Part:
        if start is None and end is None:
            return types.Part.from_uri(file_uri=self.url, mime_type="video/mp4")
        metadata = types.VideoMetadata(
            start_offset=self._duration(start or 0.0),
            end_offset=self._duration(end) if end is not None else None,
            fps=1.0,
        )
        return types.Part(
            file_data=types.FileData(file_uri=self.url, mime_type="video/mp4"),
            video_metadata=metadata,
        )


class LocalAudioMediaSource(AnalysisMediaSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._temporary = tempfile.TemporaryDirectory(prefix="chord-analysis-clips-")
        self._cache: dict[tuple[float, float], Path] = {}

    def _clip_path(self, start: float, end: float) -> Path:
        key = (round(max(0.0, start), 3), round(max(start + 0.1, end), 3))
        cached = self._cache.get(key)
        if cached and cached.is_file():
            return cached
        destination = Path(self._temporary.name) / f"clip-{len(self._cache):03d}.mp3"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{key[0]:.3f}",
                "-i",
                str(self.path),
                "-t",
                f"{key[1] - key[0]:.3f}",
                "-vn",
                "-ac",
                "2",
                "-ar",
                "32000",
                "-b:a",
                "96k",
                str(destination),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self._cache[key] = destination
        return destination

    def part(self, start: float | None = None, end: float | None = None) -> types.Part:
        selected = self.path
        if start is not None and end is not None and end > start + 0.1:
            selected = self._clip_path(start, end)
        return types.Part.from_bytes(data=selected.read_bytes(), mime_type="audio/mpeg")

    def close(self) -> None:
        self._temporary.cleanup()
