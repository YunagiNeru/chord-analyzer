from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


_ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}


@dataclass(frozen=True)
class YoutubeAudioSource:
    audio_path: Path
    video_id: str
    title: str
    uploader: str | None
    webpage_url: str
    duration_seconds: float


def _validate_youtube_url(url: str) -> None:
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("YouTube URLは http または https で指定してください。")

    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"対応していないYouTubeホストです: {host or '(empty)'}")


def _find_downloaded_file(work_dir: Path) -> Path:
    candidates = [
        path
        for path in work_dir.glob("source.*")
        if path.is_file()
        and path.suffix not in {".part", ".ytdl", ".json"}
    ]

    if not candidates:
        raise RuntimeError("yt-dlpが取得した音声ファイルを確認できませんでした。")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def download_youtube_audio(
    url: str,
    work_dir: Path,
    *,
    max_duration_seconds: float = 1_800.0,
) -> YoutubeAudioSource:
    _validate_youtube_url(url)

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlpがインストールされていません。"
            " setup_cloud_shell.shを再実行するか、"
            " python -m pip install yt-dlp を実行してください。"
        ) from exc

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpegコマンドが見つかりません。")

    work_dir = work_dir.resolve()
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    common_options = {
        "noplaylist": True,
        "cachedir": False,
        "quiet": False,
        "no_warnings": False,
    }

    with YoutubeDL(common_options) as ydl:
        info = ydl.extract_info(url, download=False)

    if not isinstance(info, dict):
        raise RuntimeError("YouTube動画情報を取得できませんでした。")

    if info.get("is_live") or info.get("live_status") in {
        "is_live",
        "is_upcoming",
        "post_live",
    }:
        raise RuntimeError("ライブ配信または配信予定動画は解析対象外です。")

    duration = float(info.get("duration") or 0.0)
    if duration <= 0:
        raise RuntimeError("YouTube動画の長さを取得できませんでした。")

    if duration > max_duration_seconds:
        raise RuntimeError(
            "YouTube動画が解析上限を超えています: "
            f"duration={duration:.1f}s, limit={max_duration_seconds:.1f}s"
        )

    download_options = {
        **common_options,
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(work_dir / "source.%(ext)s"),
        "overwrites": True,
    }

    with YoutubeDL(download_options) as ydl:
        downloaded_info = ydl.extract_info(url, download=True)

    source_path = _find_downloaded_file(work_dir)
    wav_path = work_dir / "audio.wav"

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ],
        check=True,
    )

    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        raise RuntimeError("YouTube音声のWAV変換に失敗しました。")

    if source_path != wav_path:
        source_path.unlink(missing_ok=True)

    metadata = downloaded_info if isinstance(downloaded_info, dict) else info

    return YoutubeAudioSource(
        audio_path=wav_path,
        video_id=str(metadata.get("id") or info.get("id") or "unknown"),
        title=str(metadata.get("title") or info.get("title") or "Unknown title"),
        uploader=(
            str(metadata.get("uploader") or info.get("uploader"))
            if metadata.get("uploader") or info.get("uploader")
            else None
        ),
        webpage_url=str(
            metadata.get("webpage_url")
            or info.get("webpage_url")
            or url
        ),
        duration_seconds=duration,
    )
