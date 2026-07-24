from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from urllib.parse import parse_qs, urlparse

import httpx


ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass(slots=True)
class YouTubeMetadata:
    video_id: str
    canonical_url: str
    title: str | None = None
    channel_title: str | None = None
    duration_seconds: float | None = None
    embeddable: bool | None = None
    privacy_status: str | None = None
    live_broadcast_content: str | None = None
    metadata_source: str = "url"

    def to_prompt_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    candidate = ""
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/", 1)[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif (
            parsed.path.startswith("/shorts/")
            or parsed.path.startswith("/embed/")
            or parsed.path.startswith("/live/")
        ):
            pieces = parsed.path.strip("/").split("/")
            candidate = pieces[1] if len(pieces) > 1 else ""
    if not VIDEO_ID_RE.fullmatch(candidate):
        raise ValueError("YouTube動画IDをURLから取得できませんでした。")
    return candidate


def parse_iso8601_duration(value: str | None) -> float | None:
    if not value:
        return None
    match = ISO_DURATION_RE.fullmatch(value)
    if not match:
        return None
    days = float(match.group("days") or 0)
    hours = float(match.group("hours") or 0)
    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0)
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total if total > 0 else None


def resolve_youtube_metadata(
    url: str,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 12.0,
) -> YouTubeMetadata:
    video_id = extract_video_id(url)
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    result = YouTubeMetadata(video_id=video_id, canonical_url=canonical_url)

    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            if api_key:
                response = client.get(
                    "https://www.googleapis.com/youtube/v3/videos",
                    params={
                        "part": "snippet,contentDetails,status",
                        "id": video_id,
                        "key": api_key,
                    },
                )
                if response.status_code == 200:
                    payload = response.json()
                    items = payload.get("items") or []
                    if items:
                        item = items[0]
                        snippet = item.get("snippet") or {}
                        content = item.get("contentDetails") or {}
                        status = item.get("status") or {}
                        result.title = snippet.get("title")
                        result.channel_title = snippet.get("channelTitle")
                        result.duration_seconds = parse_iso8601_duration(
                            content.get("duration")
                        )
                        result.embeddable = status.get("embeddable")
                        result.privacy_status = status.get("privacyStatus")
                        result.live_broadcast_content = snippet.get(
                            "liveBroadcastContent"
                        )
                        result.metadata_source = "youtube-data-api"
                        return result

            response = client.get(
                "https://www.youtube.com/oembed",
                params={"url": canonical_url, "format": "json"},
            )
            if response.status_code == 200:
                payload = response.json()
                result.title = payload.get("title")
                result.channel_title = payload.get("author_name")
                result.metadata_source = "youtube-oembed"
    except (httpx.HTTPError, ValueError):
        # Metadata improves precision but must never block the official Gemini
        # YouTube URL analysis path.
        return result
    return result
