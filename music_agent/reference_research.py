from __future__ import annotations

import re
from dataclasses import dataclass, field

from google.genai import types

from .model_gateway import ModelGateway
from .prompts import REFERENCE_ANALYSIS_SYSTEM_PROMPT, build_reference_prompt
from .schemas import ReferenceSource


BPM_RE = re.compile(r"(?<!\d)([4-9]\d|1\d\d|2[0-4]\d)\s*BPM\b", re.IGNORECASE)
KEY_RE = re.compile(
    r"\b([A-G](?:#|b)?(?:\s*(?:major|minor|メジャー|マイナー)))\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ReferenceResearchResult:
    summary: str = ""
    sources: list[ReferenceSource] = field(default_factory=list)
    bpm_candidates: list[float] = field(default_factory=list)
    key_candidates: list[str] = field(default_factory=list)


def research_references(
    gateway: ModelGateway,
    metadata: dict[str, object],
) -> ReferenceResearchResult:
    response = gateway.generate_text(
        contents=build_reference_prompt(metadata),
        system_instruction=REFERENCE_ANALYSIS_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=2_048,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    text = (response.text or "").strip()
    sources: list[ReferenceSource] = []
    candidate = response.candidates[0] if response.candidates else None
    grounding = getattr(candidate, "grounding_metadata", None) if candidate else None
    if grounding:
        for chunk in getattr(grounding, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            url = getattr(web, "uri", None) if web else None
            title = getattr(web, "title", None) if web else None
            if url and not any(item.url == url for item in sources):
                sources.append(
                    ReferenceSource(
                        title=title or url,
                        url=url,
                        sourceType="google-search",
                        facts=[],
                    )
                )
    bpm_candidates = sorted({float(value) for value in BPM_RE.findall(text)})
    key_candidates = list(dict.fromkeys(match.strip() for match in KEY_RE.findall(text)))
    return ReferenceResearchResult(
        summary=text,
        sources=sources[:8],
        bpm_candidates=bpm_candidates[:6],
        key_candidates=key_candidates[:6],
    )
