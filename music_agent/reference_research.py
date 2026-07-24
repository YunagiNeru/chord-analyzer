from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from google.genai import types

from .model_gateway import ModelGateway
from .prompts import REFERENCE_ANALYSIS_SYSTEM_PROMPT, build_reference_prompt
from .schemas import ReferenceSource


BPM_PATTERNS = (
    re.compile(r"\bBPM\s*[=:：]?\s*([4-9]\d|1\d\d|2[0-9]\d|3[0-2]\d)\b", re.IGNORECASE),
    re.compile(r"\b([4-9]\d|1\d\d|2[0-9]\d|3[0-2]\d)\s*BPM\b", re.IGNORECASE),
)
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
    bpm_support: dict[float, int] = field(default_factory=dict)

    @property
    def primary_bpm(self) -> float | None:
        if not self.bpm_candidates:
            return None
        return sorted(
            self.bpm_candidates,
            key=lambda value: (
                -self.bpm_support.get(value, 0),
                value,
            ),
        )[0]


def _extract_bpms(text: str) -> tuple[list[float], dict[float, int]]:
    values: list[float] = []
    for pattern in BPM_PATTERNS:
        values.extend(float(value) for value in pattern.findall(text))

    counter = Counter(values)
    candidates = sorted(counter)
    support: dict[float, int] = {}
    for candidate in candidates:
        support[candidate] = sum(
            count
            for value, count in counter.items()
            if abs(value - candidate) <= 1.5
        )
    return candidates[:8], support


def _append_grounding_sources(
    response: Any,
    sources: list[ReferenceSource],
) -> None:
    candidate = response.candidates[0] if response.candidates else None
    grounding = getattr(candidate, "grounding_metadata", None) if candidate else None
    if not grounding:
        return
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


def _trusted_bpms(
    raw_candidates: list[float],
    support: dict[float, int],
    source_count: int,
) -> list[float]:
    return [
        value
        for value in raw_candidates
        if support.get(value, 0) >= 2
        or (support.get(value, 0) >= 1 and source_count >= 2)
    ]


def research_references(
    gateway: ModelGateway,
    metadata: dict[str, object],
) -> ReferenceResearchResult:
    sources: list[ReferenceSource] = []
    texts: list[str] = []

    response = gateway.generate_text(
        contents=build_reference_prompt(metadata),
        system_instruction=REFERENCE_ANALYSIS_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=2_048,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    texts.append((response.text or "").strip())
    _append_grounding_sources(response, sources)

    combined_text = "\n".join(texts)
    raw_bpm_candidates, bpm_support = _extract_bpms(combined_text)
    bpm_candidates = _trusted_bpms(
        raw_bpm_candidates,
        bpm_support,
        len(sources),
    )

    if not bpm_candidates:
        retry_prompt = (
            build_reference_prompt(metadata)
            + "\n前回は独立資料で裏付けられたBPMを抽出できませんでした。"
            "対象はprovidedMetadataと完全一致する同一録音です。"
            "BPMだけを最低2資料で検索し、資料ごとに `BPM=数値` と明記してください。"
        )
        retry_response = gateway.generate_text(
            contents=retry_prompt,
            system_instruction=REFERENCE_ANALYSIS_SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=1_536,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        texts.append((retry_response.text or "").strip())
        _append_grounding_sources(retry_response, sources)
        combined_text = "\n".join(texts)
        raw_bpm_candidates, bpm_support = _extract_bpms(combined_text)
        bpm_candidates = _trusted_bpms(
            raw_bpm_candidates,
            bpm_support,
            len(sources),
        )

    key_candidates = list(
        dict.fromkeys(match.strip() for match in KEY_RE.findall(combined_text))
    )[:8]
    facts: list[str] = []
    if raw_bpm_candidates:
        facts.append(
            "BPM候補: "
            + ", ".join(
                f"{value:g}(support={bpm_support.get(value, 0)})"
                for value in raw_bpm_candidates
            )
        )
    if key_candidates:
        facts.append("キー候補: " + ", ".join(key_candidates))
    for source in sources:
        source.facts = list(facts)

    return ReferenceResearchResult(
        summary=combined_text,
        sources=sources[:8],
        bpm_candidates=bpm_candidates,
        key_candidates=key_candidates,
        bpm_support=bpm_support,
    )
