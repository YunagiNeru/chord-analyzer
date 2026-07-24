from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from google.genai import types

from .model_gateway import ModelGateway
from .prompts import REFERENCE_ANALYSIS_SYSTEM_PROMPT, build_reference_prompt
from .reference_research import ReferenceResearchResult
from .schemas import ReferenceSource


BPM_RE = re.compile(
    r"\bBPM\s*[=:：]?\s*([4-9]\d|1\d\d|2[0-9]\d|3[0-2]\d)\b|"
    r"\b([4-9]\d|1\d\d|2[0-9]\d|3[0-2]\d)\s*BPM\b",
    re.IGNORECASE,
)
KEY_RE = re.compile(
    r"\b([A-G](?:#|b)?\s*(?:major|minor|メジャー|マイナー))\b",
    re.IGNORECASE,
)


def _facts_from_text(text: str) -> list[str]:
    facts: list[str] = []
    for match in BPM_RE.finditer(text):
        value = match.group(1) or match.group(2)
        if value:
            facts.append(f"BPM={int(value)}")
    for value in KEY_RE.findall(text):
        normalized = re.sub(r"\s+", " ", value.strip())
        facts.append(f"KEY={normalized}")
    return list(dict.fromkeys(facts))


def _grounded_sources(response: Any) -> list[ReferenceSource]:
    candidate = response.candidates[0] if getattr(response, "candidates", None) else None
    grounding = getattr(candidate, "grounding_metadata", None) if candidate else None
    if not grounding:
        return []

    chunks = list(getattr(grounding, "grounding_chunks", None) or [])
    sources: list[ReferenceSource] = []
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        url = getattr(web, "uri", None) if web else None
        title = getattr(web, "title", None) if web else None
        sources.append(
            ReferenceSource(
                title=title or url or "Unknown source",
                url=url or "about:blank",
                sourceType="google-search",
                facts=[],
            )
        )

    support_facts: dict[int, list[str]] = defaultdict(list)
    for support in getattr(grounding, "grounding_supports", None) or []:
        segment = getattr(support, "segment", None)
        segment_text = getattr(segment, "text", "") if segment else ""
        facts = _facts_from_text(segment_text or "")
        indices = list(getattr(support, "grounding_chunk_indices", None) or [])
        for index in indices:
            if isinstance(index, int) and 0 <= index < len(sources):
                support_facts[index].extend(facts)

    deduplicated: list[ReferenceSource] = []
    seen_urls: set[str] = set()
    for index, source in enumerate(sources):
        if source.url in seen_urls:
            continue
        seen_urls.add(source.url)
        source.facts = list(dict.fromkeys(support_facts.get(index, [])))
        deduplicated.append(source)
    return deduplicated


def _collect_candidates(
    sources: list[ReferenceSource],
) -> tuple[list[float], dict[float, int], list[str]]:
    bpm_sources: dict[float, set[str]] = defaultdict(set)
    key_candidates: list[str] = []
    for source in sources:
        for fact in source.facts:
            if fact.startswith("BPM="):
                try:
                    value = float(fact.split("=", 1)[1])
                except ValueError:
                    continue
                bpm_sources[value].add(source.url)
            elif fact.startswith("KEY="):
                key_candidates.append(fact.split("=", 1)[1].strip())
    support = {value: len(urls) for value, urls in bpm_sources.items()}
    bpms = sorted(
        [value for value, count in support.items() if count >= 2],
        key=lambda value: (-support[value], value),
    )
    return bpms[:8], support, list(dict.fromkeys(key_candidates))[:8]


def research_references_attributed(
    gateway: ModelGateway,
    metadata: dict[str, object],
) -> ReferenceResearchResult:
    prompt = (
        build_reference_prompt(metadata)
        + "\n各数値・調性は、その事実を記載している資料の近くに明示してください。"
        "複数資料の情報を一つの文へ混ぜないでください。"
    )
    response = gateway.generate_text(
        contents=prompt,
        system_instruction=REFERENCE_ANALYSIS_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=2_048,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    sources = _grounded_sources(response)
    bpms, support, keys = _collect_candidates(sources)
    return ReferenceResearchResult(
        summary=(response.text or "").strip(),
        sources=sources[:8],
        bpm_candidates=bpms,
        key_candidates=keys,
        bpm_support=support,
    )
