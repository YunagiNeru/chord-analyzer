from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

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
        ranked = sorted(
            self.bpm_candidates,
            key=lambda value: (
                -self.bpm_support.get(value, 0),
                value,
            ),
        )
        candidate = ranked[0]
        support = self.bpm_support.get(candidate, 0)
        # A single ungrounded mention is not strong enough to override audio.
        if support >= 2 or (support >= 1 and len(self.sources) >= 2):
            return candidate
        return None


def _extract_bpms(text: str) -> tuple[list[float], dict[float, int]]:
    values: list[float] = []
    for pattern in BPM_PATTERNS:
        values.extend(float(value) for value in pattern.findall(text))

    # Nearby values from different catalogues are treated as one tempo family.
    # This preserves 167/170 as separate evidence but ranks the denser cluster.
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

    bpm_candidates, bpm_support = _extract_bpms(text)
    key_candidates = list(
        dict.fromkeys(match.strip() for match in KEY_RE.findall(text))
    )[:8]
    facts: list[str] = []
    if bpm_candidates:
        facts.append(
            "BPM候補: "
            + ", ".join(
                f"{value:g}(support={bpm_support.get(value, 0)})"
                for value in bpm_candidates
            )
        )
    if key_candidates:
        facts.append("キー候補: " + ", ".join(key_candidates))
    for source in sources:
        source.facts = list(facts)

    return ReferenceResearchResult(
        summary=text,
        sources=sources[:8],
        bpm_candidates=bpm_candidates,
        key_candidates=key_candidates,
        bpm_support=bpm_support,
    )
