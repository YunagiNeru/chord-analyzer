from __future__ import annotations

from . import pipeline_v2 as pipeline_module
from .chord_symbol import canonicalize_symbol
from .validators import validate_quality as base_validate_quality


_MAJOR_TYPES = {"verse", "pre_chorus", "chorus", "post_chorus"}


def _merged_state_count(section) -> int:
    events = sorted(
        [chord for measure in section.measures for chord in measure.chords],
        key=lambda item: (item.startSeconds, item.endSeconds),
    )
    count = 0
    previous_symbol: str | None = None
    previous_end = -1.0
    for event in events:
        symbol = canonicalize_symbol(event.symbol)
        if symbol != previous_symbol or event.startSeconds > previous_end + 0.003:
            count += 1
        previous_symbol = symbol
        previous_end = max(previous_end, event.endSeconds)
    return count


def validate_quality_extended(result) -> list[str]:
    errors = list(base_validate_quality(result))
    bpm = result.track.bpm or 120.0
    try:
        beats_per_bar = max(1, int((result.track.timeSignature or "4/4").split("/", 1)[0]))
    except (TypeError, ValueError, IndexError):
        beats_per_bar = 4
    bar_duration = (60.0 / bpm) * beats_per_bar

    for section in result.sections:
        bars = (section.endSeconds - section.startSeconds) / max(0.001, bar_duration)
        agreement = section.agreement if section.agreement is not None else 0.0
        if (
            section.type in _MAJOR_TYPES
            and bars >= 7.5
            and _merged_state_count(section) <= 1
            and agreement < 0.60
        ):
            errors.append(
                f"implausible_static_section:{section.id}:{bars:.2f}bars:agreement={agreement:.3f}"
            )
    return sorted(set(errors))


def install_quality_extension() -> None:
    if getattr(pipeline_module, "_quality_extension_installed", False):
        return
    pipeline_module._quality_extension_installed = True
    pipeline_module.validate_quality = validate_quality_extended
