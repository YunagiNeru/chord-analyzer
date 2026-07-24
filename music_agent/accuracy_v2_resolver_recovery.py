from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from . import accuracy_v2_finalization_v2 as finalization_v2
from . import pipeline_v2 as pipeline_module
from .accuracy_v2_finalization import ResolverWork, _apply_state_resolution, _split_uncertain_ranges, _state_count
from .diagnostics import DiagnosticsRecorder


@dataclass(slots=True)
class RecoveryWork:
    batch: list[ResolverWork]
    label: str
    depth: int = 0


def _max_recovery_calls() -> int:
    try:
        value = int(os.environ.get("MAX_RESOLVER_RECOVERY_CALLS", "12"))
    except ValueError:
        value = 12
    return max(0, min(48, value))


def _batch_clip(batch: list[ResolverWork], *, grid, duration: float) -> tuple[float, float]:
    bar_duration = grid.beat_duration * grid.beats_per_bar
    return (
        max(0.0, min(item.target.startSeconds for item in batch) - bar_duration),
        min(duration, max(item.target.endSeconds for item in batch) + bar_duration),
    )


def _generate_batch(
    *,
    resolver_gateway,
    media,
    batch: list[ResolverWork],
    grid,
    duration: float,
    label: str,
    retries: int,
) -> finalization_v2.BatchResolutionDraft:
    clip_start, clip_end = _batch_clip(batch, grid=grid, duration=duration)
    return resolver_gateway.generate_typed(
        contents=[
            media.part(clip_start, clip_end),
            finalization_v2._batch_prompt(batch, grid),
        ],
        schema=finalization_v2.BatchResolutionDraft,
        system_instruction=(
            "You are a deterministic chord-state resolver. "
            "Return one independent allow-listed decision for every targetIndex. "
            "Keep the JSON compact and complete."
        ),
        temperature=0.0,
        max_output_tokens=max(768, len(batch) * 256),
        retries=retries,
        diagnostic_label=label,
    )


def _apply_decisions(
    batch: list[ResolverWork],
    decision: finalization_v2.BatchResolutionDraft,
    *,
    diagnostics: DiagnosticsRecorder,
    label: str,
) -> tuple[int, list[ResolverWork]]:
    choices: dict[int, finalization_v2.BatchResolutionChoice] = {}
    for choice in decision.choices:
        if 0 <= choice.targetIndex < len(batch):
            choices.setdefault(choice.targetIndex, choice)

    applied = 0
    retry_work: list[ResolverWork] = []
    for index, work in enumerate(batch):
        choice = choices.get(index)
        if choice is None:
            retry_work.append(work)
            continue
        try:
            if _apply_state_resolution(work, choice):
                applied += 1
        except Exception as exc:  # noqa: BLE001
            diagnostics.record_error(f"{label}-target-{index}", exc)
            retry_work.append(work)
    return applied, retry_work


def _split_for_recovery(batch: list[ResolverWork]) -> list[list[ResolverWork]]:
    if len(batch) <= 1:
        return [batch]
    midpoint = max(1, len(batch) // 2)
    return [batch[:midpoint], batch[midpoint:]]


def _patched_run_resolvers(
    self,
    *,
    media,
    resolver_gateway,
    consensus_pairs,
    grid,
    duration,
    diagnostics,
    global_warnings,
):
    outputs = [output for _, output in consensus_pairs]
    before_count = _state_count(outputs)
    all_ranges = []
    all_work: list[ResolverWork] = []
    for section, output in consensus_pairs:
        ranges, work = _split_uncertain_ranges(section, output)
        all_ranges.extend(ranges)
        all_work.extend(work)

    batches = finalization_v2.build_resolver_batches(
        all_work,
        batch_size=finalization_v2._resolver_batch_size(),
        max_batches=self.max_resolver_calls,
    )
    recovery_queue: list[RecoveryWork] = []

    with diagnostics.stage("targeted-resolution"):
        with ThreadPoolExecutor(
            max_workers=min(self.max_parallel_calls, max(1, len(batches)))
        ) as executor:
            future_map: dict[
                Future[finalization_v2.BatchResolutionDraft],
                tuple[list[ResolverWork], str],
            ] = {}
            for batch_index, batch in enumerate(batches, start=1):
                label = f"resolver-batch-{batch_index}"
                future = executor.submit(
                    _generate_batch,
                    resolver_gateway=resolver_gateway,
                    media=media,
                    batch=batch,
                    grid=grid,
                    duration=duration,
                    label=label,
                    retries=3,
                )
                future_map[future] = (batch, label)
                diagnostics.resolver_calls += 1

            for future in as_completed(future_map):
                batch, label = future_map[future]
                try:
                    decision = future.result()
                except Exception as exc:  # noqa: BLE001
                    diagnostics.record_error(label, exc)
                    for sub_index, sub_batch in enumerate(
                        _split_for_recovery(batch),
                        start=1,
                    ):
                        recovery_queue.append(
                            RecoveryWork(
                                batch=sub_batch,
                                label=f"{label}-recovery-{sub_index}",
                                depth=1,
                            )
                        )
                    continue

                applied, retry_work = _apply_decisions(
                    batch,
                    decision,
                    diagnostics=diagnostics,
                    label=label,
                )
                if applied == 0 and not retry_work and batch:
                    global_warnings.append(f"{label} resolved no chord states")
                if retry_work:
                    recovery_queue.append(
                        RecoveryWork(
                            batch=retry_work,
                            label=f"{label}-missing-recovery",
                            depth=1,
                        )
                    )

        recovery_budget = _max_recovery_calls()
        recovery_calls = 0
        while recovery_queue and recovery_calls < recovery_budget:
            item = recovery_queue.pop(0)
            if not item.batch:
                continue
            label = f"{item.label}-size{len(item.batch)}"
            recovery_calls += 1
            diagnostics.resolver_calls += 1
            try:
                decision = _generate_batch(
                    resolver_gateway=resolver_gateway,
                    media=media,
                    batch=item.batch,
                    grid=grid,
                    duration=duration,
                    label=label,
                    retries=2,
                )
            except Exception as exc:  # noqa: BLE001
                diagnostics.record_error(label, exc)
                if len(item.batch) > 1:
                    for sub_index, sub_batch in enumerate(
                        _split_for_recovery(item.batch),
                        start=1,
                    ):
                        recovery_queue.append(
                            RecoveryWork(
                                batch=sub_batch,
                                label=f"{item.label}-{sub_index}",
                                depth=item.depth + 1,
                            )
                        )
                else:
                    global_warnings.append(
                        f"{label} failed after bounded recovery; range remains uncertain"
                    )
                continue

            _, retry_work = _apply_decisions(
                item.batch,
                decision,
                diagnostics=diagnostics,
                label=label,
            )
            if retry_work:
                if len(retry_work) == len(item.batch) and len(retry_work) > 1:
                    next_batches = _split_for_recovery(retry_work)
                else:
                    next_batches = [retry_work]
                for sub_index, sub_batch in enumerate(next_batches, start=1):
                    if sub_batch:
                        recovery_queue.append(
                            RecoveryWork(
                                batch=sub_batch,
                                label=f"{item.label}-missing-{sub_index}",
                                depth=item.depth + 1,
                            )
                        )

        remaining_recovery_states = sum(len(item.batch) for item in recovery_queue)
        if remaining_recovery_states:
            global_warnings.append(
                "resolver recovery budget exhausted; "
                f"{remaining_recovery_states} states remain for the quality gate"
            )

    after_count = _state_count(outputs)
    if before_count - after_count >= 4 and after_count < before_count * 0.70:
        diagnostics.record_required_failure(
            f"resolver_state_collapse:{before_count}->{after_count}"
        )
    diagnostics.chord_state_count = after_count
    return all_ranges


def install_accuracy_v2_resolver_recovery() -> None:
    if getattr(pipeline_module, "_accuracy_v2_resolver_recovery_installed", False):
        return
    pipeline_module._accuracy_v2_resolver_recovery_installed = True
    pipeline_module.AccuracyPipelineV2._run_resolvers = _patched_run_resolvers
