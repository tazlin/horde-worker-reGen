"""Human-readable rendering of benchmark progress events for the CLI live view and ``monitor``.

Kept separate from the channel models so the rendering is reusable and unit-testable: the live ramp
console (a sink) and the ``monitor`` tail loop both format events through :func:`format_progress_event`.
"""

from __future__ import annotations

from horde_worker_regen.benchmark.progress_channel import (
    BenchmarkProgressEvent,
    LevelFinished,
    LevelProgress,
    LevelStarted,
    ProgressSink,
    RampFinished,
    RampStarted,
)


def format_progress_event(event: BenchmarkProgressEvent, *, verbose: bool = False) -> str | None:
    """Return a one-line human-readable rendering of a progress event, or None to skip it.

    With ``verbose`` set, level-progress lines also carry the compact per-process state summary, for
    operators who want to watch the worker's cold start phase-by-phase.
    """
    if isinstance(event, RampStarted):
        tiers = ", ".join(event.tiers) or "-"
        gpu = event.gpu_name or "unknown GPU"
        return f"> Ramp {event.run_id}: {event.num_levels} levels - tiers={tiers} - mode={event.process_mode} - {gpu}"
    if isinstance(event, LevelStarted):
        position = f"{event.level_index + 1}/{event.num_levels}" if event.num_levels else "soak"
        expected = f" - {event.jobs_expected} jobs" if event.jobs_expected is not None else ""
        return f"  |- [{position}] {event.level_id} ({event.stage}/{event.tier}/{event.axis}){expected} ..."
    if isinstance(event, LevelProgress):
        return f"     . {event.level_id}: {_progress_detail(event, verbose=verbose)}"
    if isinstance(event, LevelFinished):
        return f"  '- {event.level_id}: {event.outcome.upper()}{_finished_detail(event)}"
    if isinstance(event, RampFinished):
        findings = f" - {event.num_findings} findings" if event.num_findings else ""
        return (
            f"= Ramp complete: {event.levels_passed}/{event.levels_total} passed{findings} - "
            f"report={event.report_path}"
        )
    return None


def _progress_detail(event: LevelProgress, *, verbose: bool = False) -> str:
    """Render the metric fragment of a :class:`LevelProgress` event.

    Leads with the worker's current phase so a level that is still cold-starting (no jobs, no it/s yet)
    still shows what it is doing, and flags any process restarts, which are the tell of a respawn storm.
    """
    parts: list[str] = []
    if event.phase:
        parts.append(event.phase)
    if event.jobs_expected is not None:
        parts.append(f"{event.jobs_completed}/{event.jobs_expected} jobs")
    else:
        parts.append(f"{event.jobs_completed} jobs")
    if event.jobs_faulted:
        parts.append(f"{event.jobs_faulted} faulted")
    if event.iterations_per_second is not None:
        parts.append(f"{event.iterations_per_second:.2f} it/s")
    if event.vram_used_mb is not None:
        parts.append(f"VRAM {event.vram_used_mb} MB")
    if event.gpu_busy_percent is not None:
        parts.append(f"GPU {event.gpu_busy_percent:.0f}%")
    if event.num_process_recoveries:
        parts.append(f"(!) {event.num_process_recoveries} process restart(s)")
    parts.append(f"t={event.elapsed_seconds:.0f}s")
    if verbose and event.process_summary:
        parts.append(event.process_summary)
    return " - ".join(parts)


def _finished_detail(event: LevelFinished) -> str:
    """Render the trailing detail (it/s and notes) of a :class:`LevelFinished` event."""
    parts: list[str] = []
    if event.its_p50 is not None:
        parts.append(f"it/s p50 {event.its_p50:.2f}")
    notes = "; ".join(event.reasons + event.advisories)
    if notes:
        parts.append(notes)
    return f" ({' - '.join(parts)})" if parts else ""


class ConsoleProgressSink(ProgressSink):
    """A sink that prints each event as a formatted line to stdout (the CLI live view)."""

    def __init__(self, *, verbose: bool = False) -> None:
        """Create the sink; ``verbose`` adds the per-process state summary to progress lines."""
        self._verbose = verbose

    def emit(self, event: BenchmarkProgressEvent) -> None:
        """Print the event's rendering, if it has one."""
        line = format_progress_event(event, verbose=self._verbose)
        if line is not None:
            print(line)  # noqa: T201


__all__ = ["ConsoleProgressSink", "format_progress_event"]
