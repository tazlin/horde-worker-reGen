"""Human-readable rendering of benchmark progress events for the CLI live view and ``monitor``.

Kept separate from the channel models so the rendering is reusable and unit-testable: the live ramp
console (a sink) and the ``monitor`` tail loop both format events through :func:`format_progress_event`.
"""

from __future__ import annotations

from horde_worker_regen.benchmark.progress_channel import (
    BenchmarkProgressEvent,
    LevelFinished,
    LevelPlanRow,
    LevelProgress,
    LevelStarted,
    ProgressSink,
    RampFinished,
    RampPlanned,
    RampStarted,
    RampStarting,
)


def format_progress_event(event: BenchmarkProgressEvent, *, verbose: bool = False) -> str | None:
    """Return a one-line human-readable rendering of a progress event, or None to skip it.

    With ``verbose`` set, level-progress lines also carry the compact per-process state summary, for
    operators who want to watch the worker's cold start phase-by-phase.
    """
    if isinstance(event, RampStarting):
        phase = f" - {event.phase}" if event.phase else ""
        return f"> Ramp {event.run_id} starting (mode={event.process_mode}){phase} ..."
    if isinstance(event, RampStarted):
        tiers = ", ".join(event.tiers) or "-"
        gpu = event.gpu_name or "unknown GPU"
        return f"> Ramp {event.run_id}: {event.num_levels} levels - tiers={tiers} - mode={event.process_mode} - {gpu}"
    if isinstance(event, RampPlanned):
        return format_plan_table(event.rows)
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
        lines = [
            f"= Ramp complete: {event.levels_passed}/{event.levels_total} passed{findings} - "
            f"report={event.report_path}",
        ]
        lines.extend(_provenance_lines(event))
        for warning in event.consistency_warnings:
            lines.append(f"  (!) consistency: {warning}")
        return "\n".join(lines)
    return None


def _provenance_lines(event: RampFinished) -> list[str]:
    """Render the recommendation's per-setting provenance under the completion line, if present."""
    if not event.suggestion_decisions:
        return []
    lines = ["  why each suggested value:"]
    for decision in event.suggestion_decisions:
        detail = f" - {decision.detail}" if decision.detail else ""
        lines.append(f"    - {decision.setting}={decision.value_text} [{decision.basis_label}]{detail}")
    return lines


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


def _format_mb(value_mb: int | None) -> str:
    """Render a VRAM figure in GB (e.g. ``2.1G``), or ``-`` when unknown."""
    if value_mb is None:
        return "-"
    return f"{value_mb / 1024:.1f}G"


def _format_disk(row: LevelPlanRow) -> str:
    """Render the disk cell as ``free/needed`` GB when free space is known, else just ``needed``."""
    needed = f"{row.min_disk_free_gb:.0f}G"
    if row.free_disk_bytes is None:
        return needed
    return f"{row.free_disk_bytes / 1024**3:.0f}/{needed}"


def _format_controlnet(row: LevelPlanRow) -> str:
    """Render the controlnet cell: ``-`` (n/a), ``MISSING`` (extra absent), or the annotator ROM size.

    When the extra is absent the prospective annotator size is still shown (when known) so an operator
    weighing whether to install it sees both the gap and its disk cost.
    """
    if not row.requires_controlnet:
        return "-"
    size = f"~{row.controlnet_annotator_bytes / 1024**3:.1f}G" if row.controlnet_annotator_bytes > 0 else ""
    if row.controlnet_installed is False:
        return f"MISSING {size}".rstrip()
    if row.controlnet_annotators_present:
        return "ok"
    return size or "ok"


def _format_verdict(row: LevelPlanRow) -> str:
    """Render a level's plan verdict: ``RUN``, ``DOWNLOAD FIRST (...)``, or ``SKIP (reason)``.

    Three states so the console matches the TUI: a level that fits the machine but is missing downloadable
    artifacts reads ``DOWNLOAD FIRST`` (runnable once fetched), distinct from a ``RUN`` or a hard ``SKIP``.
    """
    if row.needs_download:
        return f"DOWNLOAD FIRST ({row.download_summary})" if row.download_summary else "DOWNLOAD FIRST"
    if row.will_run:
        return "RUN"
    return f"SKIP ({row.verdict})"


def format_plan_table(rows: list[LevelPlanRow]) -> str:
    """Render the resource plan as an aligned text table (LEVEL / VRAM / DISK / NET / KEY / CN / VERDICT).

    The DISK cell reads ``free/needed`` so a shortfall is visible at a glance, the CN cell shows whether a
    controlnet level can run (and its annotator-download ROM), and a trailing line prompts the ``download``
    subcommand whenever any level still needs models or annotators, so a slow mid-run fetch does not quietly
    skew the timing.
    """
    header = ("LEVEL", "VRAM", "DISK", "NET", "KEY", "CN", "VERDICT")
    body: list[tuple[str, str, str, str, str, str, str]] = []
    for row in rows:
        body.append(
            (
                row.level_id,
                _format_mb(row.estimated_vram_mb),
                _format_disk(row),
                "yes" if row.requires_network else "-",
                "civitai" if row.requires_civitai_key else "-",
                _format_controlnet(row),
                _format_verdict(row),
            ),
        )

    widths = [len(col) for col in header]
    for cells in body:
        widths = [max(width, len(cell)) for width, cell in zip(widths, cells, strict=True)]

    def _line(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()

    lines = ["Resource plan (verdicts reflect the detected machine):", _line(header)]
    lines.extend(_line(cells) for cells in body)
    # A single nag from the per-row download-first verdict (which reckons missing image models, controlnet
    # checkpoints, and confirmed-absent annotators); unknown presence stays silent (the level pre-warms first).
    if any(row.needs_download for row in rows):
        lines.append(
            "Some levels need models, controlnet checkpoints, or annotators that are not downloaded yet. Run "
            "`horde-benchmark download` first so the timed run is not slowed (and skewed) by downloading "
            "mid-benchmark.",
        )
    return "\n".join(lines)


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


__all__ = ["ConsoleProgressSink", "format_plan_table", "format_progress_event"]
