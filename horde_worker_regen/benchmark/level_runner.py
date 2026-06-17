"""Run exactly one benchmark level in an isolated process.

Invoked by the controller as ``python -m horde_worker_regen.benchmark.level_runner`` so
that a level that OOMs, hangs, or hard-crashes the worker takes down only this process;
the controller records the death as a finding and the ramp continues.

Writes (into ``--out``):
- ``level_<id>.json`` — the :class:`LevelRunResult` (written even on failure, via finally)
- ``level_<id>.log`` — the full loguru output of the run
- ``level_<id>.heartbeat`` — touched every few seconds so the controller can tell a hang
  from a slow level
- ``level_<id>.faulthandler`` — Python tracebacks on hard faults
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import faulthandler
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from loguru import logger

if TYPE_CHECKING:
    from horde_worker_regen.benchmark.progress_channel import LevelLiveSnapshot
    from horde_worker_regen.process_management.run_metrics import RunMetricsSnapshot

_HEARTBEAT_INTERVAL_SECONDS = 5.0
_PROGRESS_INTERVAL_SECONDS = 2.0
_STALL_DUMP_SECONDS = 240.0
"""No change in the run's progress signature for this long triggers a diagnostic thread-stack dump.

Set comfortably above a slow-but-healthy cold start / model load (which still advances the phase and job
counters, so its signature keeps changing) and below the controller's hard-kill deadline, so a genuine
wedge is captured *with* stacks well before the process is killed with none. A dump neither kills nor
blocks the run and fires at most once per stall episode, so a rare false positive on an unusually slow
single phase costs only one extra diagnostic dump.
"""
_STALL_CHECK_INTERVAL_SECONDS = 15.0
"""How often the stall-watchdog thread re-checks for a lack of progress."""
_HANG_DUMP_MARGIN_SECONDS = 30.0
"""Seconds after a level's own timeout to fire the C-level faulthandler backstop (a GIL-proof dump)."""


def _build_live_snapshot(metrics: RunMetricsSnapshot, elapsed_seconds: float) -> LevelLiveSnapshot:
    """Distill the live run metrics into the lean latest-only snapshot the controller republishes."""
    from horde_worker_regen.benchmark.progress_channel import LevelLiveSnapshot

    jobs = metrics.jobs
    jobs_faulted = sum(1 for job in jobs if job.faulted)

    latest_its: float | None = None
    for job in jobs:
        if job.phase_metrics is not None and job.phase_metrics.sampling is not None:
            sampled_its = job.phase_metrics.sampling.iterations_per_second
            if sampled_its > 0:
                latest_its = sampled_its

    vram_used_mb: int | None = None
    if metrics.vram_used_high_water_mb_per_process:
        vram_used_mb = max(metrics.vram_used_high_water_mb_per_process.values())

    return LevelLiveSnapshot(
        jobs_completed=len(jobs),
        jobs_faulted=jobs_faulted,
        iterations_per_second=latest_its,
        vram_used_mb=vram_used_mb,
        gpu_busy_percent=metrics.gpu_utilization_mean_percent,
        elapsed_seconds=elapsed_seconds,
        phase=metrics.phase,
        process_summary=metrics.process_state_summary,
        num_process_recoveries=metrics.num_process_recoveries,
    )


def _write_live_snapshot(live_path: Path, metrics: RunMetricsSnapshot, elapsed_seconds: float) -> None:
    """Write the latest live metrics atomically, best-effort (a missed sample is harmless).

    The atomic temp-then-replace guarantees the controller, which reads this file concurrently, never
    sees a half-written line.
    """
    with contextlib.suppress(OSError):
        temp_path = live_path.with_suffix(".tmp")
        temp_path.write_text(_build_live_snapshot(metrics, elapsed_seconds).model_dump_json(), encoding="utf-8")
        os.replace(temp_path, live_path)


def _start_heartbeat_thread(heartbeat_path: Path) -> None:
    def _beat() -> None:
        while True:
            with contextlib.suppress(OSError):
                heartbeat_path.write_text(str(time.time()))
            time.sleep(_HEARTBEAT_INTERVAL_SECONDS)

    threading.Thread(target=_beat, daemon=True).start()


def _dump_thread_stacks(faulthandler_file: TextIO, elapsed_seconds: float) -> None:
    """Dump every thread's stack to the faulthandler file because the run looks wedged.

    Unlike the heartbeat (which proves only that *a* thread can still run) this is triggered by a lack
    of *work* progress, and unlike ``faulthandler.enable`` (which fires only on fatal signals) it fires
    on a soft hang. The dump is the single most useful artefact for a silent wedge such as the observed
    pre-spawn startup hang, where the manager log simply stopped and the ``.faulthandler`` file was empty.
    """
    logger.critical(
        f"No measurable progress for {elapsed_seconds:.0f}s; dumping all thread stacks to the "
        f".faulthandler file for diagnosis (the run may be wedged).",
    )
    with contextlib.suppress(Exception):
        faulthandler.dump_traceback(file=faulthandler_file)
        faulthandler_file.flush()


class _StallWatchdog:
    """Detects a stalled run and dumps thread stacks once per stall episode.

    Pure and clock-injectable so the stall timing is unit-testable. :meth:`note_progress` is called from
    the harness progress callback; :meth:`check` is polled from a watchdog thread. Reaching the threshold
    with an unchanged progress signature triggers the dump; any signature change rearms it.
    """

    def __init__(
        self,
        *,
        stall_seconds: float,
        dump: Callable[[float], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure the stall threshold, the dump action, and (injectable) clock."""
        self._stall_seconds = stall_seconds
        self._dump = dump
        self._clock = clock
        self._signature: object = None
        self._last_change = clock()
        self._dumped = False

    def note_progress(self, signature: object) -> None:
        """Record the run's latest progress signature; a change resets the stall timer and rearms the dump."""
        if signature != self._signature:
            self._signature = signature
            self._last_change = self._clock()
            self._dumped = False

    def check(self) -> bool:
        """Dump thread stacks once if there has been no progress for ``stall_seconds``; return if it dumped."""
        if self._dumped:
            return False
        elapsed = self._clock() - self._last_change
        if elapsed >= self._stall_seconds:
            self._dumped = True
            self._dump(elapsed)
            return True
        return False


def _start_stall_watchdog_thread(watchdog: _StallWatchdog) -> None:
    def _watch() -> None:
        while True:
            time.sleep(_STALL_CHECK_INTERVAL_SECONDS)
            with contextlib.suppress(Exception):
                watchdog.check()

    threading.Thread(target=_watch, daemon=True).start()


def run_level(level_json_path: Path, out_dir: Path, *, process_mode: str = "real") -> int:
    """Run one level and write its result; returns the process exit code."""
    # Tracing is opt-in (AIWORKER_REGEN_ENABLE_TELEMETRY); force it off before importing the
    # harness/hordelib so spawned worker children inherit the kill switch. hordelib's per-op
    # ComfyUI spans otherwise starve the GPU loop and skew the very measurements this collects.
    from horde_worker_regen.telemetry import enforce_telemetry_default_off

    enforce_telemetry_default_off()

    # Set AIWORKER_CACHE_HOME (and friends) from bridgeData.yaml before hordelib resolves its weights
    # root in the spawned inference children. Normally inherited from the controller, but a directly
    # invoked level (`--only-level` remediation, manual `python -m ...level_runner`) needs it too.
    from horde_worker_regen.benchmark.worker_env import ensure_worker_env

    ensure_worker_env(process_mode)

    if process_mode == "real":
        with contextlib.suppress(Exception):
            from horde_model_reference import resolve_weights_root

            logger.info(f"Benchmark model weights root: {resolve_weights_root(os.getenv('AIWORKER_CACHE_HOME'))}")

    from horde_worker_regen.benchmark.ladder import RampLevel
    from horde_worker_regen.benchmark.report import HarnessSummary, LevelRunResult
    from horde_worker_regen.harness import HarnessConfig, run_harness

    level = RampLevel.model_validate_json(level_json_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"level_{level.id}.log"
    result_path = out_dir / f"level_{level.id}.json"
    heartbeat_path = out_dir / f"level_{level.id}.heartbeat"
    faulthandler_path = out_dir / f"level_{level.id}.faulthandler"
    live_path = out_dir / f"level_{level.id}.live.json"

    # Keep the faulthandler file handle open for the process lifetime.
    faulthandler_file = faulthandler_path.open("w", encoding="utf-8")
    faulthandler.enable(file=faulthandler_file)
    # C-level backstop: dump all thread stacks shortly after the level's own timeout (before the
    # controller hard-kills a wedged subprocess), so even a hang that holds the GIL -- which the Python
    # stall-watchdog thread below could not interrupt -- still leaves a trace.
    faulthandler.dump_traceback_later(
        level.timeout_seconds + _HANG_DUMP_MARGIN_SECONDS,
        repeat=False,
        file=faulthandler_file,
    )

    # Progress-aware stall watchdog: dumps thread stacks if the run stops making progress. This catches a
    # soft wedge (e.g. the startup hang before any child spawns) that faulthandler.enable() -- fatal
    # signals only -- never would, and that the always-on heartbeat thread otherwise masks.
    stall_watchdog = _StallWatchdog(
        stall_seconds=_STALL_DUMP_SECONDS,
        dump=lambda elapsed: _dump_thread_stacks(faulthandler_file, elapsed),
    )

    def _on_progress(metrics: RunMetricsSnapshot, elapsed_seconds: float) -> None:
        _write_live_snapshot(live_path, metrics, elapsed_seconds)
        jobs_faulted = sum(1 for job in metrics.jobs if job.faulted)
        stall_watchdog.note_progress(
            (metrics.phase, len(metrics.jobs), jobs_faulted, metrics.num_process_recoveries),
        )

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_path, level="DEBUG", enqueue=True)

    _start_heartbeat_thread(heartbeat_path)
    _start_stall_watchdog_thread(stall_watchdog)

    result = LevelRunResult(level_id=level.id, harness=HarnessSummary())
    exit_code = 0
    try:
        scenario = level.scenario
        if scenario.soak_seconds is not None:
            image_templates, alchemy_templates = scenario.to_soak_templates()
            config = HarnessConfig(
                soak_seconds=scenario.soak_seconds,
                soak_image_templates=image_templates,
                soak_alchemy_templates=alchemy_templates,
                process_mode=process_mode,  # type: ignore[arg-type]
                skip_api=True,
                timeout_seconds=level.timeout_seconds,
                bridge_data_overrides=dict(level.bridge_data_overrides),
                audit=True,
                on_progress=_on_progress,
                progress_interval_seconds=_PROGRESS_INTERVAL_SECONDS,
            )
        else:
            arrival = scenario.arrival_schedule()
            config = HarnessConfig(
                # An empty list is a real (alchemy-only) scenario; None would trigger
                # the default image scenario.
                scenario=scenario.expand_image_jobs(),
                alchemy_forms=scenario.expand_alchemy_forms() or None,
                arrival=arrival if arrival.kind != "all_at_once" else None,
                process_mode=process_mode,  # type: ignore[arg-type]
                skip_api=True,
                timeout_seconds=level.timeout_seconds,
                bridge_data_overrides=dict(level.bridge_data_overrides),
                audit=True,
                on_progress=_on_progress,
                progress_interval_seconds=_PROGRESS_INTERVAL_SECONDS,
            )
        harness_result = run_harness(config)
        harness_dict = dataclasses.asdict(harness_result)
        # Drop the nested metrics snapshot before filtering kwargs into HarnessSummary;
        # the typed metrics object is passed through from harness_result directly below.
        harness_dict.pop("metrics", None)
        harness_kwargs = {k: v for k, v in harness_dict.items() if k in HarnessSummary.model_fields and v is not None}
        result = LevelRunResult(
            level_id=level.id,
            harness=HarnessSummary(**harness_kwargs),
            metrics=harness_result.metrics,
        )
        if not harness_result.succeeded:
            exit_code = 1
    except Exception as e:
        logger.exception(f"Level runner failed: {e}")
        result.runner_error = f"{type(e).__name__}: {e}"
        exit_code = 2
    finally:
        faulthandler.cancel_dump_traceback_later()
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    return exit_code


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running a single level."""
    parser = argparse.ArgumentParser(description="Run one benchmark level in isolation.")
    parser.add_argument("--level-json", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--process-mode", default="real", choices=("fake", "dry_run", "real"))
    args = parser.parse_args(argv)
    return run_level(args.level_json, args.out, process_mode=args.process_mode)


if __name__ == "__main__":
    sys.exit(main())
