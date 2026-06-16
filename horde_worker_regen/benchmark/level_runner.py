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
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from horde_worker_regen.benchmark.progress_channel import LevelLiveSnapshot
    from horde_worker_regen.process_management.run_metrics import RunMetricsSnapshot

_HEARTBEAT_INTERVAL_SECONDS = 5.0
_PROGRESS_INTERVAL_SECONDS = 2.0


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

    def _on_progress(metrics: RunMetricsSnapshot, elapsed_seconds: float) -> None:
        _write_live_snapshot(live_path, metrics, elapsed_seconds)

    # Keep the faulthandler file handle open for the process lifetime.
    faulthandler_file = faulthandler_path.open("w", encoding="utf-8")
    faulthandler.enable(file=faulthandler_file)

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_path, level="DEBUG", enqueue=True)

    _start_heartbeat_thread(heartbeat_path)

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
        result = LevelRunResult(
            level_id=level.id,
            harness=HarnessSummary(**{k: v for k, v in harness_dict.items() if k in HarnessSummary.model_fields}),
            metrics=harness_result.metrics,
        )
        if not harness_result.succeeded:
            exit_code = 1
    except Exception as e:
        logger.exception(f"Level runner failed: {e}")
        result.runner_error = f"{type(e).__name__}: {e}"
        exit_code = 2
    finally:
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
