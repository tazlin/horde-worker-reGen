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
import sys
import threading
import time
from pathlib import Path

from loguru import logger

_HEARTBEAT_INTERVAL_SECONDS = 5.0


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

    from horde_worker_regen.benchmark.ladder import RampLevel
    from horde_worker_regen.benchmark.report import HarnessSummary, LevelRunResult
    from horde_worker_regen.harness import HarnessConfig, run_harness

    level = RampLevel.model_validate_json(level_json_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"level_{level.id}.log"
    result_path = out_dir / f"level_{level.id}.json"
    heartbeat_path = out_dir / f"level_{level.id}.heartbeat"
    faulthandler_path = out_dir / f"level_{level.id}.faulthandler"

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
