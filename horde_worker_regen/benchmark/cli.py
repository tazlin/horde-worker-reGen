"""The `horde-benchmark` CLI: progressive worker benchmarking.

Subcommands:
- ``ramp``: run the ramp ladder via the canned-job harness (reproducible, no API).
- ``report``: re-render the markdown report from an existing output directory.
- ``live``: open-loop load generation against a live AI-Horde API (separate phase).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from horde_worker_regen.benchmark.enums import BenchTier

if TYPE_CHECKING:
    from horde_worker_regen.benchmark.report import BenchmarkReport


def _add_ramp_parser(subparsers: argparse._SubParsersAction) -> None:
    ramp = subparsers.add_parser("ramp", help="Run the progressive ramp benchmark via the harness.")
    ramp.add_argument(
        "--tiers",
        default="sd15,sdxl",
        help="Comma-separated model tiers to attempt (sd15, sdxl, flux, qwen). flux/qwen are opt-in "
        "(very large: 17-20 GB download, 13-16 GB VRAM) and auto-skip when the machine cannot hold them; "
        "qwen is a beta model sourced from the pending reference.",
    )
    ramp.add_argument(
        "--process-mode",
        default="real",
        choices=("fake", "dry_run", "real"),
        help="real = GPU benchmark; fake/dry_run exercise the ramp machinery without inference.",
    )
    ramp.add_argument("--out", type=Path, default=None, help="Output directory (default: benchmark_results/<ts>).")
    ramp.add_argument("--jobs-per-level", type=int, default=4)
    ramp.add_argument("--level-timeout", type=float, default=900.0, help="Per-level timeout in seconds.")
    ramp.add_argument("--resume", action="store_true", help="Reuse existing level results in --out.")
    ramp.add_argument("--only-level", default=None, help="Run a single level id (remediation loop).")
    ramp.add_argument("--skip-downloads", action="store_true", help="Skip levels that need network access.")
    ramp.add_argument("--include-downloads", action="store_true", help="Include the ad-hoc download levels.")
    ramp.add_argument("--no-alchemy", action="store_true", help="Skip the alchemy levels.")
    ramp.add_argument("--no-features", action="store_true", help="Skip the feature levels (stage C).")
    ramp.add_argument("--no-concurrency", action="store_true", help="Skip the concurrency levels (stage B).")
    ramp.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the post-ramp sustained-load validation soak.",
    )
    ramp.add_argument(
        "--soak-minutes",
        type=float,
        default=5.0,
        help="Duration of each per-tier validation soak (minutes).",
    )
    ramp.add_argument(
        "--warm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse one warm worker across fixed-scenario levels instead of cold-starting a fresh worker "
        "(and respawning every inference process) per level. On by default; pass --no-warm to run each "
        "level in its own isolated subprocess (full crash isolation at the cost of per-level startup).",
    )
    ramp.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-process state in the live view and raise spawned worker logging to TRACE.",
    )
    ramp.add_argument(
        "--no-abort-on-catastrophe",
        action="store_true",
        help="Keep running after a level hangs/crashes instead of aborting the whole ramp "
        "(by default the first catastrophic failure stops the ramp, since the worker stack is shared).",
    )


def _parse_tiers(raw_tiers: str) -> list[BenchTier] | None:
    """Parse the comma-separated ``--tiers`` value into tiers, or None on an unknown tier (logged)."""
    tiers: list[BenchTier] = []
    for token in (token.strip() for token in raw_tiers.split(",")):
        if not token:
            continue
        try:
            tiers.append(BenchTier(token))
        except ValueError:
            logger.error(f"Unknown tier {token!r}; valid tiers: {', '.join(tier.value for tier in BenchTier)}")
            return None
    return tiers


def _run_ramp(args: argparse.Namespace) -> int:
    from horde_worker_regen.benchmark.controller import BenchmarkController, detect_machine_info
    from horde_worker_regen.benchmark.ladder import LadderOptions, build_default_ladder
    from horde_worker_regen.benchmark.worker_env import ensure_worker_env

    tiers = _parse_tiers(args.tiers)
    if tiers is None:
        return 2

    # The harness never reads bridgeData.yaml, so set AIWORKER_CACHE_HOME (and friends) here, before
    # spawning level subprocesses, so the real inference children resolve the worker's actual model
    # directory instead of hordelib's empty ./models fallback. Children inherit this process's env.
    # Passing the tiers also opts into the beta reference when a beta tier (qwen) is requested.
    ensure_worker_env(args.process_mode, tiers)

    out_dir: Path = args.out if args.out is not None else Path("benchmark_results") / time.strftime("%Y%m%d-%H%M%S")

    # Detect the machine once: the ladder uses the VRAM to size the post-processing sweep, and the
    # controller reuses the same info instead of re-detecting.
    machine = detect_machine_info()

    options = LadderOptions(
        tiers=tiers,
        jobs_per_level=args.jobs_per_level,
        include_concurrency=not args.no_concurrency,
        include_features=not args.no_features,
        include_alchemy=not args.no_alchemy,
        include_downloads=args.include_downloads,
        level_timeout_seconds=args.level_timeout,
        total_vram_mb=machine.total_vram_mb,
    )
    ladder = build_default_ladder(options)
    logger.info(f"Ramp ladder has {len(ladder)} level(s); output in {out_dir}")

    from horde_worker_regen.benchmark.progress_channel import PROGRESS_FILENAME, JsonlProgressSink, MultiProgressSink
    from horde_worker_regen.benchmark.progress_console import ConsoleProgressSink

    # Tee progress to a durable JSONL log (for the TUI / `monitor` to tail) and a live console view.
    progress_sink = MultiProgressSink(
        [JsonlProgressSink(out_dir / PROGRESS_FILENAME), ConsoleProgressSink(verbose=args.verbose)],
    )

    controller = BenchmarkController(
        ladder,
        out_dir,
        process_mode=args.process_mode,
        resume=args.resume,
        only_level=args.only_level,
        skip_downloads=args.skip_downloads,
        validate=not args.no_validate,
        soak_seconds=args.soak_minutes * 60.0,
        progress_sink=progress_sink,
        warm=args.warm,
        verbose=args.verbose,
        abort_on_catastrophe=not args.no_abort_on_catastrophe,
        machine=machine,
    )
    try:
        report = controller.run()
    finally:
        progress_sink.close()
    _record_benchmark_in_app_state(report, out_dir)

    passed = sum(1 for level in report.levels if level.outcome == "passed")
    print(f"\nBenchmark complete: {passed}/{len(report.levels)} levels passed.")  # noqa: T201
    print(f"Report: {out_dir / 'report.md'}")  # noqa: T201
    if report.findings:
        print(f"Robustness findings: {len(report.findings)} (see the remediation queue in the report)")  # noqa: T201
    print("\nSuggested bridgeData:")  # noqa: T201
    print(report.suggested_bridge_data.as_yaml_block())  # noqa: T201
    return 0


def _record_benchmark_in_app_state(report: BenchmarkReport, out_dir: Path) -> None:
    """Record a finished CLI ramp as the canonical run-to-run benchmark, best-effort.

    Both the CLI and the TUI funnel through the same ``build_benchmark_record`` adapter so the durable
    pointer is identical regardless of how the ramp was launched. Failure here only loses bookkeeping,
    so it is logged at debug rather than raised.
    """
    try:
        from horde_worker_regen.app_state import AppStateStore, build_benchmark_record

        record = build_benchmark_record(report, results_dir=out_dir)
        AppStateStore().record_benchmark(record)
        logger.info(f"Recorded benchmark {record.run_id} in app state ({AppStateStore().path}).")
    except Exception as app_state_error:  # noqa: BLE001 - app-state bookkeeping must not fail the ramp
        logger.debug(f"Could not record benchmark in app state: {app_state_error}")


_MONITOR_POLL_SECONDS = 0.5
_MONITOR_IDLE_POLLS_BEFORE_EXIT = 3
"""Consecutive empty polls (with a report present) after which `monitor` concludes a finished run."""


def _run_monitor(args: argparse.Namespace) -> int:
    """Tail a run's ``progress.jsonl`` and render it live, for attaching to or replaying a ramp."""
    from horde_worker_regen.benchmark.progress_channel import PROGRESS_FILENAME, ProgressTailer, RampFinished
    from horde_worker_regen.benchmark.progress_console import format_progress_event

    out_dir: Path = args.out_dir
    progress_path = out_dir / PROGRESS_FILENAME
    if not progress_path.exists():
        logger.error(f"No {PROGRESS_FILENAME} found in {out_dir}")
        return 1

    tailer = ProgressTailer(progress_path)
    saw_ramp_finished = False
    idle_polls = 0
    while not saw_ramp_finished:
        events = tailer.poll()
        if not events:
            idle_polls += 1
            if (out_dir / "report.json").exists() and idle_polls >= _MONITOR_IDLE_POLLS_BEFORE_EXIT:
                break
            time.sleep(_MONITOR_POLL_SECONDS)
            continue
        idle_polls = 0
        for event in events:
            line = format_progress_event(event, verbose=args.verbose)
            if line is not None:
                print(line)  # noqa: T201
            if isinstance(event, RampFinished):
                saw_ramp_finished = True
    return 0


def _run_report(args: argparse.Namespace) -> int:
    from horde_worker_regen.benchmark.controller import load_existing_report
    from horde_worker_regen.benchmark.report import render_markdown

    report = load_existing_report(args.out_dir)
    if report is None:
        logger.error(f"No report.json found in {args.out_dir}")
        return 1
    markdown = render_markdown(report)
    (args.out_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)  # noqa: T201
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="horde-benchmark", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_ramp_parser(subparsers)

    report = subparsers.add_parser("report", help="Re-render the markdown report from an output directory.")
    report.add_argument("out_dir", type=Path)

    monitor = subparsers.add_parser("monitor", help="Tail a run's progress.jsonl live (attach or replay).")
    monitor.add_argument("out_dir", type=Path)
    monitor.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show the per-process state summary on each progress line.",
    )

    subparsers.add_parser("live", help="Open-loop load generation against a live API (not yet implemented).")

    args = parser.parse_args(argv)

    if args.command == "ramp":
        return _run_ramp(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "monitor":
        return _run_monitor(args)
    if args.command == "live":
        logger.error("The live load-generation path is a separate phase and not implemented yet.")
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
