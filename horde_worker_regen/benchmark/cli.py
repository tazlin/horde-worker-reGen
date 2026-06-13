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

from loguru import logger


def _add_ramp_parser(subparsers: argparse._SubParsersAction) -> None:
    ramp = subparsers.add_parser("ramp", help="Run the progressive ramp benchmark via the harness.")
    ramp.add_argument(
        "--tiers",
        default="sd15,sdxl",
        help="Comma-separated model tiers to attempt (sd15, sdxl, flux). flux is opt-in.",
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


def _run_ramp(args: argparse.Namespace) -> int:
    from horde_worker_regen.benchmark.controller import BenchmarkController
    from horde_worker_regen.benchmark.ladder import LadderOptions, build_default_ladder

    out_dir: Path = args.out if args.out is not None else Path("benchmark_results") / time.strftime("%Y%m%d-%H%M%S")

    options = LadderOptions(
        tiers=[tier.strip() for tier in args.tiers.split(",") if tier.strip()],
        jobs_per_level=args.jobs_per_level,
        include_concurrency=not args.no_concurrency,
        include_features=not args.no_features,
        include_alchemy=not args.no_alchemy,
        include_downloads=args.include_downloads,
        level_timeout_seconds=args.level_timeout,
    )
    ladder = build_default_ladder(options)
    logger.info(f"Ramp ladder has {len(ladder)} level(s); output in {out_dir}")

    controller = BenchmarkController(
        ladder,
        out_dir,
        process_mode=args.process_mode,
        resume=args.resume,
        only_level=args.only_level,
        skip_downloads=args.skip_downloads,
        validate=not args.no_validate,
        soak_seconds=args.soak_minutes * 60.0,
    )
    report = controller.run()

    passed = sum(1 for level in report.levels if level.outcome == "passed")
    print(f"\nBenchmark complete: {passed}/{len(report.levels)} levels passed.")  # noqa: T201
    print(f"Report: {out_dir / 'report.md'}")  # noqa: T201
    if report.findings:
        print(f"Robustness findings: {len(report.findings)} (see the remediation queue in the report)")  # noqa: T201
    print("\nSuggested bridgeData:")  # noqa: T201
    print(report.suggested_bridge_data.as_yaml_block())  # noqa: T201
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

    subparsers.add_parser("live", help="Open-loop load generation against a live API (not yet implemented).")

    args = parser.parse_args(argv)

    if args.command == "ramp":
        return _run_ramp(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "live":
        logger.error("The live load-generation path is a separate phase and not implemented yet.")
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
