"""``horde-log``: triage worker logs into sessions, timelines, and actionable findings.

A worker incident leaves its evidence smeared across an append-across-restarts ``bridge.log``, a clutch
of per-subprocess logs, zipped rotations, and (when local) the action ledger. This CLI does the log
archeology a human would otherwise do by hand: segment the file into per-launch sessions, stitch the
orchestrator log to the subprocess that actually crashed, and report what went wrong with remediation.

Subcommands are added per phase; ``sessions`` is the foundation (everything else operates on the same
segmentation).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from .bundle import LogBundle
from .correlate import build_session_context, build_timeline
from .diagnose import diagnose, select_sessions
from .duty_log_report import add_duty_report_arguments, run_duty_report
from .job_lifecycle import job_lifecycle_for
from .replay_fixture import ReplayFixtureError, render_replay_fixture_summary, write_replay_fixture
from .sessions import WorkerSession, segment_bundle_sessions
from .support_bundle import build_support_bundle
from .triage_report import (
    count_text,
    finding_to_dict,
    job_lifecycle_to_dict,
    render_findings,
    render_job_lifecycle,
    render_sessions,
    render_timeline,
    session_to_dict,
    timeline_entry_to_dict,
)
from .watch import WatchState, watch_pass

_DEFAULT_PATH = Path("logs")


def _load_sessions(path: Path) -> tuple[LogBundle, list[WorkerSession]]:
    """Build the bundle for ``path`` and segment its orchestrator logs (worker and benchmark) into sessions."""
    bundle = LogBundle.from_path(path)
    sessions = segment_bundle_sessions(bundle)
    return bundle, sessions


def _select(sessions: list[WorkerSession], args: argparse.Namespace) -> list[WorkerSession]:
    """Apply the worker (``--last``, ``--session``) and benchmark (``--last-benchmark``, ``--benchmark``) selectors."""
    return select_sessions(
        sessions,
        last=args.last,
        session_index=args.session,
        last_benchmark=args.last_benchmark,
        benchmark_index=args.benchmark,
    )


def _run_sessions(args: argparse.Namespace) -> int:
    """List the worker sessions and benchmark runs in a log path with their end-reason and peak recoveries."""
    bundle, sessions = _load_sessions(args.path)
    selected = _select(sessions, args)
    if args.json:
        print(json.dumps([session_to_dict(s) for s in selected], indent=2))
    else:
        print(render_sessions(selected, root=bundle.root, stitch=bundle.rotation_stitch))
    return 0


def _run_diagnose(args: argparse.Namespace) -> int:
    """Run all detectors over the selected session(s) and print ranked findings with remediation."""
    results = diagnose(
        args.path,
        last=args.last,
        session_index=args.session,
        last_benchmark=args.last_benchmark,
        benchmark_index=args.benchmark,
    )
    if not results:
        print("No matching sessions.")
        return 1
    if args.json:
        payload = [
            {"session": session_to_dict(r.session), "findings": [finding_to_dict(f) for f in r.findings]}
            for r in results
        ]
        print(json.dumps(payload, indent=2))
    else:
        print("\n\n".join(render_findings(r.session, r.findings) for r in results))
    return 0


def _run_timeline(args: argparse.Namespace) -> int:
    """Print the merged parent/child/ledger event stream for the selected session(s)."""
    bundle, sessions = _load_sessions(args.path)
    selected = _select(sessions, args)
    if not selected:
        print("No matching sessions.")
        return 1
    grep = re.compile(args.grep) if args.grep else None
    all_entries = []
    for session in selected:
        context = build_session_context(session, bundle)
        entries = build_timeline(context, include_child_loop=args.child or args.process is not None)
        for entry in entries:
            if args.process is not None and entry.process_id != args.process:
                continue
            if grep is not None and not grep.search(entry.text):
                continue
            all_entries.append(entry)
    if args.json:
        print(json.dumps([timeline_entry_to_dict(e) for e in all_entries], indent=2))
    else:
        print(render_timeline(all_entries))
    return 0


def _run_jobs(args: argparse.Namespace) -> int:
    """Print the per-job lifecycle table and the session's wait/concurrency summary.

    A separate subcommand rather than an option on ``job`` or ``timeline``: ``job`` takes a required job
    id and traces exactly one job across processes, and ``timeline`` is the raw merged event stream.
    Neither has a shape that a whole-session table fits into without a mutually exclusive argument set.
    """
    bundle, sessions = _load_sessions(args.path)
    selected = _select(sessions, args)
    if not selected:
        print("No matching sessions.")
        return 1
    limit = None if args.limit == 0 else args.limit
    rendered = []
    payload = []
    for session in selected:
        model = job_lifecycle_for(build_session_context(session, bundle))
        if args.json:
            payload.append(job_lifecycle_to_dict(session, model))
        else:
            rendered.append(render_job_lifecycle(session, model, limit=limit))
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n\n".join(rendered))
    return 0


def _job_matches(text: str, job_id: str | None, query: str) -> bool:
    """Whether a timeline entry refers to ``query`` (full id, or its 8-char truncation in parent lines)."""
    if job_id is not None and (job_id.startswith(query) or query.startswith(job_id)):
        return True
    return query[:8] in text


def _run_job(args: argparse.Namespace) -> int:
    """Trace one job across the parent and the slot that ran it."""
    bundle, sessions = _load_sessions(args.path)
    matched = []
    for session in sessions:
        context = build_session_context(session, bundle)
        entries = build_timeline(context, include_child_loop=True)
        hits = [e for e in entries if _job_matches(e.text, e.job_id, args.job_id)]
        matched.extend(hits)
    if not matched:
        print(f"No events found for job {args.job_id}.")
        return 1
    if args.json:
        print(json.dumps([timeline_entry_to_dict(e) for e in matched], indent=2))
    else:
        print(render_timeline(matched))
    return 0


def _run_bundle(args: argparse.Namespace) -> int:
    """Build a redacted support bundle (zip) to send a maintainer."""
    out = args.out or Path(f"horde_support_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    result = build_support_bundle(
        args.path,
        out,
        last=args.last,
        session_index=args.session,
        last_benchmark=args.last_benchmark,
        benchmark_index=args.benchmark,
        full_logs=args.full_logs,
        cache_inventory=args.cache_inventory,
        probe_gpu=args.probe_gpu,
        redact_identifiers=not args.keep_identifiers,
        config_path=args.config,
    )
    size_mb = result.size_bytes / (1024 * 1024)
    print(
        f"Wrote {result.out_path} ({size_mb:.1f} MB, {count_text(result.member_count, 'file')}, "
        f"{count_text(result.session_count, 'session')} diagnosed).",
    )
    print(
        f"Redacted {count_text(result.redaction_count, 'occurrence')} of a secret or identifier. "
        "Redaction can miss things. Read through the contents before sending.",
    )
    return 0


def _run_watch(args: argparse.Namespace) -> int:
    """Poll the logs and alert on newly-appearing warnings/criticals and rising recoveries."""
    state = WatchState()
    print(f"Watching {args.path} every {args.interval}s for new findings (Ctrl-C to stop)...")
    try:
        while True:
            alerts, state = watch_pass(LogBundle.from_path(args.path), state)
            for alert in alerts:
                print(alert, flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")
        return 0


def _run_replay_fixture(args: argparse.Namespace) -> int:
    """Write the selected session's job stream and platform-neutral footprint store as a replay fixture."""
    try:
        summary = write_replay_fixture(
            args.path,
            args.out_dir,
            session_index=args.session,
            last=args.last,
            stats_dir=args.stats,
            footprints_path=args.footprints,
        )
    except ReplayFixtureError as fixture_error:
        print(f"Could not build the replay fixture: {fixture_error}")
        return 1
    print(render_replay_fixture_summary(summary, args.out_dir))
    return 0


def _add_common_source_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared positional path + selection/format flags to a subcommand parser."""
    parser.add_argument(
        "path",
        nargs="?",
        default=_DEFAULT_PATH,
        type=Path,
        help="A logs directory, a single log file, or a .zip of logs (default: logs/).",
    )
    _add_session_selector_args(parser, verb="Only")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")


def _add_session_selector_args(parser: argparse.ArgumentParser, *, verb: str) -> None:
    """Add the worker-session and benchmark-run selectors; each kind is numbered from #0 on its own."""
    parser.add_argument("--last", action="store_true", help=f"{verb} the most recent worker session.")
    parser.add_argument("--session", type=int, default=None, metavar="N", help=f"{verb} worker session #N.")
    parser.add_argument(
        "--last-benchmark",
        action="store_true",
        help=f"{verb} the most recent benchmark run (from bridge_harness.log).",
    )
    parser.add_argument("--benchmark", type=int, default=None, metavar="N", help=f"{verb} benchmark run #N.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``horde-log`` argument parser with its subcommands."""
    parser = argparse.ArgumentParser(
        prog="horde-log",
        description="Read worker logs: list each session, show its events and report what went wrong.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sessions_parser = subparsers.add_parser(
        "sessions",
        help="List worker sessions, then benchmark runs, and how each ended.",
    )
    _add_common_source_args(sessions_parser)
    sessions_parser.set_defaults(func=_run_sessions)

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Report what went wrong in each session and what to do about it.",
    )
    _add_common_source_args(diagnose_parser)
    diagnose_parser.set_defaults(func=_run_diagnose)

    timeline_parser = subparsers.add_parser(
        "timeline",
        help="Events from the main process log, the child process logs and the action ledger, in time order.",
    )
    _add_common_source_args(timeline_parser)
    timeline_parser.add_argument("--process", type=int, default=None, metavar="N", help="Only process #N.")
    timeline_parser.add_argument("--grep", default=None, metavar="RE", help="Only entries matching this regex.")
    timeline_parser.add_argument("--child", action="store_true", help="Include every line of the child process logs.")
    timeline_parser.set_defaults(func=_run_timeline)

    jobs_parser = subparsers.add_parser(
        "jobs",
        help="List every job with its wait before generation, its generation time and its wait to be submitted.",
    )
    _add_common_source_args(jobs_parser)
    jobs_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Show only the first N jobs (default: 50, 0 for all). The summary always covers every job.",
    )
    jobs_parser.set_defaults(func=_run_jobs)

    job_parser = subparsers.add_parser(
        "job", help="Show one job's events in the main process and in the process that ran it."
    )
    job_parser.add_argument("job_id", help="The horde job id (full UUID or its leading 8 characters).")
    job_parser.add_argument(
        "path",
        nargs="?",
        default=_DEFAULT_PATH,
        type=Path,
        help="A logs directory, a single log file, or a .zip of logs (default: logs/).",
    )
    job_parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    job_parser.set_defaults(func=_run_job)

    bundle_parser = subparsers.add_parser(
        "bundle",
        help="Build a redacted support bundle (zip) to send a maintainer.",
    )
    bundle_parser.add_argument(
        "path",
        nargs="?",
        default=_DEFAULT_PATH,
        type=Path,
        help="A logs directory or single log file to bundle (default: logs/).",
    )
    bundle_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .zip path (default: horde_support_<timestamp>.zip).",
    )
    _add_session_selector_args(bundle_parser, verb="Diagnose only")
    bundle_parser.add_argument(
        "--full-logs",
        action="store_true",
        help=(
            "Include rotated logs and every stats file, and keep large files whole. By default a large file is "
            "cut to its most recent part. The bundle is much larger."
        ),
    )
    bundle_parser.add_argument(
        "--no-cache-inventory",
        dest="cache_inventory",
        action="store_false",
        help="Leave out the list of model files on disk.",
    )
    bundle_parser.add_argument(
        "--probe-gpu", action="store_true", help="Add the GPUs torch detects to the system info (slower)."
    )
    bundle_parser.add_argument(
        "--keep-identifiers",
        action="store_true",
        help="Keep the home path, username and worker name. Secrets are always redacted.",
    )
    bundle_parser.add_argument(
        "--config",
        type=Path,
        default=Path("bridgeData.yaml"),
        help=(
            "Worker config to include with its secrets redacted. The secrets to redact and the model cache "
            "location are read from it (default: bridgeData.yaml)."
        ),
    )
    bundle_parser.set_defaults(func=_run_bundle)

    watch_parser = subparsers.add_parser(
        "watch", help="Watch a running worker's logs and print new findings as they appear."
    )
    watch_parser.add_argument(
        "path",
        nargs="?",
        default=_DEFAULT_PATH,
        type=Path,
        help="A logs directory or single log file to watch (default: logs/).",
    )
    watch_parser.add_argument(
        "--interval", type=float, default=5.0, metavar="S", help="Seconds between checks (default: 5)."
    )
    watch_parser.add_argument("--once", action="store_true", help="Check once and exit (for scripts and tests).")
    watch_parser.set_defaults(func=_run_watch)

    duty_parser = subparsers.add_parser(
        "duty",
        help="Report how much of the time the GPU was generating and what it waited on the rest of the time.",
        description="The same report as horde-duty-report, with the same options.",
    )
    add_duty_report_arguments(duty_parser)
    duty_parser.set_defaults(func=lambda args: run_duty_report(args, duty_parser))

    replay_parser = subparsers.add_parser(
        "replay-fixture",
        help="Write a session's jobs and learned VRAM usage as a replay fixture for the dispatch-world simulator.",
        description=(
            "Writes session_jobs.jsonl (one row per image job, times in seconds from the session start, no "
            "prompts) and footprints.json (the learned VRAM usage file with each key's platform replaced by "
            "<platform>)."
        ),
    )
    replay_parser.add_argument(
        "path",
        type=Path,
        help="A support bundle, its logs directory, or a worker's logs directory.",
    )
    replay_parser.add_argument("out_dir", type=Path, help="Directory to write the two fixture files into.")
    replay_selector = replay_parser.add_mutually_exclusive_group(required=True)
    replay_selector.add_argument(
        "--session",
        type=int,
        default=None,
        metavar="N",
        help="Worker session #N, as horde-log sessions numbers it.",
    )
    replay_selector.add_argument("--last", action="store_true", help="The most recent worker session.")
    replay_parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help=(
            "Stats directory containing stats-v*.jsonl[.gz] files. Default: the bundle's stats directory, "
            "otherwise .horde_worker_regen/stats."
        ),
    )
    replay_parser.add_argument(
        "--footprints",
        type=Path,
        default=None,
        help=(
            "The learned VRAM usage file (vram_footprints.json). Default: config/vram_footprints.json beside "
            "the stats directory in a support bundle, otherwise the worker's file."
        ),
    )
    replay_parser.set_defaults(func=_run_replay_fixture)

    return parser


def main() -> None:
    """CLI entry point for ``horde-log``."""
    parser = build_parser()
    args = parser.parse_args()
    # ``duty`` takes the duty report's own inputs (``--stats``, ``--logs``) and reports a missing one itself.
    if "path" in vars(args) and not args.path.exists():
        parser.error(f"path not found: {args.path}")
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
