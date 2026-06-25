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
from .detectors import run_detectors
from .sessions import WorkerSession, segment_sessions
from .support_bundle import build_support_bundle
from .triage_report import (
    finding_to_dict,
    render_findings,
    render_sessions,
    render_timeline,
    session_to_dict,
    timeline_entry_to_dict,
)
from .watch import WatchState, watch_pass

_DEFAULT_PATH = Path("logs")


def _load_sessions(path: Path) -> tuple[LogBundle, list[WorkerSession]]:
    """Build the bundle for ``path`` and segment its orchestrator log into sessions."""
    bundle = LogBundle.from_path(path)
    sessions = segment_sessions(bundle.orchestrator_records())
    return bundle, sessions


def _select_sessions(
    sessions: list[WorkerSession],
    *,
    last: bool,
    index: int | None,
) -> list[WorkerSession]:
    """Apply ``--last`` / ``--session N`` selection to a session list."""
    if index is not None:
        return [s for s in sessions if s.index == index]
    if last and sessions:
        return sessions[-1:]
    return sessions


def _run_sessions(args: argparse.Namespace) -> int:
    """List the worker sessions in a log path with their end-reason and peak recoveries."""
    bundle, sessions = _load_sessions(args.path)
    selected = _select_sessions(sessions, last=args.last, index=args.session)
    if args.json:
        print(json.dumps([session_to_dict(s) for s in selected], indent=2))
    else:
        print(render_sessions(selected, root=bundle.root))
    return 0


def _run_diagnose(args: argparse.Namespace) -> int:
    """Run all detectors over the selected session(s) and print ranked findings with remediation."""
    bundle, sessions = _load_sessions(args.path)
    selected = _select_sessions(sessions, last=args.last, index=args.session)
    if not selected:
        print("No matching sessions.")
        return 1
    results = [(session, run_detectors(build_session_context(session, bundle))) for session in selected]
    if args.json:
        payload = [
            {"session": session_to_dict(session), "findings": [finding_to_dict(f) for f in findings]}
            for session, findings in results
        ]
        print(json.dumps(payload, indent=2))
    else:
        print("\n\n".join(render_findings(session, findings) for session, findings in results))
    return 0


def _run_timeline(args: argparse.Namespace) -> int:
    """Print the merged parent/child/ledger event stream for the selected session(s)."""
    bundle, sessions = _load_sessions(args.path)
    selected = _select_sessions(sessions, last=args.last, index=args.session)
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
        full_logs=args.full_logs,
        cache_inventory=args.cache_inventory,
        probe_gpu=args.probe_gpu,
        redact_identifiers=not args.keep_identifiers,
        config_path=args.config,
    )
    size_mb = result.size_bytes / (1024 * 1024)
    print(
        f"Wrote {result.out_path} ({size_mb:.1f} MB, {result.member_count} files, "
        f"{result.session_count} session(s)).",
    )
    print(
        f"Redacted {result.redaction_count} secret/identifier occurrence(s). "
        "Please skim the contents and confirm nothing sensitive remains before sending.",
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


def _add_common_source_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared positional path + selection/format flags to a subcommand parser."""
    parser.add_argument(
        "path",
        nargs="?",
        default=_DEFAULT_PATH,
        type=Path,
        help="A logs directory, a single log file, or a .zip of logs (default: logs/).",
    )
    parser.add_argument("--last", action="store_true", help="Only the most recent session.")
    parser.add_argument("--session", type=int, default=None, metavar="N", help="Only session #N.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``horde-log`` argument parser with its subcommands."""
    parser = argparse.ArgumentParser(
        prog="horde-log",
        description="Triage worker logs into sessions, timelines, and actionable findings.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sessions_parser = subparsers.add_parser("sessions", help="List worker sessions and how each ended.")
    _add_common_source_args(sessions_parser)
    sessions_parser.set_defaults(func=_run_sessions)

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Run detectors and report what went wrong, with remediation.",
    )
    _add_common_source_args(diagnose_parser)
    diagnose_parser.set_defaults(func=_run_diagnose)

    timeline_parser = subparsers.add_parser(
        "timeline",
        help="Merged parent/child/ledger event stream for a session.",
    )
    _add_common_source_args(timeline_parser)
    timeline_parser.add_argument("--process", type=int, default=None, metavar="N", help="Only slot #N.")
    timeline_parser.add_argument("--grep", default=None, metavar="RE", help="Only entries matching this regex.")
    timeline_parser.add_argument("--child", action="store_true", help="Include verbose child-loop records.")
    timeline_parser.set_defaults(func=_run_timeline)

    job_parser = subparsers.add_parser("job", help="Trace one job across the parent and its inference slot.")
    job_parser.add_argument("job_id", help="The horde job id (full UUID or its leading 8 characters).")
    job_parser.add_argument(
        "path",
        nargs="?",
        default=_DEFAULT_PATH,
        type=Path,
        help="A logs directory, a single log file, or a .zip of logs (default: logs/).",
    )
    job_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
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
        help="Output .zip path (default: horde_support_<ts>.zip).",
    )
    bundle_parser.add_argument("--last", action="store_true", help="Diagnose only the most recent session.")
    bundle_parser.add_argument("--session", type=int, default=None, metavar="N", help="Diagnose only session #N.")
    bundle_parser.add_argument(
        "--full-logs",
        action="store_true",
        help="Include rotation archives and do not tail-cap large logs (much larger bundle).",
    )
    bundle_parser.add_argument(
        "--no-cache-inventory",
        dest="cache_inventory",
        action="store_false",
        help="Skip the on-disk model listing.",
    )
    bundle_parser.add_argument("--probe-gpu", action="store_true", help="Run the GPU probe for system info (slower).")
    bundle_parser.add_argument(
        "--keep-identifiers",
        action="store_true",
        help="Do not scrub home paths / username / worker name (secrets are always redacted).",
    )
    bundle_parser.add_argument(
        "--config",
        type=Path,
        default=Path("bridgeData.yaml"),
        help="Worker config to redact and source secrets/cache from (default: bridgeData.yaml).",
    )
    bundle_parser.set_defaults(func=_run_bundle)

    watch_parser = subparsers.add_parser("watch", help="Live-watch logs and alert on new findings.")
    watch_parser.add_argument(
        "path",
        nargs="?",
        default=_DEFAULT_PATH,
        type=Path,
        help="A logs directory or single log file to watch (default: logs/).",
    )
    watch_parser.add_argument("--interval", type=float, default=5.0, metavar="S", help="Poll interval (default: 5s).")
    watch_parser.add_argument("--once", action="store_true", help="Run a single pass and exit (for scripting/tests).")
    watch_parser.set_defaults(func=_run_watch)

    return parser


def main() -> None:
    """CLI entry point for ``horde-log``."""
    parser = build_parser()
    args = parser.parse_args()
    if not args.path.exists():
        parser.error(f"path not found: {args.path}")
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
