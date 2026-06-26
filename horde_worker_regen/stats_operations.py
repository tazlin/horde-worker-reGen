"""Operations for retained worker stats JSONL files.

The live worker writes session-scoped stats files under ``.horde_worker_regen/stats``. This module keeps
post-processing those files importable by the TUI while also exposing a small CLI via :func:`main`.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any, cast

from pydantic import BaseModel, Field

from horde_worker_regen.app_state import default_app_state_dir

_STATS_FILE_PREFIX = "stats-v"
_STATS_JSONL_SUFFIX = ".jsonl"
_STATS_GZIP_SUFFIX = ".jsonl.gz"


class StatsFileOperation(BaseModel):
    """One file touched or skipped by a stats operation."""

    path: str
    output_path: str | None = None
    compressed: bool = False
    rewritten: bool = False
    skipped: bool = False
    reason: str | None = None
    input_bytes: int = 0
    output_bytes: int = 0
    lines_read: int = 0
    lines_written: int = 0
    samples_kept: int = 0
    samples_dropped: int = 0


class StatsOperationReport(BaseModel):
    """Summary returned by stats file operations."""

    stats_dir: str
    files: list[StatsFileOperation] = Field(default_factory=list)

    @property
    def files_changed(self) -> int:
        """How many files were compressed or rewritten."""
        return sum(1 for item in self.files if item.compressed or item.rewritten)

    @property
    def samples_dropped(self) -> int:
        """Total dropped ``stats_sample`` events across all processed files."""
        return sum(item.samples_dropped for item in self.files)

    @property
    def bytes_before(self) -> int:
        """Total input bytes for files considered by the operation."""
        return sum(item.input_bytes for item in self.files)

    @property
    def bytes_after(self) -> int:
        """Total output/current bytes for files considered by the operation."""
        return sum(item.output_bytes for item in self.files)


def default_stats_dir() -> Path:
    """Return the default worker stats directory in the current working directory."""
    return default_app_state_dir() / "stats"


def compress_old_stats_files(stats_dir: Path | None = None, *, include_latest: bool = False) -> StatsOperationReport:
    """Compress retained uncompressed stats JSONL files with gzip.

    Args:
        stats_dir: Directory containing ``stats-v*.jsonl`` files. Defaults to
            ``.horde_worker_regen/stats`` in the current working directory.
        include_latest: When false, skip the newest uncompressed JSONL file so a running worker's active file is
            not touched.

    Returns:
        A structured report suitable for direct TUI consumption.
    """
    directory = stats_dir or default_stats_dir()
    files = _stats_files(directory, compressed=False)
    latest = _latest_path(files) if not include_latest else None
    report = StatsOperationReport(stats_dir=str(directory))
    for path in files:
        input_bytes = _file_size(path)
        if latest is not None and path == latest:
            report.files.append(
                StatsFileOperation(
                    path=str(path),
                    skipped=True,
                    reason="latest stats file skipped",
                    input_bytes=input_bytes,
                    output_bytes=input_bytes,
                ),
            )
            continue
        report.files.append(_compress_one(path, input_bytes=input_bytes))
    return report


def downsample_stats_files(
    frequency_seconds: float,
    stats_dir: Path | None = None,
    *,
    include_latest: bool = False,
) -> StatsOperationReport:
    """Downsample ``stats_sample`` events while preserving all other JSONL events.

    The operation rewrites files in place, preserving compression state. It keeps the first sample it sees and then
    keeps later samples only when their timestamp is at least ``frequency_seconds`` after the last kept sample.
    ``job_completed`` and unknown events are always retained.

    Args:
        frequency_seconds: Minimum seconds between retained ``stats_sample`` events. Must be positive.
        stats_dir: Directory containing stats JSONL files. Defaults to ``.horde_worker_regen/stats``.
        include_latest: When false, skip the newest stats file so a running worker's active file is not touched.

    Returns:
        A structured report suitable for direct TUI consumption.
    """
    if frequency_seconds <= 0:
        raise ValueError("frequency_seconds must be positive")
    directory = stats_dir or default_stats_dir()
    files = _stats_files(directory, compressed=True)
    latest = _latest_path(files) if not include_latest else None
    report = StatsOperationReport(stats_dir=str(directory))
    last_kept_sample_timestamp: float | None = None
    for path in files:
        input_bytes = _file_size(path)
        if latest is not None and path == latest:
            report.files.append(
                StatsFileOperation(
                    path=str(path),
                    skipped=True,
                    reason="latest stats file skipped",
                    input_bytes=input_bytes,
                    output_bytes=input_bytes,
                ),
            )
            continue
        operation, last_kept_sample_timestamp = _downsample_one(
            path,
            frequency_seconds=frequency_seconds,
            input_bytes=input_bytes,
            last_kept_sample_timestamp=last_kept_sample_timestamp,
        )
        report.files.append(operation)
    return report


def _stats_files(directory: Path, *, compressed: bool) -> list[Path]:
    """Return stats files in stable chronological-ish order."""
    if not directory.exists():
        return []
    candidates: Iterable[Path]
    if compressed:
        candidates = [
            *directory.glob(f"{_STATS_FILE_PREFIX}*{_STATS_JSONL_SUFFIX}"),
            *directory.glob(f"{_STATS_FILE_PREFIX}*{_STATS_GZIP_SUFFIX}"),
        ]
    else:
        candidates = directory.glob(f"{_STATS_FILE_PREFIX}*{_STATS_JSONL_SUFFIX}")
    return sorted((path for path in candidates if path.is_file()), key=_file_order_key)


def _file_order_key(path: Path) -> tuple[float, str]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (mtime, path.name)


def _latest_path(paths: list[Path]) -> Path | None:
    return max(paths, key=_file_order_key) if paths else None


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _compress_one(path: Path, *, input_bytes: int) -> StatsFileOperation:
    target = path.with_name(path.name + ".gz")
    if target.exists():
        return StatsFileOperation(
            path=str(path),
            output_path=str(target),
            skipped=True,
            reason="compressed target already exists",
            input_bytes=input_bytes,
            output_bytes=_file_size(target),
        )
    temp = target.with_name(target.name + ".tmp")
    try:
        with path.open("rb") as source, gzip.open(temp, "wb") as destination:
            shutil.copyfileobj(source, destination)
        os.replace(temp, target)
        path.unlink()
    except OSError as error:
        with contextlib.suppress(OSError):
            temp.unlink()
        return StatsFileOperation(
            path=str(path),
            output_path=str(target),
            skipped=True,
            reason=str(error),
            input_bytes=input_bytes,
            output_bytes=input_bytes,
        )
    return StatsFileOperation(
        path=str(path),
        output_path=str(target),
        compressed=True,
        input_bytes=input_bytes,
        output_bytes=_file_size(target),
    )


def _downsample_one(
    path: Path,
    *,
    frequency_seconds: float,
    input_bytes: int,
    last_kept_sample_timestamp: float | None,
) -> tuple[StatsFileOperation, float | None]:
    lines_out: list[str] = []
    lines_read = 0
    samples_kept = 0
    samples_dropped = 0
    try:
        compressed = path.name.endswith(_STATS_GZIP_SUFFIX)
        with _open_text(path, "rt", compressed=compressed) as handle:
            for line in handle:
                lines_read += 1
                keep, timestamp = _should_keep_line(
                    line,
                    frequency_seconds=frequency_seconds,
                    last_kept_sample_timestamp=last_kept_sample_timestamp,
                )
                if keep:
                    lines_out.append(line)
                    if timestamp is not None:
                        samples_kept += 1
                        last_kept_sample_timestamp = timestamp
                else:
                    samples_dropped += 1
    except OSError as error:
        return (
            StatsFileOperation(
                path=str(path),
                skipped=True,
                reason=str(error),
                input_bytes=input_bytes,
                output_bytes=input_bytes,
            ),
            last_kept_sample_timestamp,
        )

    if samples_dropped == 0:
        return (
            StatsFileOperation(
                path=str(path),
                skipped=True,
                reason="already at or below requested frequency",
                input_bytes=input_bytes,
                output_bytes=input_bytes,
                lines_read=lines_read,
                lines_written=lines_read,
                samples_kept=samples_kept,
            ),
            last_kept_sample_timestamp,
        )

    temp = path.with_name(path.name + ".tmp")
    try:
        with _open_text(temp, "wt", compressed=compressed) as handle:
            handle.writelines(lines_out)
        os.replace(temp, path)
    except OSError as error:
        with contextlib.suppress(OSError):
            temp.unlink()
        return (
            StatsFileOperation(
                path=str(path),
                skipped=True,
                reason=str(error),
                input_bytes=input_bytes,
                output_bytes=input_bytes,
                lines_read=lines_read,
                lines_written=lines_read,
                samples_kept=samples_kept,
                samples_dropped=0,
            ),
            last_kept_sample_timestamp,
        )

    return (
        StatsFileOperation(
            path=str(path),
            rewritten=True,
            input_bytes=input_bytes,
            output_bytes=_file_size(path),
            lines_read=lines_read,
            lines_written=len(lines_out),
            samples_kept=samples_kept,
            samples_dropped=samples_dropped,
        ),
        last_kept_sample_timestamp,
    )


def _open_text(
    path: Path,
    mode: str,
    *,
    compressed: bool,
) -> IO[str]:
    if compressed:
        return cast(IO[str], gzip.open(path, mode, encoding="utf-8"))
    return path.open(mode, encoding="utf-8")


def _should_keep_line(
    line: str,
    *,
    frequency_seconds: float,
    last_kept_sample_timestamp: float | None,
) -> tuple[bool, float | None]:
    try:
        payload: Any = json.loads(line)
    except ValueError:
        return True, None
    if not isinstance(payload, dict) or payload.get("event") != "stats_sample":
        return True, None
    sample = payload.get("sample")
    if not isinstance(sample, dict):
        return True, None
    timestamp = sample.get("timestamp")
    if not isinstance(timestamp, int | float):
        return True, None
    timestamp_float = float(timestamp)
    if last_kept_sample_timestamp is None or timestamp_float - last_kept_sample_timestamp >= frequency_seconds:
        return True, timestamp_float
    return False, timestamp_float


def build_parser() -> argparse.ArgumentParser:
    """Build the ``horde-stats`` argument parser."""
    parser = argparse.ArgumentParser(prog="horde-stats", description="Operate on retained worker stats JSONL files.")
    parser.add_argument(
        "--stats-dir", type=Path, default=None, help="Stats directory; defaults to .horde_worker_regen/stats."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress = subparsers.add_parser("compress", help="Compress all stats JSONL files except the latest by default.")
    compress.add_argument("--include-latest", action="store_true", help="Also compress the newest stats file.")

    downsample = subparsers.add_parser("downsample", help="Downsample stats_sample events to a minimum interval.")
    downsample.add_argument(
        "frequency_seconds", type=float, help="Minimum seconds between retained stats_sample events."
    )
    downsample.add_argument("--include-latest", action="store_true", help="Also rewrite the newest stats file.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for stats operations."""
    args = build_parser().parse_args(argv)
    if args.command == "compress":
        report = compress_old_stats_files(args.stats_dir, include_latest=args.include_latest)
    elif args.command == "downsample":
        report = downsample_stats_files(
            args.frequency_seconds,
            args.stats_dir,
            include_latest=args.include_latest,
        )
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(f"Unknown command: {args.command}")
    print(_format_report(report))


def _format_report(report: StatsOperationReport) -> str:
    return (
        f"Stats directory: {report.stats_dir}\n"
        f"Files considered: {len(report.files)}\n"
        f"Files changed: {report.files_changed}\n"
        f"Samples dropped: {report.samples_dropped}\n"
        f"Bytes: {report.bytes_before} -> {report.bytes_after}"
    )


if __name__ == "__main__":
    main()
