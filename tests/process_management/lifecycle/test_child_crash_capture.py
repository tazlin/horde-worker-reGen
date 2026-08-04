"""Tests for the spawned-worker startup crash-capture backstops."""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

from horde_worker_regen.process_management.lifecycle.child_crash_capture import (
    _FAULTHANDLER_MAX_BYTES,
    _rearm_if_oversized,
    enable_child_faulthandler,
    neutralize_inherited_argv,
    read_last_startup_crash,
    write_startup_crash,
)
from horde_worker_regen.tui.log_tailer import discover_bridge_logs_grouped


def test_write_startup_crash_creates_discoverable_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A startup crash is written to a discoverable logs/bridge_{role}_startup.log with a full traceback."""
    monkeypatch.chdir(tmp_path)
    try:
        raise ImportError("No module named 'diffusers'")
    except ImportError as error:
        write_startup_crash("main", error)

    log_path = tmp_path / "logs" / "bridge_main_startup.log"
    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8")
    # The full exception chain, not just the message, so the operator can see where it failed.
    assert "Traceback (most recent call last):" in contents
    assert "No module named 'diffusers'" in contents
    # The same "| LEVEL |" shape as the other bridge logs so the Logs tab parses a level token.
    assert " | CRITICAL | " in contents

    # The Logs tab globs bridge*.log; the file must show up as its own process entry.
    grouped = discover_bridge_logs_grouped(tmp_path / "logs")
    assert "main_startup" in grouped


def test_write_startup_crash_embeds_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The os pid and launch id are stamped into the line so an offline tool can join it exactly."""
    monkeypatch.chdir(tmp_path)
    try:
        raise AssertionError("Torch not compiled with CUDA enabled")
    except AssertionError as error:
        write_startup_crash("inference_1", error, os_pid=4600, launch_identifier=2)

    contents = (tmp_path / "logs" / "bridge_inference_1_startup.log").read_text(encoding="utf-8")
    assert "(os_pid=4600, launch=2)" in contents
    assert "Torch not compiled with CUDA enabled" in contents


def test_read_last_startup_crash_returns_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The reader lifts the most recent crash's exception summary from the appended startup file."""
    monkeypatch.chdir(tmp_path)
    try:
        raise AssertionError("Torch not compiled with CUDA enabled")
    except AssertionError as error:
        write_startup_crash("inference_1", error, os_pid=4600, launch_identifier=2)

    assert read_last_startup_crash("inference_1") == "AssertionError: Torch not compiled with CUDA enabled"


def test_read_last_startup_crash_missing_file_is_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A slot that never wrote a startup crash yields None, not an error."""
    monkeypatch.chdir(tmp_path)
    assert read_last_startup_crash("inference_9") is None


def test_write_startup_crash_is_lazy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No file is created unless a crash is actually recorded (so the Logs tab is not cluttered)."""
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "logs" / "bridge_main_startup.log").exists()


def test_write_startup_crash_swallows_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The emergency writer must never raise over the original crash, even if the write itself fails."""
    monkeypatch.chdir(tmp_path)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", _boom)
    # Must not raise.
    write_startup_crash("main", RuntimeError("original"))


def _annotator_like_parser() -> argparse.ArgumentParser:
    """A parser shaped like the ComfyUI leres/pix2pix annotator options that read ``sys.argv``.

    It defines several ``--output_*`` options, so an inherited ``--out`` is an ambiguous abbreviation
    (argparse's default ``allow_abbrev=True``), which is what makes ``parse_known_args`` call
    ``sys.exit(2)`` when the child inherits the benchmark's ``--out`` flag.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_nc", type=int, default=1)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--output_resolution", type=int, required=False)
    return parser


def test_neutralize_inherited_argv_reduces_to_program_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child's argv is cut down to just the program name; the inherited flags are dropped."""
    monkeypatch.setattr(sys, "argv", ["run_worker", "run", "--tiers", "sd15,sdxl", "--out", "results/x"])
    neutralize_inherited_argv()
    assert sys.argv == ["run_worker"]


def test_neutralize_inherited_argv_prevents_annotator_argparse_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inherited ``--out`` crashes an annotator-like argparse; neutralizing argv first prevents it.

    This pins the exact failure mode: a spawned child that inherits the benchmark's ``--out`` flag lets a
    library ``parse_known_args`` call ``sys.exit(2)`` mid-inference (only depth/normal controlnet
    preprocessors build such a parser), which surfaces only as an unexplained process recovery.
    """
    dirty = ["run_worker", "run", "--tiers", "sd15,sdxl", "--out", "results/x"]

    # Without the fix, the inherited --out is an ambiguous prefix and argparse exits(2).
    monkeypatch.setattr(sys, "argv", list(dirty))
    with pytest.raises(SystemExit) as exit_info:
        _annotator_like_parser().parse_known_args()
    assert exit_info.value.code == 2

    # With the fix applied first, the same parse sees no inherited flags and succeeds.
    monkeypatch.setattr(sys, "argv", list(dirty))
    neutralize_inherited_argv()
    namespace, _extras = _annotator_like_parser().parse_known_args()
    assert namespace.output_dir is None


def test_neutralize_inherited_argv_handles_empty_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralizing an already-empty argv is a harmless no-op (never raises)."""
    monkeypatch.setattr(sys, "argv", [])
    neutralize_inherited_argv()
    assert sys.argv == []


def test_enable_child_faulthandler_opens_file_and_enables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The faulthandler backstop opens its per-process file and enables faulthandler without raising."""
    monkeypatch.chdir(tmp_path)
    try:
        enable_child_faulthandler("inference_0")
        assert (tmp_path / "logs" / "bridge_inference_0.faulthandler").exists()
        assert faulthandler.is_enabled()
    finally:
        faulthandler.disable()


def test_enable_child_faulthandler_writes_a_dated_banner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each arming writes an identifying, dated banner so a dump can be attributed to a launch.

    faulthandler's own output carries no timestamp and the file is shared across every launch of a role, so
    without the banner a dump cannot be lined up against the timestamped logs.
    """
    monkeypatch.chdir(tmp_path)
    try:
        enable_child_faulthandler("safety_0")
    finally:
        faulthandler.disable()

    banner = (tmp_path / "logs" / "bridge_safety_0.faulthandler").read_text(encoding="utf-8")

    assert "safety_0" in banner
    assert f"pid={os.getpid()}" in banner
    assert str(datetime.now().year) in banner


def test_enable_child_faulthandler_bounds_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An oversized fault file is rolled aside rather than grown without bound.

    A role that faults on most of its launches would otherwise accumulate megabytes whose oldest entries
    predate everything else in a log bundle.
    """
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    fault_path = log_dir / "bridge_inference_1.faulthandler"
    fault_path.write_text("x" * (_FAULTHANDLER_MAX_BYTES + 1), encoding="utf-8")

    try:
        enable_child_faulthandler("inference_1")
    finally:
        faulthandler.disable()

    assert fault_path.with_suffix(".faulthandler.1").exists()
    assert fault_path.stat().st_size < _FAULTHANDLER_MAX_BYTES


def test_rolled_segment_carries_a_closing_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A rolled file ends with a marker naming its role, so its dumps stay attributable.

    A roll can cut a process's dump history at an arbitrary point, leaving a segment whose armed banner
    is in an earlier generation; the closing marker is what still identifies it.
    """
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    fault_path = log_dir / "bridge_inference_3.faulthandler"
    fault_path.write_text("x" * (_FAULTHANDLER_MAX_BYTES + 1), encoding="utf-8")

    try:
        enable_child_faulthandler("inference_3")
    finally:
        faulthandler.disable()

    rolled = fault_path.with_suffix(".faulthandler.1").read_text(encoding="utf-8")
    assert "inference_3" in rolled
    assert "segment closed" in rolled
    assert fault_path.name in rolled, "the marker must point at where capture continued"


def test_running_process_rearms_when_its_fault_file_passes_the_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A file that grows past the bound mid-run is rolled and re-armed without waiting for a relaunch.

    Enforcing the bound only at startup lets a process that keeps faulting write its whole history under
    one banner, so a later roll splits that stream at an arbitrary byte.
    """
    monkeypatch.chdir(tmp_path)
    try:
        enable_child_faulthandler("inference_4")
        fault_path = tmp_path / "logs" / "bridge_inference_4.faulthandler"
        with fault_path.open("a", encoding="utf-8") as handle:
            handle.write("dump\n" * (_FAULTHANDLER_MAX_BYTES // 5))

        assert _rearm_if_oversized("inference_4") is True
        assert faulthandler.is_enabled()
    finally:
        faulthandler.disable()

    assert "faulthandler armed" in fault_path.read_text(encoding="utf-8"), "the new segment has no banner"
    assert fault_path.stat().st_size < _FAULTHANDLER_MAX_BYTES
    rolled = fault_path.with_suffix(".faulthandler.1")
    assert "segment closed" in rolled.read_text(encoding="utf-8")


def test_every_segment_a_reader_sees_is_self_identifying(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Across repeated mid-run rolls, no segment is left as bare dumps with nothing naming its writer."""
    monkeypatch.chdir(tmp_path)
    fault_path = tmp_path / "logs" / "bridge_inference_5.faulthandler"
    try:
        enable_child_faulthandler("inference_5")
        for _ in range(3):
            with fault_path.open("a", encoding="utf-8") as handle:
                handle.write("Current thread 0x0000 (most recent call first):\n" * 30000)
            _rearm_if_oversized("inference_5")
    finally:
        faulthandler.disable()

    segments = [fault_path, fault_path.with_suffix(".faulthandler.1")]
    for segment in segments:
        text = segment.read_text(encoding="utf-8")
        assert "faulthandler armed" in text or "segment closed" in text, f"{segment.name} identifies nobody"
        assert "inference_5" in text


def test_rearm_leaves_a_small_file_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The size watch is a no-op below the bound, so a healthy file is not churned every interval."""
    monkeypatch.chdir(tmp_path)
    try:
        enable_child_faulthandler("inference_6")
        assert _rearm_if_oversized("inference_6") is False
    finally:
        faulthandler.disable()

    assert not (tmp_path / "logs" / "bridge_inference_6.faulthandler.1").exists()


def test_enable_child_faulthandler_keeps_a_small_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A fault file under the bound is appended to, so recent launch history survives.

    The control for the roll: bounding history must not discard the immediately preceding runs.
    """
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    fault_path = log_dir / "bridge_inference_2.faulthandler"
    fault_path.write_text("earlier launch dump\n", encoding="utf-8")

    try:
        enable_child_faulthandler("inference_2")
    finally:
        faulthandler.disable()

    assert "earlier launch dump" in fault_path.read_text(encoding="utf-8")
    assert not fault_path.with_suffix(".faulthandler.1").exists()
