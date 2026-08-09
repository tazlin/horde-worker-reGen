"""Tests for log-bundle discovery, rotation stitching, and record caching."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from horde_worker_regen.analysis import bundle as bundle_module
from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.log_ingest import LogRecord
from horde_worker_regen.analysis.sessions import segment_sessions


def test_record_accessors_cache_parsed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated analysis lookups must not re-read and re-parse the same process logs."""
    calls: list[tuple[Path, ...]] = []

    def _fake_read_records(*paths: Path) -> list[LogRecord]:
        calls.append(paths)
        return []

    monkeypatch.setattr(bundle_module, "read_records", _fake_read_records)
    bundle = LogBundle(
        root=tmp_path,
        orchestrator_paths=[tmp_path / "bridge.log"],
        child_loop_paths={1: [tmp_path / "bridge_1.log"]},
        startup_paths={1: [tmp_path / "bridge_inference_1_startup.log"]},
    )

    bundle.orchestrator_records()
    bundle.orchestrator_records()
    bundle.child_records(1)
    bundle.child_records(1)
    bundle.startup_records(1)
    bundle.startup_records(1)

    assert calls == [
        (tmp_path / "bridge.log",),
        (tmp_path / "bridge_1.log",),
        (tmp_path / "bridge_inference_1_startup.log",),
    ]


_STARTUP_LINE = (
    "2026-06-24 10:00:00.000 | DEBUG    | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process"
)


def _line(ts: str, message: str) -> str:
    return f"2026-06-24 {ts} | INFO     | a.b:c:1 - {message}"


def _write_rotation(directory: Path, stamp: str, text: str, *, compress: bool) -> Path:
    """Write a loguru-style rotation of ``bridge.log``, zipped or plain as the sink would leave it."""
    plain = directory / f"bridge.{stamp}.log"
    plain.write_text(text, encoding="utf-8")
    if not compress:
        return plain
    archive = directory / f"bridge.{stamp}.log.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(plain, arcname=plain.name)
    plain.unlink()
    return archive


class TestRotationStitching:
    """A single ``bridge.log`` target must still see the rotated predecessors of the run it captures."""

    @staticmethod
    def _run_with_one_rotation(tmp_path: Path, *, compress: bool) -> Path:
        """A run whose first half rotated out: the launch line is in the rotation, not in ``bridge.log``."""
        _write_rotation(
            tmp_path,
            "2026-06-24_10-00-00_000001",
            "\n".join([_STARTUP_LINE, _line("10:30:00.000", "first half")]) + "\n",
            compress=compress,
        )
        active = tmp_path / "bridge.log"
        active.write_text(_line("10:30:00.500", "second half") + "\n", encoding="utf-8")
        return active

    @pytest.mark.parametrize("compress", [True, False], ids=["zipped", "plain"])
    def test_a_single_file_target_stitches_its_predecessor(self, tmp_path: Path, compress: bool) -> None:
        """Targeting the active file alone must not report the mid-run rotation as the session start."""
        active = self._run_with_one_rotation(tmp_path, compress=compress)
        bundle = LogBundle.from_path(active)
        session = segment_sessions(bundle.orchestrator_records())[0]
        assert session.start_is_lower_bound is False, "the launch line lives in the stitched predecessor"
        assert session.start_ts is not None
        assert session.start_ts.hour == 10 and session.start_ts.minute == 0

    def test_the_stitched_files_are_disclosed(self, tmp_path: Path) -> None:
        """What was folded in is named, so a span is never silently widened."""
        active = self._run_with_one_rotation(tmp_path, compress=True)
        bundle = LogBundle.from_path(active)
        assert bundle.rotation_stitch is not None
        assert [path.name for path in bundle.rotation_stitch.included] == [
            "bridge.2026-06-24_10-00-00_000001.log.zip",
        ]
        assert bundle.rotation_stitch.excluded == []
        assert "bridge.2026-06-24_10-00-00_000001.log.zip" in bundle.rotation_stitch.describe()

    def test_a_non_abutting_rotation_is_excluded_and_named(self, tmp_path: Path) -> None:
        """A rotation from an older run does not abut, so it is left out and said to be left out."""
        _write_rotation(
            tmp_path,
            "2026-06-24_08-00-00_000001",
            "\n".join([_STARTUP_LINE, _line("08:10:00.000", "an older run")]) + "\n",
            compress=True,
        )
        active = self._run_with_one_rotation(tmp_path, compress=True)
        bundle = LogBundle.from_path(active)
        assert bundle.rotation_stitch is not None
        assert [path.name for path in bundle.rotation_stitch.included] == [
            "bridge.2026-06-24_10-00-00_000001.log.zip",
        ]
        assert [path.name for path in bundle.rotation_stitch.excluded] == [
            "bridge.2026-06-24_08-00-00_000001.log.zip",
        ]
        assert "excluded" in bundle.rotation_stitch.describe()

    def test_no_rotations_leaves_no_disclosure(self, tmp_path: Path) -> None:
        """Nothing to stitch means nothing to disclose."""
        active = tmp_path / "bridge.log"
        active.write_text(_STARTUP_LINE + "\n", encoding="utf-8")
        assert LogBundle.from_path(active).rotation_stitch is None

    def test_targeting_a_rotation_itself_does_not_stitch(self, tmp_path: Path) -> None:
        """An operator who names one archive gets that archive, not the run around it."""
        rotation = _write_rotation(
            tmp_path,
            "2026-06-24_10-00-00_000001",
            _STARTUP_LINE + "\n",
            compress=False,
        )
        (tmp_path / "bridge.log").write_text(_line("10:30:00.500", "second half") + "\n", encoding="utf-8")
        bundle = LogBundle.from_path(rotation)
        assert bundle.rotation_stitch is None
        assert bundle.orchestrator_paths == [rotation]

    def test_a_directory_target_discloses_what_it_read(self, tmp_path: Path) -> None:
        """A directory already reads every rotation; the disclosure names them so the span is accountable."""
        self._run_with_one_rotation(tmp_path, compress=True)
        bundle = LogBundle.from_path(tmp_path)
        assert bundle.rotation_stitch is not None
        assert [path.name for path in bundle.rotation_stitch.included] == [
            "bridge.2026-06-24_10-00-00_000001.log.zip",
        ]
