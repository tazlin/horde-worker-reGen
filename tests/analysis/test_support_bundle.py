"""Tests for the support-bundle generator, led by the safety guarantee: secrets never ship.

The single most important property of this feature is that a generated bundle contains no API key. These
tests build a bundle from a synthetic worker directory whose config and logs carry a real-looking key,
then read every member of the produced zip and assert the secret appears nowhere.
"""

from __future__ import annotations

import gzip
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from horde_worker_regen.analysis import support_bundle
from horde_worker_regen.analysis.support_bundle import build_support_bundle
from horde_worker_regen.analysis.triage_report import BENCHMARK_RUN_TAG

_API_KEY = "abcdEFGH1234ijklMNOP56"
_CIVITAI = "cd92292204eaa0759418fdebc5ae6d79"
_WORKER = "tazlin-tui-example"


def _recovery(ts: str) -> str:
    """A parent recovery-diagnostics line for slot 1 crashing on start (os_pid matches the child log)."""
    return (
        f"2026-06-24 {ts} | ERROR | horde_worker_regen.process_management.lifecycle.process_lifecycle:_log_recovery_diagnostics:367 - "
        "Recovery diagnostics for process 1 (os_pid=4600, launch=2): reason='inference process replaced (crashed or hung)'; "
        "last_state=PROCESS_STARTING; exitcode=1; last_heartbeat_type=OTHER; since_last_heartbeat=8.0s; "
        "since_last_message=8.0s; last_job=None; recent_actions=[]"
    )


_BRIDGE_LOG = (
    "2026-06-24 18:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process\n"
    f"2026-06-24 18:00:05.000 | INFO | x:y:1 -   dreamer_name: {_WORKER} | (v12.29.0) | num_models: 113 | "
    "max_power: 32 (1024x1024) | max_threads: 1 | queue_size: 3 | safety_on_gpu: True\n"
    # An env-var echo of the key in a subprocess traceback (the realistic leak path).
    f"2026-06-24 18:00:10.000 | ERROR | x:y:2 - environ has AIHORDE_API_KEY={_API_KEY} during crash\n"
    + _recovery("18:00:11.000")
    + "\n"
    + _recovery("18:00:20.000")
    + "\n"
)


def _worker_dir(tmp_path: Path) -> Path:
    """A synthetic worker directory: a config with secrets and a logs/ holding a key-leaking log."""
    (tmp_path / "bridgeData.yaml").write_text(
        f"api_key: {_API_KEY}\ncivitai_api_token: {_CIVITAI}\ndreamer_name: {_WORKER}\ncache_home: {tmp_path / 'cache'}\n",
        encoding="utf-8",
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "bridge.log").write_text(_BRIDGE_LOG, encoding="utf-8")
    (logs / "bridge_inference_1_startup.log").write_text(
        f"2026-06-24 18:00:09.000 | CRITICAL | inference_1:startup - worker child (os_pid=4600, launch=2) crashed:\n"
        f"AssertionError: Torch not compiled with CUDA enabled (key {_API_KEY} leaked here too)\n",
        encoding="utf-8",
    )
    return logs


def _all_member_text(zip_path: Path) -> str:
    """Concatenate the text of every member of a zip (for a blunt 'secret appears nowhere' assertion)."""
    chunks = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            chunks.append(zf.read(name).decode("utf-8", errors="replace"))
    return "\n".join(chunks)


class TestSafety:
    """The secret must not survive into the bundle, from any source."""

    def test_api_key_absent_everywhere(self, tmp_path: Path) -> None:
        """The key leaks from the config AND a log traceback; neither survives into the zip."""
        logs = _worker_dir(tmp_path)
        out = tmp_path / "bundle.zip"
        result = build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")

        contents = _all_member_text(out)
        assert _API_KEY not in contents
        assert _CIVITAI not in contents
        assert result.redaction_count > 0

    def test_worker_name_redacted_by_default(self, tmp_path: Path) -> None:
        """Identifier redaction is on by default, so the worker name is scrubbed too."""
        logs = _worker_dir(tmp_path)
        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")
        assert _WORKER not in _all_member_text(out)

    def test_foreign_bundle_pattern_backstop(self, tmp_path: Path) -> None:
        """With no config (a bundle of someone else's logs), `api_key: ...` lines are still scrubbed."""
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "bridge.log").write_text(
            "2026-06-24 18:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process\n"
            "2026-06-24 18:00:01.000 | INFO | x:y:1 - api_key: someForeignKeyValue999\n",
            encoding="utf-8",
        )
        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "does_not_exist.yaml")
        assert "someForeignKeyValue999" not in _all_member_text(out)


class TestContents:
    """The bundle is self-describing and leads with the analysis."""

    def test_has_expected_members(self, tmp_path: Path) -> None:
        """Diagnosis, manifest, redacted config, and the logs are all present."""
        logs = _worker_dir(tmp_path)
        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        assert {"diagnose.txt", "manifest.json", "README.txt", "config/bridgeData.redacted.yaml"} <= names
        assert any(n.startswith("logs/") for n in names)

    def test_a_bundle_without_stats_says_the_export_is_off(self, tmp_path: Path) -> None:
        """The manifest counts the stats files and the README says why there are none."""
        logs = _worker_dir(tmp_path)
        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")
        with zipfile.ZipFile(out) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            readme = zf.read("README.txt").decode("utf-8")
        assert manifest["scope"]["stats_files"] == 0
        assert manifest["scope"]["stats_export_enabled"] is False
        assert "`stats_export_enabled` is off" in readme

    def test_diagnose_reports_crash_root_cause(self, tmp_path: Path) -> None:
        """The bundled diagnosis lifts the child's exception, scrubbed of the leaked key."""
        logs = _worker_dir(tmp_path)
        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")
        with zipfile.ZipFile(out) as zf:
            diagnose = zf.read("diagnose.txt").decode("utf-8")
        assert "Torch not compiled with CUDA enabled" in diagnose
        assert _API_KEY not in diagnose

    def test_stats_jsonl_files_are_included_and_redacted(self, tmp_path: Path) -> None:
        """Retained worker stats JSONL files ship with the support bundle."""
        logs = _worker_dir(tmp_path)
        stats_dir = tmp_path / ".horde_worker_regen" / "stats"
        stats_dir.mkdir(parents=True)
        (stats_dir / "stats-v1.0.0-20260620-010203-000.jsonl").write_text(
            json.dumps({"event": "stats_sample", "worker": _WORKER}) + "\n",
            encoding="utf-8",
        )
        with gzip.open(
            stats_dir / "stats-v1.0.0-20260620-010203-001.jsonl.gz",
            "wt",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps({"event": "job_completed", "token": _CIVITAI}) + "\n")

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")

        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
            stats_text = zf.read("stats/stats-v1.0.0-20260620-010203-000.jsonl").decode("utf-8")
            compressed_stats_text = zf.read("stats/stats-v1.0.0-20260620-010203-001.jsonl").decode("utf-8")
        assert "stats/stats-v1.0.0-20260620-010203-000.jsonl" in names
        assert "stats/stats-v1.0.0-20260620-010203-001.jsonl" in names
        assert _WORKER not in stats_text
        assert _CIVITAI not in compressed_stats_text


class TestStatsWindow:
    """Only stats files still being written once the bundled sessions began are shipped by default."""

    def _stats_file(self, tmp_path: Path, name: str, *, mtime: float | None = None) -> Path:
        stats_dir = tmp_path / ".horde_worker_regen" / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        path = stats_dir / name
        path.write_text(json.dumps({"event": "stats_sample"}) + "\n", encoding="utf-8")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def test_stats_older_than_every_session_are_skipped(self, tmp_path: Path) -> None:
        """A retention file last written before the earliest bundled session says nothing about it."""
        logs = _worker_dir(tmp_path)
        stale = datetime(2026, 6, 1, 12, 0, 0).timestamp()
        self._stats_file(tmp_path, "stats-v1.0.0-20260601-120000-000.jsonl", mtime=stale)
        self._stats_file(tmp_path, "stats-v1.0.0-20260624-180000-000.jsonl")

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")

        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        assert "stats/stats-v1.0.0-20260624-180000-000.jsonl" in names
        assert "stats/stats-v1.0.0-20260601-120000-000.jsonl" not in names

    def test_full_logs_ships_the_whole_retention(self, tmp_path: Path) -> None:
        """``--full-logs`` includes every retained stats file, as it does the log rotations."""
        logs = _worker_dir(tmp_path)
        stale = datetime(2026, 6, 1, 12, 0, 0).timestamp()
        self._stats_file(tmp_path, "stats-v1.0.0-20260601-120000-000.jsonl", mtime=stale)

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml", full_logs=True)

        with zipfile.ZipFile(out) as zf:
            assert "stats/stats-v1.0.0-20260601-120000-000.jsonl" in zf.namelist()


class TestTruncationPolarity:
    """Everything the bundle trims keeps its most recent end, on a record boundary."""

    @staticmethod
    def _jsonl(count: int) -> str:
        """``count`` ledger-shaped JSONL records, each carrying its index and a padding field."""
        return "".join(json.dumps({"index": i, "pad": "p" * 200}) + "\n" for i in range(count))

    def test_ledger_keeps_the_most_recent_records(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An oversized action ledger is trimmed from the front, so the newest events survive."""
        logs = _worker_dir(tmp_path)
        ledger_dir = tmp_path / ".horde_worker_regen"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        (ledger_dir / "action_ledger.jsonl").write_text(self._jsonl(200), encoding="utf-8")
        monkeypatch.setattr(support_bundle, "_MAX_FILE_BYTES", 4096)

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")
        with zipfile.ZipFile(out) as zf:
            ledger = zf.read("action_ledger.jsonl").decode("utf-8")

        indices = [json.loads(line)["index"] for line in ledger.splitlines()[1:] if line]
        assert indices, "the trimmed ledger kept no records at all"
        assert indices[-1] == 199, "the newest ledger record was dropped"
        assert indices[0] > 0, "the ledger was not actually trimmed by this test's cap"
        assert indices == list(range(indices[0], 200)), "the kept records are not a contiguous recent run"

    def test_ledger_truncation_is_marked_and_record_aligned(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The trim is announced in the same words the logs use, and no line is cut mid-record."""
        logs = _worker_dir(tmp_path)
        ledger_dir = tmp_path / ".horde_worker_regen"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        (ledger_dir / "action_ledger.jsonl").write_text(self._jsonl(200), encoding="utf-8")
        monkeypatch.setattr(support_bundle, "_MAX_FILE_BYTES", 4096)

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")
        with zipfile.ZipFile(out) as zf:
            ledger = zf.read("action_ledger.jsonl").decode("utf-8")

        lines = [line for line in ledger.splitlines() if line]
        note = json.loads(lines[0])
        assert "truncated to the most recent" in note[support_bundle._TRUNCATION_NOTE_KEY]
        for line in lines:
            json.loads(line)  # every surviving line is a whole record, including the one at the cut

    def test_stats_keep_the_most_recent_records(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An oversized stats file is trimmed with the same polarity as the ledger and the logs."""
        logs = _worker_dir(tmp_path)
        stats_dir = tmp_path / ".horde_worker_regen" / "stats"
        stats_dir.mkdir(parents=True)
        (stats_dir / "stats-v1.0.0-20260620-010203-000.jsonl").write_text(self._jsonl(200), encoding="utf-8")
        monkeypatch.setattr(support_bundle, "_MAX_FILE_BYTES", 4096)

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")
        with zipfile.ZipFile(out) as zf:
            stats = zf.read("stats/stats-v1.0.0-20260620-010203-000.jsonl").decode("utf-8")

        lines = [line for line in stats.splitlines() if line]
        assert support_bundle._TRUNCATION_NOTE_KEY in json.loads(lines[0])
        assert json.loads(lines[-1])["index"] == 199

    def test_full_logs_keeps_every_record(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--full-logs`` lifts the cap for the JSONL artifacts too."""
        logs = _worker_dir(tmp_path)
        ledger_dir = tmp_path / ".horde_worker_regen"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        (ledger_dir / "action_ledger.jsonl").write_text(self._jsonl(200), encoding="utf-8")
        monkeypatch.setattr(support_bundle, "_MAX_FILE_BYTES", 4096)

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml", full_logs=True)
        with zipfile.ZipFile(out) as zf:
            ledger = zf.read("action_ledger.jsonl").decode("utf-8")

        assert [json.loads(line)["index"] for line in ledger.splitlines() if line] == list(range(200))


class TestPerformance:
    """Support-bundle generation should not pay full-read cost for logs it tail-caps."""

    def test_capped_plain_log_reads_only_tail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A large active ``.log`` is read from the end instead of through the full physical-line reader."""
        path = tmp_path / "bridge_1.log"
        path.write_text("old prefix\n" + ("x" * 2048) + "\nnew tail\n", encoding="utf-8")
        monkeypatch.setattr(support_bundle, "_MAX_FILE_BYTES", 1024)

        def _fail_full_read(_path: Path) -> list[str]:
            raise AssertionError("oversized plain logs should be tail-read directly")

        monkeypatch.setattr(support_bundle, "_read_physical_lines", _fail_full_read)

        text = support_bundle._read_log_text(path, cap=True)

        assert text.startswith("[... truncated to the most recent")
        assert "new tail" in text
        assert "old prefix" not in text

    def test_capped_plain_log_drops_the_line_cut_by_the_seek(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The partial line the byte-offset cut lands in is dropped, so the file starts on a whole line."""
        path = tmp_path / "bridge_1.log"
        path.write_text(("head " * 400) + "\nkept line\n", encoding="utf-8")
        monkeypatch.setattr(support_bundle, "_MAX_FILE_BYTES", 64)

        text = support_bundle._read_log_text(path, cap=True)

        body = text.split("\n", 1)[1]
        assert body.strip() == "kept line"
        assert "head" not in body


class TestMidRunSessionStitching:
    """A session that begins mid-run in the active log ships its abutting rotations and their stats window."""

    _ROTATION = "bridge.2026-06-24_17-30-00_000000.log"

    def _mid_run_worker_dir(self, tmp_path: Path) -> Path:
        logs = _worker_dir(tmp_path)
        # The launch marker lives in the rotation; the active log starts a minute after the rotation ends.
        (logs / self._ROTATION).write_text(
            "2026-06-24 17:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process\n"
            f"2026-06-24 17:00:05.000 | INFO | x:y:1 -   dreamer_name: {_WORKER} | (v12.29.0) | num_models: 113 | "
            "max_power: 32 (1024x1024) | max_threads: 1 | queue_size: 3 | safety_on_gpu: True\n"
            "2026-06-24 17:59:30.000 | INFO | x:y:3 - still the same run\n",
            encoding="utf-8",
        )
        (logs / "bridge.log").write_text(
            "2026-06-24 18:00:10.000 | INFO | x:y:2 - mid-run record\n" + _recovery("18:00:11.000") + "\n",
            encoding="utf-8",
        )
        # An older rotation from a previous launch, hours before, must not be dragged in.
        (logs / "bridge.2026-06-24_09-00-00_000000.log").write_text(
            "2026-06-24 08:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process\n"
            "2026-06-24 08:30:00.000 | INFO | x:y:9 - an earlier launch\n",
            encoding="utf-8",
        )
        return logs

    def test_abutting_rotation_ships_and_stats_window_reaches_back(self, tmp_path: Path) -> None:
        """The predecessor is a member, the stats file written during it is kept, the unrelated run is not."""
        logs = self._mid_run_worker_dir(tmp_path)
        stats_dir = tmp_path / ".horde_worker_regen" / "stats"
        stats_dir.mkdir(parents=True)
        during_rotation = stats_dir / "stats-v1.0.0-20260624-170000-000.jsonl"
        during_rotation.write_text(json.dumps({"event": "stats_sample"}) + "\n", encoding="utf-8")
        mtime = datetime(2026, 6, 24, 17, 45, 0).timestamp()
        os.utime(during_rotation, (mtime, mtime))

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")

        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
            manifest = json.loads(zf.read("manifest.json"))
        assert f"logs/{self._ROTATION}" in names
        assert "logs/bridge.2026-06-24_09-00-00_000000.log" not in names
        assert "stats/stats-v1.0.0-20260624-170000-000.jsonl" in names
        assert manifest["scope"]["stitched_rotations"] == [f"logs/{self._ROTATION}"]

    def test_the_stitched_active_log_ships_whole(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An active log whose predecessors were stitched is not tail-capped, so the run has no hole."""
        monkeypatch.setattr(support_bundle, "_MAX_FILE_BYTES", 512)
        logs = self._mid_run_worker_dir(tmp_path)
        active = logs / "bridge.log"
        padding = "".join(f"2026-06-24 18:00:{i:02d}.000 | INFO | x:y:2 - filler line {i}\n" for i in range(12, 40))
        active.write_text(padding + active.read_text(encoding="utf-8"), encoding="utf-8")
        assert active.stat().st_size > 512

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")

        with zipfile.ZipFile(out) as zf:
            text = zf.read("logs/bridge.log").decode("utf-8")
        assert not text.startswith("[... truncated to the most recent")
        assert "filler line 12" in text

    def test_diagnosis_includes_ram_hold_before_the_rotation(self, tmp_path: Path) -> None:
        """Bundled analysis reads the predecessor evidence as well as shipping its bytes."""
        logs = self._mid_run_worker_dir(tmp_path)
        rotation = logs / self._ROTATION
        rotation.write_text(
            rotation.read_text(encoding="utf-8")
            + "2026-06-24 17:59:31.000 | INFO | x:y:3 - Host RAM pop hold engaged: available 8000 MB "
            "above danger floor 6343 MB, soft hold 8500 MB, preload 14500 MB, restore 32500 MB; "
            "in-flight jobs continue.\n",
            encoding="utf-8",
        )
        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml", last=True)
        with zipfile.ZipFile(out) as zf:
            diagnosis = json.loads(zf.read("diagnose.json"))
            findings = {finding["id"]: finding for finding in diagnosis[0]["findings"]}
            sessions = zf.read("sessions.txt").decode("utf-8")
        assert "host_ram_starvation" in findings
        assert "17:00:00" in sessions

    def test_a_session_with_its_launch_in_the_active_log_ships_no_rotation(self, tmp_path: Path) -> None:
        """The default bundle stays lean when the active log already holds the launch."""
        logs = _worker_dir(tmp_path)
        (logs / self._ROTATION).write_text(
            "2026-06-24 17:59:59.000 | INFO | x:y:3 - a prior run ending just before\n",
            encoding="utf-8",
        )

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")

        with zipfile.ZipFile(out) as zf:
            assert f"logs/{self._ROTATION}" not in zf.namelist()


class TestFootprintStore:
    """The learned VRAM footprint store ships beside the config, since it priced the verdicts in the logs."""

    def test_footprint_store_is_included_when_present(self, tmp_path: Path) -> None:
        """A present store is shipped verbatim under config/, where the redactor also passes over it."""
        logs = _worker_dir(tmp_path)
        state_dir = tmp_path / ".horde_worker_regen"
        state_dir.mkdir(exist_ok=True)
        (state_dir / "vram_footprints.json").write_text('{"schema_version": 2}', encoding="utf-8")

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")

        with zipfile.ZipFile(out) as zf:
            assert zf.read("config/vram_footprints.json").decode() == '{"schema_version": 2}'


class TestTextBackendLog:
    """The text backend's own output ships beside the worker's logs, capped the same way, when there is one."""

    _BACKEND_LINES = (
        "find_slot: non-consecutive token position 4096 after 4095 for sequence 1\n"
        f"decode: failed to find a memory slot for batch of size 1024 (key {_API_KEY} echoed)\n"
    )

    def test_the_backend_log_ships_redacted_when_present(self, tmp_path: Path) -> None:
        """A worker that launched a backend bundles its log, scrubbed like every other member."""
        logs = _worker_dir(tmp_path)
        (logs / "text_backend.log").write_text(self._BACKEND_LINES, encoding="utf-8")

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")

        backend_log = _read_member(out, "logs/text_backend.log")
        assert "failed to find a memory slot" in backend_log
        assert _API_KEY not in backend_log

    def test_the_backend_log_is_capped_from_the_front(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An oversized backend log keeps its most recent lines, as the worker's own logs do."""
        logs = _worker_dir(tmp_path)
        (logs / "text_backend.log").write_text(("old load line\n" * 200) + self._BACKEND_LINES, encoding="utf-8")
        monkeypatch.setattr(support_bundle, "_MAX_FILE_BYTES", 256)

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")

        backend_log = _read_member(out, "logs/text_backend.log")
        assert backend_log.startswith("[... truncated to the most recent")
        assert "failed to find a memory slot" in backend_log
        body = backend_log.split("\n", 1)[1]
        assert body.count("old load line") < 200, "the front of the log was not trimmed"
        assert all(line == "old load line" or "slot" in line for line in body.splitlines()), "a torn line survived"

    def test_a_worker_without_a_text_backend_bundles_as_before(self, tmp_path: Path) -> None:
        """No backend log on disk adds no member: the logs are exactly the worker's own."""
        logs = _worker_dir(tmp_path)

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")

        with zipfile.ZipFile(out) as zf:
            log_members = {name for name in zf.namelist() if name.startswith("logs/")}
        assert log_members == {"logs/bridge.log", "logs/bridge_inference_1_startup.log"}

    def test_a_bundle_of_one_parent_log_carries_the_backend_log_beside_it(self, tmp_path: Path) -> None:
        """Pointing the bundle at one log file still ships the backend output its wedge lines refer to."""
        logs = _worker_dir(tmp_path)
        (logs / "text_backend.log").write_text(self._BACKEND_LINES, encoding="utf-8")

        out = tmp_path / "bundle.zip"
        build_support_bundle(logs / "bridge.log", out, config_path=tmp_path / "bridgeData.yaml")

        with zipfile.ZipFile(out) as zf:
            log_members = {name for name in zf.namelist() if name.startswith("logs/")}
        assert log_members == {"logs/bridge.log", "logs/text_backend.log"}


_HARNESS_LOG = (
    "2026-06-24 18:00:03.000 | DEBUG | horde_worker_regen.process_management.process_manager:__init__:1 - "
    "Models to load: []\n"
    "2026-06-24 18:00:30.000 | INFO | x:y:1 - benchmark level running\n"
)


def _read_member(zip_path: Path, name: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(name).decode("utf-8")


class TestBenchmarkRuns:
    """A bundle lists worker sessions and benchmark runs apart, each numbered from #0."""

    def test_both_lists_appear_worker_first(self, tmp_path: Path) -> None:
        """``sessions.txt`` and ``diagnose.txt`` show the worker list, then the benchmark runs."""
        logs = _worker_dir(tmp_path)
        (logs / "bridge_harness.log").write_text(_HARNESS_LOG, encoding="utf-8")
        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")

        sessions_text = _read_member(out, "sessions.txt")
        assert sessions_text.index("1 worker session(s)") < sessions_text.index("1 benchmark run(s)")
        diagnose_text = _read_member(out, "diagnose.txt")
        worker_heading = diagnose_text.index("=== Session #0 ")
        benchmark_heading = diagnose_text.index("=== Session #0  [benchmark run: bridge_harness.log]")
        assert worker_heading < benchmark_heading
        diagnose_json = json.loads(_read_member(out, "diagnose.json"))
        assert [(entry["log_kind"], entry["session_index"]) for entry in diagnose_json] == [
            ("worker", 0),
            ("harness", 0),
        ]
        labels = [entry["session_label"] for entry in diagnose_json]
        assert labels == ["#0", f"#0  {BENCHMARK_RUN_TAG}"]
        assert all(f"=== Session {label}  " in diagnose_text for label in labels)

    def test_the_benchmark_selector_diagnoses_only_the_benchmark_run(self, tmp_path: Path) -> None:
        """``benchmark_index`` picks benchmark run #0, not worker session #0."""
        logs = _worker_dir(tmp_path)
        (logs / "bridge_harness.log").write_text(_HARNESS_LOG, encoding="utf-8")
        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml", benchmark_index=0)

        diagnose_json = json.loads(_read_member(out, "diagnose.json"))
        assert [(entry["log_kind"], entry["session_index"]) for entry in diagnose_json] == [("harness", 0)]
        assert diagnose_json[0]["session_label"] == f"#0  {BENCHMARK_RUN_TAG}"

    def test_a_worker_only_bundle_labels_its_sessions(self, tmp_path: Path) -> None:
        """Without a harness log, each ``diagnose.json`` entry still carries the label the CLI prints."""
        logs = _worker_dir(tmp_path)
        out = tmp_path / "bundle.zip"
        build_support_bundle(logs, out, config_path=tmp_path / "bridgeData.yaml")

        diagnose_json = json.loads(_read_member(out, "diagnose.json"))
        assert [(entry["log_kind"], entry["session_index"], entry["session_label"]) for entry in diagnose_json] == [
            ("worker", 0, "#0")
        ]
        assert "=== Session #0  " in _read_member(out, "diagnose.txt")
