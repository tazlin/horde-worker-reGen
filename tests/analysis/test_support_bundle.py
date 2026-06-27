"""Tests for the support-bundle generator, led by the safety guarantee: secrets never ship.

The single most important property of this feature is that a generated bundle contains no API key. These
tests build a bundle from a synthetic worker directory whose config and logs carry a real-looking key,
then read every member of the produced zip and assert the secret appears nowhere.
"""

from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

from horde_worker_regen.analysis.support_bundle import build_support_bundle

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
