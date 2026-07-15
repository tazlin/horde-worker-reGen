"""Validate the worker log-file registry and keep it in step with the sinks the code actually opens.

Two layers:

* Registry integrity and coverage: the declared specs are well-formed, mutually exclusive, and classify
  a curated set of real-world filenames (active files, rotated ``.zip`` archives, per-slot variants) the
  way the purge and the log tooling rely on.
* Drift detection (the CI job): spawn the *real* logging setups in isolated subprocesses, collect the
  loguru file sinks they actually register, and assert every one is described by a registry spec. If a
  new file sink is added in this repo or in hordelib whose name no spec covers, this fails, so the
  registry cannot silently fall behind the code that writes the logs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from horde_worker_regen.log_file_registry import (
    WORKER_LOG_FILE_SPECS,
    base_log_name,
    classify_log_file,
    is_worker_log_file,
)

_PROBE = Path(__file__).parent / "_log_sink_probe.py"

# Representative real filenames per spec, including rotated/compressed and per-slot variants. Every entry
# must classify to the named spec; together they must exercise every spec (asserted below).
_KNOWN_SAMPLES: dict[str, list[str]] = {
    "orchestrator_loop": [
        "bridge.log",
        "bridge.2026-06-22_00-55-59_013989.log.zip",
        "bridge.2026-06-22_00-55-59.log.gz",
    ],
    "child_loop": ["bridge_0.log", "bridge_3.log", "bridge_0.2026-06-22_00-55-59.log.zip"],
    "orchestrator_trace": ["trace.log", "trace.2026-06-22_00-55-59.log.zip"],
    "child_trace": ["trace_0.log", "trace_11.2026-06-22_00-55-59.log.zip"],
    "child_stdout": ["stdout_0.log", "stdout_2.log"],
    "child_stderr": ["stderr_0.log", "stderr_2.log"],
    "supervisor_loop": ["bridge_tui.log", "bridge_host.log", "bridge_tui.2026-06-22_00-55-59.log.zip"],
    "main_console": ["bridge_main_console.log"],
    "utilities_console": ["bridge_utilities_0.log", "bridge_utilities_3.log"],
    "startup_crash": [
        "bridge_main_startup.log",
        "bridge_inference_1_startup.log",
        "bridge_safety_2_startup.log",
        "bridge_download_0_startup.log",
        "bridge_cn_annotator_prewarm_startup.log",
    ],
    "faulthandler": ["bridge_main.faulthandler", "bridge_inference_1.faulthandler"],
}


def test_specs_are_wellformed_and_unique() -> None:
    """Every spec has a unique name, a non-empty description/writer, and a valid kind."""
    valid_kinds = {"loguru_sink", "raw_stream", "startup_crash", "faulthandler"}
    names = [spec.name for spec in WORKER_LOG_FILE_SPECS]
    assert len(names) == len(set(names)), "spec names must be unique"
    for spec in WORKER_LOG_FILE_SPECS:
        assert spec.description.strip(), f"{spec.name} needs a description"
        assert spec.writer.strip(), f"{spec.name} needs a writer note"
        assert spec.kind in valid_kinds, f"{spec.name} has an unknown kind {spec.kind!r}"


@pytest.mark.parametrize(
    ("filename", "expected_spec"),
    [(name, spec) for spec, names in _KNOWN_SAMPLES.items() for name in names],
)
def test_known_samples_classify_to_exactly_one_spec(filename: str, expected_spec: str) -> None:
    """Each representative filename classifies to its expected (single) spec, and matches no other."""
    matches = [spec.name for spec in WORKER_LOG_FILE_SPECS if spec.pattern.fullmatch(base_log_name(filename))]
    assert matches == [expected_spec], f"{filename!r} matched {matches}, expected [{expected_spec!r}]"


def test_every_spec_is_exercised_by_a_sample() -> None:
    """No dead specs: every declared family has at least one known sample covering it."""
    covered = set(_KNOWN_SAMPLES)
    declared = {spec.name for spec in WORKER_LOG_FILE_SPECS}
    assert declared == covered, (
        f"specs without samples: {declared - covered}; samples without specs: {covered - declared}"
    )


def test_unrelated_names_are_not_worker_logs() -> None:
    """The registry does not claim files the worker never writes."""
    for name in ("action_ledger.jsonl", "notes.txt", "config.yaml", "some_other_tool.log", "bridge", "bridge.txt"):
        assert not is_worker_log_file(name), f"{name!r} should not be a worker log"


def _run_probe(mode: str, tmp_path: Path) -> list[str]:
    """Run the sink probe for *mode* in a fresh subprocess and return the discovered basenames."""
    workdir = tmp_path / mode
    workdir.mkdir()
    out_path = workdir / "sinks.json"
    proc = subprocess.run(
        [sys.executable, str(_PROBE), mode, str(workdir), str(out_path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        if "ModuleNotFoundError" in proc.stderr or "ImportError" in proc.stderr:
            pytest.skip(f"probe dependencies unavailable for {mode}: {proc.stderr.strip().splitlines()[-1:]}")
        pytest.fail(f"probe {mode} failed ({proc.returncode}):\n{proc.stderr}")
    return json.loads(out_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("mode", "must_include"),
    [
        ("hordelib-main", {"bridge.log", "trace.log"}),
        ("hordelib-child", {"bridge_0.log", "trace_0.log"}),
        ("supervisor-tui", {"bridge_tui.log"}),
        ("supervisor-host", {"bridge_host.log"}),
    ],
)
def test_registered_loguru_sinks_are_all_in_the_registry(mode: str, must_include: set[str], tmp_path: Path) -> None:
    """The loguru file sinks each real setup registers are all described by a loguru-sink spec.

    This is the drift guard: it introspects what the code (worker + hordelib) actually opens, so adding
    or renaming a file sink without updating the registry fails here.
    """
    registered = _run_probe(mode, tmp_path)
    assert must_include.issubset(set(registered)), f"{mode} registered {registered}, missing {must_include}"
    for basename in registered:
        spec = classify_log_file(basename)
        assert spec is not None, f"{mode} registered sink {basename!r} is not described by any registry spec"
        assert spec.kind == "loguru_sink", f"{basename!r} classified as {spec.kind}, expected loguru_sink"
