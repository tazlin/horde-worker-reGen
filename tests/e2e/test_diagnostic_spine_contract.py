"""Producer/consumer contract for the worker's diagnostic spine.

Everything offline analysis knows about a run comes from the stats JSONL stream, and the per-second
``stats_sample`` record is its spine: duty attribution, slot-duty buckets and concurrency occupancy are all
differences between adjacent samples. A worker with nothing attached (a soak, a benchmark, a service under a
supervisor that went away) is exactly the run whose stream is read afterwards, so the spine has to be written
whether or not anything is listening.

This runs the real manager headless over fake children and reads the file the run left behind, joining the
producer (the snapshot builder that records the sample) to the consumers that have to make sense of it: the
duty analyzer, and the per-job records a disaggregated job's costs land on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from horde_worker_regen.analysis.session_duty import analyze_stats_files, discover_stats_sessions
from horde_worker_regen.harness import HarnessConfig, HarnessResult, run_harness_async
from horde_worker_regen.process_management.ipc.messages import PipelineStageTag
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager, SystemResources
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.simulation._canned_scenarios import SoakImageTemplate

# Real OS child processes and a minute of simulated load: opt-in via -m slow, like the other e2e sims.
pytestmark = pytest.mark.slow

_SPINE_SECONDS = 60.0
"""How long the measured run generates work for.

Long enough that the expected sample count is a real floor rather than a rounding artefact, and that the
duty analyzer has adjacent samples to difference on both sides of several completed jobs."""

_NEGATIVE_CONTROL_SECONDS = 20.0
"""How long the reinjected-defect run generates work for; it only has to establish an absence."""

_SAMPLE_FLOOR_INTERVAL_SECONDS = 1.0
"""The cadence the headless snapshot builder is expected to hold, matching the publish floor interval."""

_SAMPLE_COUNT_SLACK = 0.5
"""Share of the ideal sample count a passing run may miss.

The cadence is a floor between snapshot builds, not a scheduler: boot, shutdown and any control-loop tick
that runs long push samples later without dropping them. Half the ideal count still separates a spine that
is being written from one that is not, which is the distinction this test exists to make."""


def _spine_system_resources() -> SystemResources:
    """A single 24GB card with room for the disaggregated lanes beside a sampler."""
    return SystemResources(
        total_ram_bytes=32 * 1024 * 1024 * 1024,
        device_map=TorchDeviceMap(
            root={
                0: TorchDeviceInfo(
                    device_name="Diagnostic spine sim 0",
                    device_index=0,
                    total_memory=24 * 1024 * 1024 * 1024,
                    kind="cuda",
                ),
            },
        ),
        per_process_overhead_mb=4200,
        marginal_process_overhead_mb=2000,
    )


def _spine_harness_config(soak_seconds: float) -> HarnessConfig:
    """A headless fake-children soak that exports its stats stream and runs the disaggregated pipeline."""
    return HarnessConfig(
        process_mode="fake",
        skip_api=True,
        soak_seconds=soak_seconds,
        soak_image_templates=[
            SoakImageTemplate(model="Juggernaut XL", width=1024, height=1024, steps=30),
            SoakImageTemplate(model="AlbedoBase XL (SDXL)", width=832, height=1216, steps=25),
        ],
        soak_drain_timeout_seconds=30.0,
        job_delay_seconds=0.5,
        timeout_seconds=soak_seconds + 120.0,
        system_resources=_spine_system_resources(),
        bridge_data_overrides={
            # Fake children fabricate their durations, so the export is off by default for them. The stream
            # itself is what is under test here, and nothing reads these timings as measurements.
            "stats_export_enabled": True,
            "enable_vram_budget": True,
            "max_threads": 2,
            "queue_size": 3,
            "enable_pipeline_disaggregation": True,
            "gpu_sampling_lease_enabled": True,
            "gpu_sampling_lease_slots": 1,
            "unload_models_from_vram_often": False,
        },
    )


def _read_stats_events(paths: list[Path]) -> list[dict[str, Any]]:
    """Return every JSONL record across a session's rotated files, in file order."""
    events: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


def _session_events(stats_dir: Path) -> tuple[str, list[Path], list[dict[str, Any]]]:
    """Return the single stats session the run wrote, as ``(session id, files, records)``."""
    sessions = discover_stats_sessions(stats_dir)
    assert len(sessions) == 1, f"expected exactly one stats session under {stats_dir}, found {sessions}"
    session_id, paths = sessions[0]
    return session_id, paths, _read_stats_events(paths)


def _expected_sample_floor(result: HarnessResult) -> int:
    """Return the fewest ``stats_sample`` records a run of this length may have written."""
    ideal = result.elapsed_seconds / _SAMPLE_FLOOR_INTERVAL_SECONDS
    return int(ideal * (1.0 - _SAMPLE_COUNT_SLACK))


@pytest.mark.e2e
async def test_headless_run_writes_a_spine_its_consumers_can_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A headless worker's stats stream carries the duty spine, and the analyzers get answers from it."""
    monkeypatch.chdir(tmp_path)
    result = await run_harness_async(_spine_harness_config(_SPINE_SECONDS))
    assert not result.timed_out, result.failure_summary()
    assert result.num_jobs_completed > 0, result.failure_summary()

    session_id, paths, events = _session_events(tmp_path / ".horde_worker_regen" / "stats")

    samples = [event for event in events if event.get("event") == "stats_sample"]
    floor = _expected_sample_floor(result)
    assert len(samples) >= floor, (
        f"the headless run recorded {len(samples)} stats_sample events over "
        f"{result.elapsed_seconds:.1f}s, below the floor of {floor} at one per "
        f"{_SAMPLE_FLOOR_INTERVAL_SECONDS:.0f}s: without them the stream carries no duty spine"
    )

    report = analyze_stats_files(session_id=session_id, paths=paths)
    assert report.sample_count > 0, f"the duty analyzer read no samples out of {paths}"
    assert report.completed_jobs > 0, f"the duty analyzer attributed no completed jobs out of {paths}"
    attributed_buckets = {name: seconds for name, seconds in report.slot_duty_seconds.items() if seconds > 0}
    assert attributed_buckets, (
        f"the duty analyzer attributed no slot-duty seconds to any bucket over {report.sample_count} "
        f"samples, so the run's slot-seconds are unaccounted for: {report.slot_duty_seconds}"
    )

    _assert_disaggregated_jobs_carry_phase_metrics(result, events)


def _assert_disaggregated_jobs_carry_phase_metrics(
    result: HarnessResult,
    events: list[dict[str, Any]],
) -> None:
    """Assert every disaggregated job's exported record carries the phase metrics its sampler measured.

    A disaggregated job emits its snapshots per lane and none for the whole job, so its exported record is
    the only place an offline reader can attribute its cost from; the sample stage's snapshot is what fills
    it. Skipped, with the reason stated, when the run served nothing through the pipeline: the contract is
    about what a disaggregated job carries, not about forcing one to exist.
    """
    assert result.metrics is not None
    sampled_job_ids = {
        record.job_id for record in result.metrics.stage_metrics if record.stage is PipelineStageTag.SAMPLE
    }
    if not sampled_job_ids:
        pytest.skip("no job reached the disaggregated sample stage in this run, so there is nothing to judge")

    exported = {event["job"]["job_id"]: event["job"] for event in events if event.get("event") == "job_completed"}
    checked = 0
    for job_id in sorted(sampled_job_ids):
        record = exported.get(job_id)
        if record is None:
            # The job was still in flight when the soak drained; it has no exported record either way.
            continue
        assert record.get("phase_metrics") is not None, (
            f"disaggregated job {job_id} was exported without phase metrics, so nothing in the stream "
            f"prices what its sampler did: {record}"
        )
        checked += 1
    assert checked > 0, (
        f"none of the {len(sampled_job_ids)} disaggregated job(s) reached the exported stream, so the "
        f"per-job half of the contract judged nothing"
    )


@pytest.mark.e2e
async def test_headless_snapshot_regression_leaves_the_stream_without_a_spine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reinjecting the no-supervisor early return strips the samples, and the contract says so by name.

    The negative control for the sample-count assertion above. The snapshot builder is what records the
    periodic sample, so a publisher that returns early whenever nothing is attached writes a stream with
    jobs in it and no spine, and every duty figure derived from it is silently empty.
    """
    monkeypatch.chdir(tmp_path)
    original = HordeWorkerProcessManager._publish_supervisor_snapshot

    def _publish_only_when_attached(manager: HordeWorkerProcessManager) -> None:
        if manager._supervisor is None:
            return
        original(manager)

    monkeypatch.setattr(
        HordeWorkerProcessManager,
        "_publish_supervisor_snapshot",
        _publish_only_when_attached,
    )

    result = await run_harness_async(_spine_harness_config(_NEGATIVE_CONTROL_SECONDS))
    assert not result.timed_out, result.failure_summary()

    _session_id, _paths, events = _session_events(tmp_path / ".horde_worker_regen" / "stats")
    samples = [event for event in events if event.get("event") == "stats_sample"]
    floor = _expected_sample_floor(result)

    with pytest.raises(AssertionError, match=r"recorded 0 stats_sample events"):
        assert len(samples) >= floor, (
            f"the headless run recorded {len(samples)} stats_sample events over "
            f"{result.elapsed_seconds:.1f}s, below the floor of {floor} at one per "
            f"{_SAMPLE_FLOOR_INTERVAL_SECONDS:.0f}s: without them the stream carries no duty spine"
        )
