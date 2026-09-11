"""The live half of the log-line contract: parse a bridge log the worker itself just wrote.

``test_log_signatures.py`` pins each pattern to a literal sample, which catches a pattern edited out of
step with its sample but not a worker emit reworded out of step with both. This test closes that: it
runs the real process manager, scheduler, message dispatcher and submitter over fake children through
the dry-run harness, captures the parent's own log, and asserts that the lifecycle model recovers every
phase the run could reach for every job it generated, and that each registered pattern matched a line
the run actually produced.

Marked ``slow`` because the harness spawns real child processes for the protocol-faithful fakes.

Lines a dry run cannot produce carry a ``dry_run_reason`` in the registry and are exempted here; the
literal pin still covers them. Today those are the two fault lines (this scenario completes every job),
the submit success line (``dry_run_skip_api`` returns before it), the per-step sampling readout (the
fake children report no step progress), and four lines a short deterministic run reaches only through
a scheduling race (a cleared preload, a displaced-entry expiry, a line skip, and a post-processing
deferral, which needs sampling in progress on the lane's card at the moment its job finishes). Everything
else is exercised, including the multi-card plan line: the harness is given a synthetic two-card topology,
a post-processing job so the post-processing lane spawns, three models over two lanes so a lane replaces
its resident model, and ``unload_models_from_vram_often`` so a lane drops its model from VRAM between jobs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger

from horde_worker_regen import harness as harness_module
from horde_worker_regen.analysis.job_lifecycle import JobLifecycleModel, LaneRole, build_job_lifecycle
from horde_worker_regen.analysis.log_ingest import read_records
from horde_worker_regen.analysis.log_signatures import SIGNATURES, LogSignature
from horde_worker_regen.harness import HarnessConfig, run_harness_async
from horde_worker_regen.process_management.process_manager import SystemResources
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.simulation._canned_scenarios import make_canned_job

pytestmark = pytest.mark.slow

_MODEL = "Deliberate"
_SECOND_MODEL = "Anything Diffusion"
_THIRD_MODEL = "stable_diffusion"
_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"


def _two_card_resources() -> SystemResources:
    """A synthetic two-card host, so the worker emits its multi-card plan line and per-card intake."""
    return SystemResources(
        total_ram_bytes=32 * 1024**3,
        device_map=TorchDeviceMap(
            root={
                index: TorchDeviceInfo(
                    device_name=f"Contract card {index}",
                    device_index=index,
                    total_memory=12 * 1024**3,
                    kind="cuda",
                )
                for index in (0, 1)
            },
        ),
        per_process_overhead_mb=1000,
    )


def _scenario() -> list[ImageGenerateJobPopResponse]:
    """Three models rotating over two lanes plus a post-processing job, so every reachable line has a producer.

    Three models on two lanes force a lane to replace its resident model at least once however the scheduler
    routes the jobs, which is what makes the full-unload line reachable; two models would settle one per lane
    and never unload. The post-processing job is what makes the worker spawn its post-processing lane at all.
    """
    rotation = (_MODEL, _SECOND_MODEL, _THIRD_MODEL)
    jobs = [make_canned_job(rotation[index % 3], width=512, height=512, ddim_steps=8) for index in range(8)]
    jobs.insert(2, make_canned_job(_MODEL, width=512, height=512, ddim_steps=8, post_processing=["RealESRGAN_x4plus"]))
    return jobs


@pytest.fixture(scope="module")
def dry_run_bridge_log(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the worker through the dry-run harness and return the parent log it wrote.

    Module-scoped: the run is the expensive part, and every assertion below reads the same log.
    """
    log_path = tmp_path_factory.mktemp("log_contract") / "bridge.log"
    sink_ids: list[int] = []

    def _arm_contract_sink() -> None:
        """Stand in for the harness's own file sink so the run lands in this test's log.

        Replaced rather than added alongside: the harness arms its sink after configuring telemetry,
        which replaces every loguru handler, so a sink added before the run would be stripped.
        """
        sink_ids.append(
            logger.add(log_path, level="DEBUG", format=_LOG_FORMAT, enqueue=False, backtrace=False, diagnose=False),
        )

    original_sink = harness_module._arm_harness_log_sink
    harness_module._arm_harness_log_sink = _arm_contract_sink
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        config = HarnessConfig(
            scenario=_scenario(),
            process_mode="fake",
            skip_api=True,
            timeout_seconds=240.0,
            job_delay_seconds=2.0,
            system_resources=_two_card_resources(),
            bridge_data_overrides={
                "gpu_device_indices": [0, 1],
                "max_threads": 1,
                "queue_size": 2,
                # Forces a lane to drop its resident model after each job, which is what makes the
                # unload line reachable at all in a run this short.
                "unload_models_from_vram_often": True,
            },
        )
        result = asyncio.run(run_harness_async(config))
    finally:
        harness_module._arm_harness_log_sink = original_sink
        for sink_id in sink_ids:
            logger.remove(sink_id)

    assert result.num_jobs_completed > 0, f"the harness completed no jobs: {result.diagnostics}"
    return log_path


@pytest.fixture(scope="module")
def dry_run_messages(dry_run_bridge_log: Path) -> list[str]:
    """The captured log's record messages, which is the text the patterns are written against.

    Patterns anchored at the start of a message (the status lines and the queue listing) only match
    against the message, never against the whole file with loguru's timestamp/level/location prefix.
    """
    return [record.message for record in read_records(dry_run_bridge_log)]


@pytest.fixture(scope="module")
def dry_run_lifecycle(dry_run_bridge_log: Path) -> JobLifecycleModel:
    """The lifecycle model parsed out of the captured log."""
    return build_job_lifecycle(list(read_records(dry_run_bridge_log)))


_REACHABLE = [signature for signature in SIGNATURES.values() if signature.dry_run_reachable]


@pytest.mark.parametrize("signature", _REACHABLE, ids=lambda s: s.name)
def test_registered_pattern_matched_a_line_the_worker_just_wrote(
    signature: LogSignature,
    dry_run_messages: list[str],
) -> None:
    """Every pattern the dry run can exercise matched a line in the log that run produced.

    This is the test that goes red when an emitting f-string in ``process_management/`` is reworded: the
    pattern still matches its recorded sample, but no longer matches anything the worker writes.
    """
    assert any(signature.pattern.search(message) for message in dry_run_messages), (
        f"{signature.name} matched nothing the worker emitted; check {signature.emitter}"
    )


def test_every_completed_job_has_a_full_lifecycle(dry_run_lifecycle: JobLifecycleModel) -> None:
    """Each job the run generated is recovered with its pop, dispatch, lane, card and generation.

    The submit phase is the one a dry run cannot reach: ``dry_run_skip_api`` completes the submit
    without the API round trip and without the success line, so ``submitted_at`` stays None here and
    the post-inference segment is unmeasurable. Every earlier phase must be present for every job.
    """
    generated = [job for job in dry_run_lifecycle.jobs.values() if job.inference_finished_at is not None]
    assert len(generated) >= 8, f"the run generated too little to prove anything: {len(generated)} job(s)"
    for job in generated:
        assert job.popped_at is not None, f"{job.job_id} has no pop"
        assert job.dispatched_at is not None, f"{job.job_id} has no dispatch"
        assert job.model, f"{job.job_id} has no model"
        assert job.process_id is not None, f"{job.job_id} has no lane"
        assert job.device_index is not None, f"{job.job_id} has no card"
        assert job.pre_inference_seconds is not None
        assert job.inference_seconds is not None
    assert any(job.safety_seconds is not None for job in generated), "no job's safety check was recovered"


def test_the_process_to_device_map_is_populated(dry_run_lifecycle: JobLifecycleModel) -> None:
    """The lane placement map names a card for every inference lane the run spawned."""
    placements = dry_run_lifecycle.current_placements()
    inference_lanes = [p for p in placements.values() if p.role is LaneRole.INFERENCE]
    assert inference_lanes, "no inference lane spawn lines were parsed"
    assert all(placement.device_index is not None for placement in inference_lanes)
    assert {placement.device_index for placement in inference_lanes} == {0, 1}
    assert any(placement.role is LaneRole.SAFETY for placement in placements.values())


def test_every_dispatched_job_is_attributed_to_a_card(dry_run_lifecycle: JobLifecycleModel) -> None:
    """A job's card comes from its lane's placement, which the run must make resolvable."""
    dispatched = dry_run_lifecycle.dispatched_jobs
    assert dispatched
    assert all(job.device_index is not None for job in dispatched)


def test_the_worker_shape_is_recovered(dry_run_lifecycle: JobLifecycleModel) -> None:
    """The multi-card plan line yields the card count, the intake budget and each card's share."""
    assert dry_run_lifecycle.card_count == 2
    assert dry_run_lifecycle.intake_budget == 6
    assert set(dry_run_lifecycle.card_intake) == {0, 1}


def test_status_prints_and_the_parent_loop_are_visible(dry_run_lifecycle: JobLifecycleModel) -> None:
    """The status census and the IPC drain stream both parse out of a real run's log."""
    assert dry_run_lifecycle.status_snapshots, "no status prints were parsed"
    assert any(snapshot.inference_lanes for snapshot in dry_run_lifecycle.status_snapshots)
    assert dry_run_lifecycle.ipc_drain_profile().drains > 0
