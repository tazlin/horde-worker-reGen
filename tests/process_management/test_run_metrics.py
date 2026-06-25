"""Unit tests for the WorkerRunMetrics aggregator (no processes, no GPU)."""

from __future__ import annotations

import time

from horde_sdk.ai_horde_api import GENERATION_STATE
from hordelib.metrics import DownloadEvent, JobPhaseMetrics, ModelLoadEvent, SamplingStats

from horde_worker_regen.process_management.ipc.messages import (
    HordeDownloadMetricsMessage,
    HordeJobMetricsMessage,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, TrackedJob
from horde_worker_regen.process_management.resources.run_metrics import WorkerRunMetrics
from horde_worker_regen.process_management.testing._dummy_jobs import dummy_job_factory


def _make_phase_metrics(*, vram_high_water: int | None = 5000) -> JobPhaseMetrics:
    return JobPhaseMetrics(
        model_loads=[
            ModelLoadEvent(model_name="Deliberate", phase="disk_to_ram", duration_seconds=4.2, timestamp=time.time()),
            ModelLoadEvent(model_name="Deliberate", phase="ram_to_vram", duration_seconds=1.1, timestamp=time.time()),
        ],
        sampling=SamplingStats(
            steps_completed=30,
            total_steps=30,
            duration_seconds=6.0,
            iterations_per_second=5.0,
        ),
        vram_used_high_water_mb=vram_high_water,
        ram_used_high_water_mb=9000,
    )


def _job_metrics_message(job_id: str, *, process_id: int = 0, is_alchemy: bool = False) -> HordeJobMetricsMessage:
    return HordeJobMetricsMessage(
        process_id=process_id,
        process_launch_identifier=0,
        info="test",
        job_id=job_id,
        is_alchemy=is_alchemy,
        phase_metrics=_make_phase_metrics(),
    )


def _finalize_job(metrics: WorkerRunMetrics, *, faulted: bool = False) -> str:
    """Finalize a synthetic tracked job and return its job id string."""
    job = dummy_job_factory("Deliberate")
    assert job.id_ is not None
    tracked = TrackedJob(
        job_id=job.id_,
        sdk_api_job_info=job,
        stage=JobStage.PENDING_SUBMIT,
        time_popped=100.0,
        stage_timestamps={
            "PENDING_INFERENCE": 100.0,
            "INFERENCE_IN_PROGRESS": 102.5,
            "PENDING_SAFETY_CHECK": 110.0,
            "PENDING_SUBMIT": 111.0,
            "FINALIZED": 112.0,
        },
    )
    job_info = HordeJobInfo(
        sdk_api_job_info=job,
        state=GENERATION_STATE.faulted if faulted else GENERATION_STATE.ok,
        time_popped=100.0,
    )
    metrics.on_job_finalized(tracked, job_info)
    return str(job.id_)


class TestJobCorrelation:
    """Correlation of child-reported phase metrics with finalized jobs."""

    def test_image_job_metrics_correlated_at_finalize(self) -> None:
        """A child report keyed by str(generation id) must end up on the finalized record."""
        metrics = WorkerRunMetrics()
        job = dummy_job_factory("Deliberate")
        assert job.id_ is not None
        metrics.on_job_metrics(_job_metrics_message(str(job.id_)))
        tracked = TrackedJob(
            job_id=job.id_,
            sdk_api_job_info=job,
            stage=JobStage.PENDING_SUBMIT,
            time_popped=100.0,
            stage_timestamps={"FINALIZED": 110.0},
        )
        job_info = HordeJobInfo(sdk_api_job_info=job, state=GENERATION_STATE.ok, time_popped=100.0)
        metrics.on_job_finalized(tracked, job_info)

        snapshot = metrics.snapshot()
        assert len(snapshot.jobs) == 1
        record = snapshot.jobs[0]
        assert record.job_id == str(job.id_)
        assert not record.is_alchemy
        assert record.phase_metrics is not None
        assert record.phase_metrics.sampling is not None
        assert record.phase_metrics.sampling.iterations_per_second == 5.0

    def test_stage_latencies_derived(self) -> None:
        """Queue-wait, e2e, and safety latencies derive from the stage timestamps."""
        metrics = WorkerRunMetrics()
        _finalize_job(metrics)

        record = metrics.snapshot().jobs[0]
        assert record.queue_wait_seconds == 2.5
        assert record.e2e_seconds == 12.0
        assert record.safety_seconds == 1.0

    def test_faulted_job_flagged(self) -> None:
        """A job finalized in the faulted state is marked faulted on its record."""
        metrics = WorkerRunMetrics()
        _finalize_job(metrics, faulted=True)
        assert metrics.snapshot().jobs[0].faulted

    def test_alchemy_metrics_recorded_immediately(self) -> None:
        """Alchemy forms never finalize through the tracker, so they record on arrival."""
        metrics = WorkerRunMetrics()
        metrics.on_job_metrics(_job_metrics_message("form-1", is_alchemy=True))

        snapshot = metrics.snapshot()
        assert len(snapshot.jobs) == 1
        assert snapshot.jobs[0].is_alchemy
        assert snapshot.jobs[0].phase_metrics is not None


class TestAggregates:
    """Aggregation of downloads, high-water marks, crash events, and counters."""

    def test_vram_high_water_per_process_keeps_max(self) -> None:
        """Per-process VRAM high-water keeps the maximum across reports."""
        metrics = WorkerRunMetrics()
        metrics.on_job_metrics(_job_metrics_message("a", process_id=0, is_alchemy=True))
        message_higher = _job_metrics_message("b", process_id=0, is_alchemy=True)
        message_higher.phase_metrics.vram_used_high_water_mb = 7777
        metrics.on_job_metrics(message_higher)

        snapshot = metrics.snapshot()
        assert snapshot.vram_used_high_water_mb_per_process[0] == 7777

    def test_download_events_accumulate(self) -> None:
        """Download events from child messages accumulate into the snapshot."""
        metrics = WorkerRunMetrics()
        metrics.on_download_metrics(
            HordeDownloadMetricsMessage(
                process_id=0,
                process_launch_identifier=0,
                info="test",
                events=[
                    DownloadEvent(
                        name="some lora",
                        category="lora",
                        size_bytes=100,
                        duration_seconds=1.0,
                        megabytes_per_second=0.0001,
                        retries=0,
                        success=True,
                        timestamp=time.time(),
                    ),
                ],
            ),
        )
        assert len(metrics.snapshot().downloads) == 1

    def test_crash_events_and_counters(self) -> None:
        """Crash events and caller-provided counters appear in the snapshot."""
        metrics = WorkerRunMetrics()
        metrics.record_process_crash(
            process_id=1,
            process_launch_identifier=2,
            last_state="INFERENCE_STARTING",
            reason="inference process replaced (crashed or hung)",
        )
        snapshot = metrics.snapshot(
            num_process_recoveries=1,
            num_job_slowdowns=2,
            time_spent_no_jobs_available=3.5,
            disk_min_free_bytes={"C:/": 123},
        )
        assert snapshot.process_crash_events[0].process_id == 1
        assert snapshot.num_process_recoveries == 1
        assert snapshot.num_job_slowdowns == 2
        assert snapshot.time_spent_no_jobs_available == 3.5
        assert snapshot.disk_min_free_bytes == {"C:/": 123}

    def test_churn_events_recorded_with_timestamps(self) -> None:
        """Each churn kind records a timestamp the snapshot exposes for per-window counting."""
        metrics = WorkerRunMetrics()
        before = time.time()
        metrics.record_churn("model_swap")
        metrics.record_churn("model_swap")
        metrics.record_churn("vram_eviction")

        churn = metrics.snapshot().churn_event_times
        assert len(churn["model_swap"]) == 2
        assert len(churn["vram_eviction"]) == 1
        assert churn["process_cycle"] == []
        assert all(stamp >= before for stamp in churn["model_swap"])

    def test_reset_clears_churn_events(self) -> None:
        """A benchmark-level reset clears churn history alongside the other aggregates."""
        metrics = WorkerRunMetrics()
        metrics.record_churn("process_cycle")
        metrics.reset()
        assert metrics.snapshot().churn_event_times["process_cycle"] == []
