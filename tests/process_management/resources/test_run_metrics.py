"""Unit tests for the WorkerRunMetrics aggregator (no processes, no GPU)."""

from __future__ import annotations

import time
from pathlib import Path

from horde_sdk.ai_horde_api import GENERATION_STATE
from hordelib.metrics import DownloadEvent, JobPhaseMetrics, ModelLoadEvent, SamplingStats
from pytest import MonkeyPatch

from horde_worker_regen.process_management.ipc.messages import (
    HordeDownloadMetricsMessage,
    HordeJobMetricsMessage,
)
from horde_worker_regen.process_management.ipc.supervisor_channel import StatsSample
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, TrackedJob
from horde_worker_regen.process_management.resources.run_metrics import WorkerRunMetrics
from horde_worker_regen.process_management.simulation._dummy_jobs import dummy_job_factory


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


def _finalize_job(metrics: WorkerRunMetrics, *, faulted: bool = False, n_iter: int = 1) -> str:
    """Finalize a synthetic tracked job and return its job id string."""
    job = dummy_job_factory("Deliberate")
    assert job.id_ is not None
    if n_iter != 1:
        job = job.model_copy(update={"payload": job.payload.model_copy(update={"n_iter": n_iter})})
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

    def test_alchemy_phase_metrics_held_until_form_recorded(self) -> None:
        """A child's alchemy phase metrics are held (not recorded alone); the coordinator records the form."""
        metrics = WorkerRunMetrics()
        metrics.on_job_metrics(_job_metrics_message("form-1", is_alchemy=True))

        # No job record yet: the form's full record (name + pop->submit timing) is recorded at submit.
        assert metrics.snapshot().jobs == []

    def test_record_alchemy_form_builds_record_and_rollup(self) -> None:
        """Recording a finished form yields a record (absorbing held phase metrics) and a by-form rollup."""
        metrics = WorkerRunMetrics()
        metrics.on_job_metrics(_job_metrics_message("form-1", is_alchemy=True))  # held phase metrics
        metrics.record_alchemy_form(
            form_id="form-1",
            form="RealESRGAN_x4plus",
            e2e_seconds=3.5,
            faulted=False,
            width=1024,
            height=768,
        )

        record = metrics.snapshot().jobs[0]
        assert record.is_alchemy
        assert record.model_name == "RealESRGAN_x4plus"
        assert record.e2e_seconds == 3.5
        assert record.faulted is False
        assert (record.width, record.height) == (1024, 768)
        assert record.phase_metrics is not None  # absorbed the held child metrics

        rollups = metrics.form_rollups()
        assert len(rollups) == 1
        assert rollups[0].model == "RealESRGAN_x4plus"
        assert rollups[0].jobs == 1
        assert rollups[0].e2e_seconds == 3.5

    def test_form_rollups_accumulate_per_form(self) -> None:
        """Multiple forms of the same name fold into one rollup row, so an average can be derived."""
        metrics = WorkerRunMetrics()
        metrics.record_alchemy_form(form_id="a", form="caption", e2e_seconds=2.0, faulted=False)
        metrics.record_alchemy_form(form_id="b", form="caption", e2e_seconds=4.0, faulted=True)

        rollups = {row.model: row for row in metrics.form_rollups()}
        assert rollups["caption"].jobs == 2
        assert rollups["caption"].e2e_seconds == 6.0


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
            job_slowdown_events=4,
            paging_victim_replacements=3,
            time_spent_no_jobs_available=3.5,
            disk_min_free_bytes={"C:/": 123},
        )
        assert snapshot.process_crash_events[0].process_id == 1
        assert snapshot.num_process_recoveries == 1
        assert snapshot.num_job_slowdowns == 2
        assert snapshot.job_slowdown_events == 4
        assert snapshot.paging_victim_replacements == 3
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


class TestStatsRollupsAndExport:
    """Worker-owned stats rollups and JSONL export."""

    def test_finalized_jobs_update_rollups_incrementally(self) -> None:
        """Model and baseline rollups include MPxsteps, sampling, E2E, and batch>1 job counts."""
        metrics = WorkerRunMetrics(baseline_resolver=lambda _model: "stable_diffusion_1")
        job_id = _finalize_job(metrics, n_iter=2)
        metrics.on_job_metrics(_job_metrics_message(job_id))

        model_rows = metrics.model_rollups()
        baseline_rows = metrics.baseline_rollups()

        assert model_rows[0].model == "Deliberate"
        assert model_rows[0].baseline == "stable_diffusion_1"
        assert model_rows[0].jobs == 1
        assert model_rows[0].batch_gt_one_jobs == 1
        assert model_rows[0].megapixelsteps == 512 * 512 / 1_000_000 * 30 * 2
        assert model_rows[0].e2e_seconds == 12.0
        assert baseline_rows[0].baseline == "stable_diffusion_1"

    def test_sampling_and_e2e_seconds_are_separate(self) -> None:
        """Sampling seconds come from child phase metrics while E2E comes from tracker timestamps."""
        metrics = WorkerRunMetrics()
        job = dummy_job_factory("Deliberate")
        assert job.id_ is not None
        metrics.on_job_metrics(_job_metrics_message(str(job.id_)))
        tracked = TrackedJob(
            job_id=job.id_,
            sdk_api_job_info=job,
            stage=JobStage.PENDING_SUBMIT,
            time_popped=100.0,
            stage_timestamps={"FINALIZED": 112.0},
        )
        metrics.on_job_finalized(
            tracked,
            HordeJobInfo(sdk_api_job_info=job, state=GENERATION_STATE.ok, time_popped=100.0),
        )

        row = metrics.model_rollups()[0]
        assert row.sampling_seconds == 6.0
        assert row.e2e_seconds == 12.0

    def test_alchemy_forms_roll_up_by_form_not_into_image_tables(self) -> None:
        """A recorded alchemy form is retained as a job and rolls up by form, never into image tables."""
        metrics = WorkerRunMetrics()
        metrics.record_alchemy_form(form_id="form-1", form="caption", e2e_seconds=1.0, faulted=False)

        assert metrics.snapshot().jobs[0].is_alchemy
        assert metrics.model_rollups() == []
        assert metrics.baseline_rollups() == []
        assert [row.model for row in metrics.form_rollups()] == ["caption"]

    def test_jsonl_export_writes_sample_and_job_events(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Export writes typed stats_sample and job_completed events under the session stats directory."""
        monkeypatch.chdir(tmp_path)
        metrics = WorkerRunMetrics()
        metrics.set_stats_export(True, worker_version="1.2.3")
        metrics.record_stats_sample(StatsSample(timestamp=10.0, jobs_submitted=1))
        _finalize_job(metrics)

        files = list((tmp_path / ".horde_worker_regen" / "stats").glob("stats-v1.2.3-*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text(encoding="utf-8").splitlines()
        assert '"event":"stats_sample"' in lines[0]
        assert '"event":"job_completed"' in lines[1]

    def test_jsonl_export_rotates_and_uses_versioned_filenames(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Rotation starts numbered files, and different worker versions naturally use different names."""
        import horde_worker_regen.process_management.resources.run_metrics as run_metrics

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(run_metrics, "_STATS_ROTATE_BYTES", 10)
        first = WorkerRunMetrics()
        first.set_stats_export(True, worker_version="1.0.0")
        first.record_stats_sample(StatsSample(timestamp=10.0, jobs_submitted=1))
        first.record_stats_sample(StatsSample(timestamp=11.0, jobs_submitted=2))

        second = WorkerRunMetrics()
        second.set_stats_export(True, worker_version="2.0.0")
        second.record_stats_sample(StatsSample(timestamp=10.0, jobs_submitted=1))

        names = sorted(path.name for path in (tmp_path / ".horde_worker_regen" / "stats").glob("*.jsonl"))
        assert any("stats-v1.0.0" in name and "-001.jsonl" in name for name in names)
        assert any("stats-v2.0.0" in name for name in names)

    def test_total_size_warning_triggers_over_threshold(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Stats export state warns once retained JSONL files exceed the configured threshold."""
        import horde_worker_regen.process_management.resources.run_metrics as run_metrics

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(run_metrics, "_STATS_WARNING_BYTES", 1)
        metrics = WorkerRunMetrics()
        metrics.set_stats_export(True, worker_version="1.0.0")
        metrics.record_stats_sample(StatsSample(timestamp=10.0, jobs_submitted=1))

        assert metrics.stats_export_state().warning_over_50_mib
