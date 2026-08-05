"""Unit tests for the WorkerRunMetrics aggregator (no processes, no GPU)."""

from __future__ import annotations

import time
from pathlib import Path

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry
from hordelib.metrics import DownloadEvent, JobPhaseMetrics, ModelLoadEvent, SamplingStats
from pytest import MonkeyPatch

from horde_worker_regen.process_management.ipc.messages import (
    HordeDownloadMetricsMessage,
    HordeJobMetricsMessage,
    PipelineStageTag,
)
from horde_worker_regen.process_management.ipc.supervisor_channel import StatsSample
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, TrackedJob
from horde_worker_regen.process_management.resources.run_metrics import (
    DecisionKind,
    DecisionVerdict,
    JobMetricsRecord,
    ResourceStateKind,
    WorkerRunMetrics,
)
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


def _finalize_job(
    metrics: WorkerRunMetrics,
    *,
    faulted: bool = False,
    n_iter: int = 1,
    kudos_reward: float | None = None,
    payload_overrides: dict[str, object] | None = None,
    loras: list[LorasPayloadEntry] | None = None,
    aux_models_prepared_at: float | None = None,
    queue_depth_at_dispatch: int | None = None,
    post_processing_depth_at_dispatch: int | None = None,
    served_whole_card: bool | None = None,
    serving_process_age_seconds: float | None = None,
) -> str:
    """Finalize a synthetic tracked job and return its job id string."""
    job = dummy_job_factory("Deliberate")
    assert job.id_ is not None
    job_id = job.id_
    payload_updates: dict[str, object] = {}
    if n_iter != 1:
        payload_updates["n_iter"] = n_iter
    if loras is not None:
        payload_updates["loras"] = loras
    if payload_overrides:
        payload_updates.update(payload_overrides)
    if payload_updates:
        job = job.model_copy(update={"payload": job.payload.model_copy(update=payload_updates)})
    tracked = TrackedJob(
        job_id=job_id,
        sdk_api_job_info=job,
        stage=JobStage.PENDING_SUBMIT,
        time_popped=100.0,
        kudos_reward=kudos_reward,
        stage_timestamps={
            "PENDING_INFERENCE": 100.0,
            "INFERENCE_IN_PROGRESS": 102.5,
            "PENDING_SAFETY_CHECK": 110.0,
            "PENDING_SUBMIT": 111.0,
            "FINALIZED": 112.0,
        },
        aux_models_prepared_at=aux_models_prepared_at,
        queue_depth_at_dispatch=queue_depth_at_dispatch,
        post_processing_depth_at_dispatch=post_processing_depth_at_dispatch,
        served_whole_card=served_whole_card,
        serving_process_age_seconds=serving_process_age_seconds,
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

    def test_submit_reward_reaches_the_job_record(self) -> None:
        """The reward the tracker holds at finalize is what the job's metrics record carries.

        The horde only states a figure in the submit response, so a job's earnings are unknowable until
        its submits land; the record has to take the tracker's recorded reward rather than derive one.
        """
        metrics = WorkerRunMetrics()
        _finalize_job(metrics, kudos_reward=17.5)
        assert metrics.snapshot().jobs[0].kudos_reward == 17.5

    def test_unrewarded_job_records_no_kudos(self) -> None:
        """A job the horde paid nothing for records an unknown reward, never a zero."""
        metrics = WorkerRunMetrics()
        _finalize_job(metrics, faulted=True)
        assert metrics.snapshot().jobs[0].kudos_reward is None

    def test_pricing_features_taken_from_the_pop_payload(self) -> None:
        """The sampler, schedule, and cfg the horde priced are what the record carries.

        These complete the request side of a payload-to-measured-seconds pair, so they must be the
        popped values rather than defaults.
        """
        metrics = WorkerRunMetrics()
        _finalize_job(
            metrics,
            payload_overrides={"sampler_name": "k_dpmpp_2m", "scheduler": "exponential", "cfg_scale": 4.5},
        )

        record = metrics.snapshot().jobs[0]
        assert record.sampler_name == "k_dpmpp_2m"
        assert record.scheduler == "exponential"
        assert record.cfg_scale == 4.5

    def test_scheduler_derived_from_karras_when_unstated(self) -> None:
        """Without a named schedule, the legacy karras bool decides it, where false means normal."""
        metrics = WorkerRunMetrics()
        _finalize_job(metrics, payload_overrides={"scheduler": None, "karras": True})
        _finalize_job(metrics, payload_overrides={"scheduler": None, "karras": False})

        records = metrics.snapshot().jobs
        assert records[0].scheduler == "karras"
        assert records[1].scheduler == "normal"

    def test_alchemy_form_has_no_pricing_features(self) -> None:
        """An alchemy form has no image pop payload, so its record leaves the three fields unknown."""
        metrics = WorkerRunMetrics()
        metrics.record_alchemy_form(form_id="f", form="caption", e2e_seconds=1.0, faulted=False)

        record = metrics.snapshot().jobs[0]
        assert record.sampler_name is None
        assert record.scheduler is None
        assert record.cfg_scale is None

    def test_record_parses_without_the_pricing_features(self) -> None:
        """A record written before these fields existed still parses, with the fields unknown."""
        record = JobMetricsRecord.model_validate({"job_id": "old", "model_name": "Deliberate"})
        assert record.sampler_name is None
        assert record.scheduler is None
        assert record.cfg_scale is None

    def test_alchemy_form_carries_its_reward(self) -> None:
        """An alchemy form's record carries the reward its own submit response paid."""
        metrics = WorkerRunMetrics()
        metrics.record_alchemy_form(form_id="f", form="caption", e2e_seconds=1.0, faulted=False, kudos_reward=3.0)
        metrics.record_alchemy_form(form_id="g", form="caption", e2e_seconds=1.0, faulted=True)

        records = {record.job_id: record for record in metrics.snapshot().jobs}
        assert records["f"].kudos_reward == 3.0
        assert records["g"].kudos_reward is None

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


class TestWorkerConditionFields:
    """The per-job conditions a cost analysis controls for: load, auxiliary wait, contention, residency."""

    def test_model_load_seconds_counts_only_this_jobs_model(self) -> None:
        """A job is charged the loads of its own checkpoint, never a neighbouring model's staging."""
        metrics = WorkerRunMetrics()
        job = dummy_job_factory("Deliberate")
        assert job.id_ is not None
        message = _job_metrics_message(str(job.id_))
        message.phase_metrics.model_loads.append(
            ModelLoadEvent(model_name="AlbedoBase XL", phase="disk_to_ram", duration_seconds=9.9, timestamp=1.0),
        )
        metrics.on_job_metrics(message)
        tracked = TrackedJob(
            job_id=job.id_,
            sdk_api_job_info=job,
            stage=JobStage.PENDING_SUBMIT,
            time_popped=100.0,
            stage_timestamps={"FINALIZED": 110.0},
        )
        metrics.on_job_finalized(
            tracked,
            HordeJobInfo(sdk_api_job_info=job, state=GENERATION_STATE.ok, time_popped=100.0),
        )

        assert metrics.snapshot().jobs[0].model_load_seconds == 4.2 + 1.1

    def test_resident_model_records_a_measured_zero_load(self) -> None:
        """A job that loaded nothing reads 0.0, which is distinct from the unknown of an uncorrelated job."""
        metrics = WorkerRunMetrics()
        job = dummy_job_factory("Deliberate")
        assert job.id_ is not None
        message = _job_metrics_message(str(job.id_))
        message.phase_metrics.model_loads.clear()
        metrics.on_job_metrics(message)
        tracked = TrackedJob(
            job_id=job.id_,
            sdk_api_job_info=job,
            stage=JobStage.PENDING_SUBMIT,
            time_popped=100.0,
            stage_timestamps={"FINALIZED": 110.0},
        )
        metrics.on_job_finalized(
            tracked,
            HordeJobInfo(sdk_api_job_info=job, state=GENERATION_STATE.ok, time_popped=100.0),
        )

        assert metrics.snapshot().jobs[0].model_load_seconds == 0.0

    def test_uncorrelated_job_leaves_the_load_cost_unknown(self) -> None:
        """Without child phase metrics there is nothing to charge, so the field stays absent."""
        metrics = WorkerRunMetrics()
        _finalize_job(metrics)
        assert metrics.snapshot().jobs[0].model_load_seconds is None

    def test_lora_wait_is_the_blocked_share_of_the_queue_wait(self) -> None:
        """A LoRA job's wait runs from the pop to auxiliary readiness, not to dispatch."""
        metrics = WorkerRunMetrics()
        _finalize_job(metrics, loras=[LorasPayloadEntry(name="detail")], aux_models_prepared_at=101.5)

        record = metrics.snapshot().jobs[0]
        assert record.queue_wait_seconds == 2.5
        assert record.lora_wait_seconds == 1.5

    def test_lora_wait_is_clamped_into_the_queue_wait(self) -> None:
        """Readiness stamped after dispatch cannot have delayed a dispatch that already happened."""
        metrics = WorkerRunMetrics()
        _finalize_job(metrics, loras=[LorasPayloadEntry(name="detail")], aux_models_prepared_at=108.0)
        assert metrics.snapshot().jobs[0].lora_wait_seconds == 2.5

    def test_job_without_loras_records_no_lora_wait(self) -> None:
        """A job with no LoRAs has no auxiliary wait to measure, so the field is absent rather than zero."""
        metrics = WorkerRunMetrics()
        _finalize_job(metrics, aux_models_prepared_at=101.0)
        assert metrics.snapshot().jobs[0].lora_wait_seconds is None

    def test_dispatch_conditions_reach_the_record(self) -> None:
        """Contention, whole-card residency, and process age are carried from the tracker as stamped."""
        metrics = WorkerRunMetrics()
        _finalize_job(
            metrics,
            queue_depth_at_dispatch=3,
            post_processing_depth_at_dispatch=1,
            served_whole_card=True,
            serving_process_age_seconds=42.5,
        )

        record = metrics.snapshot().jobs[0]
        assert record.queue_depth_at_dispatch == 3
        assert record.post_processing_depth_at_dispatch == 1
        assert record.whole_card is True
        assert record.process_age_seconds == 42.5

    def test_undispatched_job_leaves_the_conditions_unknown(self) -> None:
        """A job that never reached dispatch reports no conditions rather than a fabricated zero."""
        metrics = WorkerRunMetrics()
        _finalize_job(metrics)

        record = metrics.snapshot().jobs[0]
        assert record.queue_depth_at_dispatch is None
        assert record.post_processing_depth_at_dispatch is None
        assert record.whole_card is None
        assert record.process_age_seconds is None

    def test_record_parses_without_the_worker_condition_fields(self) -> None:
        """A record written before these fields existed still parses, with every one unknown."""
        record = JobMetricsRecord.model_validate({"job_id": "old", "model_name": "Deliberate"})
        assert record.model_load_seconds is None
        assert record.lora_wait_seconds is None
        assert record.queue_depth_at_dispatch is None
        assert record.post_processing_depth_at_dispatch is None
        assert record.whole_card is None
        assert record.process_age_seconds is None


class TestStageMetrics:
    """Stage-tagged lane metrics are retained per stage without disturbing the whole-job records."""

    def _stage_message(self, job_id: str, stage: PipelineStageTag) -> HordeJobMetricsMessage:
        return HordeJobMetricsMessage(
            process_id=3,
            process_launch_identifier=0,
            info="stage",
            job_id=job_id,
            stage=stage,
            phase_metrics=_make_phase_metrics(),
        )

    def test_stage_messages_are_retained_per_stage(self) -> None:
        """Each disaggregated stage of one job survives as its own stage record with the disk->RAM event."""
        metrics = WorkerRunMetrics()
        metrics.on_job_metrics(self._stage_message("job-1", PipelineStageTag.TEXT_ENCODE))
        metrics.on_job_metrics(self._stage_message("job-1", PipelineStageTag.VAE_ENCODE))
        metrics.on_job_metrics(self._stage_message("job-1", PipelineStageTag.VAE_DECODE))

        stage_records = metrics.snapshot().stage_metrics
        assert [record.stage for record in stage_records] == [
            PipelineStageTag.TEXT_ENCODE,
            PipelineStageTag.VAE_ENCODE,
            PipelineStageTag.VAE_DECODE,
        ]
        assert all(record.job_id == "job-1" for record in stage_records)
        disk_to_ram_per_stage = [
            sum(1 for load in record.phase_metrics.model_loads if load.phase == "disk_to_ram")
            for record in stage_records
            if record.phase_metrics is not None
        ]
        assert disk_to_ram_per_stage == [1, 1, 1]

    def test_stage_messages_do_not_enter_jobs_or_rollups(self) -> None:
        """Stage records stay out of the finalized-job list and the model rollups (existing consumers)."""
        metrics = WorkerRunMetrics()
        metrics.on_job_metrics(self._stage_message("job-1", PipelineStageTag.VAE_DECODE))

        snapshot = metrics.snapshot()
        assert snapshot.jobs == []
        assert metrics.model_rollups() == []
        assert metrics.baseline_rollups() == []

    def test_untagged_message_still_correlates_at_finalize_with_no_stage(self) -> None:
        """A whole-job (untagged) message keeps its historical correlation and lands with stage=None."""
        metrics = WorkerRunMetrics()
        job = dummy_job_factory("Deliberate")
        assert job.id_ is not None
        metrics.on_job_metrics(_job_metrics_message(str(job.id_)))  # untagged
        _finalize_job_from(metrics, job)

        snapshot = metrics.snapshot()
        assert snapshot.stage_metrics == []
        assert len(snapshot.jobs) == 1
        assert snapshot.jobs[0].stage is None
        assert snapshot.jobs[0].phase_metrics is not None

    def test_reset_clears_stage_metrics(self) -> None:
        """A benchmark-level reset drops retained stage records alongside the other aggregates."""
        metrics = WorkerRunMetrics()
        metrics.on_job_metrics(self._stage_message("job-1", PipelineStageTag.TEXT_ENCODE))
        metrics.reset()
        assert metrics.snapshot().stage_metrics == []


def _finalize_job_from(metrics: WorkerRunMetrics, job: object) -> None:
    """Finalize a specific job object (so a held phase-metrics correlation can be asserted)."""
    tracked = TrackedJob(
        job_id=job.id_,  # type: ignore[attr-defined]
        sdk_api_job_info=job,  # type: ignore[arg-type]
        stage=JobStage.PENDING_SUBMIT,
        time_popped=100.0,
        stage_timestamps={"FINALIZED": 110.0},
    )
    metrics.on_job_finalized(
        tracked,
        HordeJobInfo(sdk_api_job_info=job, state=GENERATION_STATE.ok, time_popped=100.0),  # type: ignore[arg-type]
    )


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


def _read_export_events(tmp_path: Path) -> list[dict[str, object]]:
    """Return every JSONL export event across the session's stats files, oldest-first."""
    import json

    events: list[dict[str, object]] = []
    for path in sorted((tmp_path / ".horde_worker_regen" / "stats").glob("stats-v*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    return events


def _decisions(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [event for event in events if event.get("event") == "decision"]


class TestSessionAndResourceEvents:
    """Session boundary and resource-state export events (opt-in stats JSONL)."""

    def test_session_start_carries_config_snapshot(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """A session_start event serializes with its discriminator and the flat config snapshot."""
        monkeypatch.chdir(tmp_path)
        metrics = WorkerRunMetrics()
        metrics.set_stats_export(True, worker_version="9.9.9")
        metrics.record_session_start(config={"max_power": 32, "high_performance_mode": True}, timestamp=1.0)

        events = _read_export_events(tmp_path)
        assert len(events) == 1
        assert events[0]["event"] == "session_start"
        assert events[0]["worker_version"] == "9.9.9"
        assert events[0]["config"] == {"max_power": 32, "high_performance_mode": True}

    def test_session_end_reports_duration_and_totals(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """session_end derives duration from the recorded start and carries terminal totals."""
        monkeypatch.chdir(tmp_path)
        metrics = WorkerRunMetrics()
        metrics.set_stats_export(True, worker_version="1.0.0")
        metrics.record_session_start(config={}, timestamp=100.0)
        metrics.record_session_end(reason="graceful_shutdown", jobs_submitted=7, jobs_faulted=1, timestamp=160.0)

        end_events = [event for event in _read_export_events(tmp_path) if event["event"] == "session_end"]
        assert len(end_events) == 1
        assert end_events[0]["duration_seconds"] == 60.0
        assert end_events[0]["reason"] == "graceful_shutdown"
        assert end_events[0]["jobs_submitted"] == 7
        assert end_events[0]["jobs_faulted"] == 1

    def test_session_end_counters_partition_the_emitted_job_records(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Without explicit totals, the counters sum to the job_completed records in the same file."""
        monkeypatch.chdir(tmp_path)
        metrics = WorkerRunMetrics()
        metrics.set_stats_export(True, worker_version="1.0.0")
        metrics.record_session_start(config={}, timestamp=100.0)
        _finalize_job(metrics)
        _finalize_job(metrics)
        _finalize_job(metrics, faulted=True)
        metrics.record_alchemy_form(form_id="form-1", form="caption", e2e_seconds=1.0, faulted=False)
        metrics.record_session_end(reason="graceful_shutdown", timestamp=160.0)

        events = _read_export_events(tmp_path)
        completed = [event for event in events if event["event"] == "job_completed"]
        end_event = events[-1]
        assert end_event["event"] == "session_end"
        assert end_event["jobs_submitted"] == 3
        assert end_event["jobs_faulted"] == 1
        assert end_event["jobs_submitted"] + end_event["jobs_faulted"] == len(completed)

    def test_session_end_counts_only_records_the_stream_holds(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Jobs finalized before the export was switched on are not counted: they are not in the file."""
        monkeypatch.chdir(tmp_path)
        metrics = WorkerRunMetrics()
        _finalize_job(metrics)
        metrics.set_stats_export(True, worker_version="1.0.0")
        _finalize_job(metrics)
        metrics.record_session_end(reason="graceful_shutdown", timestamp=160.0)

        events = _read_export_events(tmp_path)
        completed = [event for event in events if event["event"] == "job_completed"]
        assert len(completed) == 1
        assert events[-1]["jobs_submitted"] == 1
        assert events[-1]["jobs_faulted"] == 0

    def test_a_benchmark_level_reset_keeps_the_stream_totals(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """A level boundary clears the aggregates but not the tally of what the file already holds."""
        monkeypatch.chdir(tmp_path)
        metrics = WorkerRunMetrics()
        metrics.set_stats_export(True, worker_version="1.0.0")
        _finalize_job(metrics)
        metrics.reset()
        _finalize_job(metrics)
        metrics.record_session_end(reason="graceful_shutdown", timestamp=160.0)

        events = _read_export_events(tmp_path)
        completed = [event for event in events if event["event"] == "job_completed"]
        assert events[-1]["jobs_submitted"] == len(completed) == 2

    def test_resource_state_event_serializes(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """A resource_state transition serializes with its discriminator and flat inputs."""
        monkeypatch.chdir(tmp_path)
        metrics = WorkerRunMetrics()
        metrics.set_stats_export(True, worker_version="1.0.0")
        event = metrics.record_resource_state(
            state_kind=ResourceStateKind.GOVERNOR,
            state="saturated",
            device_index=0,
            reason="device_free_governor_transition",
            inputs={"device_free_mb": 512.0},
            timestamp=5.0,
        )

        assert event is not None
        events = [item for item in _read_export_events(tmp_path) if item["event"] == "resource_state"]
        assert len(events) == 1
        assert events[0]["state_kind"] == "governor"
        assert events[0]["state"] == "saturated"
        assert events[0]["inputs"] == {"device_free_mb": 512.0}

    def test_no_events_written_when_export_disabled(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """With export off, decision/resource/session recorders write nothing and open no file."""
        monkeypatch.chdir(tmp_path)
        metrics = WorkerRunMetrics()
        assert (
            metrics.record_decision(
                decision_kind=DecisionKind.PP_DEFERRAL,
                subject="job-1",
                verdict=DecisionVerdict.DEFER,
                timestamp=1.0,
            )
            is None
        )
        metrics.record_resource_state(state_kind=ResourceStateKind.WDDM_PAGING, state="active", timestamp=1.0)
        metrics.record_session_start(config={}, timestamp=1.0)
        assert not (tmp_path / ".horde_worker_regen" / "stats").exists()


class TestDecisionCoalescing:
    """The record_decision chokepoint enforces the >1 Hz no-verbatim-repeat rule."""

    def test_sustained_identical_decision_emits_once(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Many identical sub-second evaluations collapse to a single opening DecisionEvent."""
        monkeypatch.chdir(tmp_path)
        metrics = WorkerRunMetrics()
        metrics.set_stats_export(True, worker_version="1.0.0")

        for tick in range(50):
            metrics.record_decision(
                decision_kind=DecisionKind.PP_DEFERRAL,
                subject="job-1",
                verdict=DecisionVerdict.DEFER,
                reason="pp_does_not_fit",
                inputs={"available_mb": 100.0},
                timestamp=10.0 + tick * 0.1,  # 50 ticks across 5 seconds, all under the heartbeat interval
            )

        decisions = _decisions(_read_export_events(tmp_path))
        assert len(decisions) == 1
        assert decisions[0]["verdict"] == "defer"
        assert decisions[0]["repeat_count"] == 0
        assert decisions[0]["resolved"] is False

    def test_heartbeat_after_interval_carries_repeat_count(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Once the heartbeat interval elapses, one further record carries the accumulated repeat count."""
        import horde_worker_regen.process_management.resources.run_metrics as run_metrics

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(run_metrics, "_DECISION_HEARTBEAT_SECONDS", 30.0)
        metrics = WorkerRunMetrics()
        metrics.set_stats_export(True, worker_version="1.0.0")

        # Opening record at t=0, then sub-heartbeat repeats, then one call past the interval.
        for tick in range(31):
            metrics.record_decision(
                decision_kind=DecisionKind.PP_DEFERRAL,
                subject="job-1",
                verdict=DecisionVerdict.DEFER,
                timestamp=float(tick),
            )

        decisions = _decisions(_read_export_events(tmp_path))
        assert len(decisions) == 2  # one opening + one heartbeat
        heartbeat = decisions[1]
        assert heartbeat["repeat_count"] == 30  # the 30 suppressed ticks between t=1 and t=30
        assert heartbeat["resolved"] is False

    def test_signature_change_emits_distinct_record(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """A change of verdict or reason is a fresh transition, not a coalesced repeat."""
        monkeypatch.chdir(tmp_path)
        metrics = WorkerRunMetrics()
        metrics.set_stats_export(True, worker_version="1.0.0")

        metrics.record_decision(
            decision_kind=DecisionKind.INFERENCE_DISPATCH,
            subject="job-1",
            verdict=DecisionVerdict.DEFER,
            reason="reason_a",
            timestamp=1.0,
        )
        metrics.record_decision(
            decision_kind=DecisionKind.INFERENCE_DISPATCH,
            subject="job-1",
            verdict=DecisionVerdict.DENY,
            reason="reason_b",
            timestamp=1.5,
        )

        decisions = _decisions(_read_export_events(tmp_path))
        assert [item["verdict"] for item in decisions] == ["defer", "deny"]
        assert all(item["resolved"] is False for item in decisions)

    def test_resolving_verdict_closes_open_decision(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """A resolving verdict for a tracked subject emits one final resolved record."""
        monkeypatch.chdir(tmp_path)
        metrics = WorkerRunMetrics()
        metrics.set_stats_export(True, worker_version="1.0.0")

        metrics.record_decision(
            decision_kind=DecisionKind.PP_DEFERRAL,
            subject="job-1",
            verdict=DecisionVerdict.DEFER,
            timestamp=1.0,
        )
        metrics.record_decision(
            decision_kind=DecisionKind.PP_DEFERRAL,
            subject="job-1",
            verdict=DecisionVerdict.NO_OP,
            reason="left_pending_queue",
            timestamp=2.0,
        )

        decisions = _decisions(_read_export_events(tmp_path))
        assert len(decisions) == 2
        assert decisions[0]["resolved"] is False
        assert decisions[1]["resolved"] is True

    def test_resolving_verdict_for_untracked_subject_is_dropped(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """A resolving verdict for a subject that was never deferred writes nothing (no admit-only lines)."""
        monkeypatch.chdir(tmp_path)
        metrics = WorkerRunMetrics()
        metrics.set_stats_export(True, worker_version="1.0.0")

        assert (
            metrics.record_decision(
                decision_kind=DecisionKind.PP_DEFERRAL,
                subject="job-1",
                verdict=DecisionVerdict.NO_OP,
                timestamp=1.0,
            )
            is None
        )
        assert _decisions(_read_export_events(tmp_path)) == []
