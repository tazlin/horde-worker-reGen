"""CPU-only integration of bounded assessment policy with actual local HTTP submissions."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import aiohttp
import pytest
from hordelib.metrics import JobPhaseMetrics, SamplingStats
from ruamel.yaml import YAML

from horde_worker_regen.benchmark.assessment import (
    OperatorAssessment,
    OrdinaryAssessmentWorker,
    assessment_config,
    lane_equivalent_throughput,
    observed_sampling_concurrency,
    phase_wall_throughput,
    reference_sampling_rates,
    unavailable_feature,
    wall_noise_margin,
)
from horde_worker_regen.benchmark.assessment_policy import AssessmentCase
from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityKind
from horde_worker_regen.benchmark.capabilities.catalog import CatalogOptions
from horde_worker_regen.benchmark.capabilities.report_render import render_markdown
from horde_worker_regen.benchmark.capabilities.result import (
    CapabilityReport,
    EvidenceScope,
    EvidenceStatus,
    OperatingProfile,
    ReferenceRateBasis,
    SuggestedBridgeData,
)
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.local_horde import IssuedWork, LocalHorde
from horde_worker_regen.process_management.ipc.messages import SamplingWindow
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CapacityChangeResult,
    CardSnapshot,
    FeatureReadinessSummary,
    ProcessSnapshot,
    SupervisorCommand,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.process_management.models.feature_readiness import (
    FeatureReadiness,
    FeatureReadinessState,
    GatedFeature,
)
from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord, StatsJobCompletedEvent
from horde_worker_regen.process_management.scheduling.clearance_lease import CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind


def _sampling_stats(rate: float, seconds: float) -> SamplingStats:
    steps = round(rate * seconds)
    return SamplingStats(
        steps_completed=steps, total_steps=steps, duration_seconds=seconds, iterations_per_second=rate
    )


def _rated_record(
    identifier: str, model_name: str, start: float, end: float, rate: float, device: int = 0
) -> JobMetricsRecord:
    return JobMetricsRecord(
        job_id=identifier,
        model_name=model_name,
        device_index=device,
        sampling_window=SamplingWindow(first_step=start, last_step=end),
        sampling_seconds=end - start,
        phase_metrics=JobPhaseMetrics(sampling=_sampling_stats(rate, end - start)),
    )


class HttpTestWorker:
    """Provide synthetic inference metrics while exercising pop/upload/submit over HTTP.

    Each job holds its lane from pop to finalization for ``job_wall_seconds``, and the lanes take jobs in
    turn, so at one lane a check's jobs finish one after another and at two lanes two at a time. A
    ``multi_lane_slowdown`` above one stretches every job of a run with more than one lane.
    """

    starts = 0
    closes = 0
    sequential_sampling = False
    metrics_overrides: dict[str, object] = {}
    sampling_rate = 10.0
    job_wall_seconds = 1.0
    multi_lane_slowdown = 1.0

    def __init__(self, service: LocalHorde, directory: Path, config: dict[str, object]) -> None:
        """Keep the local transport and its isolated stats directory."""
        self.service = service
        self.directory = directory
        self.task: asyncio.Task[None] | None = None
        self.running = False
        self.next_sampling_start = 0.0

    async def start(self) -> None:
        """Expose two differently sized cards without starting an accelerator process."""
        type(self).starts += 1
        self.running = True
        self.service.snapshot = WorkerStateSnapshot(
            config=WorkerConfigSummary(dreamer_name="test", worker_version="test"),
            per_card=[
                CardSnapshot(device_index=0, total_vram_mb=8192),
                CardSnapshot(device_index=1, total_vram_mb=16384),
            ],
        )

    def alive(self) -> bool:
        """Whether the test worker can keep processing supplied work."""
        return self.running

    async def configure(self, case: AssessmentCase, deadline: float) -> str | None:
        """Install one test workload consumer after its predecessor has completed."""
        if self.task is not None:
            await self.task
        self.task = asyncio.create_task(self._consume(case))
        return None

    async def _consume(self, case: AssessmentCase) -> None:
        # Synthetic windows run seconds past the wall clock, so each run starts after the last one's windows
        # end, as the worker's sequential runs do.
        sampling_start = max(time.monotonic(), self.next_sampling_start)
        self.next_sampling_start = sampling_start + 1000.0
        async with aiohttp.ClientSession() as client:
            for job_index in range(case.work_count):
                while True:
                    async with client.post(
                        f"{self.service.url}/api/v2/generate/pop",
                        json={
                            "name": "test",
                            "models": [
                                case.profile.model_name,
                                *([case.profile.companion_model] if case.profile.companion_model else []),
                            ],
                            "max_pixels": 4096**2,
                        },
                    ) as response:
                        raw = await response.json()
                    if raw["ids"]:
                        break
                    await asyncio.sleep(0.01)
                for identifier in raw["ids"]:
                    async with client.put(f"{self.service.url}/upload/{identifier}", data=b"result") as response:
                        assert response.status == 200
                    async with client.post(
                        f"{self.service.url}/api/v2/generate/submit",
                        json={
                            "id": identifier,
                            "state": "ok",
                            "generation": "R2",
                            "seed": 1,
                        },
                    ) as response:
                        assert response.status == 200
                record = self._record(case, job_index, raw["ids"][0], raw["model"], sampling_start)
                directory = self.directory / ".horde_worker_regen" / "stats"
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / "stats.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(StatsJobCompletedEvent(job=record).model_dump_json() + "\n")

    def _record(
        self, case: AssessmentCase, job_index: int, identifier: str, model_name: str, sampling_start: float
    ) -> JobMetricsRecord:
        offset = job_index * 2 if self.sequential_sampling else 0
        lanes = case.profile.threads
        job_seconds = self.job_wall_seconds * (self.multi_lane_slowdown if lanes > 1 else 1.0)
        popped = sampling_start + (job_index // lanes) * job_seconds
        profile = case.profile
        return JobMetricsRecord(
            job_id=identifier,
            model_name=model_name,
            time_popped=popped,
            stage_timestamps={"FINALIZED": popped + job_seconds},
            megapixelsteps=profile.width * profile.height * profile.steps * profile.batch / 1_000_000,
            inference_attempts=0,
            resource_failures=0,
            degraded_retry_used=False,
            host_commit_refusals=0,
            device_index=case.profile.device_index,
            sampling_window=SamplingWindow(first_step=sampling_start + offset, last_step=sampling_start + offset + 1),
            sampling_seconds=1.0,
            phase_metrics=JobPhaseMetrics(
                vram_device_used_high_water_mb=4096,
                ram_used_high_water_mb=2048,
                sampling=_sampling_stats(self.sampling_rate, 1.0),
            ),
        ).model_copy(update=self.metrics_overrides)

    async def close(self) -> None:
        """Stop the consumer after outstanding work has ended."""
        type(self).closes += 1
        self.running = False
        if self.task is not None and not self.task.done():
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task


def _operator_lease(monkeypatch: pytest.MonkeyPatch, slots: int) -> None:
    """Give the operator's copied configuration the sampling lease at the given slot count."""
    import horde_worker_regen.benchmark.assessment as assessment_module

    original = assessment_module.assessment_config

    def leased(
        models: dict[BenchTier, str], source: Path, *, alchemy_forms: list[str] | None = None
    ) -> dict[str, object]:
        config = original(models, source, alchemy_forms=alchemy_forms)
        config.update({"gpu_sampling_lease_enabled": True, "gpu_sampling_lease_slots": slots})
        return config

    monkeypatch.setattr(assessment_module, "assessment_config", leased)


async def test_shared_budget_normal_receipts_per_card_and_one_worker(tmp_path: Path) -> None:
    """A complete run must stay bounded, attribute both cards and confirm settings jointly."""
    HttpTestWorker.starts = HttpTestWorker.closes = 0
    controller = OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        seconds=10,
        max_jobs=16,
        models={BenchTier.SD15: "Deliberate"},
        worker_factory=HttpTestWorker,
    )
    report = await controller.run_async()
    assert HttpTestWorker.starts == HttpTestWorker.closes == 1
    assert report.budget is not None and report.budget.jobs_offered <= 16
    assert {item.profile.device_index for item in report.evidence} == {0, 1}
    passed = [item for item in report.evidence if item.status is EvidenceStatus.MEASURED_SUCCESS]
    assert passed and all(item.job_ids and item.device_peak_mb == 4096 for item in passed)
    assert report.suggested_bridge_data.safe_to_apply
    assert any(item.combined_validation and item.status is EvidenceStatus.MEASURED_SUCCESS for item in passed)
    assert (tmp_path / "report.md").exists()


async def test_no_downloaded_model_does_not_start_worker(tmp_path: Path) -> None:
    """An unprepared first run reports the preparation step without booting an empty worker."""
    HttpTestWorker.starts = 0
    controller = OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15]),
        out_dir=tmp_path,
        seconds=10,
        max_jobs=2,
        models={},
        worker_factory=HttpTestWorker,
    )
    report = await controller.run_async()
    assert HttpTestWorker.starts == 0
    assert report.budget is not None and "download models first" in report.budget.stop_reason
    assert not report.suggested_bridge_data.safe_to_apply


@pytest.mark.parametrize(
    "recovery",
    [
        {"inference_attempts": 1, "resource_failures": 1, "degraded_retry_used": True},
        {"host_commit_refusals": 1},
        {"inference_attempts": None},
    ],
)
async def test_joint_confirmation_requires_clean_retry_provenance(tmp_path: Path, recovery: dict[str, object]) -> None:
    """Successful submit after same-process recovery or missing provenance cannot confirm safe settings."""

    class RecoveringWorker(HttpTestWorker):
        async def _consume(self, case: AssessmentCase) -> None:
            self.metrics_overrides = recovery if case.combined else {}
            await super()._consume(case)

    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15]),
        out_dir=tmp_path,
        seconds=10,
        max_jobs=4,
        models={BenchTier.SD15: "Deliberate"},
        only_probe="sd15-baseline",
        worker_factory=RecoveringWorker,
    ).run_async()
    assert any(item.status is EvidenceStatus.MEASURED_SUCCESS for item in report.evidence)
    confirmations = [item for item in report.evidence if item.combined_validation]
    assert confirmations and all(
        item.status is EvidenceStatus.UNKNOWN and item.recoveries == 0 for item in confirmations
    )
    assert not report.suggested_bridge_data.safe_to_apply
    if recovery.get("degraded_retry_used"):
        assert all(
            item.degraded_retries == item.resource_failures == item.inference_failures == 1 for item in confirmations
        )


@pytest.mark.parametrize(
    ("metrics", "replace_process", "expected_status", "expected_details"),
    [
        ({}, False, EvidenceStatus.MEASURED_SUCCESS, ["were submitted on the selected card"]),
        (
            {"device_index": None},
            False,
            EvidenceStatus.UNKNOWN,
            ["execution card was not reported for 1 image job", "device peak memory was not reported"],
        ),
        (
            {"device_index": 1},
            False,
            EvidenceStatus.UNKNOWN,
            ["1 image job ran on a different card", "device peak memory was not reported"],
        ),
        (
            {"phase_metrics": None},
            False,
            EvidenceStatus.UNKNOWN,
            ["device peak memory was not reported for the selected card"],
        ),
        (
            {},
            True,
            EvidenceStatus.UNKNOWN,
            ["A worker process was replaced during the check", "This run could not confirm the settings"],
        ),
    ],
)
async def test_result_explains_missing_measurements_or_process_replacement(
    tmp_path: Path,
    metrics: dict[str, object],
    replace_process: bool,
    expected_status: EvidenceStatus,
    expected_details: list[str],
) -> None:
    """Incomplete measurements identify the problem while complete, unrecovered jobs still confirm settings."""

    class ReportingWorker(HttpTestWorker):
        async def _consume(self, case: AssessmentCase) -> None:
            self.metrics_overrides = metrics
            await super()._consume(case)
            if replace_process:
                assert self.service.snapshot is not None
                self.service.snapshot.num_process_recoveries += 1

    controller = OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15]),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate"},
    )
    async with LocalHorde() as service:
        worker = ReportingWorker(service, tmp_path / "worker", {})
        await worker.start()
        try:
            case = AssessmentCase(
                Capability(tier=BenchTier.SD15, kind=CapabilityKind.BASELINE),
                OperatingProfile(device_index=0, tier=BenchTier.SD15, model_name="Deliberate"),
            )
            result = await controller._execute(service, worker, case, time.monotonic() + 5)
        finally:
            await worker.close()
    assert result.status is expected_status
    assert all(detail in result.reason for detail in expected_details)
    if metrics.get("device_index") == 1:
        assert "execution card was not reported" not in result.reason


@pytest.mark.parametrize(
    ("lease_wait_seconds", "lease_timed_out", "expected_status", "expected_code"),
    [
        (CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS, False, EvidenceStatus.UNKNOWN, "clearance_timeout"),
        (5.0, True, EvidenceStatus.UNKNOWN, "clearance_timeout"),
        (5.0, False, EvidenceStatus.MEASURED_SUCCESS, "measured"),
        (0.0, False, EvidenceStatus.MEASURED_SUCCESS, "measured"),
    ],
)
async def test_jobs_sampled_only_after_the_clearance_timeout_do_not_confirm_settings(
    tmp_path: Path,
    lease_wait_seconds: float,
    lease_timed_out: bool,
    expected_status: EvidenceStatus,
    expected_code: str,
) -> None:
    """A job that waited out the clearance lease, or sampled without a grant, was never priced.

    Every job is dispatched long before its first step, as a cold model load does, so only the lease wait the
    child measured decides the grade.
    """

    class LeaseWaitWorker(HttpTestWorker):
        def _record(
            self, case: AssessmentCase, job_index: int, identifier: str, model_name: str, sampling_start: float
        ) -> JobMetricsRecord:
            record = super()._record(case, job_index, identifier, model_name, sampling_start)
            window = record.sampling_window or SamplingWindow(first_step=sampling_start, last_step=sampling_start + 1)
            return record.model_copy(
                update={
                    "stage_timestamps": {
                        **record.stage_timestamps,
                        "INFERENCE_IN_PROGRESS": time.time() - 2 * CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS,
                    },
                    "sampling_window": window.model_copy(
                        update={"lease_wait_seconds": lease_wait_seconds, "lease_timed_out": lease_timed_out}
                    ),
                    "width": case.profile.width,
                    "height": case.profile.height,
                    "batch_count": case.profile.batch,
                }
            )

    controller = OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15]),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate"},
    )
    async with LocalHorde() as service:
        worker = LeaseWaitWorker(service, tmp_path / "worker", {})
        await worker.start()
        try:
            case = AssessmentCase(
                Capability(tier=BenchTier.SD15, kind=CapabilityKind.BASELINE),
                OperatingProfile(device_index=0, tier=BenchTier.SD15, model_name="Deliberate"),
            )
            result = await controller._execute(service, worker, case, time.monotonic() + 5)
        finally:
            await worker.close()
    assert result.status is expected_status
    assert result.reason_code == expected_code
    assert result.clearance_wait_seconds == (lease_wait_seconds or None)
    if expected_status is EvidenceStatus.UNKNOWN:
        assert "only after the clearance timeout" in result.reason
        assert "512x512" in result.reason


@pytest.mark.parametrize("service_card", [1, None])
async def test_shared_service_evidence_does_not_claim_companion_image_placement(
    tmp_path: Path, service_card: int | None
) -> None:
    """A form on another lane or an unreported device remains shared-worker evidence."""

    class SharedWorker(HttpTestWorker):
        async def _consume(self, case: AssessmentCase) -> None:
            image_only = AssessmentCase(case.capability, case.profile.model_copy(update={"alchemy_forms": []}))
            await super()._consume(image_only)
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    f"{self.service.url}/api/v2/interrogate/pop",
                    json={
                        "name": "test",
                        "forms": ["caption"],
                        "amount": 1,
                        "priority_usernames": [],
                    },
                ) as response:
                    raw = await response.json()
                identifier = raw["forms"][0]["id"]
                async with client.post(
                    f"{self.service.url}/api/v2/interrogate/submit",
                    json={
                        "id": identifier,
                        "state": "ok",
                        "result": {"caption": "example"},
                    },
                ) as response:
                    assert response.status == 200
            record = JobMetricsRecord(
                job_id=identifier, workload=WorkloadKind.ALCHEMY, device_index=service_card, model_name="caption"
            )
            path = self.directory / ".horde_worker_regen" / "stats" / "stats.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(StatsJobCompletedEvent(job=record).model_dump_json() + "\n")

    controller = OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15]),
        out_dir=tmp_path,
        seconds=10,
        models={BenchTier.SD15: "Deliberate"},
    )
    async with LocalHorde() as service:
        worker = SharedWorker(service, tmp_path / "worker", {})
        await worker.start()
        try:
            case = AssessmentCase(
                Capability(tier=BenchTier.SD15, kind=CapabilityKind.ALCHEMY_CLIP),
                OperatingProfile(
                    device_index=0, tier=BenchTier.SD15, model_name="Deliberate", alchemy_forms=["caption"]
                ),
            )
            result = await controller._execute(service, worker, case, time.monotonic() + 5)
        finally:
            await worker.close()
    assert result.status is EvidenceStatus.MEASURED_SUCCESS
    assert result.scope is EvidenceScope.SHARED_WORKER
    assert "companion image" in result.reason and "shared-worker placement" in result.reason
    assert len(result.service_placements) == 1
    assert result.service_placements[0].reported_device_index == service_card
    rendered = render_markdown(
        CapabilityReport(
            operator_assessment=True, evidence=[result], suggested_bridge_data=SuggestedBridgeData(safe_to_apply=False)
        )
    )
    assert "shared worker (image card 0)" in rendered and "execution device unreported" in rendered
    assert ("reported lane card 1" if service_card is not None else "lane unreported") in rendered


async def test_sequential_success_cannot_validate_concurrency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under a lease that permits two samplers, finishing all jobs without sampling overlap leaves the cap unresolved.

    Each check runs its four jobs at one lane, then at two threads, then repeats the two-thread run once.
    """

    class SequentialWorker(HttpTestWorker):
        sequential_sampling = True

    from horde_worker_regen.benchmark.progress_channel import (
        JsonlProgressSink,
        LevelFinished,
        RampFinished,
        read_progress_events,
    )

    _operator_lease(monkeypatch, slots=2)
    controller = OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        seconds=10,
        max_jobs=28,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        only_probe="sd15-threads-2",
        worker_factory=SequentialWorker,
        progress_sink=JsonlProgressSink(tmp_path / "progress.jsonl"),
    )
    report = await controller.run_async()
    thread_checks = [item for item in report.evidence if item.capability.kind is CapabilityKind.THREADS]
    assert thread_checks and all(item.status is EvidenceStatus.UNKNOWN for item in thread_checks)
    assert all(item.reason_code == "sampling_not_overlapped" for item in thread_checks)
    assert all(item.profile.lease_slots is None and item.expected_sampling_concurrency == 2 for item in thread_checks)
    assert all(item.observed_sampling_concurrency == 1 for item in thread_checks)
    assert report.suggested_bridge_data.max_threads == 1
    assert all(item.jobs_completed == len(item.job_ids) == 12 for item in thread_checks)
    events = read_progress_events(tmp_path / "progress.jsonl")
    finished = [event for event in events if isinstance(event, LevelFinished) and "threads" in event.level_id]
    assert len(finished) == 2 and all(event.jobs_completed == event.jobs_expected == 12 for event in finished)
    final = [event for event in events if isinstance(event, RampFinished)][-1]
    assert report.budget is not None and final.run_jobs_completed == report.budget.jobs_offered


@pytest.mark.parametrize("max_jobs", [6, 9, 12, 18])
async def test_overlap_retry_preserves_confirmation_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_jobs: int
) -> None:
    """Optional retries must leave enough work to confirm every successful baseline."""
    _operator_lease(monkeypatch, slots=2)

    class SequentialWorker(HttpTestWorker):
        sequential_sampling = True

        async def start(self) -> None:
            await super().start()
            assert self.service.snapshot is not None
            if max_jobs < 12:
                self.service.snapshot.per_card = self.service.snapshot.per_card[:1]

    controller = OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        max_jobs=max_jobs,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        only_probe="sd15-threads-2",
        worker_factory=SequentialWorker,
    )
    report = await controller.run_async()
    confirmations = [item for item in report.evidence if item.combined_validation]
    assert len(confirmations) == (1 if max_jobs < 12 else 2)
    assert all(item.status is EvidenceStatus.MEASURED_SUCCESS for item in confirmations)
    assert report.suggested_bridge_data.safe_to_apply
    assert report.suggested_bridge_data.max_threads == 1
    assert report.budget is not None and report.budget.jobs_offered <= max_jobs


async def test_first_capacity_boundaries_precede_feature_catalog_under_small_budget(tmp_path: Path) -> None:
    """Representative features cannot spend the whole budget before any capacity boundary is measured."""
    controller = OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_alchemy=False),
        out_dir=tmp_path,
        seconds=10,
        # The two-thread check offers its four jobs at one lane and again at two lanes. A larger allowance
        # reaches the ControlNet check, whose job this test worker cannot pop.
        max_jobs=20,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        worker_factory=HttpTestWorker,
    )
    report = await controller.run_async()
    measured = {item.capability.kind for item in report.evidence if item.status is EvidenceStatus.MEASURED_SUCCESS}
    assert {
        CapabilityKind.BASELINE,
        CapabilityKind.BATCH,
        CapabilityKind.THREADS,
        CapabilityKind.RESOLUTION,
    } <= measured
    assert measured - {
        CapabilityKind.BASELINE,
        CapabilityKind.BATCH,
        CapabilityKind.THREADS,
        CapabilityKind.RESOLUTION,
        CapabilityKind.SUSTAINED,
    }


def test_isolated_config_passes_normal_validation_and_preserves_relative_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary worker must accept the copied config and find the original cache root."""
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIWORKER_CACHE_HOME", "relative-models")
    source = tmp_path / "bridgeData.yaml"
    source.write_text(
        "custom_models:\n  - name: local-model\n    baseline: stable_diffusion_1\n"
        "    filepath: weights/local.safetensors\n"
        "models_to_skip: [excluded-global]\nallow_inpainting: true\ngpu_overrides:\n"
        "  0:\n    allow_inpainting: true\n    models_to_skip: [excluded-card]\n",
        encoding="utf-8",
    )
    config = assessment_config({BenchTier.SD15: "Deliberate"}, source)
    reGenBridgeData.model_validate(config)
    assert "allow_inpainting" not in config
    assert "allow_inpainting" not in config["gpu_overrides"][0]
    assert config["gpu_overrides"][0]["allow_painting"] is True
    assert config["models_to_skip"] == ["excluded-global"]
    assert config["gpu_overrides"][0]["models_to_skip"] == ["excluded-card"]
    assert config["custom_models"][0]["filepath"] == str(tmp_path / "weights" / "local.safetensors")
    worker = OrdinaryAssessmentWorker(LocalHorde(), tmp_path / "isolated", config)
    assert worker._cache_root == str(tmp_path / "relative-models")


async def test_booting_resize_refusal_is_retried_before_work_is_offered(tmp_path: Path) -> None:
    """A transient startup refusal must not permanently classify the baseline as unknown."""
    service = LocalHorde()
    service.snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="test", worker_version="test"),
        supervisor_paused=True,
        per_card=[CardSnapshot(device_index=0)],
        processes=[
            ProcessSnapshot(
                process_id=1,
                process_type="INFERENCE",
                last_process_state="WAITING_FOR_JOB",
                is_alive=True,
                is_busy=False,
                current_job_id="finished-job-retained-for-display",
            )
        ],
    )

    class BootingWorker(OrdinaryAssessmentWorker):
        attempts = 0

        def alive(self) -> bool:
            """Keep the controller active without launching a subprocess."""
            return True

        async def _wait(self, predicate: Callable[[WorkerStateSnapshot], bool], deadline: float) -> bool:
            assert service.snapshot is not None
            while service._commands:
                command = service._commands.popleft()
                if command.command is SupervisorCommand.RELOAD_CONFIG:
                    service.snapshot.config_revision += 1
                if command.command is SupervisorCommand.RESIZE_CAPACITY:
                    self.attempts += 1
                    service.snapshot.capacity_change = CapacityChangeResult(
                        request_id=command.request_id,
                        device_index=0,
                        requested_threads=1,
                        accepted=self.attempts > 1,
                        reason="booting" if self.attempts == 1 else "ready",
                        retryable=self.attempts == 1,
                    )
            return predicate(service.snapshot)

    worker = BootingWorker(service, tmp_path, {})
    case = AssessmentCase(
        Capability(tier=BenchTier.SD15, kind=CapabilityKind.BASELINE),
        OperatingProfile(device_index=0, tier=BenchTier.SD15, model_name="Deliberate"),
    )
    assert await worker.configure(case, time.monotonic() + 5) is None
    assert worker.attempts == 2
    assert service._commands[-1].command is SupervisorCommand.RESUME


class ScriptedConfigureWorker(OrdinaryAssessmentWorker):
    """Answer supervisor commands from a scripted snapshot, leaving every sent command on the service queue."""

    def __init__(self, service: LocalHorde, directory: Path) -> None:
        """Start with every queued command unanswered and capacity requests accepted."""
        super().__init__(service, directory, {})
        self.answered = 0
        self.refuse_capacity = False

    def alive(self) -> bool:
        """Keep the controller active without launching a subprocess."""
        return True

    async def _wait(self, predicate: Callable[[WorkerStateSnapshot], bool], deadline: float) -> bool:
        snapshot = self._service.snapshot
        assert snapshot is not None
        commands = list(self._service._commands)
        for command in commands[self.answered :]:
            if command.command is SupervisorCommand.RELOAD_CONFIG:
                snapshot.config_revision += 1
            if command.command is SupervisorCommand.RESIZE_CAPACITY:
                snapshot.capacity_change = CapacityChangeResult(
                    request_id=command.request_id,
                    device_index=0,
                    requested_threads=command.target_threads or 1,
                    accepted=not self.refuse_capacity,
                    reason="refused" if self.refuse_capacity else "ready",
                    retryable=False,
                )
        self.answered = len(commands)
        return predicate(snapshot)


def _scripted_service() -> LocalHorde:
    service = LocalHorde()
    service.snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="test", worker_version="test"),
        supervisor_paused=True,
        per_card=[CardSnapshot(device_index=0)],
        processes=[
            ProcessSnapshot(
                process_id=identifier,
                process_type="INFERENCE",
                last_process_state="WAITING_FOR_JOB",
                is_alive=True,
                is_busy=False,
            )
            for identifier in (1, 2)
        ],
    )
    return service


def _boundary_commands(service: LocalHorde) -> list[SupervisorCommand]:
    boundary = {SupervisorCommand.PAUSE, SupervisorCommand.RELOAD_CONFIG, SupervisorCommand.RESIZE_CAPACITY}
    return [command.command for command in service._commands if command.command in boundary]


def _baseline_case(threads: int = 1) -> AssessmentCase:
    return AssessmentCase(
        Capability(tier=BenchTier.SD15, kind=CapabilityKind.BASELINE),
        OperatingProfile(device_index=0, tier=BenchTier.SD15, model_name="Deliberate", threads=threads),
    )


async def test_identical_configure_reuses_the_applied_config(tmp_path: Path) -> None:
    """An unchanged config keeps the running worker; a changed thread count applies the full protocol."""
    service = _scripted_service()
    worker = ScriptedConfigureWorker(service, tmp_path)
    full = [SupervisorCommand.PAUSE, SupervisorCommand.RELOAD_CONFIG, SupervisorCommand.RESIZE_CAPACITY]
    assert await worker.configure(_baseline_case(), time.monotonic() + 5) is None
    assert await worker.configure(_baseline_case(), time.monotonic() + 5) is None
    assert _boundary_commands(service) == full
    assert [command.command for command in service._commands].count(SupervisorCommand.RESUME) == 1
    assert await worker.configure(_baseline_case(threads=2), time.monotonic() + 5) is None
    assert _boundary_commands(service) == full * 2


async def test_configure_after_a_refusal_applies_the_full_protocol(tmp_path: Path) -> None:
    """A refused check does not retain an applied config, so the next check applies all settings again."""
    service = _scripted_service()
    worker = ScriptedConfigureWorker(service, tmp_path)
    full = [SupervisorCommand.PAUSE, SupervisorCommand.RELOAD_CONFIG, SupervisorCommand.RESIZE_CAPACITY]
    assert await worker.configure(_baseline_case(), time.monotonic() + 5) is None
    worker.refuse_capacity = True
    assert await worker.configure(_baseline_case(threads=2), time.monotonic() + 5) == "refused"
    worker.refuse_capacity = False
    assert await worker.configure(_baseline_case(threads=2), time.monotonic() + 5) is None
    assert _boundary_commands(service) == full * 3


async def test_configure_with_unfinalized_issued_work_applies_the_full_protocol(tmp_path: Path) -> None:
    """Issued work without a submit receipt must be drained before the next check starts."""
    service = _scripted_service()
    worker = ScriptedConfigureWorker(service, tmp_path)
    full = [SupervisorCommand.PAUSE, SupervisorCommand.RELOAD_CONFIG, SupervisorCommand.RESIZE_CAPACITY]
    assert await worker.configure(_baseline_case(), time.monotonic() + 5) is None
    service.work["outstanding"] = IssuedWork("outstanding", {"generation"}, issued=True)
    assert await worker.configure(_baseline_case(), time.monotonic() + 5) is None
    assert _boundary_commands(service) == full * 2


class SampledHttpTestWorker(HttpTestWorker):
    """Report a fixed sampling duration on every synthetic job record."""

    metrics_overrides: dict[str, object] = {"sampling_seconds": 1.5}


async def test_evidence_records_configure_and_sampling_seconds(tmp_path: Path) -> None:
    """Measured evidence carries the check-boundary and sampling durations the report totals."""
    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate"},
        max_jobs=4,
        worker_factory=SampledHttpTestWorker,
    ).run_async()
    measured = [item for item in report.evidence if item.status is EvidenceStatus.MEASURED_SUCCESS]
    assert measured
    for item in measured:
        assert item.configure_seconds is not None and 0 <= item.configure_seconds <= item.elapsed_seconds
        assert item.sampling_seconds == pytest.approx(1.5 * len(item.job_ids))
    assert "went to applying settings between checks" in render_markdown(report)


def test_concurrency_requires_sampling_overlap_on_the_same_card() -> None:
    """Sequential jobs, touching endpoints and a different card cannot establish two sampling lanes."""

    def record(identifier: str, start: float, end: float, device: int = 0) -> JobMetricsRecord:
        return JobMetricsRecord(
            job_id=identifier,
            device_index=device,
            sampling_window=SamplingWindow(first_step=start, last_step=end),
        )

    assert observed_sampling_concurrency([record("a", 1, 2), record("b", 2, 3)], 0) == 1
    assert observed_sampling_concurrency([record("a", 1, 3), record("b", 2, 4, 1)], 0) == 1
    assert observed_sampling_concurrency([record("a", 1, 3), record("b", 2, 4)], 0) == 2
    assert observed_sampling_concurrency([JobMetricsRecord(job_id="missing")], 0) == 0


def test_lane_equivalent_throughput_normalises_steps_to_single_lane_rates() -> None:
    """Lane-equivalent throughput measures concurrent sampling speed against single-lane reference rates."""
    references = {"fast": 10.0, "slow": 5.0}
    assert lane_equivalent_throughput([_rated_record("a", "fast", 0, 2, 10.0)], references) == pytest.approx(1.0)
    two_lanes = [_rated_record("a", "fast", 0, 2, 10.0), _rated_record("b", "slow", 0, 2, 5.0)]
    assert lane_equivalent_throughput(two_lanes, references) == pytest.approx(2.0)
    partially_unloaded = [_rated_record("a", "fast", 0, 5, 3.0), _rated_record("b", "slow", 0, 5, 1.0)]
    assert lane_equivalent_throughput(partially_unloaded, references) == pytest.approx(0.5)


def test_lane_equivalent_throughput_does_not_charge_idle_gaps() -> None:
    """Idle gaps between jobs are excluded from the sampling-window union."""
    records = [_rated_record("a", "fast", 0, 1, 10.0), _rated_record("b", "fast", 5, 6, 10.0)]
    assert lane_equivalent_throughput(records, {"fast": 10.0}) == pytest.approx(1.0)


def test_lane_equivalent_throughput_is_unknown_without_a_reference_or_rate() -> None:
    """A job that cannot be normalised leaves the figure unknown instead of partial."""
    records = [_rated_record("a", "fast", 0, 1, 10.0), _rated_record("b", "other", 0, 1, 10.0)]
    assert lane_equivalent_throughput(records, {"fast": 10.0}) is None
    unrated = JobMetricsRecord(
        job_id="c", model_name="fast", sampling_window=SamplingWindow(first_step=0, last_step=1), sampling_seconds=1
    )
    assert lane_equivalent_throughput([unrated], {"fast": 10.0}) is None
    assert lane_equivalent_throughput([], {"fast": 10.0}) is None


def test_throughput_reference_rates_prefer_jobs_that_sampled_alone_on_the_card() -> None:
    """Jobs that overlapped another window or ran on another card do not set the single-lane rate."""
    records = [
        _rated_record("a", "fast", 0, 1, 10.0),
        _rated_record("b", "fast", 2, 4, 6.0),
        _rated_record("c", "slow", 3, 5, 1.0),
        _rated_record("d", "slow", 3, 5, 2.0),
        _rated_record("e", "slow", 10, 11, 9.0, device=1),
    ]
    references = reference_sampling_rates(records, 0, ["fast", "slow", "absent"])
    assert references["fast"].iterations_per_second == 10.0
    assert references["fast"].basis is ReferenceRateBasis.SINGLE_LANE
    assert references["slow"].iterations_per_second == 2.0
    assert references["slow"].basis is ReferenceRateBasis.RUN_MAXIMUM
    assert "absent" not in references


def _timed_record(
    identifier: str, popped: float | None, finalized: float | None, megapixelsteps: float = 1.0
) -> JobMetricsRecord:
    return JobMetricsRecord(
        job_id=identifier,
        time_popped=popped,
        stage_timestamps={"FINALIZED": finalized} if finalized is not None else {},
        megapixelsteps=megapixelsteps,
        sampling_window=SamplingWindow(first_step=100.0, last_step=101.0),
    )


def test_wall_throughput_spans_the_first_pop_to_the_last_finalization() -> None:
    """Megapixelsteps and jobs are divided by the phase's wall span, including the time no job sampled."""
    records = [_timed_record("a", 0.0, 4.0, 2.0), _timed_record("b", 1.0, 10.0, 3.0)]
    throughput = phase_wall_throughput(records)
    assert throughput is not None
    assert throughput.megapixelsteps_per_second == pytest.approx(5.0 / 10.0)
    assert throughput.jobs_per_minute == pytest.approx(2 * 60 / 10.0)


def test_wall_throughput_falls_back_to_sampling_windows_and_is_unknown_without_either() -> None:
    """A job without pop and finalization stamps moves the whole phase onto the sampling windows' clock."""
    records = [_timed_record("a", 0.0, 4.0), _timed_record("b", None, 10.0)]
    throughput = phase_wall_throughput(records)
    assert throughput is not None and throughput.megapixelsteps_per_second == pytest.approx(2.0 / 1.0)
    unstamped = JobMetricsRecord(job_id="c", megapixelsteps=1.0)
    assert phase_wall_throughput([_timed_record("a", 0.0, 4.0), unstamped]) is None
    assert phase_wall_throughput([_timed_record("a", 0.0, 4.0, megapixelsteps=0.0)]) is None
    assert phase_wall_throughput([]) is None


def test_wall_noise_margin_is_the_relative_median_deviation_of_job_times() -> None:
    """One slow job does not widen the margin; a spread of typical job times does."""
    steady = [_timed_record(str(index), 0.0, seconds) for index, seconds in enumerate((2.0, 2.0, 2.0, 9.0))]
    assert wall_noise_margin(steady) == pytest.approx(0.0)
    spread = [_timed_record(str(index), 0.0, seconds) for index, seconds in enumerate((1.0, 2.0, 3.0))]
    assert wall_noise_margin(spread) == pytest.approx(1.0 / 2.0)
    assert wall_noise_margin([JobMetricsRecord(job_id="unstamped")]) == 0.0


class LaneRateWorker(HttpTestWorker):
    """Give a concurrency check's jobs each model's own rate, slowing the companion's first as a partial unload does.

    At one lane the alternating jobs sample one after another. At two threads each model's jobs are spaced by
    its own count, so two lanes overlap job for job. With the companion slowed, its first job samples at a fifth
    of its rate across four seconds and a later one samples alone at full rate.
    """

    companion_slowed = False
    primary_rate = 10.0
    companion_rate = 6.5

    async def _consume(self, case: AssessmentCase) -> None:
        self.model_counts: dict[str, int] = {}
        await super()._consume(case)

    def _record(
        self, case: AssessmentCase, job_index: int, identifier: str, model_name: str, sampling_start: float
    ) -> JobMetricsRecord:
        record = super()._record(case, job_index, identifier, model_name, sampling_start)
        if case.profile.companion_model is None:
            return record
        count = self.model_counts.get(model_name, 0)
        self.model_counts[model_name] = count + 1
        start, seconds = sampling_start + count, 1.0
        rate = self.primary_rate if model_name == case.profile.model_name else self.companion_rate
        if case.profile.threads == 1:
            start = sampling_start + job_index
        elif self.companion_slowed:
            start = sampling_start + count * 2
            if model_name == case.profile.companion_model:
                start, seconds, rate = (sampling_start, 4.0, rate / 5) if count == 0 else (start + 10, 1.0, rate)
        assert record.phase_metrics is not None
        return record.model_copy(
            update={
                "sampling_window": SamplingWindow(first_step=start, last_step=start + seconds),
                "sampling_seconds": seconds,
                "phase_metrics": record.phase_metrics.model_copy(update={"sampling": _sampling_stats(rate, seconds)}),
            }
        )


async def test_overlapping_lanes_below_one_lane_of_throughput_are_not_confirmed(tmp_path: Path) -> None:
    """Sampling overlap cannot confirm a thread count when a slowed companion reduces combined throughput."""
    from horde_worker_regen.benchmark.progress_channel import JsonlProgressSink, LevelFinished, read_progress_events

    class SlowedCompanionWorker(LaneRateWorker):
        companion_slowed = True

    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        only_probe="sd15-threads-2",
        worker_factory=SlowedCompanionWorker,
        progress_sink=JsonlProgressSink(tmp_path / "progress.jsonl"),
    ).run_async()
    thread_checks = [item for item in report.evidence if item.profile.threads == 2 and item.attempted]
    assert thread_checks
    for item in thread_checks:
        assert item.status is EvidenceStatus.UNKNOWN and item.reason_code == "throughput_below_single_lane"
        # Four jobs at one lane, then the same four at two threads.
        assert item.observed_sampling_concurrency == 2 and len(item.job_ids) == 8
        # The thrashing lane blocks the thread count although the two lanes finished the jobs in half the time.
        assert item.wall_gain == pytest.approx(2.0)
        assert "2 lanes finished 120.0 jobs per minute against 60.0 on one lane" in item.reason
        # Two primary jobs and the companion's lone job each count one lane-second; the slowed companion job
        # counts 1.3 x 4 / 6.5. The windows cover five seconds.
        assert item.lane_equivalent_throughput == pytest.approx(3.8 / 5)
        assert item.model_sampling_rates == pytest.approx({"Deliberate": 10.0, "AlbedoBase XL (SDXL)": 3.9})
        assert "together they sampled 0.76 times one lane's rate" in item.reason
        assert "AlbedoBase XL (SDXL) 3.9 it/s against" in item.reason
    assert report.suggested_bridge_data.max_threads == 1
    finished = [
        event
        for event in read_progress_events(tmp_path / "progress.jsonl")
        if isinstance(event, LevelFinished) and event.reason_code == "throughput_below_single_lane"
    ]
    assert finished and all("0.76 times one lane's rate" in event.reasons[0] for event in finished)


async def test_a_thrashing_lane_in_the_confirmation_is_judged_against_the_thread_checks_single_lane_rates(
    tmp_path: Path,
) -> None:
    """Combined confirmation uses reference rates from the thread check's single-lane jobs."""

    class SlowedConfirmationWorker(LaneRateWorker):
        async def _consume(self, case: AssessmentCase) -> None:
            self.companion_slowed = case.combined
            await super()._consume(case)

    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        only_probe="sd15-threads-2",
        worker_factory=SlowedConfirmationWorker,
    ).run_async()
    confirmations = [item for item in report.evidence if item.combined_validation and item.profile.threads == 2]
    assert confirmations
    for item in confirmations:
        assert item.status is EvidenceStatus.UNKNOWN and item.reason_code == "throughput_below_single_lane"
        assert len(item.job_ids) == 4 and item.wall_gain is None
        assert item.lane_equivalent_throughput == pytest.approx(3.8 / 5)
        assert all(
            reference.basis is ReferenceRateBasis.SINGLE_LANE for reference in item.model_reference_rates.values()
        )
    assert report.suggested_bridge_data.max_threads == 1


async def test_a_joint_unconfirmed_on_its_threads_is_confirmed_again_at_one_thread(tmp_path: Path) -> None:
    """A multi-lane joint refused for its thread count repeats at one thread, so its other limits still confirm."""

    class SlowedConfirmationWorker(LaneRateWorker):
        async def _consume(self, case: AssessmentCase) -> None:
            self.companion_slowed = case.combined
            await super()._consume(case)

    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        only_probe="sd15-threads-2",
        worker_factory=SlowedConfirmationWorker,
    ).run_async()
    follow_up_slug = Capability(tier=BenchTier.SD15, kind=CapabilityKind.SUSTAINED, magnitude=1).slug
    joints = {
        item.profile.device_index: item
        for item in report.evidence
        if item.combined_validation and item.profile.threads == 2
    }
    follow_ups = {
        item.profile.device_index: item
        for item in report.evidence
        if item.combined_validation and item.profile.threads == 1
    }
    assert joints and follow_ups.keys() == joints.keys()
    for device, joint in joints.items():
        assert joint.status is EvidenceStatus.UNKNOWN and joint.reason_code == "throughput_below_single_lane"
        follow_up = follow_ups[device]
        assert follow_up.evidence_id == f"card{device}-{follow_up_slug}-combined" != joint.evidence_id
        assert follow_up.status is EvidenceStatus.MEASURED_SUCCESS and follow_up.reason_code == "measured"
        assert follow_up.profile.companion_model is None and follow_up.profile.lease_slots is None
        assert (follow_up.profile.batch, follow_up.profile.width, follow_up.profile.height) == (
            joint.profile.batch,
            joint.profile.width,
            joint.profile.height,
        )
    suggestion = report.suggested_bridge_data
    assert suggestion.safe_to_apply and suggestion.max_threads == 1
    for follow_up in follow_ups.values():
        assert suggestion.max_batch == follow_up.profile.batch
        assert suggestion.max_power == follow_up.profile.width * follow_up.profile.height // (8 * 64 * 64)


async def test_a_joint_that_faults_is_not_repeated_at_one_thread(tmp_path: Path) -> None:
    """Only a thread-count verdict earns the single-thread follow-up. A faulted joint keeps its measured verdict."""

    class FaultingConfirmationWorker(LaneRateWorker):
        async def _consume(self, case: AssessmentCase) -> None:
            self.metrics_overrides = {"faulted": True} if case.combined else {}
            await super()._consume(case)

    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        only_probe="sd15-threads-2",
        worker_factory=FaultingConfirmationWorker,
    ).run_async()
    confirmations = [item for item in report.evidence if item.combined_validation]
    assert confirmations
    assert all(item.profile.threads == 2 for item in confirmations)
    assert all(item.status is EvidenceStatus.MEASURED_FAILURE for item in confirmations)
    assert not report.suggested_bridge_data.safe_to_apply


async def test_overlapping_lanes_at_full_throughput_confirm_two_threads(tmp_path: Path) -> None:
    """Two lanes at their reference rates report lane-equivalent throughput of 2.0 in the report and progress."""
    from horde_worker_regen.benchmark.progress_channel import JsonlProgressSink, LevelFinished, read_progress_events

    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        only_probe="sd15-threads-2",
        worker_factory=LaneRateWorker,
        progress_sink=JsonlProgressSink(tmp_path / "progress.jsonl"),
    ).run_async()
    thread_checks = [item for item in report.evidence if item.profile.threads == 2 and item.attempted]
    assert thread_checks and all(item.status is EvidenceStatus.MEASURED_SUCCESS for item in thread_checks)
    assert all(item.lane_equivalent_throughput == pytest.approx(2.0) for item in thread_checks)
    assert all(
        reference.basis is ReferenceRateBasis.SINGLE_LANE
        for item in thread_checks
        for reference in item.model_reference_rates.values()
    )
    assert report.suggested_bridge_data.max_threads == 2
    assert "The lanes together sampled 2.00 times one lane's rate" in render_markdown(report)
    finished = [
        event
        for event in read_progress_events(tmp_path / "progress.jsonl")
        if isinstance(event, LevelFinished) and "threads" in event.level_id
    ]
    assert finished and all(
        any("together sampled 2.00 times one lane's rate" in reason for reason in event.reasons) for event in finished
    )


def test_withheld_feature_is_reported_before_offering_ineligible_work() -> None:
    """Missing preparation leaves a check unknown without spending the budget on a job the worker cannot accept."""
    case = AssessmentCase(
        Capability(tier=BenchTier.SD15, kind=CapabilityKind.CONTROLNET),
        OperatingProfile(device_index=0, tier=BenchTier.SD15, model_name="Deliberate", control_type="canny"),
    )
    readiness = FeatureReadiness(
        feature=GatedFeature.CONTROLNET,
        label="ControlNet",
        state=FeatureReadinessState.WAITING,
        detail="annotators are not yet on disk",
    )
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="test", worker_version="test"),
        feature_readiness=FeatureReadinessSummary(gated=[readiness]),
    )
    assert "prepare this feature" in str(unavailable_feature(case, snapshot))
    readiness.state = FeatureReadinessState.OFFERED
    assert unavailable_feature(case, snapshot) is None


async def test_complete_assessment_derives_work_and_finishes_selected_capacity(tmp_path: Path) -> None:
    """Coverage has a finite work bound without an overall clock limit."""
    from horde_worker_regen.benchmark.progress_channel import JsonlProgressSink, LevelFinished, read_progress_events

    progress_path = tmp_path / "progress.jsonl"
    controller = OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        worker_factory=HttpTestWorker,
        progress_sink=JsonlProgressSink(progress_path),
    )
    report = await controller.run_async()
    assert report.budget is not None and report.budget.seconds is None
    assert report.budget.jobs_offered <= report.budget.max_jobs
    assert not any(item.reason_code in {"time_limit", "work_limit"} for item in report.evidence)
    # Two installed models give at most two lanes their own model, so the search stops at two threads.
    assert any(
        item.profile.threads == 2 and item.status is EvidenceStatus.MEASURED_SUCCESS for item in report.evidence
    )
    assert not any(item.profile.threads > 2 for item in report.evidence)
    finished = [event for event in read_progress_events(progress_path) if isinstance(event, LevelFinished)]
    assert all(event.jobs_completed is not None for event in finished)
    assert all(event.outcome != "skipped" for event in finished)


async def test_exact_work_allowance_reports_completion_not_exhaustion(tmp_path: Path) -> None:
    """Finishing every selected check at the work limit is a completed assessment."""
    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate"},
        worker_factory=HttpTestWorker,
        only_probe="sd15-baseline",
    ).run_async()
    assert report.budget is not None and report.budget.jobs_offered == report.budget.max_jobs
    assert report.budget.stop_reason.startswith("Selected checks finished")


def test_progress_watch_ignores_heartbeats_and_tracks_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """A healthy slow sampler may progress indefinitely; identical heartbeat traffic cannot hide a stall."""
    from horde_worker_regen.benchmark.assessment import AssessmentProgressWatch

    now = [0.0]
    monkeypatch.setattr("horde_worker_regen.benchmark.assessment.time.monotonic", lambda: now[0])
    service = LocalHorde()
    process = ProcessSnapshot(
        process_id=1,
        process_type="INFERENCE",
        is_alive=True,
        is_busy=True,
        last_process_state="INFERENCE_STARTING",
        current_job_id="job",
        last_current_step=0,
    )
    service.snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="test", worker_version="test"), processes=[process]
    )
    watch = AssessmentProgressWatch(service, 10)
    assert not watch.stalled()
    for step in range(1, 5):
        now[0] += 9
        service.last_contact = now[0]
        process.last_current_step = step
        assert not watch.stalled()
    now[0] += 11
    service.last_contact = now[0]
    assert watch.stalled() and "not progressed" in watch.reason


def test_loading_summary_distinguishes_first_repeat_missing_and_duplicate_events() -> None:
    """Missing loads are unknown, and shared metric events are counted only once."""
    from hordelib.metrics import ModelLoadEvent

    from horde_worker_regen.benchmark.assessment import model_loading_summaries

    event = ModelLoadEvent(model_name="model", phase="disk_to_ram", duration_seconds=8, timestamp=1)
    repeat = event.model_copy(update={"duration_seconds": 2, "timestamp": 3})
    records = [
        JobMetricsRecord(
            job_id=str(index), model_name="model", time_popped=index, phase_metrics=JobPhaseMetrics(model_loads=loads)
        )
        for index, loads in enumerate(([event], [event, repeat], []))
    ]
    summary = model_loading_summaries(records)
    assert len(summary) == 1
    assert summary[0].first_seconds == 8 and summary[0].repeat_seconds == [2]
    assert not model_loading_summaries([JobMetricsRecord(job_id="missing")])


def test_live_progress_reports_job_steps_and_card_metrics(tmp_path: Path) -> None:
    """Live progress uses worker measurements without converting absent values into zeros."""
    from horde_worker_regen.benchmark.progress_channel import LevelProgress, ProgressSink
    from horde_worker_regen.process_management.ipc.supervisor_channel import StatsSample

    class Capture(ProgressSink):
        events: list[LevelProgress] = []

        def emit(self, event: object) -> None:
            if isinstance(event, LevelProgress):
                self.events.append(event)

    capture = Capture()
    controller = OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15]), out_dir=tmp_path, progress_sink=capture
    )
    service = LocalHorde()
    case = AssessmentCase(
        Capability(tier=BenchTier.SD15, kind=CapabilityKind.BASELINE),
        OperatingProfile(device_index=0, tier=BenchTier.SD15, model_name="model"),
    )
    ids = service.enqueue(case.scenario().expand_image_jobs(), [])
    service.work[ids[0]].issued = True
    service.snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="test", worker_version="test"),
        per_card=[CardSnapshot(device_index=0, total_vram_mb=8192, free_vram_mb=4096)],
        latest_stats_sample=StatsSample(timestamp=time.time(), gpu_busy_fraction=0.75),
        processes=[
            ProcessSnapshot(
                process_id=1,
                process_type="INFERENCE",
                is_alive=True,
                is_busy=True,
                current_job_id=ids[0],
                last_process_state="INFERENCE_STARTING",
                last_heartbeat_type="INFERENCE_STEP",
                last_current_step=4,
                last_total_steps=16,
                last_iterations_per_second=8,
            )
        ],
    )
    controller._live_progress(service, case, ids, time.monotonic())
    progress = capture.events[-1]
    assert progress.active_jobs[0].current_step == 4 and progress.iterations_per_second == 8
    assert progress.vram_used_mb == 4096 and progress.gpu_busy_percent == 75
    service.snapshot.processes[0].is_busy = False
    service.snapshot.latest_stats_sample = None
    controller._live_progress(service, case, ids, time.monotonic())
    assert capture.events[-1].iterations_per_second is None and capture.events[-1].gpu_busy_percent is None


def test_targeted_large_thread_check_reserves_confirmation_and_follow_up() -> None:
    """A derived cap covers a selected large check without using the default search's thread ceiling."""
    from horde_worker_regen.benchmark.assessment import derived_work_budget

    profile = OperatingProfile(device_index=0, tier=BenchTier.SD15, model_name="model")
    cases = [
        AssessmentCase(Capability(tier=BenchTier.SD15, kind=CapabilityKind.BASELINE), profile),
        AssessmentCase(
            Capability(tier=BenchTier.SD15, kind=CapabilityKind.THREADS, magnitude=16),
            profile.model_copy(update={"threads": 16}),
        ),
    ]
    jobs = derived_work_budget(cases, validate=True, options=CatalogOptions(tiers=[BenchTier.SD15]), targeted=True)
    # The check runs 32 jobs at one lane and 32 at sixteen threads, the overlap follow-up repeats only the
    # sixteen-thread run, the confirmation runs only sixteen threads, and its single-thread follow-up runs one
    # job.
    assert jobs == 1 + 64 + 32 + 32 + 1


def test_default_thread_reservation_stops_at_the_installed_model_count() -> None:
    """A thread-count check reserves work only when a distinct installed model is available for every lane."""
    from horde_worker_regen.benchmark.assessment import derived_work_budget

    profile = OperatingProfile(device_index=0, tier=BenchTier.SD15, model_name="model")
    cases = [AssessmentCase(Capability(tier=BenchTier.SD15, kind=CapabilityKind.BASELINE), profile)]
    options = CatalogOptions(tiers=[BenchTier.SD15], include_features=False)

    def budget(installed_models: int) -> int:
        return derived_work_budget(cases, validate=False, options=options, installed_models=installed_models)

    # A thread magnitude reserves three runs of two jobs per thread: one lane, the thread count and the
    # overlap follow-up.
    two_thread_jobs = 2 * 2 * 3
    four_thread_jobs = 4 * 2 * 3
    assert budget(2) - budget(1) == two_thread_jobs
    assert budget(3) == budget(2)
    assert budget(4) - budget(2) == four_thread_jobs
    without_concurrency = derived_work_budget(
        cases,
        validate=False,
        options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_concurrency=False),
        installed_models=1,
    )
    # One installed model reserves the batch magnitudes only, the same as no thread search at all.
    assert budget(1) == without_concurrency + 2


@pytest.mark.parametrize(
    ("storage", "expected", "absent"),
    [
        pytest.param(
            {"/models": True, "/extra": False},
            "The models are on rotational storage (/models). 2 models stay resident, "
            "and each change to another model reloads it from disk.",
            "If the models are on",
            id="rotational",
        ),
        pytest.param(
            {"/models": False},
            "The models are on solid-state storage.",
            "configured models",
            id="solid-state",
        ),
        pytest.param(
            {"/models": None},
            "The storage type was not determined. "
            "If the models are on rotational storage, each change to another model reloads it from disk.",
            "keep at most",
            id="undetermined",
        ),
    ],
)
def test_storage_advice_uses_resolved_process_count_without_restricting_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage: dict[str, bool | None],
    expected: str,
    absent: str,
) -> None:
    """Storage advice follows the probed media type and counts the operator's models without restricting them."""
    from horde_worker_regen.benchmark.capabilities.result import MachineInfo
    from horde_worker_regen.bridge_data.load_config import BridgeDataLoader

    controller = OperatorAssessment(catalog_options=CatalogOptions(tiers=[BenchTier.SD15]), out_dir=tmp_path)
    controller._source_config = {"models_to_load": ["one"]}
    controller._operator_config = {
        "models_to_load": ["unused"],
        "gpu_overrides": {0: {"models_to_load": ["one", "two", "three"]}},
    }
    monkeypatch.setattr(
        BridgeDataLoader,
        "_resolve_meta_instructions",
        staticmethod(lambda bridge, manager, **_: bridge.image_models_to_load),
    )
    report = CapabilityReport(operator_assessment=True, evidence=[], machine=MachineInfo(storage=storage))
    from horde_worker_regen.benchmark.capabilities.result import WorkloadEvidence

    report.evidence.append(
        WorkloadEvidence(
            evidence_id="combined",
            capability=Capability(tier=BenchTier.SD15, kind=CapabilityKind.SUSTAINED),
            profile=OperatingProfile(device_index=0, tier=BenchTier.SD15, model_name="one"),
            status=EvidenceStatus.MEASURED_SUCCESS,
            reason="completed",
            combined_validation=True,
        )
    )
    notes = controller._storage_notes(report)
    assert notes[0].startswith("Model loading includes deserialization")
    assert any(expected in note for note in notes)
    assert not any(absent in note for note in notes)
    if storage != {"/models": False}:
        assert any("3 configured models" in note and "2 inference processes" in note for note in notes)
    assert controller._operator_config["gpu_overrides"][0]["models_to_load"] == ["one", "two", "three"]
    assert controller._source_config["models_to_load"] == ["one"]


def test_caption_is_enabled_and_advertised_by_ordinary_form_expansion(tmp_path: Path) -> None:
    """A selected caption check must survive the ordinary form-config and opt-in gates."""
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.jobs.alchemy_popper import expand_offered_forms

    source = tmp_path / "source.yaml"
    source.write_text("forms: [nsfw, post-process]\nalchemy_caption_enabled: false\n", encoding="utf-8")
    config = assessment_config({BenchTier.SD15: "Deliberate"}, source, alchemy_forms=["caption"])
    worker_config = reGenBridgeData.model_validate(config)
    assert worker_config.alchemy_caption_enabled
    assert "caption" in expand_offered_forms(worker_config)


async def test_unresponsive_worker_is_the_stop_reason(tmp_path: Path) -> None:
    """A worker that stops reporting ends the run with an unresponsive-worker reason."""
    from horde_worker_regen.benchmark.capabilities.capability import CapabilityKind
    from horde_worker_regen.benchmark.progress_channel import JsonlProgressSink, RampFinished, read_progress_events

    class SilentWorker(HttpTestWorker):
        """Serve the baselines, then fall silent as a wedged worker would."""

        async def configure(self, case: AssessmentCase, deadline: float) -> str | None:
            if case.capability.kind is CapabilityKind.BASELINE:
                return await super().configure(case, deadline)
            if self.task is not None:
                await self.task
            self.service.last_contact = time.monotonic() - 31
            return None

    controller = OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate"},
        worker_factory=SilentWorker,
        progress_sink=JsonlProgressSink(tmp_path / "progress.jsonl"),
    )
    report = await controller.run_async()
    assert report.budget is not None
    assert report.budget.stop_reason.startswith("The worker stopped responding")
    assert "worker_unresponsive" in {item.reason_code for item in report.evidence}
    final = [event for event in read_progress_events(tmp_path / "progress.jsonl") if isinstance(event, RampFinished)]
    assert final[-1].stop_reason == report.budget.stop_reason


async def test_single_model_concurrency_check_is_not_applicable(tmp_path: Path) -> None:
    """One offered model cannot give each lane its own model, so no thread work is offered or reserved."""
    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate"},
        worker_factory=HttpTestWorker,
    ).run_async()
    thread_checks = [item for item in report.evidence if item.capability.kind is CapabilityKind.THREADS]
    assert thread_checks and all(item.reason_code == "not_applicable" and not item.job_ids for item in thread_checks)
    assert all(item.status is EvidenceStatus.UNKNOWN and not item.attempted for item in thread_checks)
    assert report.suggested_bridge_data.max_threads == 1
    assert report.budget is not None and report.budget.max_jobs < 20


async def test_two_model_concurrency_check_loads_both_models_and_alternates_jobs(tmp_path: Path) -> None:
    """Each lane gets its own model: both are configured for the card and the offered jobs alternate."""
    seen: list[list[str]] = []

    class RecordingWorker(HttpTestWorker):
        async def configure(self, case: AssessmentCase, deadline: float) -> str | None:
            if case.profile.threads > 1:
                seen.append([job.model for job in case.scenario().expand_image_jobs()])
            return await super().configure(case, deadline)

    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        only_probe="sd15-threads-2",
        worker_factory=RecordingWorker,
    ).run_async()
    assert seen and seen[0] == ["Deliberate", "AlbedoBase XL (SDXL)", "Deliberate", "AlbedoBase XL (SDXL)"]
    confirmed = [item for item in report.evidence if item.combined_validation and item.profile.threads == 2]
    assert confirmed and all(item.profile.companion_model == "AlbedoBase XL (SDXL)" for item in confirmed)
    assert report.suggested_bridge_data.max_threads == 2


async def test_two_lanes_finishing_the_jobs_sooner_confirm_threads_with_the_lease(tmp_path: Path) -> None:
    """The same jobs finished in half the wall time at two lanes confirm two threads under a one-slot lease."""
    from horde_worker_regen.benchmark.progress_channel import JsonlProgressSink, LevelFinished, read_progress_events

    lane_counts: list[tuple[int, int | None]] = []

    class RecordingWorker(HttpTestWorker):
        async def configure(self, case: AssessmentCase, deadline: float) -> str | None:
            if case.profile.companion_model is not None:
                lane_counts.append((case.profile.threads, case.profile.lease_slots))
            return await super().configure(case, deadline)

    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        only_probe="sd15-threads-2",
        worker_factory=RecordingWorker,
        progress_sink=JsonlProgressSink(tmp_path / "progress.jsonl"),
    ).run_async()
    # Each card's thread check runs one lane under the operator's settings first. The combined confirmations
    # run only the thread count the checks ranked.
    assert lane_counts == [(1, None), (2, 1)] * 2 + [(2, 1)] * 2
    two_thread = [item for item in report.evidence if item.profile.threads == 2 and item.attempted]
    assert two_thread and all(item.status is EvidenceStatus.MEASURED_SUCCESS for item in two_thread)
    thread_checks = [item for item in two_thread if not item.combined_validation]
    confirmations = [item for item in two_thread if item.combined_validation]
    assert len(thread_checks) == len(confirmations) == 2
    for item in confirmations:
        assert item.profile.lease_slots == 1 and len(item.job_ids) == 4
        assert item.wall_gain is None and item.wall_gain_required is None and item.jobs_per_minute is None
    for item in thread_checks:
        assert item.profile.lease_slots == 1 and item.expected_sampling_concurrency == 1
        assert item.wall_gain == pytest.approx(2.0) and item.wall_gain_required == pytest.approx(1.0)
        assert item.jobs_per_minute == pytest.approx(120.0)
        assert item.single_lane_jobs_per_minute == pytest.approx(60.0)
        assert item.wall_megapixelsteps_per_second == pytest.approx(
            2 * (item.single_lane_wall_megapixelsteps_per_second or 0.0)
        )
    suggestion = report.suggested_bridge_data
    assert suggestion.max_threads == 2
    assert suggestion.gpu_sampling_lease_enabled is True and suggestion.gpu_sampling_lease_slots == 1
    exported = YAML(typ="safe").load(suggestion.as_yaml_block())
    assert exported["gpu_sampling_lease_enabled"] is True and exported["gpu_sampling_lease_slots"] == 1
    assert "2 lanes finished 120.0 jobs per minute against 60.0 on one lane" in render_markdown(report)
    finished = [
        event
        for event in read_progress_events(tmp_path / "progress.jsonl")
        if isinstance(event, LevelFinished) and "threads" in event.level_id
    ]
    assert finished and all(
        any("Throughput ratio: 2.00 (minimum 1.00)" in reason for reason in event.reasons) for event in finished
    )


async def test_two_lanes_without_a_wall_gain_are_not_confirmed(tmp_path: Path) -> None:
    """Jobs that take twice as long at two lanes finish no sooner, so the thread count stays at one."""

    class SlowerTogetherWorker(HttpTestWorker):
        multi_lane_slowdown = 2.0

    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        only_probe="sd15-threads-2",
        worker_factory=SlowerTogetherWorker,
    ).run_async()
    thread_checks = [item for item in report.evidence if item.profile.threads == 2 and item.attempted]
    assert thread_checks
    for item in thread_checks:
        assert item.status is EvidenceStatus.UNKNOWN and item.reason_code == "wall_gain_below_noise"
        assert item.wall_gain == pytest.approx(1.0) and len(item.job_ids) == 8
        assert "2 lanes finished 60.0 jobs per minute against 60.0 on one lane" in item.reason
        assert "The throughput ratio was 1.00" in item.reason
    assert report.suggested_bridge_data.max_threads == 1
    assert report.suggested_bridge_data.gpu_sampling_lease_enabled is None


def test_case_config_writes_the_lease_candidate_and_keeps_an_operator_lease(tmp_path: Path) -> None:
    """A profile's slot count enables the lease in the check's config; None keeps the operator's settings."""
    service = _scripted_service()
    assert service.snapshot is not None
    operator = {"gpu_sampling_lease_enabled": True, "gpu_sampling_lease_slots": 3}
    worker = OrdinaryAssessmentWorker(service, tmp_path, operator)
    case = _baseline_case(threads=2)
    kept = worker._case_config(case, service.snapshot)
    assert kept["gpu_sampling_lease_enabled"] is True and kept["gpu_sampling_lease_slots"] == 3
    candidate = AssessmentCase(case.capability, case.profile.model_copy(update={"lease_slots": 1}))
    applied = worker._case_config(candidate, service.snapshot)
    assert applied["gpu_sampling_lease_enabled"] is True and applied["gpu_sampling_lease_slots"] == 1
    unleased = OrdinaryAssessmentWorker(service, tmp_path, {})._case_config(case, service.snapshot)
    assert "gpu_sampling_lease_enabled" not in unleased


async def test_lease_slots_bound_the_overlap_a_thread_check_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the operator's lease at one slot, sequential sampling is the configured behaviour and confirms two threads.

    The check uses the operator's existing lease settings, and the suggestion preserves them.
    """

    class SequentialWorker(HttpTestWorker):
        sequential_sampling = True

    _operator_lease(monkeypatch, slots=1)
    report = await OperatorAssessment(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], include_features=False, include_alchemy=False),
        out_dir=tmp_path,
        models={BenchTier.SD15: "Deliberate", BenchTier.SDXL: "AlbedoBase XL (SDXL)"},
        only_probe="sd15-threads-2",
        worker_factory=SequentialWorker,
    ).run_async()
    thread_checks = [item for item in report.evidence if item.capability.kind is CapabilityKind.THREADS]
    assert thread_checks and all(item.status is EvidenceStatus.MEASURED_SUCCESS for item in thread_checks)
    assert all(
        item.expected_sampling_concurrency == 1 and item.observed_sampling_concurrency == 1 for item in thread_checks
    )
    assert all("lanes staged" in item.reason for item in thread_checks)
    assert all(item.profile.lease_slots is None for item in thread_checks)
    assert report.suggested_bridge_data.max_threads == 2
    assert report.suggested_bridge_data.gpu_sampling_lease_enabled is None
    assert report.suggested_bridge_data.gpu_sampling_lease_slots is None


def test_storage_note_names_a_page_file_on_the_model_drive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A page file on the model drive is a host condition the operator should know about."""
    from horde_worker_regen.benchmark.capabilities.result import MachineInfo
    from horde_worker_regen.process_management.scheduling import host_commit

    monkeypatch.setattr(host_commit, "page_file_drive_roots", lambda: frozenset({"t:\\"}))
    controller = OperatorAssessment(catalog_options=CatalogOptions(tiers=[BenchTier.SD15]), out_dir=tmp_path)
    report = CapabilityReport(
        operator_assessment=True,
        machine=MachineInfo(storage={"T:\\models": False, "D:\\more": False}),
    )
    notes = controller._storage_notes(report)
    assert any(note.startswith("A page file shares drive T:") for note in notes)
    assert not any("drive D:" in note for note in notes)


async def test_withheld_service_offer_returns_without_waiting_for_job_timeout(tmp_path: Path) -> None:
    """An explicitly withheld form remains unavailable instead of consuming the inactivity timeout."""

    class WithheldWorker(HttpTestWorker):
        async def _consume(self, case: AssessmentCase) -> None:
            image = AssessmentCase(case.capability, case.profile.model_copy(update={"alchemy_forms": []}))
            await super()._consume(image)
            async with aiohttp.ClientSession() as client:
                for _ in range(2):
                    async with client.post(
                        f"{self.service.url}/api/v2/interrogate/pop",
                        json={"name": "test", "forms": ["nsfw"], "priority_usernames": [], "amount": 1},
                    ) as response:
                        assert response.status == 200

    async with LocalHorde() as service:
        worker = WithheldWorker(service, tmp_path / "worker", {})
        await worker.start()
        controller = OperatorAssessment(
            catalog_options=CatalogOptions(tiers=[BenchTier.SD15]), out_dir=tmp_path, max_jobs=4
        )
        case = AssessmentCase(
            Capability(tier=BenchTier.SD15, kind=CapabilityKind.ALCHEMY_CLIP),
            OperatingProfile(device_index=0, tier=BenchTier.SD15, model_name="Deliberate", alchemy_forms=["caption"]),
        )
        try:
            result = await controller._execute(service, worker, case, time.monotonic() + 10)
            assert result.status is EvidenceStatus.UNKNOWN and result.reason_code == "unavailable"
            assert "not offering caption" in result.reason
        finally:
            await worker.close()
