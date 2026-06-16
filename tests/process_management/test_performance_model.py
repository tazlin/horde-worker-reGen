"""Unit tests for the expected-time-to-complete model (signatures, seed, self-calibration, persistence)."""

from __future__ import annotations

import json
from pathlib import Path

from horde_worker_regen.process_management._canned_scenarios import make_canned_job
from horde_worker_regen.process_management.performance_model import (
    BENCHMARK_BASELINE_STEPS,
    PERF_MODEL_SCHEMA_VERSION,
    BatchBucket,
    PerformanceModel,
    ResolutionBucket,
    StepsBucket,
    baseline_signature,
    load_seed_its_by_signature,
    signature_from_job,
)

_SD15_BASELINE = "stable_diffusion_1"


def test_signature_buckets_a_plain_job() -> None:
    """A 512x512 / 30-step / batch-1 job lands in the expected buckets with no feature flags."""
    job = make_canned_job(width=512, height=512, ddim_steps=30, n_iter=1)
    signature = signature_from_job(job, _SD15_BASELINE)

    assert signature is not None
    assert signature.baseline == _SD15_BASELINE
    assert signature.resolution_bucket == ResolutionBucket.TINY
    assert signature.steps_bucket == StepsBucket.MEDIUM
    assert signature.batch_bucket == BatchBucket.SINGLE
    assert not signature.has_controlnet
    assert not signature.has_hires_fix
    assert signature.total_sampling_iterations == 30
    assert signature.is_baseline_like


def test_signature_none_for_unknown_baseline() -> None:
    """A job cannot be characterized without a known baseline."""
    job = make_canned_job(ddim_steps=30)
    assert signature_from_job(job, None) is None


def test_controlnet_and_hires_change_key_and_iterations() -> None:
    """Controlnet/hires flags both partition the calibration key; hires also doubles the iteration count."""
    plain = signature_from_job(make_canned_job(ddim_steps=30), _SD15_BASELINE)
    controlnet = signature_from_job(make_canned_job(ddim_steps=30, control_type="canny"), _SD15_BASELINE)
    hires = signature_from_job(make_canned_job(ddim_steps=30, hires_fix=True), _SD15_BASELINE)

    assert plain is not None and controlnet is not None and hires is not None
    assert plain.key != controlnet.key != hires.key
    assert plain.key != hires.key
    assert controlnet.has_controlnet
    assert hires.total_sampling_iterations == 60
    assert not hires.is_baseline_like


def test_batch_does_not_multiply_iterations_but_changes_key() -> None:
    """Batching lowers per-step it/s rather than adding iterations, so it is keyed separately."""
    single = signature_from_job(make_canned_job(ddim_steps=30, n_iter=1), _SD15_BASELINE)
    batch4 = signature_from_job(make_canned_job(ddim_steps=30, n_iter=4), _SD15_BASELINE)

    assert single is not None and batch4 is not None
    assert batch4.batch_bucket == BatchBucket.SMALL
    assert batch4.total_sampling_iterations == 30
    assert single.key != batch4.key
    assert not batch4.is_baseline_like


def test_seed_answers_baseline_like_only_until_calibrated() -> None:
    """The seed answers for the baseline signature it was keyed at, and nothing else, until calibration."""
    baseline = baseline_signature(baseline=_SD15_BASELINE, resolution=512)
    model = PerformanceModel(seed_its_by_signature={baseline.key: 8.0})

    assert model.expected_its(baseline) == 8.0
    assert model.expected_sampling_seconds(baseline) == BENCHMARK_BASELINE_STEPS / 8.0

    controlnet = signature_from_job(make_canned_job(ddim_steps=30, control_type="canny"), _SD15_BASELINE)
    assert controlnet is not None
    assert model.expected_its(controlnet) is None
    assert model.expected_sampling_seconds(controlnet) is None


def test_calibration_overrides_seed_after_min_samples() -> None:
    """Once a signature has enough observations its learned median supersedes the seed."""
    baseline = baseline_signature(baseline=_SD15_BASELINE, resolution=512)
    model = PerformanceModel(seed_its_by_signature={baseline.key: 8.0}, min_samples=3)

    model.observe(baseline, 4.0)
    model.observe(baseline, 4.0)
    assert model.expected_its(baseline) == 8.0  # still seed: not enough samples yet

    model.observe(baseline, 4.0)
    assert model.expected_its(baseline) == 4.0  # learned median now trusted
    assert model.sample_count(baseline) == 3


def test_observation_window_is_bounded() -> None:
    """The per-signature rolling window keeps only the most recent samples."""
    baseline = baseline_signature(baseline=_SD15_BASELINE, resolution=512)
    model = PerformanceModel(max_samples_per_signature=3, min_samples=1)

    for rate in (1.0, 2.0, 3.0, 4.0, 5.0):
        model.observe(baseline, rate)

    assert model.sample_count(baseline) == 3
    assert model.expected_its(baseline) == 4.0  # median of the last three: 3,4,5


def test_non_positive_observation_is_ignored() -> None:
    """A zero/negative it/s sample (e.g. a job with no measured sampling) is not folded in."""
    baseline = baseline_signature(baseline=_SD15_BASELINE, resolution=512)
    model = PerformanceModel()
    model.observe(baseline, 0.0)
    model.observe(baseline, -1.0)
    assert model.sample_count(baseline) == 0


def test_persistence_round_trip(tmp_path: Path) -> None:
    """Saved calibration reloads on the next construction at the same path."""
    path = tmp_path / "perf_model.json"
    baseline = baseline_signature(baseline=_SD15_BASELINE, resolution=512)

    model = PerformanceModel(path=path, min_samples=2)
    model.observe(baseline, 6.0)
    model.observe(baseline, 8.0)
    model.save()

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == PERF_MODEL_SCHEMA_VERSION

    reloaded = PerformanceModel(path=path, min_samples=2)
    assert reloaded.sample_count(baseline) == 2
    assert reloaded.expected_its(baseline) == 7.0


def test_in_memory_model_never_writes(tmp_path: Path) -> None:
    """With no path the model keeps everything in memory and creates no file."""
    baseline = baseline_signature(baseline=_SD15_BASELINE, resolution=512)
    model = PerformanceModel()
    model.observe(baseline, 5.0)
    model.save()
    assert not any(tmp_path.iterdir())


def test_save_degrades_when_path_unwritable(tmp_path: Path) -> None:
    """A perf-model write failure degrades to memory and never raises."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")

    baseline = baseline_signature(baseline=_SD15_BASELINE, resolution=512)
    model = PerformanceModel(path=blocker / "nested" / "perf_model.json")
    model.observe(baseline, 5.0)
    model.save()  # must not raise

    assert model._file_disabled is True
    assert model.sample_count(baseline) == 1


def test_load_seed_from_benchmark_report(tmp_path: Path) -> None:
    """A written report.json yields a seed keyed by each tier's exact baseline signature."""
    report = {
        "tier_baselines_its": {"sd15": 9.0, "sdxl": 3.5},
    }
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")

    seed = load_seed_its_by_signature(tmp_path)

    sd15 = baseline_signature(baseline="stable_diffusion_1", resolution=512)
    sdxl = baseline_signature(baseline="stable_diffusion_xl", resolution=1024)
    assert seed[sd15.key] == 9.0
    assert seed[sdxl.key] == 3.5


def test_load_seed_absent_report_is_empty(tmp_path: Path) -> None:
    """A missing report.json seeds nothing rather than raising."""
    assert load_seed_its_by_signature(tmp_path) == {}


def test_metrics_then_finalize_calibrates() -> None:
    """A per-job metrics message followed by finalization teaches the model that job's it/s."""
    from unittest.mock import Mock

    from hordelib.metrics import JobPhaseMetrics, SamplingStats

    from horde_worker_regen.process_management.job_tracker import JobStage, TrackedJob
    from horde_worker_regen.process_management.messages import HordeJobMetricsMessage

    job = make_canned_job(width=512, height=512, ddim_steps=30, n_iter=1)
    model = PerformanceModel(baseline_resolver=lambda _name: _SD15_BASELINE, min_samples=1)

    sampling = SamplingStats(steps_completed=30, total_steps=30, duration_seconds=5.0, iterations_per_second=6.0)
    message = HordeJobMetricsMessage(
        process_id=0,
        process_launch_identifier=0,
        info="",
        job_id=str(job.id_),
        phase_metrics=JobPhaseMetrics(sampling=sampling),
    )
    model.on_job_metrics(message)

    tracked = TrackedJob(job_id=job.id_, sdk_api_job_info=job, stage=JobStage.PENDING_SUBMIT)
    model.on_job_finalized(tracked, Mock())

    signature = signature_from_job(job, _SD15_BASELINE)
    assert signature is not None
    assert model.expected_its(signature) == 6.0


def test_alchemy_metrics_are_ignored() -> None:
    """Alchemy forms have no sampling signature and must not be cached for calibration."""
    from hordelib.metrics import JobPhaseMetrics, SamplingStats

    from horde_worker_regen.process_management.messages import HordeJobMetricsMessage

    model = PerformanceModel()
    sampling = SamplingStats(steps_completed=10, total_steps=10, duration_seconds=2.0, iterations_per_second=5.0)
    message = HordeJobMetricsMessage(
        process_id=0,
        process_launch_identifier=0,
        info="",
        job_id="form-1",
        is_alchemy=True,
        phase_metrics=JobPhaseMetrics(sampling=sampling),
    )
    model.on_job_metrics(message)
    assert model._its_by_job_id == {}


def test_forget_job_drops_cached_rate() -> None:
    """A job that will not finalize can have its cached it/s discarded to bound the cache."""
    from hordelib.metrics import JobPhaseMetrics, SamplingStats

    from horde_worker_regen.process_management.messages import HordeJobMetricsMessage

    model = PerformanceModel()
    sampling = SamplingStats(steps_completed=30, total_steps=30, duration_seconds=5.0, iterations_per_second=6.0)
    message = HordeJobMetricsMessage(
        process_id=0,
        process_launch_identifier=0,
        info="",
        job_id="job-9",
        phase_metrics=JobPhaseMetrics(sampling=sampling),
    )
    model.on_job_metrics(message)
    assert "job-9" in model._its_by_job_id

    model.forget_job("job-9")
    assert "job-9" not in model._its_by_job_id
