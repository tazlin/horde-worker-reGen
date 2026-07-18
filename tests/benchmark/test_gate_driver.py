"""Tests for the disagg gate driver: mechanism-metric derivation, ABBA orchestration, and scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from hordelib.metrics import JobPhaseMetrics, ModelLoadEvent, SamplingStats

from horde_worker_regen.benchmark import gate_driver
from horde_worker_regen.benchmark.disagg_mixes import DisaggGateMix
from horde_worker_regen.benchmark.gate_driver import (
    GatePreconditionError,
    GateRunConfig,
    GateRunResult,
    GateScoringUnavailableError,
    GateVariant,
    _abba_variant_order,
    _coerce_scalar,
    _disk_to_ram_by_stage,
    _gate_lock,
    _parse_overrides,
    _resolve_kudos_ckpt,
    _score_kudos,
    _stage_latency_percentiles,
    run_gate_rung,
)
from horde_worker_regen.process_management.ipc.messages import PipelineStageTag
from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord, RunMetricsSnapshot


def _phase_metrics(
    *,
    disk_to_ram_seconds: list[float],
    ram_to_vram_seconds: list[float] | None = None,
    sampling_seconds: float | None = None,
    phase_seconds: dict[str, float] | None = None,
) -> JobPhaseMetrics:
    """Build a phase-metrics fixture with the given model-load and timing shape."""
    loads = [
        ModelLoadEvent(model_name="m", phase="disk_to_ram", duration_seconds=seconds, timestamp=0.0)
        for seconds in disk_to_ram_seconds
    ]
    loads += [
        ModelLoadEvent(model_name="m", phase="ram_to_vram", duration_seconds=seconds, timestamp=0.0)
        for seconds in (ram_to_vram_seconds or [])
    ]
    sampling = None
    if sampling_seconds is not None:
        sampling = SamplingStats(
            steps_completed=30,
            total_steps=30,
            duration_seconds=sampling_seconds,
            iterations_per_second=30.0 / sampling_seconds if sampling_seconds else 0.0,
        )
    return JobPhaseMetrics(model_loads=loads, sampling=sampling, phase_seconds=phase_seconds or {})


def _make_snapshot(
    *,
    jobs: list[JobMetricsRecord],
    stage_metrics: list[JobMetricsRecord],
    governor_saturation_events: int = 0,
) -> RunMetricsSnapshot:
    """A minimal run-metrics snapshot carrying the job/stage records under test."""
    return RunMetricsSnapshot(
        jobs=jobs,
        stage_metrics=stage_metrics,
        downloads=[],
        vram_used_high_water_mb_per_process={},
        ram_used_high_water_mb_per_process={},
        disk_min_free_bytes={},
        num_process_recoveries=0,
        num_job_slowdowns=0,
        time_spent_no_jobs_available=0.0,
        process_crash_events=[],
        governor_saturation_events=governor_saturation_events,
    )


def _mechanism_snapshot() -> RunMetricsSnapshot:
    """A fixture spanning stage-tagged and untagged records with a mix of load phases and timings."""
    stage_metrics = [
        JobMetricsRecord(
            job_id="a",
            stage=PipelineStageTag.TEXT_ENCODE,
            phase_metrics=_phase_metrics(disk_to_ram_seconds=[1.0], phase_seconds={"text_encode": 0.5}),
        ),
        JobMetricsRecord(
            job_id="a",
            stage=PipelineStageTag.SAMPLE,
            phase_metrics=_phase_metrics(
                disk_to_ram_seconds=[2.0],
                ram_to_vram_seconds=[0.3],
                sampling_seconds=8.0,
            ),
        ),
        JobMetricsRecord(
            job_id="a",
            stage=PipelineStageTag.VAE_DECODE,
            phase_metrics=_phase_metrics(disk_to_ram_seconds=[], phase_seconds={"vae_decode": 0.4}),
        ),
    ]
    jobs = [
        JobMetricsRecord(
            job_id="a",
            e2e_seconds=12.0,
            phase_metrics=_phase_metrics(disk_to_ram_seconds=[3.0]),
        ),
        JobMetricsRecord(job_id="b", faulted=True, e2e_seconds=None, phase_metrics=None),
    ]
    return _make_snapshot(jobs=jobs, stage_metrics=stage_metrics, governor_saturation_events=4)


class TestDiskToRamDerivation:
    """disk->RAM reloads are counted and timed per stage, whole-job records land under `whole_job`."""

    def test_counts_only_disk_to_ram_by_stage(self) -> None:
        """Only disk->RAM loads are counted; ram->vram is excluded; every stage key is present."""
        counts, seconds = _disk_to_ram_by_stage(_mechanism_snapshot())
        expected_keys = {stage.value for stage in PipelineStageTag} | {"whole_job"}
        assert set(counts) == expected_keys
        assert counts[PipelineStageTag.TEXT_ENCODE.value] == 1
        assert counts[PipelineStageTag.SAMPLE.value] == 1  # the ram->vram load is not counted
        assert counts[PipelineStageTag.VAE_DECODE.value] == 0
        assert counts[PipelineStageTag.VAE_ENCODE.value] == 0
        assert counts["whole_job"] == 1

    def test_sums_disk_to_ram_seconds_by_stage(self) -> None:
        """The seconds map mirrors the counts, summing only the disk->RAM durations."""
        _counts, seconds = _disk_to_ram_by_stage(_mechanism_snapshot())
        assert seconds[PipelineStageTag.TEXT_ENCODE.value] == pytest.approx(1.0)
        assert seconds[PipelineStageTag.SAMPLE.value] == pytest.approx(2.0)
        assert seconds["whole_job"] == pytest.approx(3.0)


class TestStageLatencyDerivation:
    """Stage latency percentiles come from recorded durations; untimed stages are omitted."""

    def test_p50_uses_recorded_compute_time_and_omits_untimed_stages(self) -> None:
        """Each stage's compute wall-time is summed from sampling and named phases; e2e backs whole_job."""
        p50, p95 = _stage_latency_percentiles(_mechanism_snapshot())
        assert p50[PipelineStageTag.TEXT_ENCODE.value] == pytest.approx(0.5)
        assert p50[PipelineStageTag.SAMPLE.value] == pytest.approx(8.0)
        assert p50[PipelineStageTag.VAE_DECODE.value] == pytest.approx(0.4)
        assert p50["whole_job"] == pytest.approx(12.0)
        # VAE_ENCODE produced no record, so it is omitted rather than reported as zero.
        assert PipelineStageTag.VAE_ENCODE.value not in p50
        assert PipelineStageTag.VAE_ENCODE.value not in p95


class TestKudosScoring:
    """Scoring is optional unless required, and lazily loads the checkpoint."""

    def test_no_ckpt_and_not_required_returns_none(self) -> None:
        """Without a checkpoint and without requiring it, scoring yields None rather than failing."""
        snapshot = _make_snapshot(jobs=[], stage_metrics=[])
        assert _score_kudos(snapshot, ckpt_path=None, require_kudos=False) is None

    def test_required_without_ckpt_raises(self) -> None:
        """Requiring kudos without a checkpoint fails with a clear error."""
        snapshot = _make_snapshot(jobs=[], stage_metrics=[])
        with pytest.raises(GateScoringUnavailableError, match="AI_HORDE_KUDOS_MODEL_CKPT"):
            _score_kudos(snapshot, ckpt_path=None, require_kudos=True)

    def test_scoring_path_uses_the_checkpoint(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """With a checkpoint, scoring delegates to the kudos module (mocked, so no real ckpt is needed)."""
        import horde_worker_regen.analysis.kudos_score as kudos_module

        class _FakeScorer:
            def __init__(self, path: Path) -> None:
                self.path = path

        class _FakeReport:
            kudos_per_hour = 4242.0

        monkeypatch.setattr(kudos_module, "KudosModelScorer", _FakeScorer)
        monkeypatch.setattr(kudos_module, "score_session", lambda records, scorer: _FakeReport())

        ckpt = tmp_path / "kudos.ckpt"
        ckpt.write_text("not a real checkpoint")
        snapshot = _make_snapshot(jobs=[JobMetricsRecord(job_id="a")], stage_metrics=[])
        assert _score_kudos(snapshot, ckpt_path=ckpt, require_kudos=True) == pytest.approx(4242.0)


class TestCheckpointResolution:
    """The checkpoint resolves from the explicit path or the environment, else None."""

    def test_explicit_file_wins(self, tmp_path: Path) -> None:
        """An explicit path that is a file is returned."""
        ckpt = tmp_path / "explicit.ckpt"
        ckpt.write_text("x")
        assert _resolve_kudos_ckpt(ckpt) == ckpt

    def test_env_var_used_when_no_explicit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The env var supplies the path when no explicit one is given."""
        ckpt = tmp_path / "env.ckpt"
        ckpt.write_text("x")
        monkeypatch.setenv("AI_HORDE_KUDOS_MODEL_CKPT", str(ckpt))
        assert _resolve_kudos_ckpt(None) == ckpt

    def test_missing_file_resolves_to_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A path that is not a file resolves to None."""
        monkeypatch.delenv("AI_HORDE_KUDOS_MODEL_CKPT", raising=False)
        assert _resolve_kudos_ckpt(tmp_path / "missing.ckpt") is None


class TestOverrideParsing:
    """CLI overrides parse conservatively into typed scalars."""

    def test_typed_coercion(self) -> None:
        """Bools, none, ints, floats, and strings are each recognised."""
        assert _coerce_scalar("true") is True
        assert _coerce_scalar("False") is False
        assert _coerce_scalar("none") is None
        assert _coerce_scalar("7") == 7
        assert _coerce_scalar("1.5") == pytest.approx(1.5)
        assert _coerce_scalar("hello") == "hello"

    def test_parse_key_value_pairs(self) -> None:
        """Well-formed pairs become a typed map."""
        parsed = _parse_overrides(["max_threads=2", "gpu_sampling_lease_enabled=true", "note=hi"])
        assert parsed == {"max_threads": 2, "gpu_sampling_lease_enabled": True, "note": "hi"}

    def test_malformed_pair_rejected(self) -> None:
        """A value without an `=` is rejected."""
        with pytest.raises(argparse.ArgumentTypeError, match="key=value"):
            _parse_overrides(["not_a_pair"])


class TestAbbaOrdering:
    """The A/B ladder alternates variants in ABBA order on identical seeds per rung."""

    def test_abba_order(self) -> None:
        """The per-rung order is A, B, B, A."""
        variant_a = GateVariant(label="A")
        variant_b = GateVariant(label="B")
        order = _abba_variant_order(variant_a, variant_b)
        assert [variant.label for variant in order] == ["A", "B", "B", "A"]

    def test_ladder_runs_abba_per_rung_on_identical_seed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The ladder drives each rung ABBA on one seed and writes a round-trippable report."""
        calls: list[tuple[float, int, str, int]] = []

        def _fake_run_gate_rung(config: GateRunConfig, *, order_index: int = 0) -> GateRunResult:
            calls.append((config.rung_seconds, order_index, config.variant.label, config.seed))
            return GateRunResult(
                mix=config.mix,
                rung_seconds=config.rung_seconds,
                seed=config.seed,
                variant_label=config.variant.label,
                order_index=order_index,
                mode=config.mode,
                jobs_completed=3,
            )

        monkeypatch.setattr(gate_driver, "run_gate_rung", _fake_run_gate_rung)

        results = gate_driver.run_gate_ladder(
            mix=DisaggGateMix.CHURN_DETERMINISTIC,
            output_dir=tmp_path,
            variant_a=GateVariant(label="baseline"),
            variant_b=GateVariant(label="candidate"),
            seed=99,
            rungs=(90.0, 300.0),
        )

        # Two rungs, ABBA each: labels alternate A,B,B,A and every run in a rung shares the seed.
        for rung in (90.0, 300.0):
            rung_calls = [call for call in calls if call[0] == rung]
            assert [call[2] for call in rung_calls] == ["baseline", "candidate", "candidate", "baseline"]
            assert {call[3] for call in rung_calls} == {99}

        report_path = tmp_path / "disagg_gate_report.json"
        assert report_path.is_file()
        loaded = json.loads(report_path.read_text())
        restored = [GateRunResult(**item) for item in loaded]
        assert restored == results


class TestGuards:
    """Real-mode preconditions and the output-dir lock refuse to start unsafely."""

    def test_real_mode_refused_while_pytest_running(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A real-mode rung is refused when a pytest process is detected."""
        monkeypatch.setattr(gate_driver, "_pytest_is_running", lambda: True)
        config = GateRunConfig(
            mix=DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=90.0,
            seed=1,
            variant=GateVariant(label="A"),
            output_dir=tmp_path,
            mode="real",
        )
        with pytest.raises(GatePreconditionError, match="pytest"):
            run_gate_rung(config)

    def test_lock_is_exclusive(self, tmp_path: Path) -> None:
        """A second lock acquisition fails while the first is held, and the lock clears on exit."""
        held = _gate_lock(tmp_path)
        held.__enter__()
        try:
            with pytest.raises(GatePreconditionError, match="already in progress"):
                _gate_lock(tmp_path).__enter__()
        finally:
            held.__exit__(None, None, None)
        assert not (tmp_path / "disagg_gate.lock").exists()


@pytest.mark.slow
def test_fake_mode_gate_rung_smoke(tmp_path: Path) -> None:
    """A short fake-mode rung runs through the real harness and materializes a scored result + artifact."""
    config = GateRunConfig(
        mix=DisaggGateMix.CHURN_DETERMINISTIC,
        rung_seconds=10.0,
        seed=1,
        variant=GateVariant(label="A"),
        output_dir=tmp_path,
        mode="fake",
        jobs_per_minute_estimate=12.0,
        # A high img2img share ensures the source-image-bearing path actually runs in the smoke.
        img2img_fraction=0.5,
    )
    result = run_gate_rung(config)

    assert result.jobs_completed > 0
    assert result.faults == 0  # img2img jobs carry a source image and must not fault or degrade
    assert result.kudos_per_hour is None  # no checkpoint supplied in the smoke
    assert result.job_records_path is not None
    records_path = Path(result.job_records_path)
    assert records_path.is_file()
    # The dumped job records are the rescorable artifact; there is one line per finalized job.
    dumped_lines = [line for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(dumped_lines) >= result.jobs_completed
