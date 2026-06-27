"""Tests for stats-backed session duty-cycle analysis."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

from horde_worker_regen.analysis.duty_log_report import main as duty_report_main
from horde_worker_regen.analysis.session_duty import DutyLossKind, analyze_stats_sessions


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def _sample(timestamp: float, **overrides: object) -> dict[str, object]:
    sample: dict[str, object] = {
        "timestamp": timestamp,
        "gpu_duty_percent": 0.0,
        "gpu_busy_fraction": 0.0,
        "jobs_pending_inference": 0,
        "jobs_in_progress": 0,
        "jobs_pending_safety_check": 0,
        "jobs_being_safety_checked": 0,
        "jobs_pending_submit": 0,
        "pending_megapixelsteps": 0,
        "time_spent_no_jobs_available": 0.0,
        "process_state_summary": "inf#0=WAITING_FOR_JOB",
        "churn_counts": {"model_swap": 0, "vram_eviction": 0},
    }
    sample.update(overrides)
    return {"event": "stats_sample", "sample": sample}


def _job(**overrides: object) -> dict[str, object]:
    job: dict[str, object] = {
        "job_id": "job-1",
        "faulted": False,
        "is_alchemy": False,
        "queue_wait_seconds": 2.0,
        "safety_seconds": 0.5,
        "stage_timestamps": {"PENDING_SUBMIT": 120.0, "FINALIZED": 121.0},
        "model_name": "model-a",
        "phase_metrics": {
            "model_loads": [
                {"phase": "disk_to_ram", "duration_seconds": 3.0},
                {"phase": "ram_to_vram", "duration_seconds": 1.5},
            ],
            "sampling": {"duration_seconds": 8.0, "iterations_per_second": 4.0},
            "phase_seconds": {"vae_decode": 0.7, "clip_encode": 0.2, "model_unload": 0.3},
        },
    }
    job.update(overrides)
    return {"event": "job_completed", "job": job, "baseline": "stable_diffusion_xl"}


class TestStatsIngestion:
    """Reading retained stats JSONL sessions."""

    def test_reads_uncompressed_and_gzipped_rotations(self, tmp_path: Path) -> None:
        """Rotated JSONL and JSONL.GZ files are grouped into one session."""
        stats_dir = tmp_path / "stats"
        stats_dir.mkdir()
        first = stats_dir / "stats-v1.0.0-20260620-010203-000.jsonl"
        second = stats_dir / "stats-v1.0.0-20260620-010203-001.jsonl.gz"
        _write_jsonl(first, [_sample(100.0), _job()])
        with gzip.open(second, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(_sample(101.0, gpu_duty_percent=95.0, gpu_busy_fraction=1.0)) + "\n")
            handle.write(json.dumps({"event": "future_event", "value": 1}) + "\n")

        reports = analyze_stats_sessions(stats_dir)

        assert len(reports) == 1
        report = reports[0]
        assert report.sample_count == 2
        assert report.completed_jobs == 1
        assert report.unknown_event_count == 1
        assert report.model_breakdown == {"model-a": 1}
        assert report.baseline_breakdown == {"stable_diffusion_xl": 1}
        assert report.per_job_phase_medians["model_load"] == 3.0
        assert report.per_job_phase_medians["vram_transfer"] == 1.5
        assert report.inference_queue_wait is not None
        assert report.inference_queue_wait.total_seconds == 2.0
        assert report.inference_queue_wait.top_models[0].model_name == "model-a"
        assert "scheduler_wait" not in report.per_job_phase_medians

    def test_excludes_warmup_samples_before_first_inference(self, tmp_path: Path) -> None:
        """Samples before the first inference (cold-boot model loading) are dropped from the duty figure."""
        stats_dir = tmp_path / "stats"
        stats_dir.mkdir()
        path = stats_dir / "stats-v1.0.0-20260620-010203-000.jsonl"
        _write_jsonl(
            path,
            [
                _sample(100.0, gpu_duty_percent=0.0, gpu_busy_fraction=0.0),  # boot: weights still loading
                _sample(105.0, gpu_duty_percent=0.0, gpu_busy_fraction=0.0),  # boot
                _sample(110.0, gpu_duty_percent=90.0, gpu_busy_fraction=1.0),  # after first inference
                _sample(115.0, gpu_duty_percent=90.0, gpu_busy_fraction=1.0),
                _job(stage_timestamps={"INFERENCE_IN_PROGRESS": 108.0, "FINALIZED": 116.0}),
            ],
        )

        report = analyze_stats_sessions(stats_dir)[0]

        # Only the two post-inference samples count; the cold-boot zeros are excluded from the mean.
        assert report.sample_count == 2
        assert report.mean_gpu_duty_percent == 90.0
        assert report.start_timestamp == 110.0


class TestAttribution:
    """Idle and partial-utilization buckets from samples and job phases."""

    def test_no_jobs_available_maps_to_demand_limited_idle(self, tmp_path: Path) -> None:
        """No-work samples become demand-limited idle loss."""
        stats_dir = tmp_path / "stats"
        stats_dir.mkdir()
        _write_jsonl(
            stats_dir / "stats-v1.0.0-20260620-010203-000.jsonl",
            [
                _sample(100.0, last_pop_no_jobs_available=True, time_spent_no_jobs_available=5.0),
                _sample(110.0, last_pop_no_jobs_available=True, time_spent_no_jobs_available=15.0),
            ],
        )

        report = analyze_stats_sessions(stats_dir)[0]

        bucket = next(item for item in report.buckets if item.kind == DutyLossKind.DEMAND_LIMITED)
        assert bucket.idle_seconds > 0

    def test_pending_jobs_with_idle_gpu_maps_to_scheduler_wait(self, tmp_path: Path) -> None:
        """Queued inference work with an idle GPU is scheduler wait."""
        stats_dir = tmp_path / "stats"
        stats_dir.mkdir()
        _write_jsonl(
            stats_dir / "stats-v1.0.0-20260620-010203-000.jsonl",
            [
                _sample(100.0, jobs_pending_inference=2, pending_megapixelsteps=20),
                _sample(110.0, jobs_pending_inference=2, pending_megapixelsteps=20),
            ],
        )

        report = analyze_stats_sessions(stats_dir)[0]

        bucket = next(item for item in report.buckets if item.kind == DutyLossKind.SCHEDULER_WAIT)
        assert bucket.idle_seconds > 0

    def test_dispatch_gap_counts_only_queued_time_without_active_inference(self, tmp_path: Path) -> None:
        """Queued standby time behind an active inference is not counted as a dispatch gap."""
        stats_dir = tmp_path / "stats"
        stats_dir.mkdir()
        _write_jsonl(
            stats_dir / "stats-v1.0.0-20260620-010203-000.jsonl",
            [
                _sample(
                    100.0, jobs_pending_inference=1, jobs_in_progress=0, process_state_summary="inf#0=PRELOADING_MODEL"
                ),
                _sample(110.0, jobs_pending_inference=1, jobs_in_progress=1, process_state_summary="inf#0=INFERENCE"),
                _sample(120.0, jobs_pending_inference=0, jobs_in_progress=1, process_state_summary="inf#0=INFERENCE"),
            ],
        )

        report = analyze_stats_sessions(stats_dir)[0]

        assert report.inference_dispatch_gap is not None
        assert report.inference_dispatch_gap.queued_seconds == 20.0
        assert report.inference_dispatch_gap.no_active_inference_seconds == 10.0
        assert report.inference_dispatch_gap.no_active_inference_fraction == 0.5
        assert report.inference_dispatch_gap.top_states == {"inf#0=PRELOADING_MODEL": 10.0}

    def test_busy_but_low_mean_duty_maps_to_partial_loss(self, tmp_path: Path) -> None:
        """High busy fraction with low mean duty becomes partial loss."""
        stats_dir = tmp_path / "stats"
        stats_dir.mkdir()
        _write_jsonl(
            stats_dir / "stats-v1.0.0-20260620-010203-000.jsonl",
            [
                _sample(
                    100.0,
                    gpu_duty_percent=45.0,
                    gpu_busy_fraction=1.0,
                    jobs_in_progress=1,
                    process_state_summary="inf#0=INFERENCE_STARTING model load",
                ),
                _sample(
                    110.0,
                    gpu_duty_percent=45.0,
                    gpu_busy_fraction=1.0,
                    jobs_in_progress=1,
                    process_state_summary="inf#0=INFERENCE_STARTING model load",
                ),
            ],
        )

        report = analyze_stats_sessions(stats_dir)[0]

        bucket = next(item for item in report.buckets if item.kind == DutyLossKind.MODEL_LOAD)
        assert bucket.partial_utilization_seconds > 0
        assert bucket.idle_seconds == 0

    def test_missing_fields_degrade_to_unknown(self, tmp_path: Path) -> None:
        """Sparse legacy samples are counted as unknown, not dropped."""
        stats_dir = tmp_path / "stats"
        stats_dir.mkdir()
        _write_jsonl(
            stats_dir / "stats-v1.0.0-20260620-010203-000.jsonl",
            [
                {"event": "stats_sample", "sample": {"timestamp": 100.0}},
                {"event": "stats_sample", "sample": {"timestamp": 110.0}},
            ],
        )

        report = analyze_stats_sessions(stats_dir)[0]

        bucket = next(item for item in report.buckets if item.kind == DutyLossKind.UNKNOWN)
        assert bucket.total_seconds > 0


class TestDutyReportCli:
    """The horde-duty-report CLI prefers stats and keeps log fallback."""

    def test_json_uses_stats_schema(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """JSON output uses the stats-backed schema when stats are supplied."""
        stats_dir = tmp_path / "stats"
        stats_dir.mkdir()
        _write_jsonl(stats_dir / "stats-v1.0.0-20260620-010203-000.jsonl", [_sample(100.0), _sample(101.0)])
        monkeypatch.setattr(sys, "argv", ["horde-duty-report", "--stats", str(stats_dir), "--json"])

        duty_report_main()

        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["session_id"] == "20260620-010203 v1.0.0"
        assert "buckets" in payload[0]
        assert "inference_queue_wait" in payload[0]
        assert "inference_dispatch_gap" in payload[0]

    def test_text_report_includes_inference_queue_wait(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Text output calls popped-job scheduler wait out explicitly."""
        stats_dir = tmp_path / "stats"
        stats_dir.mkdir()
        _write_jsonl(
            stats_dir / "stats-v1.0.0-20260620-010203-000.jsonl",
            [_sample(100.0), _sample(101.0), _job()],
        )
        monkeypatch.setattr(sys, "argv", ["horde-duty-report", "--stats", str(stats_dir)])

        duty_report_main()

        output = capsys.readouterr().out
        assert "inference queue wait:" in output
        assert "inference dispatch gap:" in output

    def test_log_only_fallback_preserves_existing_parser(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Explicit log analysis still uses the existing epoch parser."""
        log = tmp_path / "bridge.log"
        log_lines = [
            (
                "2026-06-19 18:21:00.000 | DEBUG | "
                "horde_worker_regen.process_management.process_manager:__init__:503 - Models to load: [...]"
            ),
            (
                "2026-06-19 18:24:00.000 | WARNING | "
                "horde_worker_regen.process_management.process_manager:_log_duty_cycle_summary:1955 - "
                "GPU duty cycle 50% over last 180s (target 90%, source=nvml, busy=78%). "
                "biggest worker-side gaps: queue wait 26.5s/job. "
                "jobs: 15 done | 3 pending | 1 in-flight; processes: inf#1=WAITING_FOR_JOB"
            ),
        ]
        log.write_text("\n".join(log_lines), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["horde-duty-report", "--logs", str(log)])

        duty_report_main()

        assert "Epoch 0" in capsys.readouterr().out
