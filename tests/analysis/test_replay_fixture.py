"""Tests for ``horde-log replay-fixture``: a session's stats rows in, the replay fixture's rows out."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hordelib.metrics import JobPhaseMetrics, JobVramFootprint, ModelLoadEvent, SamplingStats

from horde_worker_regen.analysis.log_triage_cli import build_parser
from horde_worker_regen.analysis.replay_fixture import (
    FOOTPRINTS_FILENAME,
    PLATFORM_PLACEHOLDER,
    SESSION_JOBS_FILENAME,
    ReplayFixtureError,
    build_replay_job_rows,
    platform_neutral_footprint_text,
    read_replay_job_rows,
    write_replay_fixture,
)
from horde_worker_regen.process_management.resources.run_metrics import (
    JobMetricsRecord,
    SessionStartEvent,
    StatsJobCompletedEvent,
)
from horde_worker_regen.process_management.scheduling.workload_kind import WorkloadKind

_SESSION_START = 1_782_316_800.0
"""The synthetic session's ``session_start`` epoch; every offset counts from it."""

# One launch, opened by the main-process logger-setup line that segments a bridge.log into sessions. The stats
# file's stamp below sits one second after it, on the same host clock.
_BRIDGE_LOG = """\
2026-06-24 18:00:00.000 | DEBUG    | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process
2026-06-24 18:30:00.000 | INFO     | horde_worker_regen.run_worker:main:106 - Worker has finished working.
"""
_STATS_FILENAME = "stats-v18.0.0-20260624-180001-000.jsonl"


def _session_start() -> dict[str, object]:
    return SessionStartEvent(worker_version="18.0.0", timestamp=_SESSION_START).model_dump(mode="json")


def _loaded_job() -> dict[str, object]:
    """Return a finished image job that loaded its model, sampled, decoded and was post-processed."""
    record = JobMetricsRecord(
        job_id="job-loaded",
        time_popped=_SESSION_START + 10.0,
        stage_timestamps={
            "INFERENCE_IN_PROGRESS": _SESSION_START + 12.5,
            "POST_PROCESSING": _SESSION_START + 30.0,
            "PENDING_SAFETY_CHECK": _SESSION_START + 33.25,
            "FINALIZED": _SESSION_START + 40.0,
        },
        queue_wait_seconds=2.5,
        safety_seconds=1.5,
        phase_metrics=JobPhaseMetrics(
            model_loads=[
                ModelLoadEvent(model_name="Example XL", phase="disk_to_ram", duration_seconds=3.0, timestamp=1.0),
                ModelLoadEvent(model_name="Example XL", phase="ram_to_vram", duration_seconds=1.25, timestamp=2.0),
                ModelLoadEvent(model_name="Example XL", phase="ram_to_vram", duration_seconds=0.5, timestamp=3.0),
            ],
            sampling=SamplingStats(
                steps_completed=30, total_steps=30, duration_seconds=6.0, iterations_per_second=5.0
            ),
            vram_used_high_water_mb=14000,
            vram_footprint=JobVramFootprint(
                peak_resident_weights_mb=6800.0,
                peak_device_used_mb=13900.0,
                resident_weights_after_job_mb=6800.0,
                baseline="stable_diffusion_xl",
                batch_size=2,
            ),
            phase_seconds={"clip_encode": 0.25, "vae_decode": 0.75, "pipeline_execute": 8.0},
        ),
        model_name="Example XL",
        steps=30,
        width=1024,
        height=1024,
        loras_count=1,
        post_processing=["RealESRGAN_x4plus"],
        sampler_name="k_euler",
        cfg_scale=7.0,
        batch_count=2,
        megapixelsteps=62.9,
        post_processing_chain_seconds=2.75,
        queue_depth_at_dispatch=1,
        whole_card=False,
        kudos_reward=21.5,
    )
    return StatsJobCompletedEvent(job=record, baseline="stable_diffusion_xl").model_dump(mode="json")


def _resident_job() -> dict[str, object]:
    """Return a faulted image job, popped before the loaded one, whose model was already on the card."""
    record = JobMetricsRecord(
        job_id="job-resident",
        faulted=True,
        fault_reason="RuntimeError: example",
        time_popped=_SESSION_START + 4.0,
        stage_timestamps={"INFERENCE_IN_PROGRESS": _SESSION_START + 4.5, "FINALIZED": _SESSION_START + 9.0},
        phase_metrics=JobPhaseMetrics(vram_footprint=JobVramFootprint(baseline="stable_diffusion_1")),
        model_name="Example 1.5",
        steps=20,
        width=512,
        height=512,
        sampler_name="k_dpmpp_2m",
    )
    return StatsJobCompletedEvent(job=record).model_dump(mode="json")


def _alchemy_record() -> dict[str, object]:
    record = JobMetricsRecord(job_id="form-1", workload=WorkloadKind.ALCHEMY, time_popped=_SESSION_START + 5.0)
    return StatsJobCompletedEvent(job=record).model_dump(mode="json")


def _stats_sample() -> dict[str, object]:
    return {"event": "stats_sample", "sample": {"timestamp": _SESSION_START + 1.0}}


def _footprint_store() -> dict[str, object]:
    return {
        "schema_version": 4,
        "observations": [
            {
                "key": {"model_baseline": "stable_diffusion_xl", "platform": "linux", "stage": "resident"},
                "observation": {"watermark_mb": 6800.0, "recent_mb": [6790.0, 6800.0]},
            },
            {
                "key": {"model_baseline": "safety_process", "platform": "linux", "stage": "safety"},
                "observation": {"watermark_mb": 1052.0, "recent_mb": [153.0]},
            },
        ],
    }


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """Return a support-bundle layout holding one session's log, stats file and footprint store."""
    root = tmp_path / "bundle"
    (root / "logs").mkdir(parents=True)
    (root / "stats").mkdir()
    (root / "config").mkdir()
    (root / "logs" / "bridge.log").write_text(_BRIDGE_LOG, encoding="utf-8")
    payloads = [_session_start(), _stats_sample(), _loaded_job(), _alchemy_record(), _resident_job()]
    (root / "stats" / _STATS_FILENAME).write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    (root / "config" / "vram_footprints.json").write_text(json.dumps(_footprint_store()), encoding="utf-8")
    return root


class TestJobRows:
    """Rows built from a session's stats payloads."""

    def test_image_jobs_only_in_pop_order(self) -> None:
        """Alchemy records and samples are dropped, and rows sort by pop offset rather than file order."""
        stream = build_replay_job_rows(
            [_session_start(), _stats_sample(), _loaded_job(), _alchemy_record(), _resident_job()],
        )
        assert stream.session_start_epoch == _SESSION_START
        assert [row.job_id for row in stream.rows] == ["job-resident", "job-loaded"]

    def test_loaded_job_columns(self) -> None:
        """Offsets, load sums, phase seconds, the post-processing window and peaks come from the record."""
        row = build_replay_job_rows([_session_start(), _loaded_job()]).rows[0]
        assert row.offset_s == 10.0
        assert row.dispatch_offset_s == 12.5
        assert row.finalized_offset_s == 40.0
        assert row.detached_offset_s is None
        assert row.model == "Example XL"
        assert row.baseline == "stable_diffusion_xl"
        assert row.batch == 2
        assert row.loras == 1
        assert row.post_processing == ("RealESRGAN_x4plus",)
        assert row.ram_to_vram_s == 1.75
        assert row.disk_to_ram_s == 3.0
        assert row.sampling_s == 6.0
        assert row.steps_completed == 30
        assert row.clip_encode_s == 0.25
        assert row.vae_decode_s == 0.75
        assert row.model_unload_s is None
        assert row.pipeline_execute_s == 8.0
        assert row.post_processing_s == 3.25
        assert row.post_processing_chain_s == 2.75
        assert row.kudos == 21.5
        assert row.peak_device_used_mb == 13900.0
        assert row.vram_used_high_water_mb == 14000

    def test_job_with_no_footprint_batch_or_loads(self) -> None:
        """A footprint with no batch is batch 1, and a job that loaded nothing sums to zero load seconds."""
        row = build_replay_job_rows([_session_start(), _resident_job()]).rows[0]
        assert row.batch == 1
        assert row.ram_to_vram_s == 0
        assert row.disk_to_ram_s == 0
        assert row.faulted is True
        assert row.fault_reason == "RuntimeError: example"
        assert row.post_processing_s is None
        assert row.sampling_s is None

    def test_job_before_session_start_is_refused(self) -> None:
        """A job row with no session start before it has no zero to count from."""
        with pytest.raises(ReplayFixtureError, match="before the session's session_start"):
            build_replay_job_rows([_loaded_job(), _session_start()])

    def test_session_without_start_is_refused(self) -> None:
        """Stats with no session_start row cannot be offset."""
        with pytest.raises(ReplayFixtureError, match="no session_start"):
            build_replay_job_rows([_stats_sample()])

    def test_unpopped_record_is_counted_and_left_out(self) -> None:
        """An image record with no pop time has no offset, so it is counted and not written."""
        unpopped = StatsJobCompletedEvent(job=JobMetricsRecord(job_id="job-unpopped")).model_dump(mode="json")
        stream = build_replay_job_rows([_session_start(), unpopped])
        assert stream.rows == []
        assert stream.unpopped_count == 1


class TestFootprintStore:
    """The platform-neutral copy of the learned store."""

    def test_every_platform_is_replaced(self) -> None:
        """Each key's platform becomes the placeholder, and the replaced platforms are reported."""
        text, platforms = platform_neutral_footprint_text(json.dumps(_footprint_store()))
        store = json.loads(text)
        assert platforms == ["linux"]
        assert {entry["key"]["platform"] for entry in store["observations"]} == {PLATFORM_PLACEHOLDER}
        assert store["observations"][1]["observation"]["recent_mb"] == [153.0]
        assert text.endswith("}\n")

    def test_store_without_observations_is_refused(self) -> None:
        """A file that is not a footprint store is reported, never written as one."""
        with pytest.raises(ReplayFixtureError, match="observations"):
            platform_neutral_footprint_text(json.dumps({"schema_version": 4}))


class TestCommand:
    """``horde-log replay-fixture`` against a bundle layout."""

    def test_writes_both_files_from_the_bundle(self, bundle: Path, tmp_path: Path) -> None:
        """The session's rows and the neutral store land in the out directory, and the rows read back."""
        out_dir = tmp_path / "fixture"
        args = build_parser().parse_args(["replay-fixture", str(bundle), "--session", "0", str(out_dir)])
        assert args.func(args) == 0

        rows = read_replay_job_rows(out_dir / SESSION_JOBS_FILENAME)
        assert [row.job_id for row in rows] == ["job-resident", "job-loaded"]
        lines = (out_dir / SESSION_JOBS_FILENAME).read_bytes().split(b"\n")
        assert lines[-1] == b""
        assert json.loads(lines[0])["ram_to_vram_s"] == 0
        stored = json.loads((out_dir / FOOTPRINTS_FILENAME).read_text(encoding="utf-8"))
        assert {entry["key"]["platform"] for entry in stored["observations"]} == {PLATFORM_PLACEHOLDER}

    def test_logs_directory_finds_the_bundle_stats(self, bundle: Path, tmp_path: Path) -> None:
        """Given the bundle's logs directory, the stats and store beside it are used."""
        summary = write_replay_fixture(bundle / "logs", tmp_path / "fixture", session_index=0)
        assert [path.name for path in summary.stats_files] == [_STATS_FILENAME]
        assert summary.job_count == 2
        assert summary.faulted_count == 1
        assert summary.rewritten_platforms == ("linux",)

    def test_rebuild_is_byte_identical(self, bundle: Path, tmp_path: Path) -> None:
        """A rebuild from the same files writes the same bytes, so a committed fixture can be regenerated."""
        write_replay_fixture(bundle, tmp_path / "first", session_index=None, last=True)
        write_replay_fixture(bundle, tmp_path / "second", session_index=0)
        for filename in (SESSION_JOBS_FILENAME, FOOTPRINTS_FILENAME):
            assert (tmp_path / "first" / filename).read_bytes() == (tmp_path / "second" / filename).read_bytes()

    def test_session_with_no_stats_nearby_is_reported(
        self,
        bundle: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A stats file stamped far from the session's start is not taken for it."""
        stats_file = bundle / "stats" / _STATS_FILENAME
        stats_file.rename(stats_file.with_name("stats-v18.0.0-20260624-190000-000.jsonl"))
        args = build_parser().parse_args(["replay-fixture", str(bundle), "--session", "0", str(tmp_path / "out")])
        assert args.func(args) == 1
        assert "no stats session" in capsys.readouterr().out

    def test_unknown_session_is_reported(self, bundle: Path, tmp_path: Path) -> None:
        """A session number the logs do not have is an error, never an empty fixture."""
        with pytest.raises(ReplayFixtureError, match="no matching worker session"):
            write_replay_fixture(bundle, tmp_path / "out", session_index=5)

    def test_a_session_selector_is_required(self, bundle: Path, tmp_path: Path) -> None:
        """Without --session or --last the command does not guess which session to write."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["replay-fixture", str(bundle), str(tmp_path / "out")])
