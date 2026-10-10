"""Unit tests for the duty report: the bridge.log epoch parser, the clearance-hold digest and the CLI."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from horde_worker_regen.analysis.duty_log_report import (
    analyze_log,
    build_epoch_report,
    epoch_lines_for_stats_session,
    main,
    parse_clearance_hold,
    parse_duty_window,
    render_clearance_hold_digest,
    render_report,
    split_into_epochs,
    summarize_clearance_holds,
)
from horde_worker_regen.analysis.log_triage_cli import build_parser
from horde_worker_regen.process_management.resources.vram_footprints import (
    FOOTPRINT_STORE_FILENAME,
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
    ResolutionBucket,
)

# Two sessions in one appended log: each opens with a process-manager __init__ burst (the epoch
# boundary), prints a status block (identity + perf-mode config), then periodic duty-cycle lines. The
# second session is the "tuned" one (lower churn, higher duty) so a test can assert the split.
_LOG = """\
2026-06-19 18:21:00.000 | DEBUG    | horde_worker_regen.process_management.process_manager:__init__:503 - Models to load: [...]
2026-06-19 18:21:00.100 | DEBUG    | horde_worker_regen.process_management.process_manager:__init__:560 - Total RAM: 63.9 GB
2026-06-19 18:21:05.000 | INFO     | horde_worker_regen.reporting.status_reporter:_print_worker_info:425 -   dreamer_name: tazlin-tui-example | (v12.16.0) | horde user: Tazlin#6572 | num_models: 111 | custom_models: False | max_power: 32 (1024x1024) | max_threads: 1 | queue_size: 3 | safety_on_gpu: True
2026-06-19 18:21:05.100 | INFO     | horde_worker_regen.reporting.status_reporter:_print_worker_info:428 -   unload_models_from_vram_often: True | high_performance_mode: True | moderate_performance_mode: False | high_memory_mode: False
2026-06-19 18:24:00.000 | WARNING  | horde_worker_regen.process_management.process_manager:_log_duty_cycle_summary:1955 - GPU duty cycle 50% over last 180s (target 90%, source=nvml, busy=78%). biggest worker-side gaps: queue wait 26.5s/job, submit 3.4s/job; reload churn: 23 model swaps, 18 VRAM evictions. jobs: 15 done | 3 pending | 1 in-flight; processes: inf#1=WAITING_FOR_JOB
2026-06-19 18:27:00.000 | WARNING  | horde_worker_regen.process_management.process_manager:_log_duty_cycle_summary:1955 - GPU duty cycle 60% over last 180s (target 90%, source=nvml, busy=82%). biggest worker-side gaps: queue wait 16.5s/job, submit 4.2s/job; reload churn: 11 model swaps. jobs: 16 done | 3 pending | 1 in-flight; processes: inf#1=WAITING_FOR_JOB
2026-06-19 18:30:00.000 | WARNING  | horde_worker_regen.utils.disk_monitor:sample:71 - Low disk space on G:\\x: 0.1 GB free (floor: 20.0 GB). Model downloads and result writes may start failing.
2026-06-19 19:00:00.000 | DEBUG    | horde_worker_regen.process_management.process_manager:__init__:503 - Models to load: [...]
2026-06-19 19:00:05.000 | INFO     | horde_worker_regen.reporting.status_reporter:_print_worker_info:428 -   unload_models_from_vram_often: False | high_performance_mode: False | moderate_performance_mode: False | high_memory_mode: False
2026-06-19 19:03:00.000 | DEBUG    | horde_worker_regen.process_management.process_manager:_log_duty_cycle_summary:1946 - GPU duty cycle 92% over last 180s (target 90%, source=nvml, busy=96%). biggest worker-side gaps: queue wait 4.0s/job. jobs: 30 done | 4 pending | 1 in-flight; processes: inf#1=INFERENCE_STARTING
"""


class TestEpochSplitting:
    """Splitting an appended log into per-session epochs on the manager-init boundary."""

    def test_init_burst_collapses_to_one_boundary(self) -> None:
        """The two __init__ lines of session one are one epoch, not two."""
        epochs = split_into_epochs(_LOG.splitlines())
        assert len(epochs) == 2

    def test_preamble_before_first_boundary_is_dropped(self) -> None:
        """Lines before the first boundary do not create a phantom epoch."""
        lines = ["2026-06-19 18:20:00.000 | INFO | something before any session", *_LOG.splitlines()]
        assert len(split_into_epochs(lines)) == 2


class TestDutyLineParsing:
    """Parsing one duty-cycle line, including the decimal-safe gap and churn capture."""

    def test_gaps_with_decimals_parse(self) -> None:
        """The decimal in '26.5s/job' must not truncate the gap capture (regression guard)."""
        message = (
            "GPU duty cycle 50% over last 180s (target 90%, source=nvml, busy=78%). "
            "biggest worker-side gaps: queue wait 26.5s/job, submit 3.4s/job; "
            "reload churn: 23 model swaps, 18 VRAM evictions. jobs: 15 done"
        )
        window = parse_duty_window(message, None)
        assert window is not None
        assert window.duty_percent == 50
        assert window.busy_percent == 78
        assert window.gaps == {"queue wait": 26.5, "submit": 3.4}
        assert window.churn == {"model swaps": 23, "VRAM evictions": 18}

    def test_multi_card_breakdown_still_parses(self) -> None:
        """A multi-GPU worker's per-card parenthetical rides beside the headline without hiding the line."""
        message = (
            "GPU duty cycle 61% (card 0: 88%, card 1: 34%) over last 180s "
            "(target 90%, source=nvml, busy=78%). biggest worker-side gaps: queue wait 4.0s/job. jobs: 15 done"
        )
        window = parse_duty_window(message, None)
        assert window is not None
        assert window.duty_percent == 61
        assert window.window_seconds == 180
        assert window.busy_percent == 78
        assert window.gaps == {"queue wait": 4.0}

    def test_non_duty_line_returns_none(self) -> None:
        """A line that is not a duty-cycle report parses to None."""
        assert parse_duty_window("just some other log line", None) is None


class TestEpochReport:
    """End-to-end: the per-epoch aggregates a tuning comparison reads."""

    def test_two_epochs_with_config_and_aggregates(self) -> None:
        """Each epoch carries its own config, duty stats, churn totals, and disk-pressure low-water."""
        reports = analyze_log(_LOG.splitlines())
        assert len(reports) == 2

        first, second = reports
        assert first.config.num_models == 111
        assert first.config.unload_models_from_vram_often is True
        assert first.config.high_performance_mode is True
        assert first.mean_duty() == 55.0  # (50 + 60) / 2
        assert first.churn_totals() == {"model swaps": 34, "VRAM evictions": 18}
        assert first.min_disk_free_gb == 0.1

        # Second epoch is the tuned one: residency on, lower churn, at target.
        assert second.config.unload_models_from_vram_often is False
        assert second.config.high_performance_mode is False
        assert second.mean_duty() == 92.0
        assert second.churn_totals() == {}
        assert second.band_distribution()[">=90%"] == 1

    def test_top_gaps_ranked_biggest_first(self) -> None:
        """The aggregated per-job gaps are ordered largest-first for the report."""
        reports = analyze_log(_LOG.splitlines())
        gaps = list(reports[0].mean_gaps())
        assert gaps[0] == "queue wait"  # 21.5 mean >> submit ~3.8


_GOV_PREFIX = "horde_worker_regen.process_management.scheduling.pop_governor_registry:_default_log:200"


def _gov_enter(ts: str, name: str = "large_model_reentry") -> str:
    return f"2026-06-19 {ts} | INFO | {_GOV_PREFIX} - Pop governor ENTER: {name} (cooling down); expected ~180s"


def _gov_exit(ts: str, name: str = "large_model_reentry") -> str:
    return f"2026-06-19 {ts} | INFO | {_GOV_PREFIX} - Pop governor EXIT: {name} after 3m00s (1x this session, 3m00s total)"


def test_epoch_attributes_pop_governor_spell_time() -> None:
    """An ENTER/EXIT pair within an epoch is reconstructed into per-governor engaged seconds and rendered."""
    lines = [
        "2026-06-19 18:21:00.000 | DEBUG | horde_worker_regen.process_management.process_manager:__init__:503 - Models to load: [...]",
        _gov_enter("18:22:00.000"),
        "2026-06-19 18:23:00.000 | WARNING | horde_worker_regen.process_management.process_manager:_log_duty_cycle_summary:1955 - GPU duty cycle 50% over last 180s (target 90%, source=nvml, busy=78%). jobs: 1 done | 0 pending | 0 in-flight; processes: inf#1=WAITING_FOR_JOB",
        _gov_exit("18:25:00.000"),
    ]

    report = build_epoch_report(0, lines)

    assert report.governor_seconds.get("large_model_reentry") == 180.0
    rendered = render_report([report])
    assert "pop governors engaged" in rendered
    assert "the large-model re-entry cooldown" in rendered


def test_open_governor_spell_counts_to_epoch_end() -> None:
    """A spell still open at the last record is attributed up to that point, not dropped."""
    lines = [
        "2026-06-19 18:21:00.000 | DEBUG | horde_worker_regen.process_management.process_manager:__init__:503 - Models to load: [...]",
        _gov_enter("18:22:00.000", name="whole_card_residency"),
        "2026-06-19 18:24:00.000 | WARNING | horde_worker_regen.process_management.process_manager:_log_duty_cycle_summary:1955 - GPU duty cycle 50% over last 180s (target 90%, source=nvml, busy=78%). jobs: 1 done | 0 pending | 0 in-flight; processes: inf#1=WAITING_FOR_JOB",
    ]

    report = build_epoch_report(0, lines)

    assert report.governor_seconds.get("whole_card_residency") == 120.0


_HOLD_PREFIX = (
    "INFO     | horde_worker_regen.process_management.scheduling.inference_scheduler:_log_clearance_hold:6042 - "
)
_HOLD_TAIL = (
    ". The staged child waits for room (or samples via its lease-acquire timeout); this repeats at most once per "
    "distinct cause."
)
_DEFER_HOLD = (
    "Clearance held for process 4 (Flux.1-Schnell fp8 (Compact)): defer; candidate 13338MB (child already holds "
    "0MB) vs device free 4194MB, available 3038MB, reserve 0MB, outstanding reservations 554MB, noise buffer 602MB; "
    "reclaim run: none" + _HOLD_TAIL
)
"""A literal line ``InferenceScheduler._log_clearance_hold`` wrote on a worker; the regex is pinned to it."""
_MUTEX_HOLD = (
    "Clearance held for process 5 (Z-Image-Turbo): post_processing_coresidency; post-processing co-residency mutex "
    "held the card for an in-flight or pending chain" + _HOLD_TAIL
)


def _hold_line(stamp: str, process: int, model: str, candidate: int, available: int, reclaim: str) -> str:
    return (
        f"{stamp} | {_HOLD_PREFIX}Clearance held for process {process} ({model}): defer; candidate {candidate}MB "
        f"(child already holds 24MB) vs device free 9000MB, available {available}MB, reserve 0MB, outstanding "
        f"reservations 1000MB, noise buffer 602MB; reclaim run: {reclaim}{_HOLD_TAIL}"
    )


def _init_line(stamp: str) -> str:
    return f"{stamp} | DEBUG    | horde_worker_regen.process_management.process_manager:__init__:503 - Models to load"


class TestClearanceHoldParsing:
    """The clearance-hold line's arithmetic, per model."""

    def test_logged_defer_line_parses(self) -> None:
        """Every figure of a real defer line is read, so a reworded emitter fails here."""
        hold = parse_clearance_hold(f"2026-10-09 23:51:05.639 | {_HOLD_PREFIX}{_DEFER_HOLD}")

        assert hold is not None
        assert hold.timestamp == datetime(2026, 10, 9, 23, 51, 5, 639000)
        assert (hold.process_id, hold.model, hold.decision) == (4, "Flux.1-Schnell fp8 (Compact)", "defer")
        assert (hold.candidate_mb, hold.child_holds_mb, hold.device_free_mb, hold.available_mb) == (
            13338,
            0,
            4194,
            3038,
        )
        assert (hold.reserve_mb, hold.outstanding_mb, hold.noise_buffer_mb) == (0, 554, 602)
        assert hold.reclaim_run == "none"

    def test_unpriced_and_unmeasured_terms_parse_as_none(self) -> None:
        """The emitter's ``unpriced``, ``unreported`` and ``n/a`` spellings leave their figures unset."""
        line = (
            "Clearance held for process 3 (SDXL 1.0): defer; candidate unpriced (child already holds unreported) vs "
            "device free n/a, available n/a, reserve 512MB, outstanding reservations 0MB, noise buffer 602MB; "
            "reclaim run: release_cache, evict_idle_model" + _HOLD_TAIL
        )

        hold = parse_clearance_hold(line)

        assert hold is not None
        assert (hold.candidate_mb, hold.child_holds_mb, hold.device_free_mb, hold.available_mb) == (
            None,
            None,
            None,
            None,
        )
        assert hold.reserve_mb == 512
        assert hold.reclaim_run == "release_cache, evict_idle_model"

    def test_outstanding_by_unit_suffix_leaves_the_reclaim_value_whole(self) -> None:
        """The emitter's per-unit breakdown after the reclaim field is not read as part of the reclaim value."""
        line = (
            "Clearance held for process 4 (AlbedoBase XL 3.1): defer; candidate 9907MB (child already holds 24MB) vs "
            "device free 10851MB, available 9221MB, reserve 0MB, outstanding reservations 1029MB, noise buffer 602MB; "
            "reclaim run: none; outstanding by unit: dispatch_admission:4bb3a3bc@p3 1029MB (reports 1706MB)"
            + _HOLD_TAIL
        )

        hold = parse_clearance_hold(line)

        assert hold is not None
        assert hold.reclaim_run == "none"
        assert hold.outstanding_mb == 1029

    def test_mutex_hold_parses_without_arithmetic(self) -> None:
        """A post-processing co-residency hold is a hold with no figures."""
        hold = parse_clearance_hold(_MUTEX_HOLD)

        assert hold is not None
        assert hold.decision == "post_processing_coresidency"
        assert hold.candidate_mb is None
        assert hold.reclaim_run is None

    def test_digest_takes_medians_and_reclaim_distribution_per_model(self) -> None:
        """Holds group by model with median candidate, available and outstanding, and how many ran no reclaim."""
        lines = [
            _hold_line("2026-06-20 01:03:00.000", 1, "Model A", 8000, 3000, "none"),
            _hold_line("2026-06-20 01:04:00.000", 2, "Model A", 9000, 4000, "evict_idle_model"),
            _hold_line("2026-06-20 01:05:00.000", 1, "Model A", 10000, 5000, "none"),
            _hold_line("2026-06-20 01:06:00.000", 1, "Model B", 6000, 2000, "release_cache, evict_idle_model"),
            "2026-06-20 01:07:00.000 | INFO | something else",
        ]

        digests = summarize_clearance_holds(lines)

        assert [digest.model for digest in digests] == ["Model A", "Model B"]
        model_a = digests[0]
        assert model_a.hold_lines == 3
        assert model_a.decisions == {"defer": 3}
        assert (model_a.median_candidate_mb, model_a.median_available_mb, model_a.median_outstanding_mb) == (
            9000.0,
            4000.0,
            1000.0,
        )
        assert model_a.reclaim_runs == {"none": 2, "evict_idle_model": 1}
        rendered = "\n".join(render_clearance_hold_digest(digests))
        assert "clearance holds (distinct holds logged): 4" in rendered
        assert "Model A: 3 (defer 3); median candidate 9000MB, available 4000MB, outstanding 1000MB" in rendered
        assert "reclaim run: release_cache+evict_idle_model 1" in rendered


_SESSION_LOG_LINES = [
    _init_line("2026-06-19 22:00:00.000"),
    _hold_line("2026-06-19 22:10:00.000", 1, "Earlier Session Model", 7000, 1000, "none"),
    _init_line("2026-06-20 01:01:58.000"),
    _hold_line("2026-06-20 01:10:00.000", 2, "This Session Model", 8000, 2000, "none"),
]
"""Two sessions in one appended log; the stats file below is stamped four seconds after the second banner."""


class TestStatsSessionEpochMatch:
    """A stats session reads the log epoch that started on the host clock beside its filename stamp."""

    def test_epoch_nearest_the_stamp_is_chosen(self) -> None:
        """The epoch whose banner sits seconds before the stamp is the stats session's own."""
        lines = epoch_lines_for_stats_session(_SESSION_LOG_LINES, datetime(2026, 6, 20, 1, 2, 3))

        assert [digest.model for digest in summarize_clearance_holds(lines)] == ["This Session Model"]

    def test_no_epoch_near_the_stamp_reads_nothing(self) -> None:
        """A tail-capped log missing the session's banner yields no lines, never another session's holds."""
        assert epoch_lines_for_stats_session(_SESSION_LOG_LINES, datetime(2026, 6, 21, 9, 0, 0)) == []


def _write_bundle(root: Path) -> Path:
    """Write a minimal support bundle: one stats session, a two-session bridge.log and a footprint store."""
    stats_dir = root / "stats"
    stats_dir.mkdir(parents=True)
    sample = {
        "timestamp": 1000.0,
        "jobs_in_progress": 1,
        "process_state_summary": "inference#1=INFERENCE_PRIMED inference#2=WAITING_FOR_JOB",
        "dispatch_hold_bucket": "clearance_hold",
        "jobs_pending_inference": 1,
    }
    (stats_dir / "stats-v1.0.0-20260620-010203-000.jsonl").write_text(
        json.dumps({"event": "stats_sample", "sample": sample}) + "\n",
        encoding="utf-8",
    )
    logs_dir = root / "logs"
    logs_dir.mkdir()
    (logs_dir / "bridge.log").write_text("\n".join(_SESSION_LOG_LINES) + "\n", encoding="utf-8")
    config_dir = root / "config"
    config_dir.mkdir()
    store = LearnedFootprintStore(path=config_dir / FOOTPRINT_STORE_FILENAME)
    store.observe_peak(
        FootprintKey(
            model_baseline="stable_diffusion_xl",
            resolution_bucket=ResolutionBucket.LE_1024,
            platform="linux",
            stage=FootprintStage.SAMPLE,
        ),
        9000.0,
    )
    store.save()
    return root


class TestDutyReportCli:
    """``horde-duty-report`` and ``horde-log duty`` run one report over a bundle."""

    def test_stats_report_adds_the_session_clearance_digest_and_footprints(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """With ``--logs`` the session's own epoch supplies the holds; the bundle's store is read by default."""
        bundle = _write_bundle(tmp_path / "bundle")
        monkeypatch.setattr(
            sys,
            "argv",
            ["horde-duty-report", "--stats", str(bundle / "stats"), "--logs", str(bundle / "logs"), "--last"],
        )

        main()

        output = capsys.readouterr().out
        assert "sampling concurrency: 0 100.0% of 1 samples" in output
        assert "primed but not sampling 100.0% (1 samples" in output
        assert "This Session Model: 1 (defer 1)" in output
        assert "Earlier Session Model" not in output
        assert "== Learned footprints (" in output
        assert "stable_diffusion_xl le_1024 sample [all]: watermark 9000MB" in output

    def test_json_carries_the_clearance_digest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The JSON keeps one object per session and adds its clearance holds."""
        bundle = _write_bundle(tmp_path / "bundle")
        monkeypatch.setattr(
            sys,
            "argv",
            ["horde-duty-report", "--stats", str(bundle / "stats"), "--logs", str(bundle / "logs"), "--json"],
        )

        main()

        payload = json.loads(capsys.readouterr().out)
        assert [hold["model"] for hold in payload[0]["clearance_holds"]] == ["This Session Model"]
        assert payload[0]["sampling_concurrency"]["total_samples"] == 1

    def test_horde_log_duty_runs_the_same_report(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """``horde-log --help`` lists ``duty``, and the subcommand prints the duty report."""
        parser = build_parser()
        assert "duty" in parser.format_help()
        bundle = _write_bundle(tmp_path / "bundle")
        args = parser.parse_args(["duty", "--stats", str(bundle / "stats"), "--logs", str(bundle / "logs")])

        exit_code = args.func(args)

        output = capsys.readouterr().out
        assert exit_code == 0
        assert "== Stats session 20260620-010203 v1.0.0" in output
        assert "This Session Model: 1 (defer 1)" in output
