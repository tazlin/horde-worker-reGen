"""Unit tests for the epoch-aware bridge.log duty-cycle report (pure parsing, no worker)."""

from __future__ import annotations

from horde_worker_regen.analysis.duty_log_report import (
    analyze_log,
    build_epoch_report,
    parse_duty_window,
    render_report,
    split_into_epochs,
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
