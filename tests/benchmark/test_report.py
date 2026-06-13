"""Tests for level stats computation, bridgeData synthesis, and report rendering."""

from __future__ import annotations

from hordelib.metrics import DownloadEvent, JobPhaseMetrics, ModelLoadEvent, SamplingStats

from horde_worker_regen.benchmark.criteria import LevelStats
from horde_worker_regen.benchmark.ladder import LadderOptions, build_default_ladder
from horde_worker_regen.benchmark.report import (
    BenchmarkReport,
    HarnessSummary,
    LevelReport,
    LevelRunResult,
    compute_level_stats,
    render_markdown,
    synthesize_bridge_data,
)
from horde_worker_regen.process_management.run_metrics import JobMetricsRecord, RunMetricsSnapshot


def _run_result(*, faulted: int = 0, its: list[float] | None = None) -> LevelRunResult:
    """A synthetic raw level result with two jobs."""
    jobs = []
    for index, rate in enumerate(its or [5.0, 4.0]):
        jobs.append(
            JobMetricsRecord(
                job_id=f"job-{index}",
                queue_wait_seconds=1.0 + index,
                e2e_seconds=10.0 + index,
                phase_metrics=JobPhaseMetrics(
                    model_loads=[
                        ModelLoadEvent(model_name="m", phase="disk_to_ram", duration_seconds=3.0, timestamp=0.0),
                        ModelLoadEvent(model_name="m", phase="ram_to_vram", duration_seconds=1.0, timestamp=0.0),
                    ],
                    sampling=SamplingStats(
                        steps_completed=30,
                        total_steps=30,
                        duration_seconds=30 / rate,
                        iterations_per_second=rate,
                    ),
                    vram_used_high_water_mb=7000,
                ),
            ),
        )
    metrics = RunMetricsSnapshot(
        jobs=jobs,
        downloads=[
            DownloadEvent(
                name="some lora",
                category="lora",
                size_bytes=150 * 1024 * 1024,
                duration_seconds=10.0,
                megabytes_per_second=15.0,
                retries=0,
                success=True,
                timestamp=0.0,
            ),
        ],
        vram_used_high_water_mb_per_process={0: 7000},
        ram_used_high_water_mb_per_process={0: 12000},
        disk_min_free_bytes={"C:/": 200 * 1024**3},
        num_process_recoveries=0,
        num_job_slowdowns=0,
        time_spent_no_jobs_available=0.0,
        process_crash_events=[],
    )
    return LevelRunResult(
        level_id="A-sd15-baseline",
        harness=HarnessSummary(num_jobs_expected=2, num_jobs_completed=2 - faulted, num_jobs_faulted=faulted),
        metrics=metrics,
    )


class TestComputeLevelStats:
    """Distillation of raw run results into criteria stats."""

    def test_aggregates_sampling_and_loads(self) -> None:
        """it/s percentiles, load-phase medians, and high-waters are derived."""
        stats = compute_level_stats(_run_result(), total_vram_mb=24000)
        assert stats.its_p50 == 4.5
        assert stats.its_min == 4.0
        assert stats.model_load_disk_seconds_median == 3.0
        assert stats.model_load_vram_seconds_median == 1.0
        assert stats.vram_used_high_water_mb == 7000
        assert stats.total_vram_mb == 24000
        assert stats.download_mbps_min == 15.0
        assert stats.disk_min_free_bytes == 200 * 1024**3
        assert stats.e2e_seconds_p95 == 11.0

    def test_empty_metrics_yield_none_fields(self) -> None:
        """Absent metrics produce None stats rather than zeros (zeros would fail criteria)."""
        result = LevelRunResult(level_id="x", harness=HarnessSummary(num_jobs_expected=1, num_jobs_completed=1))
        stats = compute_level_stats(result)
        assert stats.its_p50 is None
        assert stats.vram_used_high_water_mb is None


class TestSynthesizeBridgeData:
    """bridgeData suggestions derive from the highest-passing levels."""

    def _reports(self, passing_ids: set[str]) -> list[LevelReport]:
        ladder = build_default_ladder(LadderOptions(include_downloads=True))
        return [
            LevelReport(
                level=level,
                outcome="passed" if level.id in passing_ids else "failed",
                stats=LevelStats(),
            )
            for level in ladder
        ]

    def test_everything_passing_unlocks_features(self) -> None:
        """All-passing levels yield the most permissive suggestion."""
        ladder = build_default_ladder(LadderOptions(include_downloads=True))
        suggestion = synthesize_bridge_data(self._reports({level.id for level in ladder}))
        assert suggestion.max_threads == 2
        assert suggestion.queue_size == 2
        assert suggestion.max_batch == 4
        assert suggestion.allow_controlnet
        assert suggestion.allow_post_processing
        assert suggestion.allow_lora
        assert suggestion.alchemist
        assert suggestion.alchemy_allow_concurrent
        assert "Deliberate" in suggestion.models_to_load
        assert "AlbedoBase XL (SDXL)" in suggestion.models_to_load

    def test_only_baselines_passing_is_conservative(self) -> None:
        """Only stage-A passes ⇒ conservative defaults with the proven models."""
        suggestion = synthesize_bridge_data(self._reports({"A-sd15-baseline"}))
        assert suggestion.max_threads == 1
        assert suggestion.queue_size == 1
        assert suggestion.max_batch == 1
        assert not suggestion.allow_controlnet
        assert suggestion.models_to_load == ["Deliberate"]

    def test_yaml_block_renders(self) -> None:
        """The YAML snippet includes the model list."""
        suggestion = synthesize_bridge_data(self._reports({"A-sd15-baseline"}))
        yaml_block = suggestion.as_yaml_block()
        assert "models_to_load:" in yaml_block
        assert '- "Deliberate"' in yaml_block


class TestRenderMarkdown:
    """The markdown report renders all sections."""

    def test_render_includes_levels_and_remediation(self) -> None:
        """Level table, suggested bridgeData, and remediation queue are present."""
        ladder = build_default_ladder(LadderOptions(tiers=["sd15"], include_alchemy=False, include_features=False))
        reports = [LevelReport(level=level, outcome="passed", stats=LevelStats(its_p50=5.0)) for level in ladder]
        report = BenchmarkReport(levels=reports, suggested_bridge_data=synthesize_bridge_data(reports))
        markdown = render_markdown(report)
        assert "## Levels" in markdown
        assert "A-sd15-baseline" in markdown
        assert "## Suggested bridgeData" in markdown
        assert "## Remediation queue" in markdown
        assert "No robustness findings" in markdown

    def test_report_json_round_trip(self) -> None:
        """The report survives JSON round-tripping (used for --resume and re-rendering)."""
        ladder = build_default_ladder(LadderOptions(tiers=["sd15"], include_features=False))
        reports = [LevelReport(level=level, outcome="skipped") for level in ladder]
        report = BenchmarkReport(levels=reports)
        assert BenchmarkReport.model_validate_json(report.model_dump_json()) == report
