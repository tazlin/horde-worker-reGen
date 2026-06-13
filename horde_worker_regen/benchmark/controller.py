"""The ramp controller: runs ladder levels in isolated subprocesses and builds the report.

Each level runs via ``python -m horde_worker_regen.benchmark.level_runner`` so a level
that OOMs or hangs kills only its own process tree. The controller applies pre-flight
checks (disk/VRAM burden estimates), evaluates outcomes against the criteria, classifies
robustness findings, and applies the skip rules (a failed tier baseline skips the tier's
dependent levels; a failure on an axis stops higher rungs of that axis).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

from horde_worker_regen.benchmark.criteria import TierBaseline, evaluate_level
from horde_worker_regen.benchmark.ladder import RampLevel
from horde_worker_regen.benchmark.report import (
    BenchmarkReport,
    Finding,
    HarnessSummary,
    LevelReport,
    LevelRunResult,
    MachineInfo,
    compute_level_stats,
    render_markdown,
    synthesize_bridge_data,
)

_SUBPROCESS_GRACE_SECONDS = 120.0
_LOG_TAIL_LINES = 100

_OOM_PATTERN = re.compile(r"CUDA out of memory|torch\.OutOfMemoryError|cudaErrorMemoryAllocation", re.IGNORECASE)


def detect_machine_info() -> MachineInfo:
    """Best-effort hardware detection (no-op on machines without torch/CUDA)."""
    info = MachineInfo()
    try:
        import psutil

        info.total_ram_bytes = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - purely informational
        pass
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info.gpu_name = properties.name
            info.total_vram_mb = round(properties.total_memory / (1024 * 1024))
    except Exception:  # noqa: BLE001 - purely informational
        pass
    return info


class BenchmarkController:
    """Drives the ramp: one isolated subprocess per level, then report synthesis."""

    def __init__(
        self,
        ladder: list[RampLevel],
        out_dir: Path,
        *,
        process_mode: str = "real",
        resume: bool = False,
        only_level: str | None = None,
        skip_downloads: bool = False,
    ) -> None:
        """Initialize the controller.

        Args:
            ladder: The ordered levels to run.
            out_dir: Where level artifacts and the final report are written.
            process_mode: Passed through to the harness ("fake"/"dry_run"/"real").
            resume: Skip levels that already have a result JSON in `out_dir`.
            only_level: Run just this level id (the remediation-loop primitive).
            skip_downloads: Skip levels that require network access.
        """
        self._ladder = ladder
        self._out_dir = out_dir
        self._process_mode = process_mode
        self._resume = resume
        self._only_level = only_level
        self._skip_downloads = skip_downloads

        self._tier_baselines: dict[str, TierBaseline] = {}
        self._failed_tier_baselines: set[str] = set()
        self._failed_axes: set[tuple[str, str]] = set()

    def run(self) -> BenchmarkReport:
        """Run the ramp and return (and persist) the full report."""
        self._out_dir.mkdir(parents=True, exist_ok=True)
        machine = detect_machine_info()
        reports: list[LevelReport] = []

        for level in self._ladder:
            if self._only_level is not None and level.id != self._only_level:
                continue

            report: LevelReport | None = None
            if self._resume:
                prior_result = self._load_result(self._out_dir / f"level_{level.id}.json")
                if prior_result is not None:
                    logger.info(f"Resuming level {level.id} from its existing result")
                    report = self._evaluate_result(level, prior_result, machine, log_tail=[])

            if report is None:
                skip_reason = self._pre_flight_skip_reason(level, machine)
                if skip_reason is not None:
                    logger.warning(f"Skipping level {level.id}: {skip_reason}")
                    reports.append(LevelReport(level=level, outcome="skipped", reasons=[skip_reason]))
                    continue

                logger.info(f"Running level {level.id}: {level.description}")
                report = self._run_level(level, machine)

            reports.append(report)

            if report.outcome == "passed":
                if level.establishes_tier_baseline and report.stats is not None and report.stats.its_p50 is not None:
                    self._tier_baselines[level.tier] = TierBaseline(tier=level.tier, its_p50=report.stats.its_p50)
            else:
                if level.establishes_tier_baseline:
                    self._failed_tier_baselines.add(level.tier)
                self._failed_axes.add((level.tier, level.axis))
                logger.warning(f"Level {level.id} did not pass: {'; '.join(report.reasons) or report.outcome}")

        benchmark_report = BenchmarkReport(
            machine=machine,
            levels=reports,
            suggested_bridge_data=synthesize_bridge_data(reports),
            tier_baselines_its={tier: baseline.its_p50 for tier, baseline in self._tier_baselines.items()},
        )

        (self._out_dir / "report.json").write_text(benchmark_report.model_dump_json(indent=2), encoding="utf-8")
        (self._out_dir / "report.md").write_text(render_markdown(benchmark_report), encoding="utf-8")
        return benchmark_report

    # region per-level

    def _pre_flight_skip_reason(self, level: RampLevel, machine: MachineInfo) -> str | None:
        """Return why the level should be skipped without running, or None to proceed."""
        if self._skip_downloads and level.requires_network:
            return "requires network access (--skip-downloads)"
        if level.establishes_tier_baseline and level.tier in self._failed_tier_baselines:
            pass  # a baseline level never skips itself
        elif level.tier in self._failed_tier_baselines:
            return f"tier {level.tier} baseline failed"
        if (level.tier, level.axis) in self._failed_axes and not level.establishes_tier_baseline:
            return f"axis {level.axis} already failed at a lower rung"

        free_disk = shutil.disk_usage(self._out_dir).free
        if free_disk < level.criteria.min_disk_free_gb * 1024**3:
            return f"insufficient disk: {free_disk / 1024**3:.1f} GB free"

        if self._process_mode == "real" and machine.total_vram_mb:
            try:
                from hordelib.api import estimate_job_burden

                burden = estimate_job_burden(
                    baseline=level.baseline_hordelib,
                    width=max((job.width for job in level.scenario.image_jobs), default=512),
                    height=max((job.height for job in level.scenario.image_jobs), default=512),
                    batch=max((job.n_iter for job in level.scenario.image_jobs), default=1),
                )
                if burden.vram_mb > machine.total_vram_mb:
                    return (
                        f"insufficient VRAM: estimated {burden.vram_mb} MB needed, "
                        f"{machine.total_vram_mb} MB available"
                    )
            except Exception as e:  # noqa: BLE001 - pre-flight must never block the ramp
                logger.debug(f"Pre-flight burden estimate unavailable: {e}")

        return None

    def _run_level(self, level: RampLevel, machine: MachineInfo) -> LevelReport:
        """Run one level in a subprocess and evaluate its outcome."""
        level_json_path = self._out_dir / f"level_{level.id}.def.json"
        level_json_path.write_text(level.model_dump_json(indent=2), encoding="utf-8")
        result_path = self._out_dir / f"level_{level.id}.json"
        log_path = self._out_dir / f"level_{level.id}.log"
        heartbeat_path = self._out_dir / f"level_{level.id}.heartbeat"

        command = [
            sys.executable,
            "-m",
            "horde_worker_regen.benchmark.level_runner",
            "--level-json",
            str(level_json_path),
            "--out",
            str(self._out_dir),
            "--process-mode",
            self._process_mode,
        ]

        hung = False
        try:
            completed = subprocess.run(
                command,
                timeout=level.timeout_seconds + _SUBPROCESS_GRACE_SECONDS,
                capture_output=True,
                text=True,
                check=False,
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            hung = True
            exit_code = -1

        log_tail = self._read_log_tail(log_path)
        result = self._load_result(result_path)

        if hung:
            findings = [
                Finding(
                    kind="hang",
                    level_id=level.id,
                    evidence=(
                        f"level subprocess exceeded {level.timeout_seconds + _SUBPROCESS_GRACE_SECONDS:.0f}s and "
                        f"was killed (last heartbeat: {self._heartbeat_age_description(heartbeat_path)})"
                    ),
                ),
            ]
            return LevelReport(
                level=level,
                outcome="crashed_hang",
                reasons=["subprocess hung and was killed"],
                findings=findings,
                log_tail=log_tail,
            )

        if result is None:
            return LevelReport(
                level=level,
                outcome="crashed",
                reasons=[f"level subprocess died (exit code {exit_code}) without writing a result"],
                findings=self._classify_findings(level, None, log_tail, crashed=True),
                log_tail=log_tail,
            )

        return self._evaluate_result(level, result, machine, log_tail=log_tail)

    def _evaluate_result(
        self,
        level: RampLevel,
        result: LevelRunResult,
        machine: MachineInfo,
        *,
        log_tail: list[str],
    ) -> LevelReport:
        """Apply criteria and finding classification to a (fresh or resumed) raw result."""
        stats = compute_level_stats(result, total_vram_mb=machine.total_vram_mb)
        verdict = evaluate_level(stats, level.criteria, self._tier_baselines.get(level.tier))
        findings = self._classify_findings(level, result, log_tail, crashed=False)

        reasons = list(verdict.reasons)
        if result.runner_error is not None:
            reasons.append(f"runner error: {result.runner_error}")

        return LevelReport(
            level=level,
            outcome="passed" if verdict.passed and result.runner_error is None else "failed",
            reasons=reasons,
            advisories=verdict.advisories,
            stats=stats,
            harness=result.harness,
            findings=findings,
            log_tail=log_tail if (reasons or findings) else [],
        )

    # endregion

    # region findings

    def _classify_findings(
        self,
        level: RampLevel,
        result: LevelRunResult | None,
        log_tail: list[str],
        *,
        crashed: bool,
    ) -> list[Finding]:
        """Derive robustness findings from the run result and log output."""
        findings: list[Finding] = []
        log_text = "\n".join(log_tail)

        if _OOM_PATTERN.search(log_text):
            match = _OOM_PATTERN.search(log_text)
            findings.append(
                Finding(kind="oom", level_id=level.id, evidence=f"OOM signature in log: {match.group(0) if match else ''}"),
            )
        if crashed:
            findings.append(
                Finding(kind="crash", level_id=level.id, evidence="level subprocess died without a result file"),
            )

        if result is not None:
            for failure in result.harness.audit_failures:
                kind = "double_submit" if "double submit" in failure else "lost_job"
                findings.append(Finding(kind=kind, level_id=level.id, evidence=failure))
            if result.metrics is not None:
                for crash in result.metrics.process_crash_events:
                    findings.append(
                        Finding(
                            kind="process_recovery",
                            level_id=level.id,
                            evidence=(
                                f"process {crash.process_id} replaced (last state {crash.last_state}): {crash.reason}"
                            ),
                        ),
                    )
                for download in result.metrics.downloads:
                    if not download.success:
                        findings.append(
                            Finding(
                                kind="download_stall",
                                level_id=level.id,
                                evidence=f"download of {download.name} failed after {download.retries} retries",
                            ),
                        )

        return findings

    # endregion

    @staticmethod
    def _read_log_tail(log_path: Path) -> list[str]:
        if not log_path.exists():
            return []
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return lines[-_LOG_TAIL_LINES:]

    @staticmethod
    def _heartbeat_age_description(heartbeat_path: Path) -> str:
        try:
            last_beat = float(heartbeat_path.read_text())
            return f"{time.time() - last_beat:.0f}s ago"
        except (OSError, ValueError):
            return "never"

    @staticmethod
    def _load_result(result_path: Path) -> LevelRunResult | None:
        if not result_path.exists():
            return None
        try:
            return LevelRunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - a corrupt result is treated as a crash
            logger.error(f"Failed to parse level result {result_path}: {e}")
            return None


def load_existing_report(out_dir: Path) -> BenchmarkReport | None:
    """Load a previously written report.json from a benchmark output directory."""
    report_path = out_dir / "report.json"
    if not report_path.exists():
        return None
    return BenchmarkReport.model_validate_json(report_path.read_text(encoding="utf-8"))


__all__ = [
    "BenchmarkController",
    "HarnessSummary",
    "detect_machine_info",
    "load_existing_report",
]
