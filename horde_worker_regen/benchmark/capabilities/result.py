"""The records a probe run and a full catalog run produce, plus the shared recommendation data models.

:class:`CapabilityProbeResult` is one probe's complete outcome (the capability-engine replacement for
``LevelReport``): the verdict, why, the distilled stats, the timing split, and any robustness findings.
:class:`CapabilityReport` is the whole run (the replacement for ``BenchmarkReport``): every probe result
plus the synthesized capability surface and recommendation, and the machine it ran on.

This module also owns the leaf data models the recommendation is built from (``Finding``,
``SuggestedBridgeData``, the provenance enums, the capability surface). They live here, in the
torch-free capability layer, rather than in ``report.py`` so the synthesis/rendering layer depends on
this module and not the other way around: nothing here imports the harness or the legacy report, so the
package stays import-light and free of an import cycle.
"""

from __future__ import annotations

import re
import time
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.timing import ProbeTiming
from horde_worker_regen.benchmark.criteria import LevelStats
from horde_worker_regen.benchmark.enums import BenchTier, FindingKind

if TYPE_CHECKING:
    from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
    from horde_worker_regen.process_management.resources.run_metrics import RunMetricsSnapshot

CAPABILITY_REPORT_SCHEMA_VERSION = 5
"""Bumped from the legacy benchmark report's 4 on the clean break to the capability model."""


def _current_worker_version() -> str:
    """Return the running worker version (local import keeps this module import-light)."""
    from horde_worker_regen import __version__

    return __version__


class Finding(BaseModel):
    """One robustness problem observed during a probe, for the remediation queue."""

    kind: FindingKind
    level_id: str
    evidence: str


class HarnessSummary(BaseModel):
    """The JSON-friendly subset of a HarnessResult."""

    num_jobs_expected: int = 0
    num_jobs_completed: int = 0
    num_jobs_faulted: int = 0
    num_alchemy_forms_expected: int = 0
    num_alchemy_forms_completed: int = 0
    num_alchemy_forms_faulted: int = 0
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    audit_failures: list[str] = Field(default_factory=list)
    exit_reason: str = ""
    diagnostics: list[str] = Field(default_factory=list)


class MachineInfo(BaseModel):
    """The hardware the benchmark ran on."""

    gpu_name: str | None = None
    total_vram_mb: int | None = None
    total_ram_bytes: int | None = None


class SuggestionBasis(StrEnum):
    """Why a suggested setting holds the value it does, so a reader can tell proof from absence.

    The crucial distinction is between a setting that is off because it was *tested and did not work*
    (:attr:`DISABLED_FAILED`) and one that is off merely because it was *never tested*
    (:attr:`UNTESTED_SKIPPED` / :attr:`NOT_IN_LADDER`); the recommendation looks identical in both cases
    but means very different things to an operator.
    """

    PROVEN = "proven"
    """A probe for the relevant capability passed; the value is grounded in a real result."""
    DISABLED_FAILED = "disabled_failed"
    """A probe for the capability ran and failed/crashed, so the capability is left off."""
    UNTESTED_SKIPPED = "untested_skipped"
    """The capability's probes were all skipped (machine-fit gate or unmet prerequisite); never proven."""
    NOT_IN_LADDER = "not_in_ladder"
    """The capability was not part of this catalog at all (e.g. excluded by --no-features)."""
    CAPPED_VRAM = "capped_vram"
    """Held back to keep VRAM headroom on this machine, not for lack of capability."""
    CAPPED_SOAK = "capped_soak"
    """Downgraded because the sustained-load soak did not hold up under combined load."""


class SuggestionDecision(BaseModel):
    """The provenance of one suggested setting: its value and why the synthesis chose it."""

    setting: str
    value: bool | int | list[str]
    basis: SuggestionBasis
    detail: str = ""


class SuggestedBridgeData(BaseModel):
    """A conservative bridgeData recommendation derived from the proven probes.

    Unlike a raw "highest passing rung" readout, this keeps VRAM headroom (only models that fit with
    headroom are loaded), prefers a batch size that passed without robustness findings, and has
    concurrent alchemy disabled if the sustained-load soak did not hold up. ``decisions`` records the
    basis behind every value (proven / failed / untested / capped) and ``notes`` the human-readable
    downgrades, for the report.
    """

    max_threads: int = 1
    queue_size: int = 1
    max_batch: int = 1
    allow_lora: bool = False
    allow_controlnet: bool = False
    allow_sdxl_controlnet: bool = False
    allow_post_processing: bool = False
    models_to_load: list[str] = Field(default_factory=list)
    alchemist: bool = False
    alchemy_allow_concurrent: bool = False
    alchemy_max_concurrency: int = 1
    decisions: list[SuggestionDecision] = Field(default_factory=list)
    """Per-setting provenance: the basis behind each suggested value (not part of the bridgeData)."""
    notes: list[str] = Field(default_factory=list)
    """Human-readable rationale for the conservative choices (not part of the bridgeData itself)."""

    def as_yaml_block(self) -> str:
        """Render as a bridgeData.yaml-compatible snippet."""
        lines = [
            f"max_threads: {self.max_threads}",
            f"queue_size: {self.queue_size}",
            f"max_batch: {self.max_batch}",
            f"allow_lora: {str(self.allow_lora).lower()}",
            f"allow_controlnet: {str(self.allow_controlnet).lower()}",
            f"allow_sdxl_controlnet: {str(self.allow_sdxl_controlnet).lower()}",
            f"allow_post_processing: {str(self.allow_post_processing).lower()}",
            "models_to_load:",
            *[f'  - "{model}"' for model in self.models_to_load],
            f"alchemist: {str(self.alchemist).lower()}",
            f"alchemy_allow_concurrent: {str(self.alchemy_allow_concurrent).lower()}",
            f"alchemy_max_concurrency: {self.alchemy_max_concurrency}",
        ]
        return "\n".join(lines)

    def to_bridge_overrides(self) -> dict[str, object]:
        """The worker bridge-data fields this recommendation sets (for the validation run).

        ``max_batch`` is intentionally excluded: batch size is a per-job payload value, not a
        worker config field, so the soak applies it through its job templates instead.
        """
        return {
            "max_threads": self.max_threads,
            "queue_size": self.queue_size,
            "allow_lora": self.allow_lora,
            "allow_controlnet": self.allow_controlnet,
            "allow_sdxl_controlnet": self.allow_sdxl_controlnet,
            "allow_post_processing": self.allow_post_processing,
            "models_to_load": list(self.models_to_load),
            "alchemist": self.alchemist,
            "alchemy_allow_concurrent": self.alchemy_allow_concurrent,
            "alchemy_max_concurrency": self.alchemy_max_concurrency,
        }


class TierCapability(BaseModel):
    """What one model tier proved during the run."""

    tier: BenchTier
    model_name: str
    baseline_passed: bool
    observed_its_p50: float | None = None
    max_stable_batch: int = 1
    """The largest batch rung that passed with no robustness findings."""
    peak_vram_mb: int | None = None
    fits_with_headroom: bool = False
    """Whether the tier's baseline peak VRAM left the headroom reserve free (drives models_to_load)."""


class WorkerCapabilities(BaseModel):
    """Everything the worker proved it can do, independent of the conservative recommendation."""

    tiers: list[TierCapability] = Field(default_factory=list)
    supports_hires_fix: bool = False
    supports_post_processing: bool = False
    supports_controlnet: bool = False
    """Classic SD1.5 preprocessor controlnet (canny/depth/openpose)."""
    supports_qr_code: bool = False
    """The QR-code controlnet workflow (the SDXL controlnet capability)."""
    supports_alchemy_clip: bool = False
    supports_alchemy_graph: bool = False
    supports_alchemy_concurrent: bool = False
    supports_lora: bool = False


class CapabilityProbeResult(BaseModel):
    """One probe's complete record: the capability, the verdict, why, the stats, timing, and findings."""

    capability: Capability
    verdict: CapabilityVerdict
    reasons: list[str] = Field(default_factory=list)
    """Why the capability was disproven (empty when proven)."""
    advisories: list[str] = Field(default_factory=list)
    """Non-fatal observations (e.g. a slow download, a sub-target duty cycle)."""
    stats: LevelStats | None = None
    harness: HarnessSummary | None = None
    timing: ProbeTiming | None = None
    """Where the probe's wall-clock went (warmup boot, productive inference, teardown)."""
    findings: list[Finding] = Field(default_factory=list)
    log_tail: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the probe proved its capability (``verdict is PROVEN``)."""
        return self.verdict is CapabilityVerdict.PROVEN


class CapabilityReport(BaseModel):
    """A whole catalog run: every probe's result, the capability surface, and the recommendation."""

    report_schema_version: int = CAPABILITY_REPORT_SCHEMA_VERSION
    worker_version: str = Field(default_factory=_current_worker_version)
    created_at: float = Field(default_factory=time.time)
    run_id: str = ""
    machine: MachineInfo = Field(default_factory=MachineInfo)
    probes: list[CapabilityProbeResult] = Field(default_factory=list)
    capabilities: WorkerCapabilities = Field(default_factory=WorkerCapabilities)
    suggested_bridge_data: SuggestedBridgeData = Field(default_factory=SuggestedBridgeData)
    tier_baselines_its: dict[str, float] = Field(default_factory=dict)
    """Per-tier reference sampling rate (it/s p50) established by each tier's baseline probe."""

    @property
    def findings(self) -> list[Finding]:
        """All findings across probes, in catalog order."""
        return [finding for probe in self.probes for finding in probe.findings]


_OOM_PATTERN = re.compile(r"CUDA out of memory|torch\.OutOfMemoryError|cudaErrorMemoryAllocation", re.IGNORECASE)
"""Log signatures of a CUDA out-of-memory event, distinct from a generic crash."""


def classify_findings(
    probe: CapabilityProbe,
    *,
    audit_failures: list[str],
    metrics: RunMetricsSnapshot | None,
    log_tail: list[str],
    crashed: bool,
) -> list[Finding]:
    """Derive robustness findings for a probe from its run's audit failures, metrics, and log tail.

    Pulled out of the controller so every driver (warm, subprocess, the probe-runner tests) classifies
    findings identically. ``crashed`` marks a worker that died without a usable result; the metrics and
    audit failures supply OOM, lost/double-submitted jobs, process recoveries, and download stalls. The
    findings are keyed by the probe's slug.
    """
    findings: list[Finding] = []
    probe_id = probe.probe_id

    oom_match = _OOM_PATTERN.search("\n".join(log_tail))
    if oom_match:
        findings.append(
            Finding(kind=FindingKind.OOM, level_id=probe_id, evidence=f"OOM signature in log: {oom_match.group(0)}"),
        )
    if crashed:
        findings.append(
            Finding(kind=FindingKind.CRASH, level_id=probe_id, evidence="probe worker died without a result"),
        )

    for failure in audit_failures:
        findings.append(
            Finding(
                kind=FindingKind.DOUBLE_SUBMIT if "double submit" in failure else FindingKind.LOST_JOB,
                level_id=probe_id,
                evidence=failure,
            ),
        )

    if metrics is not None:
        for crash in metrics.process_crash_events:
            findings.append(
                Finding(
                    kind=FindingKind.PROCESS_RECOVERY,
                    level_id=probe_id,
                    evidence=f"process {crash.process_id} replaced (last state {crash.last_state}): {crash.reason}",
                ),
            )
        for download in metrics.downloads:
            if not download.success:
                findings.append(
                    Finding(
                        kind=FindingKind.DOWNLOAD_STALL,
                        level_id=probe_id,
                        evidence=f"download of {download.name} failed after {download.retries} retries",
                    ),
                )

    return findings


__all__ = [
    "CAPABILITY_REPORT_SCHEMA_VERSION",
    "CapabilityProbeResult",
    "CapabilityReport",
    "Finding",
    "HarnessSummary",
    "MachineInfo",
    "SuggestedBridgeData",
    "SuggestionBasis",
    "SuggestionDecision",
    "TierCapability",
    "WorkerCapabilities",
    "classify_findings",
]
