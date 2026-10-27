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

from pydantic import BaseModel, Field, model_validator

from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.timing import ProbeTiming
from horde_worker_regen.benchmark.criteria import LevelStats
from horde_worker_regen.benchmark.enums import BenchTier, FindingKind

if TYPE_CHECKING:
    from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
    from horde_worker_regen.process_management.resources.run_metrics import RunMetricsSnapshot

CAPABILITY_REPORT_SCHEMA_VERSION = 8
"""Schema version for persisted capability reports."""
_OPERATOR_EVIDENCE_GUARANTEES_VERSION = 7
"""Minimum schema version with retry history and enforced plain-workload feature exclusions."""


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
    num_jobs_completed_with_loras: int = 0
    num_jobs_completed_without_loras: int = 0
    num_jobs_faulted_with_loras: int = 0
    num_jobs_faulted_without_loras: int = 0
    consecutive_failed_jobs_pause_count: int = 0
    """Scheduling-backoff observability for a LoRA-storm soak: the LoRA-carrying vs plain terminal split
    and how many times the consecutive-failures pop pause armed. A scheduling-clean soak keeps the two
    faulted-with/without counts and the pause count at zero (see :class:`HarnessResult`)."""


class MachineInfo(BaseModel):
    """The hardware the benchmark ran on."""

    gpu_name: str | None = None
    total_vram_mb: int | None = None
    total_ram_bytes: int | None = None
    storage: dict[str, bool | None] = Field(default_factory=dict)
    """Model directory to whether it is on rotational media (True), solid state (False) or undetermined (None)."""


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


class SuggestedCardSettings(BaseModel):
    """Per-card limits confirmed together for every model offered on that card."""

    max_threads: int = 1
    queue_size: int = 1
    max_batch: int = 1
    max_power: int = 8
    high_performance_mode: bool = False
    moderate_performance_mode: bool = False
    dynamic_models: bool = False
    models_to_load: list[str] = Field(default_factory=list)
    allow_lora: bool = False
    allow_img2img: bool = False
    allow_painting: bool = False
    allow_controlnet: bool = False
    allow_sdxl_controlnet: bool = False
    allow_post_processing: bool = False


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
    allow_img2img: bool | None = None
    allow_painting: bool | None = None
    allow_controlnet: bool = False
    allow_sdxl_controlnet: bool = False
    allow_post_processing: bool = False
    models_to_load: list[str] = Field(default_factory=list)
    alchemist: bool = False
    alchemy_allow_concurrent: bool = False
    alchemy_max_concurrency: int = 1
    max_power: int | None = None
    gpu_overrides: dict[int, SuggestedCardSettings] = Field(default_factory=dict)
    gpu_device_indices: list[int] | None = None
    clear_card_fields: dict[int, list[str]] = Field(default_factory=dict)
    """Override fields to remove when applying global settings, also retained for known-good restoration."""
    safe_to_apply: bool = True
    """Whether the assessment established at least one enforceable combined configuration."""
    high_performance_mode: bool | None = None
    moderate_performance_mode: bool | None = None
    dynamic_models: bool | None = None
    gpu_sampling_lease_enabled: bool | None = None
    """Set with the thread count when the confirmed profile ran under the sampling lease. None leaves the
    operator's lease settings as written."""
    gpu_sampling_lease_slots: int | None = None
    decisions: list[SuggestionDecision] = Field(default_factory=list)
    """Per-setting provenance: the basis behind each suggested value (not part of the bridgeData)."""
    notes: list[str] = Field(default_factory=list)
    """Human-readable rationale for the conservative choices (not part of the bridgeData itself)."""

    def as_yaml_block(self) -> str:
        """Render as a bridgeData.yaml-compatible snippet."""
        lines: list[str] = []
        if self.clear_card_fields:
            lines.extend(
                [
                    "# Before merging these settings into an existing configuration:",
                    "# Remove the following fields from each listed card's existing overrides.",
                    "# Keep unrelated fields and other cards. TUI Apply performs this cleanup automatically.",
                ]
            )
            for device_index, fields in sorted(self.clear_card_fields.items()):
                lines.append(f"# Card {device_index}: {', '.join(fields)}")
                if "allow_painting" in fields:
                    lines.append("# Also remove allow_inpainting, the alias for allow_painting.")
        lines.extend(
            [
                f"max_threads: {self.max_threads}",
                f"queue_size: {self.queue_size}",
                f"max_batch: {self.max_batch}",
                f"allow_lora: {str(self.allow_lora).lower()}",
                f"allow_controlnet: {str(self.allow_controlnet).lower()}",
                f"allow_sdxl_controlnet: {str(self.allow_sdxl_controlnet).lower()}",
                f"allow_post_processing: {str(self.allow_post_processing).lower()}",
            ]
        )
        # Omit an empty model list to preserve the operator's configuration. A bare key would load as null and
        # replace that list when merged.
        if self.models_to_load:
            lines.extend(["models_to_load:", *[f'  - "{model}"' for model in self.models_to_load]])
        lines.extend(
            [
                f"alchemist: {str(self.alchemist).lower()}",
                f"alchemy_allow_concurrent: {str(self.alchemy_allow_concurrent).lower()}",
                f"alchemy_max_concurrency: {self.alchemy_max_concurrency}",
            ]
        )
        if self.max_power is not None:
            lines.append(f"max_power: {self.max_power}")
        if self.allow_img2img is not None:
            lines.append(f"allow_img2img: {str(self.allow_img2img).lower()}")
        if self.allow_painting is not None:
            lines.append(f"allow_painting: {str(self.allow_painting).lower()}")
        for field_name in ("high_performance_mode", "moderate_performance_mode", "dynamic_models"):
            value = self.model_dump()[field_name]
            if value is not None:
                lines.append(f"{field_name}: {str(value).lower()}")
        if self.gpu_sampling_lease_enabled is not None:
            lines.append(f"gpu_sampling_lease_enabled: {str(self.gpu_sampling_lease_enabled).lower()}")
        if self.gpu_sampling_lease_slots is not None:
            lines.append(f"gpu_sampling_lease_slots: {self.gpu_sampling_lease_slots}")
        if self.gpu_device_indices is not None:
            lines.append(f"gpu_device_indices: {self.gpu_device_indices}")
        if self.gpu_overrides:
            lines.append("gpu_overrides:")
            for device_index, settings in sorted(self.gpu_overrides.items()):
                lines.append(f"  {device_index}:")
                for name, value in settings.model_dump().items():
                    if isinstance(value, list):
                        if not value:
                            continue
                        lines.append(f"    {name}:")
                        lines.extend(f'      - "{model}"' for model in value)
                    elif isinstance(value, bool):
                        lines.append(f"    {name}: {str(value).lower()}")
                    else:
                        lines.append(f"    {name}: {value}")
        return "\n".join(lines)

    def to_bridge_overrides(self) -> dict[str, object]:
        """The worker bridge-data fields this recommendation sets (for the validation run).

        ``max_batch`` is intentionally excluded: batch size is a per-job payload value, not a
        worker config field, so the soak applies it through its job templates instead.
        """
        overrides: dict[str, object] = {
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
        if self.allow_img2img is not None:
            overrides["allow_img2img"] = self.allow_img2img
        if self.allow_painting is not None:
            overrides["allow_painting"] = self.allow_painting
        return overrides


class TierCapability(BaseModel):
    """What one model tier proved during the run."""

    tier: BenchTier
    device_index: int | None = None
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


class EvidenceStatus(StrEnum):
    """How a workload claim is supported by an operator assessment."""

    MEASURED_SUCCESS = "measured_success"
    MEASURED_FAILURE = "measured_failure"
    PREDICTED = "predicted"
    UNKNOWN = "unknown"


class OperatingProfile(BaseModel):
    """The card, checkpoint and workload settings associated with a result."""

    device_index: int
    tier: BenchTier
    model_name: str
    width: int = 512
    height: int = 512
    batch: int = 1
    threads: int = 1
    queue_size: int = 1
    steps: int = 16
    work_jobs: int | None = None
    companion_model: str | None = None
    """A second installed checkpoint whose jobs alternate with the primary model's, so a concurrency
    check gives every lane its own model under a dispatch policy that keeps one model on one lane."""
    high_performance_mode: bool = False
    moderate_performance_mode: bool = False
    control_type: str | None = None
    workflow: str | None = None
    hires_fix: bool = False
    post_processing: list[str] = Field(default_factory=list)
    alchemy_forms: list[str] = Field(default_factory=list)
    lora_names: list[str] = Field(default_factory=list)
    lease_slots: int | None = None
    """The check's sampling-lease slot count, or None to use the operator's lease settings.

    A value enables the lease at that slot count, so one lane samples while the others load and encode
    their next job. The setting is global, and the recommendation carries it with the thread count."""


class EvidenceScope(StrEnum):
    """Where a workload's claim applies; a service form need not execute on the companion image's card."""

    CARD = "card"
    SHARED_WORKER = "shared_worker"


class ServicePlacement(BaseModel):
    """A service form's reported worker-lane affinity, separate from its actual execution device."""

    job_id: str
    form: str | None = None
    reported_device_index: int | None = None
    """The worker lane's reported card, if available; not proof that the form's operations used that GPU."""


class ReferenceRateBasis(StrEnum):
    """Where a model's single-lane sampling rate came from."""

    SINGLE_LANE = "single_lane"
    """The median rate for this model's jobs whose sampling did not overlap another job on the card."""
    RUN_MAXIMUM = "run_maximum"
    """The fastest rate observed for the model in the run. No job sampled alone, and the fastest rate
    understates the lane count a slower true single-lane rate would give."""


class ReferenceSamplingRate(BaseModel):
    """A model's single-lane sampling rate on one card, against which a concurrency check is normalised."""

    iterations_per_second: float
    basis: ReferenceRateBasis
    relative_spread: float = 0.0
    """The single-lane samples' median absolute deviation divided by their median.

    One sample or a run-maximum reference gives 0.
    """


class WorkloadEvidence(BaseModel):
    """A measured or estimated result with separate confirmation of combined limits."""

    evidence_id: str
    capability: Capability
    profile: OperatingProfile
    status: EvidenceStatus
    reason: str
    scope: EvidenceScope = EvidenceScope.CARD
    service_placements: list[ServicePlacement] = Field(default_factory=list)
    job_ids: list[str] = Field(default_factory=list)
    predicted_from: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    configure_seconds: float | None = None
    """Seconds the check spent applying its settings before work was offered, including a reused config."""
    sampling_seconds: float | None = None
    """Sum of the image jobs' reported sampling seconds, or None when no job reported one."""
    device_peak_mb: int | None = None
    ram_peak_mb: int | None = None
    recoveries: int = 0
    inference_failures: int = 0
    resource_failures: int = 0
    degraded_retries: int = 0
    host_commit_refusals: int = 0
    retry_provenance_complete: bool = False
    observed_sampling_concurrency: int = 0
    expected_sampling_concurrency: int = 0
    """The denoise overlap the profile can reach: its thread count, lowered to the sampling-lease slot count
    when the lease is enabled. Lanes beyond it stage the next job and never sample at the same time. Zero
    on reports written before the field existed, which readers treat as the thread count."""
    lane_equivalent_throughput: float | None = None
    """How many single lanes' worth of sampling a concurrency check delivered while any of its jobs sampled.

    A diagnostic of concurrent denoise. It blocks a thread count only when two or more sampling windows
    overlapped and it is below 1.0, which is a lane sampling on partly unloaded weights. With one sampler at
    a time it cannot see the staging the extra lanes do, so it is informational. None outside concurrency
    checks and when a job did not report a window, a rate or sampling seconds, or its model has no
    reference rate."""
    wall_megapixelsteps_per_second: float | None = None
    """Megapixelsteps the check's jobs delivered per second of wall time at the tested thread count, from the
    first pop to the last finalization. None when the jobs did not report the times or the megapixelsteps."""
    single_lane_wall_megapixelsteps_per_second: float | None = None
    """The same figure for the same jobs at one lane, measured first in the same check."""
    wall_gain: float | None = None
    """The ratio of measured throughput at the tested thread count to single-lane throughput."""
    wall_gain_required: float | None = None
    """One plus the relative median absolute deviation of the single-lane jobs' pop-to-finalize times, so a
    gain within the measured noise does not confirm a thread count."""
    jobs_per_minute: float | None = None
    """Image jobs finished per minute of wall time at the tested thread count."""
    single_lane_jobs_per_minute: float | None = None
    """Image jobs finished per minute of wall time at one lane."""
    model_sampling_rates: dict[str, float] = Field(default_factory=dict)
    """Median sampling rate per model across a concurrency check's image jobs."""
    model_reference_rates: dict[str, ReferenceSamplingRate] = Field(default_factory=dict)
    """The single-lane rate each model's check rate was compared with, and where it came from."""
    combined_validation: bool = False
    attempted: bool = False
    reason_code: str = ""
    jobs_completed: int = 0
    jobs_faulted: int = 0
    sampling_its: float | None = None


class ModelLoadingSummary(BaseModel):
    """Observed loader durations, which can include deserialization and reads from the OS cache."""

    model_name: str
    first_seconds: float
    repeat_seconds: list[float] = Field(default_factory=list)
    sampling_seconds: list[float] = Field(default_factory=list)

    @property
    def load_seconds(self) -> list[float]:
        """Return observed load durations, excluding missing measurements."""
        return [self.first_seconds, *self.repeat_seconds]


class AssessmentBudget(BaseModel):
    """Represents the shared wall-clock target and executed-work limit across all selected cards."""

    seconds: float | None = None
    max_jobs: int = 24
    jobs_offered: int = 0
    elapsed_seconds: float = 0.0
    stop_reason: str = ""


class CapabilityReport(BaseModel):
    """Results for a catalog run, including capability checks and suggested settings."""

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
    operator_assessment: bool = False
    evidence: list[WorkloadEvidence] = Field(default_factory=list)
    budget: AssessmentBudget | None = None
    model_loading: list[ModelLoadingSummary] = Field(default_factory=list)
    storage_notes: list[str] = Field(default_factory=list)
    model_guidance: list[str] = Field(default_factory=list)
    """Plain-language advice on which model families to offer, derived from the operator assessment's evidence."""

    @model_validator(mode="after")
    def _withhold_legacy_operator_suggestions(self) -> CapabilityReport:
        if self.operator_assessment and self.report_schema_version < _OPERATOR_EVIDENCE_GUARANTEES_VERSION:
            self.suggested_bridge_data.safe_to_apply = False
            note = (
                "This older assessment does not record retry history or enforce feature exclusions. "
                "Rerun before applying settings."
            )
            if note not in self.suggested_bridge_data.notes:
                self.suggested_bridge_data.notes.append(note)
        return self

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
    "AssessmentBudget",
    "CapabilityProbeResult",
    "CapabilityReport",
    "EvidenceScope",
    "EvidenceStatus",
    "Finding",
    "HarnessSummary",
    "MachineInfo",
    "OperatingProfile",
    "ReferenceRateBasis",
    "ReferenceSamplingRate",
    "ServicePlacement",
    "SuggestedBridgeData",
    "SuggestedCardSettings",
    "SuggestionBasis",
    "SuggestionDecision",
    "TierCapability",
    "WorkerCapabilities",
    "WorkloadEvidence",
    "classify_findings",
]
