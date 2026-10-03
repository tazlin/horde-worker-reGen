"""Host RAM reclaim: returning allocator-retained pages to the OS, and pricing the pages a preload can reuse.

An inference child keeps a freed model's pages resident after an unload; a preload onto that slot reuses them,
so the RAM budget prices such a swap at its marginal growth (the reuse credit) and later checks the settled RSS
against that charge. When the budget still cannot fit the head, the last resort is to cycle an idle slot whose
allocator will not give the pages back, bounded so the same slot is not cycled on consecutive attempts and so a
freshly spawned successor never qualifies. A disaggregation service lane re-pages its encoders on its next stage,
so its containment is an in-process unload rather than a cycle.

This module holds the constants, the reuse-credit ledger and the pure selections. The scheduler owns the
actuation (cycling a process, sending an unload) and the pricing inputs that need its other collaborators.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo, LoadCompletionSample
from horde_worker_regen.process_management.resources.ram_footprints import LearnedRamStore

FRESH_INFERENCE_CHILD_BASELINE_MB = 1100.0
"""Resident RSS (MB) a just-spawned inference child holds before it loads any model weights.

The interpreter, torch/CUDA import allocations and IPC scaffolding of a cold child (measured 1.03 to 1.1 GB). An
idle process's RSS above this is retained model pages the allocator kept after an unload, which a later preload
onto that slot reuses rather than allocates; the excess is what the reuse credit is computed from. The high end of
the measured range keeps the credit conservative, so the budget under-credits rather than over-admits."""

CREEP_CONTAINMENT_RSS_BYTES = 18432 * 1024 * 1024
"""Compatibility ceiling for callers without offered-model or private-memory telemetry.

The scheduler passes its model-aware ceiling and compares private working-set bytes when measured,
falling back to RSS. Clean checkpoint mappings alone are not evidence of allocator creep."""

STALE_RAM_UNLOAD_MARGIN_PERCENT = 2.0
"""Share of total host RAM added to the cold-child baseline to set the stale-unload reclaim threshold.

The threshold has to sit above anything a freshly spawned child can reach, or the reclaim cycles clean slots
forever: the successor of every cycle would itself qualify. Scaling with the host keeps that true across
machines, since a larger host's children carry proportionally larger interpreter caches and allocator arenas."""

STALE_RAM_UNLOAD_MARGIN_FLOOR_MB = 512.0
"""Smallest stale-unload margin (MB), so a small host still clears a cold child's ordinary variance."""

STALE_RAM_UNLOAD_CYCLE_MIN_INTERVAL_SECONDS = 120.0
"""How long one slot is exempt from another stale-unload cycle after being cycled.

A cycle costs a respawn plus a cold load, so a slot cycling on consecutive reclaim attempts spends more of the
pool than the RAM it returns. Sustained pressure spreads its cycles across the pool instead."""

LANE_RAM_CONTAINMENT_RSS_BYTES = 10240 * 1024 * 1024
"""Idle service-lane RSS above which the lane is asked to unload its models from RAM.

A disaggregation service lane keeps its components resident across jobs and, alternating between the hot pool
models, ratchets its resident set well past what its live encoders occupy. It re-pages its encoders on the next
stage, so the remedy is an in-process unload rather than a cycle. Far below the host RAM danger floor so the
containment runs as a bounded sawtooth before the floor is threatened, and above what a lane holding only its
working encoders occupies."""

LANE_RAM_CONTAINMENT_MIN_INTERVAL_SECONDS = 180.0
"""Minimum time between two RAM-unload requests to the same idle service lane, so the reload cost stays
negligible against the RAM returned."""

LOAD_COMPLETION_REPORT_TIMEOUT_SECONDS = 600.0
"""How long an admitted load may go without its load-completion report before its record is dropped unlearned.

Longer than a cold load of the largest checkpoint from a slow disk; a load still unreported by then failed or
was abandoned, and its record would otherwise wait forever."""

REUSE_CREDIT_RECONCILE_SLACK_MB = 2048.0
"""How far a credited admission's measured peak growth may exceed its charge before it is flagged too generous.

Absorbs ordinary per-job creep and measurement noise so only a materially over-generous credit is reported."""

RAM_RECLAIM_CYCLE_GRACE_SECONDS = 60.0
"""How long after a deliberate reclaim cycle the recovery supervisor keeps ignoring a queue wedge.

The cycle restarts the slot and the next head must then preload onto it, a window in which the queue is
unservable by the worker's own bounded action rather than a wedge. Covers the respawn plus preload window and no
more, so a cycle that never recovers still trips the supervisor."""


class ReuseCreditKind(enum.StrEnum):
    """Which marginal accounting priced a credited RAM admission."""

    WHOLE = "whole"
    PAGE_REUSE = "page_reuse"
    """The charge was reduced by the staging target's retained resident pages."""
    COMPONENT = "component"
    """The charge was a disaggregation-class job's UNet-only component charge rather than the whole checkpoint."""


@dataclass(frozen=True)
class ReuseCreditRecord:
    """A credited RAM admission awaiting the measured-truth check against its target's settled RSS.

    Held per target process; superseded if that slot is credited again before it settles.
    """

    model: str
    """The model the credited preload was staging onto the target."""
    private_at_admit_mb: float
    """The target's private working set (MB, RSS fallback) at admit time, the baseline growth is measured from."""
    effective_charge_mb: float
    """The checkpoint charge (MB) the admission priced the load at; the growth is reconciled against this."""
    admitted_at: float
    """When the admission was recorded, on the scheduler clock children stamp their reports with."""
    kind: ReuseCreditKind = ReuseCreditKind.PAGE_REUSE
    process_launch_identifier: int | None = None
    baseline: str | None = None
    """The model's baseline, under which the load also counts as evidence for its siblings."""
    size_mb: float | None = None
    """The staged file's size (MB), which scales baseline evidence to another checkpoint."""


@dataclass(frozen=True)
class ReuseCreditDiscrepancy:
    """A settled admission whose measured peak growth exceeded its charge by more than the slack."""

    process_id: int
    record: ReuseCreditRecord
    growth_mb: float


class RamCycleReason(enum.Enum):
    """Why an idle inference slot is being cycled to return RAM to the OS."""

    CREEP = enum.auto()
    """Its private working set (RSS fallback) exceeds the model-aware creep ceiling."""
    STALE_UNLOAD = enum.auto()
    """It did not release RAM after an unload request; the allocator retains the freed model's pages."""


def stale_ram_unload_replace_bytes(total_ram_mb: float) -> float:
    """RSS above which a model-less idle slot is judged to still hold a freed model's pages.

    The cold-child baseline plus a host-scaled margin, so a freshly spawned child can never reach it. That
    invariant is what keeps the reclaim from cycling the very successor its last cycle spawned.
    """
    margin_mb = max(total_ram_mb * (STALE_RAM_UNLOAD_MARGIN_PERCENT / 100.0), STALE_RAM_UNLOAD_MARGIN_FLOOR_MB)
    return (FRESH_INFERENCE_CHILD_BASELINE_MB + margin_mb) * 1024 * 1024


def staging_reuse_credit_mb(target: HordeProcessInfo) -> float:
    """The retained, reusable resident RSS (MB) a preload onto ``target`` can reuse instead of allocating.

    A busy target's pages are in live use and earn no credit; a fresh slot at baseline yields zero, which
    collapses the verdict to the ordinary full charge.
    """
    if target.is_process_busy():
        return 0.0
    target_rss_mb = max(0, target.ram_usage_bytes) / (1024 * 1024)
    return max(0.0, target_rss_mb - FRESH_INFERENCE_CHILD_BASELINE_MB)


def select_ram_cycle_victim(
    process_infos: Iterable[HordeProcessInfo],
    *,
    now: float,
    protect_process_id: int | None,
    cycled_at: Mapping[int, float],
    stale_replace_bytes: float,
    creep_ceiling_bytes: float = CREEP_CONTAINMENT_RSS_BYTES,
) -> tuple[HordeProcessInfo, RamCycleReason] | None:
    """Choose the idle inference slot to cycle for RAM, creep victims ahead of stale-unload victims.

    A slot just routed a preload is mid-stage and is spared from both triggers. Creep containment ignores the
    unload state, the resident model and ``protect_process_id``, since a crept slot is leak either way. The
    stale-unload trigger takes a model-less slot whose last control flag is the RAM unload, judged only against a
    reading sampled after that request, above ``stale_replace_bytes``, at most once per slot per
    :data:`STALE_RAM_UNLOAD_CYCLE_MIN_INTERVAL_SECONDS`; the protected slot (the head's staging target, whose
    retained pages the reuse credit priced) is exempt.
    """
    creep_victim: HordeProcessInfo | None = None
    stale_victim: HordeProcessInfo | None = None
    for process_info in process_infos:
        if process_info.process_type != HordeProcessType.INFERENCE:
            continue
        if process_info.is_process_busy():
            continue
        if process_info.last_control_flag == HordeControlFlag.PRELOAD_MODEL:
            continue
        if creep_victim is None and process_info.ram_working_set_bytes >= creep_ceiling_bytes:
            creep_victim = process_info
            continue
        if stale_victim is not None:
            continue
        if process_info.process_id == protect_process_id:
            continue
        if process_info.loaded_horde_model_name is not None:
            continue
        if process_info.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
            continue
        if not process_info.ram_reading_postdates_unload():
            continue
        last_cycled_at = cycled_at.get(process_info.process_id)
        if last_cycled_at is not None and (now - last_cycled_at) < STALE_RAM_UNLOAD_CYCLE_MIN_INTERVAL_SECONDS:
            continue
        if process_info.ram_usage_bytes < stale_replace_bytes:
            continue
        stale_victim = process_info

    if creep_victim is not None:
        return creep_victim, RamCycleReason.CREEP
    if stale_victim is not None:
        return stale_victim, RamCycleReason.STALE_UNLOAD
    return None


def idle_lanes_over_ram_ceiling(
    process_infos: Iterable[HordeProcessInfo],
    *,
    now: float,
    contained_at: Mapping[int, float],
) -> list[HordeProcessInfo]:
    """The idle service lanes holding more resident RAM than the containment ceiling and not recently asked."""
    lanes: list[HordeProcessInfo] = []
    for process_info in process_infos:
        if process_info.process_type not in (HordeProcessType.COMPONENT, HordeProcessType.VAE_LANE):
            continue
        if not process_info.is_process_alive() or not process_info.can_accept_job():
            continue
        if process_info.ram_usage_bytes < LANE_RAM_CONTAINMENT_RSS_BYTES:
            continue
        last_contained = contained_at.get(process_info.process_id)
        if last_contained is not None and (now - last_contained) < LANE_RAM_CONTAINMENT_MIN_INTERVAL_SECONDS:
            continue
        lanes.append(process_info)
    return lanes


class RamReclaimLedger:
    """The reuse credits awaiting reconciliation and the clocks that bound RAM reclaim actuations."""

    def __init__(self, clock: Callable[[], float]) -> None:
        """Track reclaim state against ``clock``."""
        self._clock = clock
        self.pending_reuse_credits: dict[int, ReuseCreditRecord] = {}
        """Credited admissions awaiting the measured-truth check, keyed by target process id."""
        self.stale_cycled_at: dict[int, float] = {}
        """When each slot was last cycled by the stale-unload reclaim, keyed by process id (which outlives the
        replaced process object). Creep containment is deliberately not throttled by it."""
        self.lane_contained_at: dict[int, float] = {}
        """When each service lane was last asked to unload its models from RAM, keyed by process id."""
        self.learned_ram = LearnedRamStore()
        self.cycle_at: float = 0.0
        """When an idle slot was last deliberately cycled to reclaim RAM; 0.0 when no cycle is in flight."""
        self._last_announced_admission: tuple[int, str | None, int] | None = None

    def announce_admission(self, key: tuple[int, str | None, int]) -> bool:
        """Whether a credited admission with this (target, model, rounded charge) key is new since the last one.

        Edge-triggered so the sub-second loop cannot re-log an unchanged decision.
        """
        if key == self._last_announced_admission:
            return False
        self._last_announced_admission = key
        return True

    def record_credit(
        self,
        target: HordeProcessInfo,
        *,
        model: str,
        effective_charge_mb: float,
        kind: ReuseCreditKind,
        baseline: str | None = None,
        size_mb: float | None = None,
    ) -> None:
        """Record a credited admission onto ``target`` for the measured-truth check once its load settles."""
        self.pending_reuse_credits[target.process_id] = ReuseCreditRecord(
            model=model,
            private_at_admit_mb=max(0, target.ram_working_set_bytes) / (1024 * 1024),
            effective_charge_mb=effective_charge_mb,
            admitted_at=self._clock(),
            kind=kind,
            process_launch_identifier=target.process_launch_identifier,
            baseline=baseline,
            size_mb=size_mb,
        )

    def void_credit(self, process_id: int) -> None:
        """Drop a slot's pending credit: a cycled slot's successor cold-loads, so the swap never settles."""
        self.pending_reuse_credits.pop(process_id, None)

    def settle_credits(self, process_map: Mapping[int, HordeProcessInfo]) -> list[ReuseCreditDiscrepancy]:
        """Retire every admission whose load-completion report arrived, returning those that outgrew their charge.

        Only the report that completed this admission's load counts: same launch, same model, sampled after
        the admission. Until it arrives the record stays pending, so a delayed report is still learned from.
        A record whose launch was replaced, or whose report never arrives within
        :data:`LOAD_COMPLETION_REPORT_TIMEOUT_SECONDS`, is dropped without learning. Growth below the charge is
        the intended outcome of marginal load accounting and is not reported.
        """
        now = self._clock()
        discrepancies: list[ReuseCreditDiscrepancy] = []
        for process_id, record in list(self.pending_reuse_credits.items()):
            process_info = process_map.get(process_id)
            if process_info is None or (
                record.process_launch_identifier is not None
                and record.process_launch_identifier != process_info.process_launch_identifier
            ):
                del self.pending_reuse_credits[process_id]
                continue
            sample = process_info.load_completion_sample
            if sample is None or not self._completes(record, sample):
                if now - record.admitted_at >= LOAD_COMPLETION_REPORT_TIMEOUT_SECONDS:
                    del self.pending_reuse_credits[process_id]
                continue
            del self.pending_reuse_credits[process_id]
            settled_mb = sample.private_bytes / (1024 * 1024) - record.private_at_admit_mb
            peak_mb = sample.peak_private_bytes / (1024 * 1024) - record.private_at_admit_mb
            if record.effective_charge_mb > 0:
                self.learned_ram.observe(
                    record.model,
                    record.kind,
                    settled_mb,
                    peak_mb,
                    baseline=record.baseline,
                    size_mb=record.size_mb,
                )
            if peak_mb > record.effective_charge_mb + REUSE_CREDIT_RECONCILE_SLACK_MB:
                discrepancies.append(ReuseCreditDiscrepancy(process_id, record, peak_mb))
        return discrepancies

    @staticmethod
    def _completes(record: ReuseCreditRecord, sample: LoadCompletionSample) -> bool:
        """Whether ``sample`` is the report that completed ``record``'s load."""
        return (
            sample.model == record.model
            and (
                record.process_launch_identifier is None
                or sample.process_launch_identifier == record.process_launch_identifier
            )
            and sample.sampled_at is not None
            and sample.sampled_at > record.admitted_at
        )

    def note_cycle(self) -> None:
        """Open the reclaim-cycle grace: a slot is respawning and the next head must preload onto it."""
        self.cycle_at = self._clock()

    def cycle_grace_active(self) -> bool:
        """Whether a deliberate reclaim cycle is still inside its bounded respawn and preload window."""
        if self.cycle_at == 0.0:
            return False
        return (self._clock() - self.cycle_at) < RAM_RECLAIM_CYCLE_GRACE_SECONDS
