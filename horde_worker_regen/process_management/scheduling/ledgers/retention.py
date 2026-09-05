"""VRAM retention: leaving a job's weights on the card for the same-model successor predicted to follow it.

hordelib evicts the model after every job so sibling GPU instances never collectively overcommit; the next
same-model job then pays a RAM-to-VRAM reload, the dominant non-sampling cost on small jobs. A retention grant
suppresses that eviction for one job. Eviction elsewhere is on-demand and verified (the device-free governor and
the reclaim ladder), so a grant does not need to be stingy about fit: it needs to be stingy about evidence,
because a copy nothing comes back for holds the card, prices every later grant against itself, and saves nothing.

The evidence is trailing, never a queue lookahead: the pop cycle refills the queue after a dispatch drains it, so
a same-model successor is almost never visible at the dispatch instant even when one arrives milliseconds later.
What the slot has already been asked to run predicts the successor the queue cannot yet show.

A grant is bounded rather than permanent. A cross-model dispatch onto the retaining slot evicts before it loads,
the reclaim ladder treats retained residents as first-class candidates, and a hold that goes unreused past the
stale horizon is swept off a card that stays under pressure.

This module holds the retention state and the pure arithmetic. The scheduler owns the actuation (issuing unloads)
and the pricing inputs that need its other collaborators.
"""

from __future__ import annotations

import enum
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from loguru import logger

from horde_worker_regen.process_management.ipc.messages import ModelLoadState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap

RETENTION_REPEAT_EVIDENCE_DISPATCHES = 3
"""How many of a slot's trailing dispatches are searched for a repeat before retention is granted on it.

A worker offering one model repeats on every dispatch and clears this at once; a wide rotation repeats rarely
and is granted correspondingly little. Three is short enough that a slot which has moved on stops earning grants
within a job or two, and long enough that a two-model alternation on one slot still reads as repeating.
"""

RETENTION_STALE_HOLD_SECONDS = 60.0
"""How long a retained copy may go unreused before the prediction that issued it counts as falsified.

The dispatch history that issued a grant can never refute it afterwards (the grant's own dispatch heads that
history), so only the successor failing to arrive can. Expressed in seconds of demand rather than jobs or bytes
so it means the same thing on any card and any offer size.
"""

RETENTION_PRESSURE_REVOKE_SECONDS = 15.0
"""How long a card must be continuously off HEALTHY before stale retained residents are revoked.

Debounced because a reload costs seconds of earning time and a momentary dip often clears on its own.
"""

RETENTION_EVICTION_CONFIRMATION_PASSES = 3
"""Scheduling passes a dispatch waits for its own retention eviction to be evidenced at the device.

A child frees before it reports. Past the bound the record is dropped so a child whose reports never arrive
cannot park the queue; the measured admission gate still prices the dispatch against device truth every pass.
"""


class RetentionDenialReason(enum.Enum):
    """Which gate refused a VRAM retention grant.

    A worker that grants nothing is indistinguishable at the duty figure from one whose grants are all evicted
    before reuse. Bucketing the refusals separates traffic retention cannot help (no repeat evidence) from traffic
    it could help on a card that will not carry it (static fit, governor state).
    """

    ACTUATION_DISABLED = "actuation_disabled"
    """The legacy comfy unload regime is configured, so the child returns the card whatever the grant says."""
    BUDGET_INACTIVE = "budget_inactive"
    """Measured VRAM budgeting is off, so nothing can price what holding the weights would cost."""
    WDDM_PAGING = "wddm_paging"
    """The driver is already demand-paging the worker's allocations; holding weights can only deepen it."""
    NO_REPEAT_EVIDENCE = "no_repeat_evidence"
    """The slot's trailing dispatches do not contain this model, so nothing predicts a same-model successor."""
    GOVERNOR_STATE = "governor_state"
    """The card is PRESSURE or SATURATED, so the reclaim ladder holds priority over a new resident."""
    UNPRICEABLE = "unpriceable"
    """A sibling context or an existing retained resident shares the card at a cost not yet measured."""
    STATIC_FIT = "static_fit"
    """The card's total cannot absorb this job's peak beside what retention already holds on it."""


@dataclass
class PendingRetentionEviction:
    """A retained resident whose eviction was issued and has not yet been seen to land on the card.

    Residency tracking clears the instant the unload is sent, but the bytes come back only when the child has
    freed them. The record carries the card's free reading and the slot's reservation as they stood when the
    unload went out, so the child's post-free reports are what release a dispatch held against it.
    """

    model: str
    reserved_baseline_mb: float | None
    device_free_baseline_mb: float | None
    passes_waited: int = 0


@dataclass(frozen=True)
class RetentionFit:
    """The static-fit arithmetic behind a retention grant or a dispatch hold, with the figures that produced it.

    The margin on top of the peak is the admission noise buffer, not the operator's ``vram_reserve_mb``: that
    reserve is a sampling headroom term, and stacking it on an activation-inclusive learned peak denies retention
    on a small card by a few dozen MB every job. ``granted`` is True when the peak is unknown, since a static gate
    cannot refuse on a figure it does not have.
    """

    predicted_mb: float | None
    noise_mb: float
    total_vram_mb: float
    static_charges_mb: float
    retained_resident_mb: float
    committed_reserve_mb: float

    @property
    def effective_available_mb(self) -> float:
        """Card total net of sibling contexts, retained residents and in-flight commitments."""
        return self.total_vram_mb - self.static_charges_mb - self.retained_resident_mb - self.committed_reserve_mb

    @property
    def granted(self) -> bool:
        """Whether the peak plus the noise buffer fits in what the card has left."""
        return self.predicted_mb is None or (self.predicted_mb + self.noise_mb) <= self.effective_available_mb

    def describe(self) -> str:
        """The gate figures for a log line."""
        retained = f", retained residents {self.retained_resident_mb:.0f}MB" if self.retained_resident_mb > 0 else ""
        return (
            f"static: peak {self.predicted_mb} + noise {self.noise_mb:.0f} vs {self.effective_available_mb:.0f}MB "
            f"(total {self.total_vram_mb:.0f}MB minus sibling contexts, the job's own post-processing, and "
            f"in-flight commitments{retained})"
        )


@dataclass(frozen=True)
class RetentionUnpriceable:
    """A static fit that could not be judged, and which gate could not price it."""

    reason: RetentionDenialReason
    detail: str | None
    """Why the gate could not price it, for the log line; None when the refusal is tallied silently."""


def _on_card(process_info: HordeProcessInfo, device_index: int | None) -> bool:
    return device_index is None or process_info.device_index == device_index


def sibling_retained_resident_present(
    processes: Iterable[HordeProcessInfo],
    *,
    target_id: int,
    device_index: int | None,
) -> bool:
    """Whether a slot other than ``target_id`` is holding weights on this card between jobs."""
    return any(
        p.process_id != target_id and _on_card(p, device_index) and p.retained_resident_model is not None
        for p in processes
    )


def idle_retained_resident_mb(processes: Iterable[HordeProcessInfo], device_index: int | None) -> float:
    """Device memory (MB) idle inference slots on a card hold under a retention grant.

    Priced by each slot's measured allocator reservation, which is what an eviction returns. A retained slot
    with no reservation reading adds nothing, and a busy slot's weights are in use rather than reclaimable.
    """
    return sum(
        float(p.process_reserved_mb)
        for p in processes
        if p.process_type == HordeProcessType.INFERENCE
        and _on_card(p, device_index)
        and p.retained_resident_model is not None
        and not p.is_process_busy()
        and p.process_reserved_mb is not None
    )


def idle_lane_component_charges_mb(
    processes: Iterable[HordeProcessInfo],
    *,
    target_id: int,
    device_index: int | None,
) -> float:
    """VRAM (MB) idle lanes report holding in their component caches between jobs.

    Read from what each lane reports rather than predicted, since a cache's contents are the lane's own history.
    A lane mid-stage is priced by that stage's own admission and is not charged again here.
    """
    return sum(
        sum(max(0.0, held.approx_ram_mb) for held in p.held_components)
        for p in processes
        if p.process_id != target_id and _on_card(p, device_index) and not p.is_process_busy() and p.held_components
    )


_CONTEXT_HOLDING_TYPES = frozenset(
    {
        HordeProcessType.INFERENCE,
        HordeProcessType.POST_PROCESS,
        HordeProcessType.VAE_LANE,
        HordeProcessType.COMPONENT,
    },
)


def sibling_context_count(
    processes: Iterable[HordeProcessInfo],
    *,
    target_id: int,
    device_index: int | None,
    safety_on_gpu: bool,
) -> int:
    """How many other live GPU processes on this card hold a CUDA context, with or without a model."""
    return sum(
        1
        for p in processes
        if p.process_id != target_id
        and _on_card(p, device_index)
        and (p.process_type in _CONTEXT_HOLDING_TYPES or (p.process_type == HordeProcessType.SAFETY and safety_on_gpu))
    )


def retained_resident_charges_mb(
    processes: Iterable[HordeProcessInfo],
    *,
    target_id: int,
    dispatched_model: str,
    device_index: int | None,
    include_target_retained: bool,
    footprint_mb: Callable[[HordeProcessInfo, str], float | None],
) -> float | None:
    """VRAM (MB) of weights already held on the card by earlier retention grants, or None when one is unpriceable.

    Every later grant's peak has to fit beside these; charging only live contexts lets a run of grants sum past
    the card. A same-model re-grant on the target charges nothing (the reuse is the point), and a caller whose
    load is preceded by an eviction of the target's own retained weights passes ``include_target_retained=False``.
    """
    charges_mb = 0.0
    for p in processes:
        if not _on_card(p, device_index):
            continue
        retained_model = p.retained_resident_model
        if retained_model is None:
            continue
        if p.process_id == target_id and (retained_model == dispatched_model or not include_target_retained):
            continue
        footprint = footprint_mb(p, retained_model)
        if footprint is None:
            return None
        charges_mb += max(0.0, footprint)
    return charges_mb


class RetentionLedger:
    """Session tallies, per-slot dispatch history, hold ages and in-flight evictions for VRAM retention."""

    def __init__(self, clock: Callable[[], float]) -> None:
        """Track retention state on ``clock``, the clock every retention window is measured on."""
        self._clock = clock
        self._grants_issued = 0
        self._grant_denials: dict[RetentionDenialReason, int] = {}
        self._reuses = 0
        self._evicted_unused = 0
        self._revokes = 0
        self._affinity_reorders = 0
        self._reorder_pareto_vetoes = 0
        self._slot_dispatch_history: dict[int, deque[str]] = {}
        self._pending_evictions: dict[int, PendingRetentionEviction] = {}
        self._pressure_since: dict[int, float] = {}
        self.wddm_paging_active: bool = False
        """The parent's measured WDDM demand-paging verdict on the worker's own children. While set, retention
        is denied outright: holding weights in a regime the driver is already paging can only deepen it."""
        self.wddm_paging_victims_shared_mb_by_pid: dict[int, float] = {}
        """The child PIDs whose VRAM the driver most recently demoted, mapped to their shared (system-backed) GPU
        MB. Refreshed on every active verdict so a watchdog reads a current set; cleared the moment paging clears."""
        self.wddm_paging_victims_at: float = 0.0
        """When the victim set was last recorded."""

    def note_wddm_paging(self, victims_shared_mb_by_pid: Mapping[int, float], *, active: bool) -> bool:
        """Record the parent's paging verdict; return whether this is the rising edge.

        The rising edge is the one moment to reclaim idle resident VRAM: a repeat of an active verdict only
        refreshes the victim set, and a cleared verdict empties it so a stale set cannot outlive the pressure
        that produced it.
        """
        was_active = self.wddm_paging_active
        self.wddm_paging_active = active
        if active:
            self.wddm_paging_victims_shared_mb_by_pid = dict(victims_shared_mb_by_pid)
            self.wddm_paging_victims_at = self._clock()
        else:
            self.wddm_paging_victims_shared_mb_by_pid = {}
        return active and not was_active

    def wddm_paging_victims(self, max_age_seconds: float) -> dict[int, float]:
        """The victim set while it is younger than ``max_age_seconds``, else empty.

        A stale or absent verdict yields nothing, so a caller can never act on a paging episode that has
        already cleared or whose telemetry has stopped arriving.
        """
        if not self.wddm_paging_victims_shared_mb_by_pid:
            return {}
        if (self._clock() - self.wddm_paging_victims_at) > max_age_seconds:
            return {}
        return dict(self.wddm_paging_victims_shared_mb_by_pid)

    @property
    def grants_issued(self) -> int:
        """Dispatches this session whose weights were left on the card."""
        return self._grants_issued

    @property
    def grant_denials(self) -> Mapping[RetentionDenialReason, int]:
        """Refused grants this session, keyed by the gate that refused."""
        return self._grant_denials

    @property
    def reuses(self) -> int:
        """Dispatches this session that landed on a slot already retaining the job's model: uploads avoided.

        Read against :attr:`evicted_unused`; the two partition every retention episode.
        """
        return self._reuses

    @property
    def evicted_unused(self) -> int:
        """Retained copies given back this session before any successor reused them.

        A slot whose child died carries its residency out with the process and is not counted here.
        """
        return self._evicted_unused

    @property
    def revokes(self) -> int:
        """Retained copies the sustained-pressure sweep took back as stale this session."""
        return self._revokes

    @property
    def affinity_reorders(self) -> int:
        """Jobs this session seated ahead of the queue head onto weights a slot already retained."""
        return self._affinity_reorders

    @property
    def reorder_pareto_vetoes(self) -> int:
        """Reorder candidates refused because the head's own load had nowhere else to go."""
        return self._reorder_pareto_vetoes

    def note_grant(self) -> None:
        """Tally a granted retention."""
        self._grants_issued += 1

    def note_denial(self, reason: RetentionDenialReason) -> None:
        """Tally a refused grant against the gate that refused it."""
        self._grant_denials[reason] = self._grant_denials.get(reason, 0) + 1

    def note_reuse_if_retained(self, process_info: HordeProcessInfo, model: str) -> None:
        """Tally a dispatch landing on the model its slot retains, and end that hold's episode.

        The bet paid, so the hold that follows this job is a new prediction with the full horizon to be met.
        """
        if process_info.retained_resident_model != model:
            return
        self._reuses += 1
        process_info.retained_resident_since = None

    def note_evicted_unused(self, process_info: HordeProcessInfo) -> None:
        """Tally a retained copy about to be given back without any successor having reused it."""
        if process_info.retained_resident_model is not None:
            self._evicted_unused += 1

    def note_revoke(self) -> None:
        """Tally a stale hold revoked under pressure."""
        self._revokes += 1

    def note_affinity_reorder(self) -> None:
        """Tally a job seated ahead of the head onto retained weights."""
        self._affinity_reorders += 1

    def note_reorder_pareto_veto(self) -> None:
        """Tally a reorder candidate refused for lack of a spare load target."""
        self._reorder_pareto_vetoes += 1

    def record_slot_dispatch(self, process_id: int, model: str) -> None:
        """Record a committed dispatch of ``model`` to a slot, newest first.

        Recorded after the grant verdict, never before: the gate asks what the slot ran previously, and a job
        already in the history would satisfy the repeat test with itself.
        """
        history = self._slot_dispatch_history.get(process_id)
        if history is None:
            history = deque(maxlen=RETENTION_REPEAT_EVIDENCE_DISPATCHES + 1)
            self._slot_dispatch_history[process_id] = history
        history.appendleft(model)

    def slot_has_repeat_evidence(self, process_id: int, model: str, *, exclude_latest: bool = False) -> bool:
        """Whether the slot's trailing dispatches contain ``model``, so a same-model successor is predicted.

        ``exclude_latest`` skips the newest dispatch so a live grant can be re-asked the question its issuance
        answered, rather than a looser one its own dispatch would satisfy.
        """
        history = self._slot_dispatch_history.get(process_id)
        if history is None:
            return False
        window = list(history)[1:] if exclude_latest else list(history)
        return model in window[:RETENTION_REPEAT_EVIDENCE_DISPATCHES]

    def stamp_hold_ages(self, processes: Iterable[HordeProcessInfo]) -> None:
        """Start the clock on any retention episode that has begun holding since the last tick.

        The settle that creates a retention runs on the completion path, which has no scheduler clock, so the
        stamp is taken here. An episode already stamped keeps its start; an ended one carries no stamp forward.
        """
        now = self._clock()
        for process_info in processes:
            if process_info.retained_resident_model is None:
                process_info.retained_resident_since = None
            elif process_info.retained_resident_since is None:
                process_info.retained_resident_since = now

    def hold_is_stale(self, process_info: HordeProcessInfo) -> bool:
        """Whether the slot's retained weights have gone unreused past the falsification horizon.

        An unstamped hold is never stale: its age is unknown and must not be revoked on an assumption.
        """
        held_since = process_info.retained_resident_since
        if held_since is None:
            return False
        return (self._clock() - held_since) >= RETENTION_STALE_HOLD_SECONDS

    def record_pending_eviction(self, process_id: int, pending: PendingRetentionEviction) -> None:
        """Track an issued retention eviction until the card evidences the room is back."""
        self._pending_evictions[process_id] = pending

    def eviction_pending(self, processes_by_id: Mapping[int, HordeProcessInfo], device_index: int | None) -> bool:
        """Whether a retention eviction issued for this card has not yet been evidenced at the device."""
        for process_id in self._pending_evictions:
            process_info = processes_by_id.get(process_id)
            if process_info is not None and _on_card(process_info, device_index):
                return True
        return False

    def prune_confirmed_evictions(
        self,
        processes_by_id: Mapping[int, HordeProcessInfo],
        model_map: HordeModelMap,
        *,
        measured_free_mb: Callable[[HordeProcessInfo], float | None],
    ) -> None:
        """Drop the pending evictions the card has evidenced, leaving only the unlanded ones.

        A child frees first and reports after. A risen device free reading, a fallen slot reservation, the map no
        longer placing the weights on the slot, the slot no longer naming the model, or the slot being gone all
        evidence the room is back. Absent all of them the record is kept for a bounded number of passes.
        """
        for process_id, pending in list(self._pending_evictions.items()):
            process_info = processes_by_id.get(process_id)
            if process_info is None or process_info.loaded_horde_model_name != pending.model:
                del self._pending_evictions[process_id]
                continue
            reserved_mb = process_info.process_reserved_mb
            if (
                pending.reserved_baseline_mb is not None
                and reserved_mb is not None
                and float(reserved_mb) < pending.reserved_baseline_mb
            ):
                del self._pending_evictions[process_id]
                continue
            device_free_mb = measured_free_mb(process_info)
            if (
                pending.device_free_baseline_mb is not None
                and device_free_mb is not None
                and device_free_mb > pending.device_free_baseline_mb
            ):
                del self._pending_evictions[process_id]
                continue
            model_info = model_map.root.get(pending.model)
            map_places_weights_here = (
                model_info is not None
                and model_info.process_id == process_id
                and model_info.horde_model_load_state in (ModelLoadState.LOADED_IN_VRAM, ModelLoadState.IN_USE)
            )
            if not map_places_weights_here:
                del self._pending_evictions[process_id]
                continue
            pending.passes_waited += 1
            if pending.passes_waited >= RETENTION_EVICTION_CONFIRMATION_PASSES:
                logger.debug(
                    f"Retention eviction of {pending.model} on process {process_id} went unevidenced for "
                    f"{pending.passes_waited} passes; the dispatch waiting on it now stands on the measured "
                    "admission gate alone.",
                )
                del self._pending_evictions[process_id]

    def pressure_sustained(self, device_index: int, *, healthy: bool) -> bool:
        """Whether a card has now been continuously off HEALTHY for the revoke debounce.

        A HEALTHY commit clears the card's pressure start, so a momentary dip never reaches the sweep.
        """
        if healthy:
            self._pressure_since.pop(device_index, None)
            return False
        since = self._pressure_since.setdefault(device_index, self._clock())
        return (self._clock() - since) >= RETENTION_PRESSURE_REVOKE_SECONDS
