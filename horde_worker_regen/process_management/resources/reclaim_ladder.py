"""The verified LIFO reclaim ladder: the worker's single owner of device-VRAM pressure relief.

When the device-free governor calls a card SATURATED (device-level free below the hard floor), the card is
at or past the WDDM paging cliff and memory must come back now. This module owns that reclaim as one engine
so there are never two mechanisms independently evicting against the same card. It does two things a naive
"unload something" call cannot:

- It reclaims in LIFO order (most-recently-materialized tenant first). Under WDDM the driver demotes the
  least-recently-touched allocator, so the newest idle resident is both the likeliest squatter and the
  cheapest to give back (its weights are still warm in RAM). The rung order is fixed: unload the newest idle
  resident model, then release the reclaimable allocator caches on idle processes, then evict the older idle
  residents, then pause the post-processing / VAE / component lanes, then move safety off the GPU. An
  actively-sampling process is never a rung: it is the one process the driver did not demote, and tearing it
  down would trade a slow job for a faulted one.

- It verifies. Freeing on WDDM is externally checkable: NVML device-used drops once a real release lands.
  After issuing a rung the engine watches the following governor samples and compares the realized device-free
  gain against the rung's promised figure (the tenant's footprint / reclaimable reservation). A rung that
  yields less than half of what it promised *and shows no further progress for its whole time budget* is
  logged against the tenant it named, recorded as a calibration event, and the engine escalates to the next
  rung rather than trusting the estimate. When the whole ladder is exhausted and the card is still SATURATED,
  the episode is marked unresolved: nothing the worker can give back relieved the card, which is the signal a
  later kill rung reads.

Verification is budgeted in wall-clock seconds rather than in samples because no rung frees synchronously.
Every rung is an asynchronous actuation: an unload is an IPC the child services between allocations, and a
multi-gigabyte WDDM release lands seconds after the parent sent it; a lane pause or a safety move returns its
memory only once the OS has torn the process down. Counting samples grades a working release as a failure
whenever the control loop ticks faster than the driver frees, which escalates the whole ladder in the seconds
before the rung that was already working takes effect. Each budget is a fixed latency allowance plus a term
scaled by the megabytes the rung promised, so the same judgement holds on an 8GB card and a 24GB one rather
than being tuned to whichever card it was measured on. Observed progress extends the budget, so a release still
arriving keeps its rung, and a rung that has moved nothing for its full budget is a genuine shortfall.

The engine is driven from the parent's single-threaded control loop, one call per governor tick per card, and
holds all cross-tick verification state per device. It touches no process state itself: a
:class:`ReclaimLadderActuator` (implemented by the scheduler, which owns process lifecycle) performs each rung
and reports a calibration shortfall, exactly as the arbiter describes actuations for a caller to run.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from loguru import logger

from horde_worker_regen.process_management.resources.vram_arbiter import (
    ActuatorCommand,
    ActuatorCommandKind,
    VramActuator,
)

_VERIFICATION_BASE_SECONDS = 4.0
"""Fixed part of an in-process/IPC rung's progress-free budget: the latency before any memory can come back.

Applies to the rungs actuated through a child's control pipe (model unload, allocator-cache release). Those are
asynchronous: the parent sends the command, the child picks it up between its own allocations, and only then
does the driver begin returning memory. This covers that round trip; the size-dependent part below covers the
release itself."""

_VERIFICATION_SECONDS_PER_GB = 1.5
"""Seconds of progress-free budget an in-process/IPC rung earns per gigabyte it promised to return.

A WDDM release is not an instant bookkeeping change: the driver hands the block back at a rate set by how much
is being released and by the host it runs on, so an 11GB checkpoint settles materially later than a 3GB one and
the same release settles later on a slower machine. A budget scaled by the promise therefore means the same
thing on an 8GB card and a 24GB one, where any fixed figure would be tuned to whichever card it was measured
on: generous on the small one, and short enough on the large one to misjudge exactly the multi-gigabyte
releases that matter most. Deliberately several times the pace a healthy release actually achieves, since the
cost of waiting is one more sample over the cliff while the cost of escalating early is the rest of the
ladder."""

_TEARDOWN_VERIFICATION_BASE_SECONDS = 12.0
"""Fixed part of a teardown rung's progress-free budget (a lane pause, safety off-GPU).

A process's device memory does not return to the driver until the OS has torn the process down, so these rungs
pay a full process exit before their first megabyte arrives. Larger than :data:`_VERIFICATION_BASE_SECONDS` by
that exit, which is a whole-process cost rather than a per-byte one."""

_TEARDOWN_VERIFICATION_SECONDS_PER_GB = _VERIFICATION_SECONDS_PER_GB
"""Per-gigabyte part of a teardown rung's budget: once the process is gone the release scales as any other."""

_VERIFICATION_PROGRESS_MB = 64.0
"""Realized free (MB) beyond a rung's previous high-water mark that counts as the rung still working.

Device-free is a shared figure that a settling allocator or a foreign app moves by tens of megabytes either
way, so the threshold sits above that noise: only a genuine new high extends the rung's budget. Measured
against the rung's own high-water rather than the previous sample, so a reading that oscillates cannot extend
the budget indefinitely; the total extension a rung can earn is bounded by the memory the card can return."""

_SAFETY_RUNG_COOLDOWN_SECONDS = 300.0
"""Minimum gap between two safety-off-GPU actuations on one card.

Moving safety off the GPU ends and rebuilds the safety process, and the placement policy brings it back once
the card fits it again, so the rung and its restore together are a full process cycle that stalls result
submission for as long as the rebuild takes. Without a dwell, every pressure episode buys another cycle and a
card under recurring pressure spends its session rebuilding safety instead of submitting. A rung refused for
this cooldown is skipped exactly as an inactive rung is: the engine moves to whatever relief is left, and an
episode with nothing left is unresolved rather than relieved by a cycle that would not have held."""

_VERIFICATION_YIELD_FRACTION = 0.5
"""Fraction of a rung's promised free the realized device-free gain must reach to count as verified.

Set at one half: device-free is a shared figure that a foreign app or a settling allocator can move by a few
hundred MB either way, so demanding the full promised delta would flag honest releases as short. Realizing
less than half of a promised multi-GB unload, by contrast, means the rung did not do what its estimate
claimed, and the estimate feeds a calibration event."""


class ReclaimRungKind(enum.StrEnum):
    """The kind of pressure-relief action one ladder rung performs."""

    UNLOAD_IDLE_MODEL = "unload_idle_model"
    """Unload an idle resident model's weights from VRAM back to RAM (rungs (a) newest and (c) older)."""
    RELEASE_IDLE_CACHE = "release_idle_cache"
    """Release an idle process's reclaimable allocator cache back to the card without evicting a model."""
    PAUSE_PP_LANE = "pause_pp_lane"
    """Pause the dedicated post-processing lane so its context and models free."""
    PAUSE_VAE_LANE = "pause_vae_lane"
    """Pause the dedicated VAE/image lane so its context and models free."""
    PAUSE_COMPONENT_LANE = "pause_component_lane"
    """Pause the component/text-encode lane so its context and models free."""
    SAFETY_OFF_GPU = "safety_off_gpu"
    """Move the on-GPU safety context off the card to reclaim it (the last rung before a kill)."""


LANE_PAUSE_RUNG_KINDS = frozenset(
    {
        ReclaimRungKind.PAUSE_PP_LANE,
        ReclaimRungKind.PAUSE_VAE_LANE,
        ReclaimRungKind.PAUSE_COMPONENT_LANE,
    },
)
"""The rung kinds that stop a dedicated lane off the GPU, which the issuer must later restore on its own.

A lane pause has no external restore trigger: unlike safety (re-promoted by the runtime safety-placement
policy once the card fits it), a paused lane stays down until something restarts it. Whichever subsystem
issued the pause therefore owns the restore for exactly these rungs, unwinding them when the condition it
paused for ends. Safety is excluded on purpose: it is restored by the placement policy, not by a rung issuer."""

_TEARDOWN_RUNG_KINDS = LANE_PAUSE_RUNG_KINDS | frozenset({ReclaimRungKind.SAFETY_OFF_GPU})
"""Rung kinds whose memory is freed by a process exiting, so they get the longer verification budget."""


@dataclass(frozen=True)
class ReclaimRung:
    """One ordered pressure-relief action, carrying its promised free and the tenant it acts on.

    ``promised_freed_mb`` is the device memory the rung is expected to return (a resident model's footprint,
    a process's reclaimable reservation, a lane's or safety's context charge); the engine verifies the
    realized gain against it. ``tenant_label`` names the process/lane/model for the shortfall log line.
    """

    kind: ReclaimRungKind
    device_index: int | None
    promised_freed_mb: float
    tenant_label: str
    target_process_id: int | None = None


@dataclass(frozen=True)
class IdleResidentModel:
    """An idle inference process holding a resident model, an unload candidate ranked by recency."""

    process_id: int
    tenant_label: str
    materialized_monotonic: float
    """When this model last became VRAM-resident (higher is newer); the LIFO ranking key."""
    footprint_mb: float
    """The model's device footprint (MB), the rung's promised free."""


@dataclass(frozen=True)
class CacheReleaseTarget:
    """An idle process holding reclaimable allocator cache (no resident model), ranked by recency."""

    process_id: int
    tenant_label: str
    materialized_monotonic: float
    reclaimable_mb: float
    """Reserved-minus-allocated device memory (MB) an ``empty_cache`` would return, the rung's promised free."""


@dataclass(frozen=True)
class LaneReclaimCandidate:
    """A lane or safety context that can be paused/moved off the card to reclaim it."""

    kind: ReclaimRungKind
    tenant_label: str
    promised_mb: float


@dataclass(frozen=True)
class ContextReduction:
    """A live-inference-context reduction the ladder owes a regrowth for once the card recovers.

    Reducing a card's live inference-process count returns the retained per-context VRAM the driver never
    gives back while a process lives, and it is the one reclaim action that shrinks the worker's serving
    capacity. Like a lane pause it has no external restore trigger, so the ladder records it as a restore
    obligation and unwinds it with the rest when the card returns HEALTHY; without that the card keeps
    whatever depth an episode of pressure left it at for the remainder of the session.
    """

    device_index: int | None
    """The card the reduction acted on; ``None`` is the card-agnostic worker-wide (single-GPU) scope."""


RestoreObligation = ReclaimRung | ContextReduction
"""One recorded action an episode must unwind when its card recovers, in LIFO order."""


@dataclass(frozen=True)
class LadderCandidates:
    """The raw, already-idle-filtered inputs the pure ladder builder orders into rungs.

    The scheduler assembles this from live state, excluding every actively-sampling process, so the builder
    (and its tests) never sees a busy tenant. ``lanes`` is already in the fixed pause order (post-processing,
    then VAE, then component), restricted to lanes currently on the GPU.
    """

    device_index: int | None
    idle_residents: tuple[IdleResidentModel, ...] = ()
    cache_targets: tuple[CacheReleaseTarget, ...] = ()
    lanes: tuple[LaneReclaimCandidate, ...] = ()
    safety: LaneReclaimCandidate | None = None


def build_reclaim_ladder(candidates: LadderCandidates) -> tuple[ReclaimRung, ...]:
    """Order the candidates into the fixed reclaim sequence with LIFO ranking among like rungs.

    The sequence is: the newest idle resident model, then each reclaimable allocator cache (newest first),
    then the older idle residents (newest first), then the lane pauses in their given order, then safety off
    the GPU. Ranking by ``materialized_monotonic`` descending puts the most-recently-materialized tenant, the
    likeliest WDDM squatter, first within each group. Every input is already idle-filtered, so an
    actively-sampling process can never appear as a rung.

    Args:
        candidates: The idle-filtered reclaim inputs for one card.

    Returns:
        The ordered rungs, empty when nothing on the card can be reclaimed.
    """
    device_index = candidates.device_index
    residents_newest_first = sorted(
        candidates.idle_residents,
        key=lambda resident: resident.materialized_monotonic,
        reverse=True,
    )
    rungs: list[ReclaimRung] = []

    if residents_newest_first:
        newest = residents_newest_first[0]
        rungs.append(
            ReclaimRung(
                kind=ReclaimRungKind.UNLOAD_IDLE_MODEL,
                device_index=device_index,
                promised_freed_mb=newest.footprint_mb,
                tenant_label=newest.tenant_label,
                target_process_id=newest.process_id,
            ),
        )

    for target in sorted(candidates.cache_targets, key=lambda t: t.materialized_monotonic, reverse=True):
        rungs.append(
            ReclaimRung(
                kind=ReclaimRungKind.RELEASE_IDLE_CACHE,
                device_index=device_index,
                promised_freed_mb=target.reclaimable_mb,
                tenant_label=target.tenant_label,
                target_process_id=target.process_id,
            ),
        )

    for resident in residents_newest_first[1:]:
        rungs.append(
            ReclaimRung(
                kind=ReclaimRungKind.UNLOAD_IDLE_MODEL,
                device_index=device_index,
                promised_freed_mb=resident.footprint_mb,
                tenant_label=resident.tenant_label,
                target_process_id=resident.process_id,
            ),
        )

    for lane in candidates.lanes:
        rungs.append(
            ReclaimRung(
                kind=lane.kind,
                device_index=device_index,
                promised_freed_mb=lane.promised_mb,
                tenant_label=lane.tenant_label,
            ),
        )

    if candidates.safety is not None:
        rungs.append(
            ReclaimRung(
                kind=ReclaimRungKind.SAFETY_OFF_GPU,
                device_index=device_index,
                promised_freed_mb=candidates.safety.promised_mb,
                tenant_label=candidates.safety.tenant_label,
            ),
        )

    return tuple(rungs)


class ReclaimLadderActuator(Protocol):
    """The execution surface the engine drives; the scheduler implements it (it owns process lifecycle).

    Each method performs one rung and reports whether it acted (a target that has already gone away returns
    False, and the engine moves on without waiting to verify a no-op). ``record_calibration_event`` folds a
    verified shortfall back into the worker's calibration (a raise-only footprint observation where a key
    applies, else a counter), so a rung whose promised free the hardware did not deliver improves the estimate
    that priced it.
    """

    def unload_idle_model(self, process_id: int, device_index: int | None) -> bool:
        """Unload the resident model on ``process_id`` from VRAM back to RAM."""
        ...

    def release_idle_cache(self, process_id: int) -> bool:
        """Release ``process_id``'s reclaimable allocator cache back to the card."""
        ...

    def pause_post_process_lane(self, device_index: int | None) -> bool:
        """Pause the post-processing lane off the GPU."""
        ...

    def pause_vae_lane(self, device_index: int | None) -> bool:
        """Pause the VAE/image lane off the GPU."""
        ...

    def pause_component_lane(self, device_index: int | None) -> bool:
        """Pause the component/text-encode lane off the GPU."""
        ...

    def safety_off_gpu(self, device_index: int | None) -> bool:
        """Move the on-GPU safety context off the card."""
        ...

    def restore_post_process_lane(self, device_index: int | None) -> bool:
        """Restart the post-processing lane the ladder paused, once the card has recovered."""
        ...

    def restore_vae_lane(self, device_index: int | None) -> bool:
        """Restart the VAE/image lane the ladder paused, once the card has recovered."""
        ...

    def restore_component_lane(self, device_index: int | None) -> bool:
        """Restart the component/text-encode lane the ladder paused, once the card has recovered."""
        ...

    def restore_live_contexts(self, device_index: int | None) -> bool:
        """Regrow the card's inference-process pool toward its configured size, once the card has recovered."""
        ...

    def record_calibration_event(self, rung: ReclaimRung, *, promised_mb: float, realized_mb: float) -> None:
        """Record that ``rung`` freed ``realized_mb`` against a promised ``promised_mb`` (a shortfall)."""
        ...


def execute_reclaim_rung(rung: ReclaimRung, actuator: ReclaimLadderActuator) -> bool:
    """Dispatch one rung onto the actuator, returning whether it acted.

    The single mapping from a rung kind to the actuator call that performs it, shared by every subsystem that
    issues rungs so a rung means the same action wherever it is driven from. A rung whose target has already
    gone away (or whose kind carries no target) reports False, which the caller reads as "nothing was freed".

    Args:
        rung: The rung to perform.
        actuator: The surface that owns the process actions.

    Returns:
        True when the actuator reported that it acted.
    """
    if rung.kind is ReclaimRungKind.UNLOAD_IDLE_MODEL and rung.target_process_id is not None:
        return actuator.unload_idle_model(rung.target_process_id, rung.device_index)
    if rung.kind is ReclaimRungKind.RELEASE_IDLE_CACHE and rung.target_process_id is not None:
        return actuator.release_idle_cache(rung.target_process_id)
    if rung.kind is ReclaimRungKind.PAUSE_PP_LANE:
        return actuator.pause_post_process_lane(rung.device_index)
    if rung.kind is ReclaimRungKind.PAUSE_VAE_LANE:
        return actuator.pause_vae_lane(rung.device_index)
    if rung.kind is ReclaimRungKind.PAUSE_COMPONENT_LANE:
        return actuator.pause_component_lane(rung.device_index)
    if rung.kind is ReclaimRungKind.SAFETY_OFF_GPU:
        return actuator.safety_off_gpu(rung.device_index)
    return False


def restore_reclaim_rung(rung: ReclaimRung, actuator: ReclaimLadderActuator) -> bool:
    """Dispatch one lane-restore onto the actuator, returning whether it acted.

    Defined for the :data:`LANE_PAUSE_RUNG_KINDS` only; any other kind reports False. The actuator routes the
    call through its owner-guarded restore path, so a lane another owner paused is left untouched.

    Args:
        rung: The lane-pause rung to unwind.
        actuator: The surface that owns the process actions.

    Returns:
        True when the actuator reported that it acted.
    """
    if rung.kind is ReclaimRungKind.PAUSE_PP_LANE:
        return actuator.restore_post_process_lane(rung.device_index)
    if rung.kind is ReclaimRungKind.PAUSE_VAE_LANE:
        return actuator.restore_vae_lane(rung.device_index)
    if rung.kind is ReclaimRungKind.PAUSE_COMPONENT_LANE:
        return actuator.restore_component_lane(rung.device_index)
    return False


def unwind_restore_obligation(obligation: RestoreObligation, actuator: ReclaimLadderActuator) -> bool:
    """Dispatch one recorded restore obligation onto the actuator, returning whether it acted.

    The single mapping from an obligation to the actuator call that undoes it, so a lane pause and a live-
    context reduction are unwound through one surface in one LIFO order. Each call routes through the
    actuator's own guarded restore path, so an action another owner is responsible for is left untouched.

    Args:
        obligation: The lane-pause rung or context reduction to unwind.
        actuator: The surface that owns the process actions.

    Returns:
        True when the actuator reported that it acted.
    """
    if isinstance(obligation, ContextReduction):
        return actuator.restore_live_contexts(obligation.device_index)
    return restore_reclaim_rung(obligation, actuator)


@dataclass
class _PendingVerification:
    """A rung awaiting verification: its promise, the device-free baseline at issue, and its progress clock."""

    rung: ReclaimRung
    baseline_free_mb: float
    progress_at: float
    """When the rung last realized new free (its issue time until it does), the instant its budget runs from."""
    best_realized_mb: float = 0.0
    """The highest realized free this rung has reached, so an oscillating reading cannot extend its budget."""
    samples_waited: int = 0
    """How many governor samples the rung has been graded over, for the shortfall notice."""


@dataclass
class _Episode:
    """One card's live reclaim record: its frozen ladder, cursor, pending rung, outcome, and what it owes back.

    ``restore_obligations`` records, in the order they were taken, every action this episode actually performed
    that the engine must undo when the card recovers: a lane pause it issued, and a live-context reduction the
    per-cycle admission path requested. An action that was a no-op (a lane already paused by another owner) is
    never recorded, so the engine never tries to restore something it did not do.

    ``ladder`` is built lazily on the first SATURATED tick, so the rungs stay frozen against the topology at
    the moment the card crossed the cliff even when the episode was opened earlier by a recorded obligation.
    """

    ladder: tuple[ReclaimRung, ...] | None = None
    next_index: int = 0
    pending: _PendingVerification | None = None
    unresolved: bool = False
    restore_obligations: list[RestoreObligation] = field(default_factory=list)
    retention_logged: bool = False
    """Whether this episode has already disclosed that a refused restore left it holding a debt."""

    def reset_ladder_progress(self) -> None:
        """Drop this episode's ladder, cursor, pending rung and outcome, keeping only what it still owes.

        Leaves the episode in the same shape as one opened by a recorded obligation alone, so a card that
        saturates again builds a fresh ladder against the topology of that moment.
        """
        self.ladder = None
        self.next_index = 0
        self.pending = None
        self.unresolved = False


class VerifiedReclaimLadder:
    """The parent-side, single-owner engine that runs and verifies the reclaim ladder per card.

    Driven once per governor tick per card via :meth:`on_tick`. It issues at most one rung per tick, then
    watches the following ticks' device-free readings for the rung's verification budget before escalating. All
    per-device episode state lives here; the engine performs no process actions itself. The run-wide counters
    (:attr:`rungs_issued`, :attr:`verified_frees_mb`, :attr:`verification_shortfalls`) are calibration
    visibility; :meth:`is_saturation_unresolved` reports whether a card's current episode exhausted the ladder
    while still SATURATED, the signal a later kill rung reads.
    """

    def __init__(self) -> None:
        """Initialise with zeroed counters and no per-device episodes."""
        self.rungs_issued = 0
        self.verified_frees_mb = 0.0
        self.verification_shortfalls = 0
        self.safety_rungs_refused = 0
        """Safety rungs skipped because the card's previous safety actuation is still inside its cooldown."""
        self._episodes: dict[int | None, _Episode] = {}
        self._safety_actuated_at: dict[int | None, float] = {}

    def on_tick(
        self,
        device_index: int,
        *,
        saturated: bool,
        healthy: bool = False,
        device_free_mb: float,
        actuator: ReclaimLadderActuator,
        ladder_builder: Callable[[], tuple[ReclaimRung, ...]],
        context_restore_ready: bool = True,
        now: float | None = None,
    ) -> None:
        """Advance the reclaim episode for one card by one governor sample.

        When the card is SATURATED, a pending rung is verified first (crediting a realized free or, once the
        rung has spent its whole verification budget without realizing further free, logging the shortfall and
        escalating), then the next rung is issued if the ladder is not exhausted. An exhausted ladder on a
        still-SATURATED card marks the episode unresolved.

        When the card is not SATURATED the episode is winding down, but the engine holds it (issuing no further
        rungs) until the card returns fully HEALTHY, then unwinds: it undoes every restore obligation it took
        (each lane it paused, each live-context reduction it made) in reverse order (LIFO), and clears the
        episode. Holding through the intermediate PRESSURE band (below the soft floor but above the hard floor)
        matters because those actions free real CUDA contexts: undoing them the instant saturation lifts would
        re-add the contexts while the card is still tight and risk re-crossing the cliff, so the restore waits
        for the governor's debounced HEALTHY signal. Safety, if the ladder cycled it off, is not restored here:
        the runtime safety-placement policy re-promotes it once the card demonstrably fits it.

        Obligations recorded against the card-agnostic worker-wide scope (the ``None`` key, used by the
        per-cycle admission path on a single-GPU host) are unwound alongside the sampled card's own, since that
        scope is only ever produced where there is exactly one governed card.

        Args:
            device_index: The card this tick is for.
            saturated: Whether the device-free governor calls the card SATURATED this sample.
            healthy: Whether the governor calls the card HEALTHY this sample (device-free above the soft
                floor), the debounced signal that a winding-down episode may restore its paused lanes.
            device_free_mb: The card's NVML device-level free VRAM (MB) this sample.
            actuator: The surface that performs each rung, restores paused lanes, and records calibration
                shortfalls.
            ladder_builder: Builds the ordered rungs when a new episode begins; called at most once per
                episode so the ladder is frozen against the topology at the moment the card crossed the cliff.
            context_restore_ready: Whether a live-context reduction may be unwound this sample. A lane pause
                is cheap to undo, but regrowing the pool costs a full process cold start, and the freed VRAM
                is exactly why the card reads HEALTHY: unwinding on the first healthy sample regrows the
                context, re-inflates the footprint, and buys the next reduction, which is a cold start per
                cycle for no net change. The caller supplies the dwell and the evidence that the demand the
                reduction was made for has cleared; lane rungs are unaffected.
            now: The monotonic-scale instant of this sample, which every verification budget and the safety
                rung's cooldown are measured on. Defaults to :func:`time.monotonic`; a caller that drives the
                control loop on its own clock passes that clock so the budgets are measured on the same
                timeline as everything else it gates.
        """
        if now is None:
            now = time.monotonic()
        if not saturated:
            if healthy:
                for key in (device_index, None):
                    episode = self._episodes.get(key)
                    if episode is None:
                        continue
                    if self._unwind_restore_obligations(
                        episode,
                        actuator,
                        context_restore_ready=context_restore_ready,
                    ):
                        self._episodes.pop(key, None)
                        continue
                    # An obligation the actuator refused is still owed, so the episode survives as a debt
                    # record and re-attempts on a later HEALTHY sample. Its ladder progress is dropped: the
                    # saturation it was running is over, and a fresh one must build its rungs against the
                    # topology of the moment it crosses the cliff, exactly as an episode opened by a recorded
                    # obligation alone does.
                    episode.reset_ladder_progress()
                    self._log_retained_obligations(key, episode)
            return

        episode = self._episodes.get(device_index)
        if episode is None:
            episode = _Episode()
            self._episodes[device_index] = episode
        if episode.ladder is None:
            episode.ladder = tuple(ladder_builder())

        if episode.pending is not None and not self._verify(episode, device_free_mb, actuator, now):
            return

        self._issue_next(episode, device_free_mb, actuator, device_index=device_index, now=now)

    def record_context_reduction(self, device_index: int | None) -> None:
        """Record that a card's live inference-context count was reduced under admission pressure.

        The per-cycle admission path reduces the live context count directly (it is the one relief that
        returns the per-context VRAM a live process retains), which shrinks the card's serving capacity until
        something grows it back. Booking it here puts the regrowth on the same footing as a ladder-issued lane
        pause: the engine owes the card its pool back and unwinds the obligation with the rest, LIFO, once the
        governor calls the card HEALTHY. Recording is idempotent per card: a head that re-asks every cycle
        books one obligation, not one per ask, so the unwind regrows the pool once.

        Args:
            device_index: The card whose contexts were reduced; ``None`` is the card-agnostic worker-wide
                (single-GPU) scope.
        """
        episode = self._episodes.get(device_index)
        if episode is None:
            episode = _Episode()
            self._episodes[device_index] = episode
        reduction = ContextReduction(device_index=device_index)
        if reduction in episode.restore_obligations:
            return
        episode.restore_obligations.append(reduction)

    def has_context_reduction(self, device_index: int | None) -> bool:
        """Whether an outstanding live-context reduction on ``device_index`` still owes the card its pool back."""
        episode = self._episodes.get(device_index)
        if episode is None:
            return False
        return any(isinstance(obligation, ContextReduction) for obligation in episode.restore_obligations)

    def discharge_context_reduction(self, device_index: int | None) -> bool:
        """Drop an outstanding live-context reduction on ``device_index`` because the pool was already regrown.

        The obligation records a debt; it is not the debt itself. When a caller outside the episode's own LIFO
        unwind has regrown the pool the reduction shrank, leaving the record in place would have that caller
        act again on every later observation and have a subsequent unwind regrow a pool that is no longer
        shrunk. Discharging it keeps exactly one regrowth per reduction. A lane pause needs no equivalent
        because the lifecycle's own pause state is what a second restore attempt reads.

        Args:
            device_index: The card whose reduction is discharged; ``None`` is the card-agnostic worker-wide
                (single-GPU) scope.

        Returns:
            True when an outstanding reduction was removed.
        """
        episode = self._episodes.get(device_index)
        if episode is None:
            return False
        remaining = [
            obligation for obligation in episode.restore_obligations if not isinstance(obligation, ContextReduction)
        ]
        if len(remaining) == len(episode.restore_obligations):
            return False
        episode.restore_obligations[:] = remaining
        return True

    def is_saturation_unresolved(self, device_index: int) -> bool:
        """Whether ``device_index``'s current SATURATED episode exhausted the ladder without relieving it."""
        episode = self._episodes.get(device_index)
        return episode is not None and episode.unresolved

    def episode_holds_paused_lane(self, device_index: int) -> bool:
        """Whether a live saturation episode on ``device_index`` currently owns one or more lane pauses.

        A caller reasoning about whether a ladder-owned lane pause has a live claimant reads this: while an
        episode is recorded with paused lanes, that episode's own LIFO unwind (on the card returning HEALTHY)
        is the responsible restore path, so no external backstop should lift the pause. A card that has already
        returned HEALTHY has had its episode restored and removed, so this reports False and the pause, if still
        held, is an orphan the backstop may reclaim.
        """
        episode = self._episodes.get(device_index)
        if episode is None:
            return False
        return any(
            isinstance(obligation, ReclaimRung) and obligation.kind in LANE_PAUSE_RUNG_KINDS
            for obligation in episode.restore_obligations
        )

    def _verify(
        self,
        episode: _Episode,
        device_free_mb: float,
        actuator: ReclaimLadderActuator,
        now: float,
    ) -> bool:
        """Verify the pending rung against realized device-free; return True once it resolves (freed or short).

        A rung resolves as verified the moment realized free reaches :data:`_VERIFICATION_YIELD_FRACTION` of
        its promise (crediting the realized gain), or as a shortfall once its whole verification budget has
        passed without it realizing further free (logging, recording a calibration event, and letting the
        caller escalate). The budget is :data:`_TEARDOWN_VERIFICATION_BUDGET_SECONDS` for teardown-class rungs
        (a lane pause or safety off-GPU, whose memory only returns once the process has exited) and
        :data:`_VERIFICATION_BUDGET_SECONDS` otherwise, and it runs from the last sample that realized a new
        high-water of free rather than from the issue: a release still arriving keeps its rung, so a working
        multi-gigabyte give-back is never graded short for landing more slowly than the control loop samples.
        While the rung is still inside its budget this returns False so the engine waits another tick rather
        than issuing the next rung.

        A rung whose promise is not a positive figure (the tenant's reservation was never reported, so it was
        priced at zero) is *unverifiable* rather than verified: the yield-fraction test would reduce to
        "realized at least nothing" and certify the first sample of a rung that freed nothing at all, sprinting
        the engine down the whole ladder on a rung it never graded. Such a rung serves its full budget, credits
        nothing to :attr:`verified_frees_mb`, and resolves without being counted either verified or short, so
        the engine escalates on an honest absence of evidence.
        """
        pending = episode.pending
        assert pending is not None
        pending.samples_waited += 1
        realized_mb = device_free_mb - pending.baseline_free_mb
        promised_mb = pending.rung.promised_freed_mb

        if realized_mb >= pending.best_realized_mb + _VERIFICATION_PROGRESS_MB:
            pending.best_realized_mb = realized_mb
            pending.progress_at = now
        within_budget = (now - pending.progress_at) < self._verification_budget_for(pending.rung)

        if promised_mb <= 0.0:
            if within_budget:
                return False
            logger.debug(
                f"Reclaim rung {pending.rung.kind.value} on {pending.rung.tenant_label} "
                f"(device {pending.rung.device_index}) carried no promised free, so its realized "
                f"~{max(0.0, realized_mb):.0f}MB cannot be graded; escalating without crediting it.",
            )
            episode.pending = None
            return True

        if realized_mb >= _VERIFICATION_YIELD_FRACTION * promised_mb:
            self.verified_frees_mb += max(0.0, realized_mb)
            episode.pending = None
            return True

        if not within_budget:
            self.verification_shortfalls += 1
            logger.warning(
                f"Reclaim rung {pending.rung.kind.value} on {pending.rung.tenant_label} "
                f"(device {pending.rung.device_index}) freed only ~{max(0.0, realized_mb):.0f}MB of a "
                f"promised ~{promised_mb:.0f}MB, and has realized nothing further for "
                f"{self._verification_budget_for(pending.rung):.0f}s over {pending.samples_waited} "
                "samples; escalating.",
            )
            actuator.record_calibration_event(pending.rung, promised_mb=promised_mb, realized_mb=realized_mb)
            episode.pending = None
            return True

        return False

    def _issue_next(
        self,
        episode: _Episode,
        device_free_mb: float,
        actuator: ReclaimLadderActuator,
        *,
        device_index: int | None,
        now: float,
    ) -> None:
        """Issue the next rung that actually acts, or mark the episode unresolved when the ladder is exhausted.

        A rung whose target has already gone away (the actuator returns False) frees nothing to verify, so the
        engine advances to the next rung in the same tick rather than opening a verification window on a no-op.
        A safety rung still inside :data:`_SAFETY_RUNG_COOLDOWN_SECONDS` of the card's previous safety
        actuation is skipped the same way: cycling safety again this soon rebuilds the process for relief the
        last cycle already failed to hold. The first rung that acts opens a fresh verification budget and stops
        the tick.
        """
        ladder = episode.ladder or ()
        while episode.next_index < len(ladder):
            rung = ladder[episode.next_index]
            episode.next_index += 1
            if rung.kind is ReclaimRungKind.SAFETY_OFF_GPU and self._safety_rung_in_cooldown(device_index, now):
                continue
            if execute_reclaim_rung(rung, actuator):
                self.rungs_issued += 1
                if rung.kind is ReclaimRungKind.SAFETY_OFF_GPU:
                    self._safety_actuated_at[device_index] = now
                if rung.kind in LANE_PAUSE_RUNG_KINDS:
                    # Only a lane pause that actually acted is the engine's to restore later; record it so the
                    # episode-end unwind restarts exactly the lanes this engine stopped.
                    episode.restore_obligations.append(rung)
                episode.pending = _PendingVerification(
                    rung=rung,
                    baseline_free_mb=device_free_mb,
                    progress_at=now,
                )
                return
        episode.unresolved = True

    def _safety_rung_in_cooldown(self, device_index: int | None, now: float) -> bool:
        """Whether ``device_index``'s last safety actuation is recent enough to refuse another one.

        Counted once per refusal so the cost of the dwell is visible; the ladder advances past a refused rung
        within the tick, so this cannot repeat per sample while an episode is running.
        """
        actuated_at = self._safety_actuated_at.get(device_index)
        if actuated_at is None or (now - actuated_at) >= _SAFETY_RUNG_COOLDOWN_SECONDS:
            return False
        self.safety_rungs_refused += 1
        logger.debug(
            f"Reclaim ladder: refusing the safety-off-GPU rung on device {device_index}; the previous safety "
            f"actuation was {now - actuated_at:.0f}s ago, inside the {_SAFETY_RUNG_COOLDOWN_SECONDS:.0f}s "
            "dwell that keeps a card under recurring pressure from spending its session rebuilding safety.",
        )
        return True

    @staticmethod
    def _verification_budget_for(rung: ReclaimRung) -> float:
        """Seconds without realized free this rung is given before it counts as short.

        A fixed part for the latency before any memory can arrive (a control-pipe round trip, or a whole
        process exit for a teardown rung) plus a part scaled by the megabytes the rung promised, because a
        release settles at a rate set by its own size and by the host. Scaling it is what keeps the same
        judgement honest across the card sizes operators actually run: a constant sized for one of them
        misjudges a working large release on the others.
        """
        teardown = rung.kind in _TEARDOWN_RUNG_KINDS
        base_seconds = _TEARDOWN_VERIFICATION_BASE_SECONDS if teardown else _VERIFICATION_BASE_SECONDS
        seconds_per_gb = _TEARDOWN_VERIFICATION_SECONDS_PER_GB if teardown else _VERIFICATION_SECONDS_PER_GB
        promised_gb = max(0.0, rung.promised_freed_mb) / 1024.0
        return base_seconds + seconds_per_gb * promised_gb

    @staticmethod
    def _unwind_restore_obligations(
        episode: _Episode,
        actuator: ReclaimLadderActuator,
        *,
        context_restore_ready: bool = True,
    ) -> bool:
        """Undo everything this episode owes the card, in reverse order of the actions taken (LIFO unwind).

        The unwind mirrors the order the memory was taken back in: the last lane stopped (or context reduced)
        is the first restored, so a card that gave back memory action-by-action reclaims its capacity in the
        same order it released it. Each restore targets only what this engine did (the actuator routes it
        through its owner-guarded restore path), so an action another owner is responsible for is left
        untouched. Run when the card returns HEALTHY.

        An actuator that reports it did not act has not discharged the debt: the context restore stands down
        for as long as a whole-card residency owns the pool, and dropping the obligation on that answer would
        leave the card shrunk with nobody left to regrow it. Such an obligation is retained in its original
        order for a later attempt, so the debt outlives this pass. A context reduction the caller says is not
        yet ready to unwind is retained the same way, so the debt survives until its dwell is met.

        Returns:
            True when every obligation was discharged and the episode owes the card nothing.
        """
        retained: list[RestoreObligation] = []
        for obligation in reversed(episode.restore_obligations):
            if isinstance(obligation, ContextReduction) and not context_restore_ready:
                retained.append(obligation)
                continue
            if not unwind_restore_obligation(obligation, actuator):
                retained.append(obligation)
        retained.reverse()
        episode.restore_obligations[:] = retained
        if not retained:
            episode.retention_logged = False
        return not retained

    @staticmethod
    def _log_retained_obligations(device_index: int | None, episode: _Episode) -> None:
        """Name the debt an unwind could not discharge, once per episode until it clears.

        The unwind re-attempts on every HEALTHY sample, so the notice is edge-triggered: a card whose restore
        owner stands down for minutes reports the retained debt once rather than on every governor tick.
        """
        if episode.retention_logged:
            return
        episode.retention_logged = True
        kinds = ", ".join(
            sorted(
                {
                    "live contexts" if isinstance(obligation, ContextReduction) else obligation.kind.value
                    for obligation in episode.restore_obligations
                },
            ),
        )
        logger.debug(
            f"Reclaim episode on device {device_index} still owes the card {kinds}: the restore was not taken "
            "this sample (the actuator declined, or the restore is holding for its dwell), so the obligation "
            "is kept and retried once the card can take it back.",
        )

    @staticmethod
    def execute_arbiter_commands(
        commands: tuple[ActuatorCommand, ...],
        actuator: VramActuator,
        *,
        device_index: int | None,
        for_head_of_queue: bool,
    ) -> tuple[ActuatorCommand, ...]:
        """Run the arbiter's deferred-preload actuations through this single reclaim owner.

        The verified ladder (governor SATURATED path) and the arbiter's per-cycle DEFER ladder are the worker's
        two reclaim triggers; routing both through this engine keeps one execution surface so they can never
        become two mechanisms evicting the same card by different rules. This maps each described
        :class:`ActuatorCommand` onto the caller's :class:`VramActuator`, one action each, exactly as the
        preload path did inline: RELEASE_CACHE targets an idle lane, EVICT_IDLE_MODEL frees an idle resident,
        REDUCE_LIVE_CONTEXTS collapses the live context count, CYCLE_SAFETY_OFF_GPU frees the safety context.
        The arbiter guarantees RELEASE_CACHE and service-lane pause targets are idle, so live work is never
        disturbed. Returns exactly the commands whose actuator reported that it acted; callers that temporarily
        borrow a service lane use this receipt to restore only a pause they actually acquired, never a same-owner
        pause that the governor's independent saturation episode already held.
        """
        applied: list[ActuatorCommand] = []
        for command in commands:
            if command.kind is ActuatorCommandKind.RELEASE_CACHE and command.target_process_id is not None:
                acted = actuator.release_cache(command.target_process_id)
            elif command.kind is ActuatorCommandKind.EVICT_IDLE_MODEL:
                acted = actuator.evict_idle_model(device_index, for_head_of_queue=for_head_of_queue)
            elif command.kind is ActuatorCommandKind.REDUCE_LIVE_CONTEXTS:
                acted = actuator.reduce_live_contexts(device_index)
            elif command.kind is ActuatorCommandKind.PAUSE_VAE_LANE:
                acted = actuator.pause_vae_lane(device_index)
            elif command.kind is ActuatorCommandKind.PAUSE_COMPONENT_LANE:
                acted = actuator.pause_component_lane(device_index)
            elif command.kind is ActuatorCommandKind.CYCLE_SAFETY_OFF_GPU:
                acted = actuator.cycle_safety_off_gpu(device_index)
            else:
                acted = False
            if acted:
                applied.append(command)
        return tuple(applied)
