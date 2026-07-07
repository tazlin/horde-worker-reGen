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

- It verifies. Freeing on WDDM is externally checkable: NVML device-used drops within a couple of seconds of
  a real release. After issuing a rung the engine watches the next one or two governor samples and compares
  the realized device-free gain against the rung's promised figure (the tenant's footprint / reclaimable
  reservation). A rung that yields less than half of what it promised is logged against the tenant it named,
  recorded as a calibration event, and the engine escalates to the next rung rather than trusting the
  estimate. When the whole ladder is exhausted and the card is still SATURATED, the episode is marked
  unresolved: nothing the worker can give back relieved the card, which is the signal a later kill rung reads.

The engine is driven from the parent's single-threaded control loop, one call per governor tick per card, and
holds all cross-tick verification state per device. It touches no process state itself: a
:class:`ReclaimLadderActuator` (implemented by the scheduler, which owns process lifecycle) performs each rung
and reports a calibration shortfall, exactly as the arbiter describes actuations for a caller to run.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from loguru import logger

from horde_worker_regen.process_management.resources.vram_arbiter import (
    ActuatorCommand,
    ActuatorCommandKind,
    VramActuator,
)

_VERIFICATION_SAMPLES = 2
"""Governor samples a rung's realized free is given to reach the promised figure before it counts as short.

A real WDDM release shows up in NVML device-used within a couple of seconds (one to two governor ticks), so a
rung that has not yielded its promised memory after this many samples has demonstrably not worked and the
engine escalates. Fewer would misjudge a release still settling; more would leave the card over the cliff
longer than necessary. Applies to the in-process reclaim rungs (model unload, cache release), which free their
memory synchronously as the actuator returns."""

_TEARDOWN_VERIFICATION_SAMPLES = 3
"""Verification window for rungs that free memory by ending a whole process (a lane pause, safety off-GPU).

A process's device memory does not return to the driver until the OS has torn the process down, which takes
longer than one governor sample: a lane pause has been measured returning ~0MB of its promised context one
sample after it was issued, then the full figure a sample or two later. Giving these teardown-class rungs one
extra sample over the in-process rungs keeps the engine from falsely grading a working pause as short and
escalating past it while its process is still exiting. Still bounded so a genuinely stuck teardown escalates
within a few seconds."""

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


_LANE_PAUSE_RUNG_KINDS = frozenset(
    {
        ReclaimRungKind.PAUSE_PP_LANE,
        ReclaimRungKind.PAUSE_VAE_LANE,
        ReclaimRungKind.PAUSE_COMPONENT_LANE,
    },
)
"""The rung kinds that stop a dedicated lane off the GPU, which the engine must later restore on its own.

A lane pause has no external restore trigger: unlike safety (re-promoted by the runtime safety-placement
policy once the card fits it), a paused lane stays down until something restarts it. The engine therefore owns
the restore for exactly these rungs, unwinding them when the card's saturation episode ends. Safety is
excluded on purpose: it is restored by the placement policy, not by the ladder."""

_TEARDOWN_RUNG_KINDS = _LANE_PAUSE_RUNG_KINDS | frozenset({ReclaimRungKind.SAFETY_OFF_GPU})
"""Rung kinds whose memory is freed by a process exiting, so they get the longer verification window."""


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

    def record_calibration_event(self, rung: ReclaimRung, *, promised_mb: float, realized_mb: float) -> None:
        """Record that ``rung`` freed ``realized_mb`` against a promised ``promised_mb`` (a shortfall)."""
        ...


@dataclass
class _PendingVerification:
    """A rung awaiting verification: its promise, the device-free baseline at issue, and samples waited."""

    rung: ReclaimRung
    baseline_free_mb: float
    samples_waited: int = 0


@dataclass
class _Episode:
    """One contiguous SATURATED stretch on a card: its frozen ladder, cursor, pending rung, and outcome.

    ``paused_lanes`` records, in issue order, every lane-pause rung this episode actually actuated, so the
    engine can restore exactly those lanes (and only those, in LIFO order) when the episode ends. A lane whose
    pause was a no-op (already paused by another owner) is never recorded, so the engine never tries to restore
    a lane it did not stop.
    """

    ladder: tuple[ReclaimRung, ...]
    next_index: int = 0
    pending: _PendingVerification | None = None
    unresolved: bool = False
    paused_lanes: list[ReclaimRung] = field(default_factory=list)


class VerifiedReclaimLadder:
    """The parent-side, single-owner engine that runs and verifies the reclaim ladder per card.

    Driven once per governor tick per card via :meth:`on_tick`. It issues at most one rung per tick, then
    watches the next one or two ticks' device-free readings to verify the freed memory before escalating. All
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
        self._episodes: dict[int, _Episode] = {}

    def on_tick(
        self,
        device_index: int,
        *,
        saturated: bool,
        healthy: bool = False,
        device_free_mb: float,
        actuator: ReclaimLadderActuator,
        ladder_builder: Callable[[], tuple[ReclaimRung, ...]],
    ) -> None:
        """Advance the reclaim episode for one card by one governor sample.

        When the card is SATURATED, a pending rung is verified first (crediting a realized free or, after the
        rung's verification window of short samples, logging the shortfall and escalating), then the next rung
        is issued if the ladder is not exhausted. An exhausted ladder on a still-SATURATED card marks the
        episode unresolved.

        When the card is not SATURATED the episode is winding down, but the engine holds it (issuing no further
        rungs) until the card returns fully HEALTHY, then unwinds: it restores every lane it paused, in reverse
        rung order (LIFO), and clears the episode. Holding through the intermediate PRESSURE band (below the
        soft floor but above the hard floor) matters because a lane pause frees a real CUDA context: restarting
        it the instant saturation lifts would re-add that context while the card is still tight and risk
        re-crossing the cliff, so the restore waits for the governor's debounced HEALTHY signal. Safety, if the
        ladder cycled it off, is not restored here: the runtime safety-placement policy re-promotes it once the
        card demonstrably fits it.

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
        """
        if not saturated:
            episode = self._episodes.get(device_index)
            if episode is not None and healthy:
                self._restore_paused_lanes(episode, actuator)
                self._episodes.pop(device_index, None)
            return

        episode = self._episodes.get(device_index)
        if episode is None:
            episode = _Episode(ladder=tuple(ladder_builder()))
            self._episodes[device_index] = episode

        if episode.pending is not None and not self._verify(episode, device_free_mb, actuator):
            return

        self._issue_next(episode, device_free_mb, actuator)

    def is_saturation_unresolved(self, device_index: int) -> bool:
        """Whether ``device_index``'s current SATURATED episode exhausted the ladder without relieving it."""
        episode = self._episodes.get(device_index)
        return episode is not None and episode.unresolved

    def _verify(
        self,
        episode: _Episode,
        device_free_mb: float,
        actuator: ReclaimLadderActuator,
    ) -> bool:
        """Verify the pending rung against realized device-free; return True once it resolves (freed or short).

        A rung resolves as verified the moment realized free reaches :data:`_VERIFICATION_YIELD_FRACTION` of
        its promise (crediting the realized gain), or as a shortfall once it has been given its verification
        window of samples without doing so (logging, recording a calibration event, and letting the caller
        escalate). The window is :data:`_TEARDOWN_VERIFICATION_SAMPLES` for teardown-class rungs (a lane pause
        or safety off-GPU, whose memory only returns once the process has exited) and :data:`_VERIFICATION_SAMPLES`
        otherwise. While it is still within its verification window it returns False so the engine waits another
        tick rather than issuing the next rung.
        """
        pending = episode.pending
        assert pending is not None
        pending.samples_waited += 1
        realized_mb = device_free_mb - pending.baseline_free_mb
        promised_mb = pending.rung.promised_freed_mb

        if realized_mb >= _VERIFICATION_YIELD_FRACTION * promised_mb:
            self.verified_frees_mb += max(0.0, realized_mb)
            episode.pending = None
            return True

        if pending.samples_waited >= self._verification_window_for(pending.rung.kind):
            self.verification_shortfalls += 1
            logger.warning(
                f"Reclaim rung {pending.rung.kind.value} on {pending.rung.tenant_label} "
                f"(device {pending.rung.device_index}) freed only ~{max(0.0, realized_mb):.0f}MB of a "
                f"promised ~{promised_mb:.0f}MB after {pending.samples_waited} samples; escalating.",
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
    ) -> None:
        """Issue the next rung that actually acts, or mark the episode unresolved when the ladder is exhausted.

        A rung whose target has already gone away (the actuator returns False) frees nothing to verify, so the
        engine advances to the next rung in the same tick rather than opening a verification window on a no-op.
        The first rung that acts opens a fresh verification window and stops the tick.
        """
        while episode.next_index < len(episode.ladder):
            rung = episode.ladder[episode.next_index]
            episode.next_index += 1
            if self._execute(rung, actuator):
                self.rungs_issued += 1
                if rung.kind in _LANE_PAUSE_RUNG_KINDS:
                    # Only a lane pause that actually acted is the engine's to restore later; record it so the
                    # episode-end unwind restarts exactly the lanes this engine stopped.
                    episode.paused_lanes.append(rung)
                episode.pending = _PendingVerification(rung=rung, baseline_free_mb=device_free_mb)
                return
        episode.unresolved = True

    @staticmethod
    def _verification_window_for(kind: ReclaimRungKind) -> int:
        """Samples a rung of this kind is given to realize its promise before it counts as short."""
        return _TEARDOWN_VERIFICATION_SAMPLES if kind in _TEARDOWN_RUNG_KINDS else _VERIFICATION_SAMPLES

    @staticmethod
    def _restore_paused_lanes(episode: _Episode, actuator: ReclaimLadderActuator) -> None:
        """Restart every lane this episode paused, in reverse rung order (LIFO unwind).

        The unwind mirrors the pause order: the last lane stopped is the first restored, so a card that gave
        back memory pause-by-pause reclaims its lanes in the same order it released them. Each restore targets
        only the ladder-owned pause (the actuator routes it through the owner-guarded restore path), so a lane
        another owner paused is left untouched. Called once, when the card returns HEALTHY.
        """
        for rung in reversed(episode.paused_lanes):
            VerifiedReclaimLadder._restore_lane(rung, actuator)
        episode.paused_lanes.clear()

    @staticmethod
    def _restore_lane(rung: ReclaimRung, actuator: ReclaimLadderActuator) -> bool:
        """Dispatch one lane-restore onto the actuator, returning whether it acted."""
        if rung.kind is ReclaimRungKind.PAUSE_PP_LANE:
            return actuator.restore_post_process_lane(rung.device_index)
        if rung.kind is ReclaimRungKind.PAUSE_VAE_LANE:
            return actuator.restore_vae_lane(rung.device_index)
        if rung.kind is ReclaimRungKind.PAUSE_COMPONENT_LANE:
            return actuator.restore_component_lane(rung.device_index)
        return False

    @staticmethod
    def execute_arbiter_commands(
        commands: tuple[ActuatorCommand, ...],
        actuator: VramActuator,
        *,
        device_index: int | None,
        for_head_of_queue: bool,
    ) -> None:
        """Run the arbiter's deferred-preload actuations through this single reclaim owner.

        The verified ladder (governor SATURATED path) and the arbiter's per-cycle DEFER ladder are the worker's
        two reclaim triggers; routing both through this engine keeps one execution surface so they can never
        become two mechanisms evicting the same card by different rules. This maps each described
        :class:`ActuatorCommand` onto the caller's :class:`VramActuator`, one action each, exactly as the
        preload path did inline: RELEASE_CACHE targets an idle lane, EVICT_IDLE_MODEL frees an idle resident,
        REDUCE_LIVE_CONTEXTS collapses the live context count, CYCLE_SAFETY_OFF_GPU frees the safety context.
        The arbiter guarantees RELEASE_CACHE targets only idle lanes, so a busy lane is never asked to release.
        """
        for command in commands:
            if command.kind is ActuatorCommandKind.RELEASE_CACHE and command.target_process_id is not None:
                actuator.release_cache(command.target_process_id)
            elif command.kind is ActuatorCommandKind.EVICT_IDLE_MODEL:
                actuator.evict_idle_model(device_index, for_head_of_queue=for_head_of_queue)
            elif command.kind is ActuatorCommandKind.REDUCE_LIVE_CONTEXTS:
                actuator.reduce_live_contexts(device_index)
            elif command.kind is ActuatorCommandKind.CYCLE_SAFETY_OFF_GPU:
                actuator.cycle_safety_off_gpu(device_index)

    @staticmethod
    def _execute(rung: ReclaimRung, actuator: ReclaimLadderActuator) -> bool:
        """Dispatch one rung onto the actuator, returning whether it acted."""
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
