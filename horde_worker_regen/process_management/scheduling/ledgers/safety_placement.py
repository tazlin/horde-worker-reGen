"""Runtime safety placement: keeping the safety process on the GPU only while its own card can afford it.

The single safety process runs on-GPU where a driven card's effective ``safety_on_gpu`` permits it. On a card too
tight to hold safety's context beside the model sampling there, that context competes for VRAM the sampler needs,
so the scheduler moves safety to a CPU-only process and brings it back once the card proves durable room. The
policy only ever degrades the operator's placement (GPU to CPU) and restores it, never more.

Every term the policy reads is about safety's own card: the one it occupies, or the one it would land on while it
is off. Demotion needs measured pressure there, restoration needs the mirror-image forecast (measured free
covering safety beside the peak the card is committed to), and both are dwelt in seconds against what one
placement flip costs rather than in control cycles.

This module holds the per-card evidence snapshot with its two predicates, and the ledger of evidence clocks,
one-shot requests and tallies. The scheduler gathers the evidence from its collaborators and owns the actuation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState

SAFETY_GPU_LOAD_CHARGE_MB = 3044.0
"""The device VRAM (MB) charged for the safety process on the GPU: a conservative seed for the idle CLIP model
plus its CUDA context. DeepDanbooru, BLIP, the aesthetic head and evaluation activations are reclaimable and
not part of it. Erring high keeps safety off-GPU one more cycle rather than restoring it onto a card it would
over-commit. Every consumer reads the seed through the scheduler's learned safety footprint, which raises it by
any measured :attr:`FootprintStage.SAFETY` watermark."""

MEMORY_PRESSURE_PAUSE_OWNERS = frozenset({PauseOwner.RUNTIME_SAFETY_PLACEMENT, PauseOwner.RECLAIM_LADDER})
"""The safety-off-GPU requests taken because the card was short of memory, rather than to clear it for a model.

Their restores have to earn forecast headroom: the memory such a pause returned is part of what any instantaneous
gate then reads, so a restore priced on that alone hands the card back into the pressure that evicted safety. A
whole-card residency's pause instead ends when its own model drains, and its restore is the liveness path that
gives a heavy-resident card its on-GPU safety process back."""

SAFETY_PLACEMENT_RESTORE_DWELL_FACTOR = 2.0
"""How much longer restore evidence must persist than demotion evidence, as a multiple of the demotion dwell.

Both dwells are seconds derived from the measured cost of one placement flip, never a cycle count: the control
loop runs several times a second, so counting cycles lets a sub-second reading spend tens of seconds of safety
unavailability. Greater than one so the worker leaves a card that is genuinely short of memory promptly and comes
back only once its room has proven durable, so a readmit does not re-trip on the next heavy job."""

SAFETY_BACKLOG_PRIORITY_DEPTH = 2
"""Safety backlog depth above which GPU safety restoration is prioritized over placement inertia."""

SAFETY_RESTORE_PP_BACKLOG_DEPTH = 2
"""Post-processing backlog depth above which a paused safety process is kept off its card.

A depth rather than mere presence because a worker serving post-processed requests is rarely without some
post-processing work in flight, and an absolute veto would hand that steady trickle the power to keep safety off
the card for the whole run. Matched to :data:`SAFETY_BACKLOG_PRIORITY_DEPTH`: a queue this size is one ordinary
job's tail work, while anything deeper is a lane genuinely under load."""

SAFETY_RESTORE_PP_BACKLOG_MAX_AGE_SECONDS = 90.0
"""How long a shallow post-processing backlog may defer safety restoration before it stops counting.

Measured from when the backlog last became non-empty, so an emptying lane resets it and only continuously
occupied work ages. Long enough to cover the tail of a batch of jobs. The headroom evidence still decides
restoration timing; this bound only stops an unbroken trickle from pinning safety to the CPU indefinitely."""


@dataclass(frozen=True)
class SafetyPlacementInputs:
    """The per-card evidence the placement policy decides one cycle from.

    Every term is about the card safety occupies (or would land on while off-GPU), because cards are independent
    VRAM domains and a sibling card's sampling says nothing about this one. Gathered once per cycle so the
    demotion predicate, the restore forecast and the diagnostics cannot read different pictures of the card.
    """

    device_index: int | None
    measured_free_mb: float | None
    marginal_need_mb: float
    """The device VRAM (MB) the card still has to find for the heaviest peak it is committed to, net of what the
    process that will sample it already holds. Charging the whole peak instead makes a modeled non-fit permanent
    on a small card."""
    noise_buffer_mb: float
    governor_state: GovernorState
    safety_footprint_mb: float
    reclaimable_idle_mb: float = 0.0
    """Device memory (MB) idle inference processes on the card hold in resident weights.

    Whether kept warm under a retention grant or simply left loaded after the last job, those weights leave the
    moment an idle-model eviction asks, so for placement they are room the card can produce within a tick, not
    room it lacks. Counting them as used arms a demotion for as long as a resident sits idle and keeps the restore
    forecast from ever passing on a card that holds models between jobs."""

    def available_mb(self) -> float | None:
        """The measured free plus what idle retained residents would return; None without a measurement."""
        if self.measured_free_mb is None:
            return None
        return self.measured_free_mb + self.reclaimable_idle_mb

    def restore_requirement_mb(self) -> float:
        """The available room the card must show for safety to survive the peak it is committed to."""
        return self.safety_footprint_mb + self.marginal_need_mb + self.noise_buffer_mb

    def is_pressured(self) -> bool:
        """Whether safety's card is short of memory right now, on measured evidence rather than a model.

        Two facts demote: the device-free governor has left HEALTHY, or the available room no longer covers the
        marginal step the heaviest committed peak still has to take plus the noise buffer. A modeled non-fit is
        deliberately not one of them: on a small card that arithmetic can be unsatisfiable by a margin narrower
        than the buffer itself, which arms a permanent eviction against a card that is serving its work. Missing
        telemetry does not demote, and neither does a card with nothing left to find: a low free reading beside
        work the card already holds the memory for is a card full of weights, which is admission's and reclaim's
        subject rather than placement's.
        """
        if self.governor_state is not GovernorState.HEALTHY:
            return True
        available_mb = self.available_mb()
        if available_mb is None or self.marginal_need_mb <= 0.0:
            return False
        return available_mb < (self.marginal_need_mb + self.noise_buffer_mb)

    def restore_headroom_fits(self) -> bool:
        """Whether the card's available room covers safety beside the peak it is committed to.

        A forecast rather than a snapshot: readmitting safety on room the next peak will take back is how one
        restore buys the next eviction. The governor must also be HEALTHY, so a card hovering at the paging cliff
        never readmits, and missing telemetry does not restore: promotion needs positive measured evidence.
        """
        available_mb = self.available_mb()
        if available_mb is None or self.governor_state is not GovernorState.HEALTHY:
            return False
        return available_mb >= self.restore_requirement_mb()

    def describe(self) -> str:
        """Return the evidence as one diagnostic clause, for the placement log lines."""
        free_display = "unreported" if self.measured_free_mb is None else f"{self.measured_free_mb:.0f}MB"
        return (
            f"card {self.device_index}: measured free {free_display}, reclaimable idle residents "
            f"{self.reclaimable_idle_mb:.0f}MB, marginal need "
            f"{self.marginal_need_mb:.0f}MB, noise buffer {self.noise_buffer_mb:.0f}MB, safety footprint "
            f"{self.safety_footprint_mb:.0f}MB, governor {self.governor_state.name}"
        )


class SafetyPlacementLedger:
    """The evidence clocks, one-shot requests and tallies behind runtime safety placement.

    Each clock is the time its condition has held continuously since, or None while it does not hold. The
    policy's verdict is derived from a clock against a dwell measured in seconds, so no intent can outlive the
    evidence that produced it.
    """

    def __init__(self, clock: Callable[[], float]) -> None:
        """Track placement evidence against ``clock``."""
        self._clock = clock
        self.pressure_since: float | None = None
        """When measured pressure on safety's card, with safety resident, was first seen."""
        self.headroom_since: float | None = None
        """When forecast headroom on the chosen card, with safety off it, was first seen."""
        self.pp_backlog_since: float | None = None
        """When the post-processing backlog last became non-empty; ages the restore-side deferral bound."""
        self.reclaim_pause_requested = False
        """The reclaim ladder's one-shot request to take safety off the card, consumed by the reconciler."""
        self.weights_demoted = False
        """Whether the on-GPU safety process holds its CLIP weights in host RAM at the parent's request."""
        self.demotions = 0
        """Policy-initiated moves of safety off the GPU this run (not the whole-card residency's own pauses)."""
        self.promotions = 0
        """Policy-initiated restores of safety onto the GPU this run."""
        self._last_logged_inputs: tuple[object, ...] | None = None

    def dwell_met(self, since: float | None, dwell_seconds: float) -> bool:
        """Whether evidence first seen at ``since`` has now held continuously for ``dwell_seconds``."""
        if since is None:
            return False
        return (self._clock() - since) >= dwell_seconds

    def freeze_evidence(self) -> None:
        """Drop both evidence clocks, so an intentional rebuild's window cannot decide the flip that follows it."""
        self.pressure_since = None
        self.headroom_since = None

    def reset(self) -> None:
        """Clear every clock and request while the policy is inert."""
        self.freeze_evidence()
        self.reclaim_pause_requested = False
        self.pp_backlog_since = None

    def advance_evidence(self, inputs: SafetyPlacementInputs, *, safety_paused: bool, pressure_counts: bool) -> None:
        """Advance the one clock that applies to safety's current placement, clearing the other.

        A resident safety process accrues the pressure that would justify evicting it, and an evicted one accrues
        the forecast headroom that would justify bringing it back. Either clock resets the moment its condition
        stops holding. ``pressure_counts`` is False when a caller-side veto (a safety backlog, a whole-card
        residency on safety's card) means measured pressure must not feed the clock this cycle.
        """
        now = self._clock()
        if safety_paused:
            self.pressure_since = None
            if not inputs.restore_headroom_fits():
                self.headroom_since = None
            elif self.headroom_since is None:
                self.headroom_since = now
            return

        self.headroom_since = None
        if not (pressure_counts and inputs.is_pressured()):
            self.pressure_since = None
        elif self.pressure_since is None:
            self.pressure_since = now

    def track_post_processing_backlog(self, depth: int) -> None:
        """Advance the post-processing backlog clock: cleared the moment the lane drains, started when it fills."""
        if depth == 0:
            self.pp_backlog_since = None
        elif self.pp_backlog_since is None:
            self.pp_backlog_since = self._clock()

    def post_processing_defers_restore(self, depth: int) -> bool:
        """Whether post-processing work keeps a paused safety process off its card this cycle.

        A safety (re)load competes with the post-processing lane's transient demand, so live post-processing
        defers restoration, bounded on depth and age: a backlog deeper than
        :data:`SAFETY_RESTORE_PP_BACKLOG_DEPTH` always defers, while a shallow one defers only until it has been
        continuously occupied for :data:`SAFETY_RESTORE_PP_BACKLOG_MAX_AGE_SECONDS`.
        """
        if depth == 0:
            return False
        if depth > SAFETY_RESTORE_PP_BACKLOG_DEPTH:
            return True
        since = self.pp_backlog_since
        if since is None:
            return True
        return (self._clock() - since) < SAFETY_RESTORE_PP_BACKLOG_MAX_AGE_SECONDS

    def log_inputs(self, inputs: SafetyPlacementInputs, *, safety_paused: bool) -> None:
        """Log the placement evidence on a change, so a later capture can attribute a flip to what it saw.

        Edge-triggered on the rounded evidence: an unchanged picture repeats at TRACE and a change speaks at DEBUG.
        """
        free_mb = inputs.measured_free_mb
        signature: tuple[object, ...] = (
            inputs.device_index,
            None if free_mb is None else round(free_mb / 64.0),
            round(inputs.marginal_need_mb / 64.0),
            round(inputs.reclaimable_idle_mb / 64.0),
            inputs.governor_state,
            safety_paused,
        )
        message = f"Runtime safety placement inputs: {inputs.describe()}."
        if signature == self._last_logged_inputs:
            logger.trace(message)
            return
        self._last_logged_inputs = signature
        logger.debug(message)
