"""The liveness contract of the VRAM admission system, tested as a property rather than by anecdote.

A worker that is alive but cannot admit its head-of-queue job is wedged: the process pool is healthy and
idle, yet no work flows. Two production incidents wedged exactly this way, both by the admission identity
counting the head's own footprint more than once and then blaming the resulting over-count on "the worker's
own committed load holds the card":

- A 24GB card, a flux head needing exclusive residency: its preload was admitted, recorded a planned charge,
  and had its target reclaimed before the load materialised. The stale planned charge never decayed (a dead
  target's reservation never grows) and never reconciled away, so the head's re-ask counted the load once as
  the lingering plan and once as the candidate delta and deferred forever on its own footprint.
- An 8GB card, an SD1.5 model already resident and idle on the target process: dispatching it moved nothing,
  yet its resident weights were charged in the committed floor, a stale planned charge persisted, AND the
  candidate delta re-charged the weights the resident-credit contract says it must net out: a triple count
  that put the no-op dispatch over capacity.

This module encodes the invariant those incidents violated: **no permanent DEFER may be reachable from a
request's own footprint alone.** The self-count hostile tests prove the permanent-DEFER branches (the
own-committed-shortfall branch first) cannot be composed purely from the request's own resident weights, its
own not-yet-materialised plan, and its own candidate delta. The progress-property matrix drives the
evaluate/actuate loop across a varied grid of circumstances and asserts that any head whose candidate can
physically fit the card, once the reclaim machinery it is entitled to has run, is admitted within a bounded
number of cycles.

Scope of the grid is stated precisely in :data:`_SCENARIOS` and the parametrised ids so nothing is claimed
beyond what is driven here. What it does NOT cover: the scheduler's assembly of the measured snapshot from
live process state (exercised in the scheduler suites), the governor's verified-reclaim ladder timing, and
multi-card routing. The two incident repros are kept as named regressions at the foot of the module; the
scheduler-side wiring that feeds these arbiter verdicts is proved in
``tests/process_management/regressions/test_admission_self_count_wedge.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from horde_worker_regen.process_management.resources.vram_arbiter import (
    _STARVATION_DIAGNOSTIC_SECONDS,
    ActuatorCommandKind,
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequest,
    VramRequestKind,
)

_PRELOAD_FLOW = "preload_admission"
"""The ledger flow the planned overlay is registered under in the running scheduler; matched here so the
ledger-level regression exercises the same namespace the scheduler reconciles."""


@dataclass
class _CardModel:
    """A decomposable model of one card's VRAM so a simulated reclaim can act on named contributors.

    The arbiter only ever sees the aggregate committed floor and the truthful device-free reading; this model
    holds the decomposition the real card has so the simulation can apply each actuation to the contributor it
    targets (release a lane's cache, evict an idle model, tear an idle context down) and re-price the reduced
    state on the next cycle. The request's own footprint is held separately from genuinely-other load so a test
    can compose a purely self-inflicted state and prove the head still makes progress.
    """

    total_vram_mb: float
    baseline_mb: float
    candidate_delta_mb: float
    candidate_resident_weights_mb: float = 0.0
    """The candidate's own weights already materialised in VRAM on the target (nonzero only when resident)."""
    own_stale_plan_mb: float = 0.0
    """The request's own admitted-but-unmaterialised planned charge lingering from an earlier cycle."""
    other_plan_mb: float = 0.0
    """A genuinely-other process's planned charge (never netted by this request)."""
    idle_cache_by_pid: dict[int, float] = field(default_factory=dict)
    """Releasable idle-lane allocator cache (MB) keyed by process id (RELEASE_CACHE targets)."""
    idle_model_mb: float = 0.0
    """An idle resident model's weights (MB) that EVICT_IDLE_MODEL can reclaim."""
    idle_context_mb: list[float] = field(default_factory=list)
    """Bare idle CUDA contexts (MB) that only a REDUCE_LIVE_CONTEXTS teardown frees (one per cycle)."""
    live_sibling_mb: float = 0.0
    """A busy sibling's committed weights (MB): genuinely-other load that no reclaim here may take."""
    device_free_override_mb: float | None = None
    """The truthful NVML device-free reading (MB) when it must diverge from ``total - baseline - committed``.

    The committed ledger and the device-free reading are independent measurements that disagree under WDDM: the
    foreign-pressure branch admits only when the truthful reading shows physical room the ledger does not. Most
    scenarios leave this None (device-free tracks the decomposition honestly); a foreign-pressure scenario sets
    it to model the truthful reading holding room the ledger prices as over-committed."""
    committed_is_stale: bool = False
    candidate_resident: bool = False

    def committed_mb(self) -> float:
        """The aggregate committed floor the arbiter prices, summed over every live contributor."""
        return (
            self.candidate_resident_weights_mb
            + sum(self.idle_cache_by_pid.values())
            + self.idle_model_mb
            + sum(self.idle_context_mb)
            + self.live_sibling_mb
        )

    def planned_mb(self) -> float:
        """The aggregate planned overlay: the request's own stale plan plus any genuinely-other plan."""
        return self.own_stale_plan_mb + self.other_plan_mb

    def device_free_mb(self) -> float:
        """The truthful device-free reading: the override when set, else total net of baseline and committed."""
        if self.device_free_override_mb is not None:
            return self.device_free_override_mb
        return max(0.0, self.total_vram_mb - self.baseline_mb - self.committed_mb())

    def capacity_mb(self) -> float:
        """The admission capacity ``(total - baseline) - noise`` the identity tests demand against."""
        return (self.total_vram_mb - self.baseline_mb) - admission_noise_buffer_mb(self.total_vram_mb)

    def to_state(self) -> DeviceVramState:
        """Freeze this model into the per-device measurement the arbiter prices a request against."""
        return DeviceVramState(
            total_vram_mb=self.total_vram_mb,
            baseline_mb=self.baseline_mb,
            committed_vram_mb=self.committed_mb(),
            planned_unmaterialized_mb=self.planned_mb(),
            committed_is_stale=self.committed_is_stale,
            noise_buffer_mb=admission_noise_buffer_mb(self.total_vram_mb),
            idle_process_ids=frozenset(self.idle_cache_by_pid),
            device_free_mb=self.device_free_mb(),
        )

    def request(self, *, kind: VramRequestKind, is_head: bool, starved_seconds: float) -> VramRequest:
        """Build the request this card's head presents, wired with the self-net and residency the model implies."""
        return VramRequest(
            kind=kind,
            job_label="head_model",
            baseline="stable_diffusion_xl",
            device_index=0,
            target_process_id=0,
            candidate_delta_mb=self.candidate_delta_mb,
            candidate_already_resident=self.candidate_resident,
            own_planned_unmaterialized_mb=self.own_stale_plan_mb,
            is_head_of_queue=is_head,
            starved_seconds=starved_seconds,
            has_reclaimable_idle_model=self.idle_model_mb > 0.0,
            idle_contexts_teardownable=is_head and kind is VramRequestKind.PRELOAD and bool(self.idle_context_mb),
        )

    def apply(self, actuation_kinds: list[ActuatorCommandKind], targets: list[int | None]) -> bool:
        """Apply one cycle's described actuations to the decomposition; return whether anything was freed."""
        freed = False
        for kind, target in zip(actuation_kinds, targets, strict=True):
            if kind is ActuatorCommandKind.RELEASE_CACHE and target in self.idle_cache_by_pid:
                del self.idle_cache_by_pid[target]
                freed = True
            elif kind is ActuatorCommandKind.EVICT_IDLE_MODEL and self.idle_model_mb > 0.0:
                self.idle_model_mb = 0.0
                freed = True
            elif kind is ActuatorCommandKind.REDUCE_LIVE_CONTEXTS and self.idle_context_mb:
                # A context is reclaimed only by a process exiting, so one teardown frees one context per cycle.
                self.idle_context_mb.pop()
                freed = True
        return freed


def _drive_until_admitted(card: _CardModel, *, kind: VramRequestKind, is_head: bool, max_cycles: int) -> str:
    """Drive the evaluate/actuate loop until the head is admitted or the cycle budget is spent.

    Each cycle freezes the current decomposition, evaluates the head, and on a DEFER runs the described
    actuations against the model so the next cycle prices the reduced state (the real caller's contract: a
    deferred demand's actuations are executed and the request re-asks). Returns a short outcome token so the
    caller can assert both the disposition and, for a DEFER, that reclaim was actually making progress.
    """
    arbiter = VramArbiter()
    for _cycle in range(max_cycles):
        state = card.to_state()
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
        # Starvation grows unbounded so a head entitled to a context teardown becomes eligible for it.
        request = card.request(kind=kind, is_head=is_head, starved_seconds=_STARVATION_DIAGNOSTIC_SECONDS + 30.0)
        verdict = arbiter.evaluate(request)
        if verdict.disposition is VramDisposition.FITS:
            return "admitted"
        if verdict.disposition is VramDisposition.DENY:
            return "denied"
        freed = card.apply(
            [c.kind for c in verdict.required_actuations],
            [c.target_process_id for c in verdict.required_actuations],
        )
        if not freed:
            # A DEFER that describes no runnable reclaim is a terminal wedge unless the card recovers on its
            # own; the loop cannot make progress, so stop and report it for the assertion to catch.
            return "wedged_defer"
    return "unadmitted_within_budget"


# --------------------------------------------------------------------------------------------------------
# The self-inflicted-state grid: a card whose entire demand is the request's own footprint.
# --------------------------------------------------------------------------------------------------------

_SMALL_CARD = {"total_vram_mb": 8192.0, "baseline_mb": 700.0}
_LARGE_CARD = {"total_vram_mb": 24576.0, "baseline_mb": 1200.0}


@dataclass
class _SelfCountCase:
    """One purely-self-inflicted composition and the model size it is priced on."""

    label: str
    card: dict[str, float]
    candidate_delta_mb: float
    candidate_resident_weights_mb: float
    own_stale_plan_mb: float
    candidate_resident: bool


_SELF_COUNT_CASES = [
    # A resident, idle model on a small card: weights in committed, a stale plan lingering, and a candidate
    # delta that (mispriced) re-charges the weights. The dispatch moves nothing and must admit.
    _SelfCountCase("resident_idle_small", _SMALL_CARD, 4000.0, 4000.0, 3998.0, True),
    # A resident model whose reservation alone tops the noise-adjusted ceiling: still a no-op, still admits.
    _SelfCountCase("resident_over_ceiling_small", _SMALL_CARD, 3800.0, 7000.0, 3800.0, True),
    # A staged (not resident) preload on a large card with only its own stale plan lingering: the plan is
    # netted, nothing else holds the card, so the candidate fits outright.
    _SelfCountCase("staged_own_plan_only_large", _LARGE_CARD, 8229.0, 0.0, 8229.0, False),
    # The same on a small card with a candidate that still fits the empty card once the self-plan is netted.
    _SelfCountCase("staged_own_plan_only_small", _SMALL_CARD, 3500.0, 0.0, 3500.0, False),
    # A resident model on a large card with a large stale plan: triple-count shape, admits as a no-op.
    _SelfCountCase("resident_large_plan_large", _LARGE_CARD, 8229.0, 8229.0, 8229.0, True),
]


class TestSelfCountNeverWedges:
    """No permanent DEFER is reachable when the entire demand is the request's own footprint.

    This is the core liveness mandate stated as a hostile property: compose committed and planned from nothing
    but the request's own resident weights and its own stale plan, then assert the head admits. Unnetted, each
    of these defers indefinitely on "the worker's own committed load holds the card after full reclaim", because
    the head's footprint is counted two or three times over.
    """

    @pytest.mark.parametrize("case", _SELF_COUNT_CASES, ids=lambda c: c.label)
    def test_pure_self_footprint_admits_immediately(self, case: _SelfCountCase) -> None:
        """A purely self-inflicted state admits on the first evaluation: there is nothing to reclaim."""
        card = _CardModel(
            total_vram_mb=case.card["total_vram_mb"],
            baseline_mb=case.card["baseline_mb"],
            candidate_delta_mb=case.candidate_delta_mb,
            candidate_resident_weights_mb=case.candidate_resident_weights_mb,
            own_stale_plan_mb=case.own_stale_plan_mb,
            candidate_resident=case.candidate_resident,
        )
        arbiter = VramArbiter()
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: card.to_state()}))
        verdict = arbiter.evaluate(
            card.request(kind=VramRequestKind.PRELOAD, is_head=True, starved_seconds=0.0),
        )
        assert verdict.disposition is VramDisposition.FITS
        assert "own committed load holds the card" not in verdict.reason

    @pytest.mark.parametrize("case", _SELF_COUNT_CASES, ids=lambda c: c.label)
    def test_pure_self_footprint_never_reaches_own_shortfall_defer(self, case: _SelfCountCase) -> None:
        """Driven for several cycles, a purely self-inflicted state is never wedged on its own footprint."""
        for kind in (VramRequestKind.PRELOAD, VramRequestKind.MONOLITHIC_DISPATCH):
            card = _CardModel(
                total_vram_mb=case.card["total_vram_mb"],
                baseline_mb=case.card["baseline_mb"],
                candidate_delta_mb=case.candidate_delta_mb,
                candidate_resident_weights_mb=case.candidate_resident_weights_mb,
                own_stale_plan_mb=case.own_stale_plan_mb,
                candidate_resident=case.candidate_resident,
            )
            outcome = _drive_until_admitted(card, kind=kind, is_head=True, max_cycles=5)
            assert outcome == "admitted", f"{case.label}/{kind}: expected admission, got {outcome}"


class TestOwnShortfallRequiresGenuinelyOtherLoad:
    """The own-committed-shortfall permanent DEFER may only fire on load that is not the request's own.

    This is the hostile test on the first permanent-DEFER branch. It proves the branch keys on genuinely-other
    committed load (a live sibling holding the card), not on any composition of the request's own resident
    weights, own stale plan, or own candidate.
    """

    def test_live_sibling_over_capacity_still_defers_on_own_shortfall(self) -> None:
        """A live sibling whose weights alone exceed capacity legitimately holds the card: the head waits."""
        card = _CardModel(
            total_vram_mb=8192.0,
            baseline_mb=700.0,
            candidate_delta_mb=100.0,
            live_sibling_mb=7300.0,
        )
        arbiter = VramArbiter()
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: card.to_state()}))
        verdict = arbiter.evaluate(card.request(kind=VramRequestKind.PRELOAD, is_head=True, starved_seconds=0.0))
        assert verdict.disposition is VramDisposition.DEFER
        assert "own committed load holds the card" in verdict.reason

    def test_swapping_sibling_load_for_own_plan_removes_the_defer(self) -> None:
        """Moving the over-capacity mass from a live sibling to the head's own stale plan admits instead.

        The raw demand is identical (a small candidate atop a 7300 MB mass that tops capacity), but in one case
        the mass is a genuinely-other live sibling and in the other it is the request's own not-yet-materialised
        plan. The self-net removes the latter, so the branch that legitimately fired for the sibling cannot fire
        for the request's own footprint.
        """
        capacity_topping_mb = 7300.0
        sibling_card = _CardModel(
            total_vram_mb=8192.0,
            baseline_mb=700.0,
            candidate_delta_mb=100.0,
            live_sibling_mb=capacity_topping_mb,
        )
        own_plan_card = _CardModel(
            total_vram_mb=8192.0,
            baseline_mb=700.0,
            candidate_delta_mb=100.0,
            own_stale_plan_mb=capacity_topping_mb,
        )
        arbiter = VramArbiter()
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: sibling_card.to_state()}))
        assert (
            arbiter.evaluate(
                sibling_card.request(kind=VramRequestKind.PRELOAD, is_head=True, starved_seconds=0.0),
            ).disposition
            is VramDisposition.DEFER
        )
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: own_plan_card.to_state()}))
        own_verdict = arbiter.evaluate(
            own_plan_card.request(kind=VramRequestKind.PRELOAD, is_head=True, starved_seconds=0.0),
        )
        assert own_verdict.disposition is VramDisposition.FITS


# --------------------------------------------------------------------------------------------------------
# The progress-property matrix: a head entitled to reclaim is admitted within a bounded number of cycles.
# --------------------------------------------------------------------------------------------------------


@dataclass
class _ProgressScenario:
    """One reclaimable-pressure scenario and the reclaim path the head is entitled to walk to admission."""

    label: str
    card: _CardModel
    kind: VramRequestKind
    is_head: bool
    expected: str
    max_cycles: int = 8


def _progress_scenarios() -> list[_ProgressScenario]:
    """The grid of circumstances the progress property is asserted over (see module docstring for scope)."""
    scenarios: list[_ProgressScenario] = []

    # Idle-lane cache holds the card; releasing it admits the head within a cycle or two. Both card sizes.
    scenarios.append(
        _ProgressScenario(
            "small_release_cache_admits",
            _CardModel(
                total_vram_mb=8192.0,
                baseline_mb=700.0,
                candidate_delta_mb=3000.0,
                idle_cache_by_pid={3: 2500.0, 5: 2500.0},
            ),
            VramRequestKind.PRELOAD,
            True,
            "admitted",
        ),
    )
    scenarios.append(
        _ProgressScenario(
            "large_release_cache_admits",
            _CardModel(
                total_vram_mb=24576.0,
                baseline_mb=1200.0,
                candidate_delta_mb=8229.0,
                idle_cache_by_pid={2: 9000.0, 4: 9000.0},
            ),
            VramRequestKind.PRELOAD,
            True,
            "admitted",
        ),
    )

    # An idle resident model holds the card; evicting it admits the head.
    scenarios.append(
        _ProgressScenario(
            "evict_idle_model_admits",
            _CardModel(
                total_vram_mb=24576.0,
                baseline_mb=1200.0,
                candidate_delta_mb=8229.0,
                idle_model_mb=16000.0,
            ),
            VramRequestKind.PRELOAD,
            True,
            "admitted",
        ),
    )

    # The exclusive-head shape generalised: idle sibling CUDA contexts hold the deficit, freed only by teardown,
    # one per cycle. A starved exclusive head is entitled to tear them down and must be admitted within a run.
    scenarios.append(
        _ProgressScenario(
            "idle_contexts_teardown_admits",
            _CardModel(
                total_vram_mb=24576.0,
                baseline_mb=1200.0,
                candidate_delta_mb=8229.0,
                idle_context_mb=[4000.0, 4000.0, 4000.0, 4000.0],
            ),
            VramRequestKind.PRELOAD,
            True,
            "admitted",
        ),
    )

    # Mixed pressure: some releasable cache, an idle model, and idle contexts. Every rung is walked to admit.
    scenarios.append(
        _ProgressScenario(
            "mixed_reclaim_admits",
            _CardModel(
                total_vram_mb=24576.0,
                baseline_mb=1200.0,
                candidate_delta_mb=8229.0,
                idle_cache_by_pid={6: 3000.0},
                idle_model_mb=6000.0,
                idle_context_mb=[4000.0, 4000.0],
            ),
            VramRequestKind.PRELOAD,
            True,
            "admitted",
        ),
    )

    # A resident, idle candidate over a fully-committed small card: dispatch materialises nothing, admits at
    # once even though committed sits above capacity (the resident no-op-dispatch contract).
    scenarios.append(
        _ProgressScenario(
            "resident_dispatch_admits_over_capacity",
            _CardModel(
                total_vram_mb=8192.0,
                baseline_mb=700.0,
                candidate_delta_mb=4000.0,
                candidate_resident_weights_mb=4000.0,
                idle_cache_by_pid={2: 3600.0},
                candidate_resident=True,
            ),
            VramRequestKind.MONOLITHIC_DISPATCH,
            True,
            "admitted",
        ),
    )

    # A candidate that cannot fit even an emptied card denies: legitimate non-progress, not a wedge.
    scenarios.append(
        _ProgressScenario(
            "oversized_candidate_denies",
            _CardModel(
                total_vram_mb=8192.0,
                baseline_mb=700.0,
                candidate_delta_mb=9000.0,
                idle_cache_by_pid={3: 2000.0},
            ),
            VramRequestKind.PRELOAD,
            True,
            "denied",
        ),
    )

    # Foreign load the worker cannot reclaim, but the candidate physically fits the truthful device-free
    # reading: the head admits into reality. The ledger prices the card over-committed (own load fits capacity,
    # the candidate tips it over) while the truthful device-free reading independently shows physical room.
    scenarios.append(
        _ProgressScenario(
            "foreign_pressure_head_admits_into_reality",
            _CardModel(
                total_vram_mb=24576.0,
                baseline_mb=1200.0,
                candidate_delta_mb=5000.0,
                live_sibling_mb=18000.0,
                device_free_override_mb=8000.0,
            ),
            VramRequestKind.PRELOAD,
            True,
            "admitted",
            max_cycles=2,
        ),
    )

    # A stale committed ledger drops the measured floor: a fitting candidate is admitted on the planned-only
    # identity, never wedged by however large the stale floor reads.
    scenarios.append(
        _ProgressScenario(
            "stale_floor_admits_fitting_candidate",
            _CardModel(
                total_vram_mb=24576.0,
                baseline_mb=1200.0,
                candidate_delta_mb=8229.0,
                live_sibling_mb=30000.0,
                committed_is_stale=True,
            ),
            VramRequestKind.PRELOAD,
            True,
            "admitted",
            max_cycles=2,
        ),
    )

    # The self-inflicted state carried alongside genuinely-other reclaimable load: the head's own stale plan is
    # netted AND the idle cache is reclaimed, so it still admits. Proves the self-net does not disable reclaim.
    scenarios.append(
        _ProgressScenario(
            "own_plan_plus_reclaimable_cache_admits",
            _CardModel(
                total_vram_mb=24576.0,
                baseline_mb=1200.0,
                candidate_delta_mb=8229.0,
                own_stale_plan_mb=8229.0,
                idle_cache_by_pid={7: 9000.0, 8: 9000.0},
            ),
            VramRequestKind.PRELOAD,
            True,
            "admitted",
        ),
    )

    return scenarios


class TestProgressProperty:
    """Any head whose candidate can physically fit is admitted within a bounded number of evaluate/actuate cycles.

    The loop mirrors the caller's contract: a DEFER's described actuations are executed against the card model
    and the request re-asks the next cycle. A scenario that describes no runnable reclaim on a DEFER is a wedge
    and fails the property; the only acceptable non-admissions are a structural DENY (the candidate cannot fit
    an emptied card) which is asserted explicitly.
    """

    @pytest.mark.parametrize("scenario", _progress_scenarios(), ids=lambda s: s.label)
    def test_head_reaches_its_expected_outcome_within_budget(self, scenario: _ProgressScenario) -> None:
        """The head reaches admission (or a legitimate DENY) within the scenario's bounded cycle budget."""
        outcome = _drive_until_admitted(
            scenario.card,
            kind=scenario.kind,
            is_head=scenario.is_head,
            max_cycles=scenario.max_cycles,
        )
        assert outcome == scenario.expected, f"{scenario.label}: expected {scenario.expected}, got {outcome}"

    def test_non_head_is_held_behind_the_head_but_the_head_still_progresses(self) -> None:
        """A non-head (line-skip) request forfeits the over-budget admit, but its presence never wedges the head.

        A non-head request that would physically fit foreign-pressure room is denied that room (it belongs to
        the head), so it defers. The same card presented to the true head admits it into reality. This proves
        the head-only reservation does not itself become a wedge for the head it protects.
        """
        card = _CardModel(
            total_vram_mb=24576.0,
            baseline_mb=1200.0,
            candidate_delta_mb=5000.0,
            live_sibling_mb=18000.0,
            device_free_override_mb=8000.0,
        )
        arbiter = VramArbiter()
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: card.to_state()}))
        non_head = arbiter.evaluate(card.request(kind=VramRequestKind.PRELOAD, is_head=False, starved_seconds=0.0))
        assert non_head.disposition is VramDisposition.DEFER
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: card.to_state()}))
        head = arbiter.evaluate(card.request(kind=VramRequestKind.PRELOAD, is_head=True, starved_seconds=0.0))
        assert head.disposition is VramDisposition.FITS


# --------------------------------------------------------------------------------------------------------
# Named incident regressions (arbiter- and ledger-level).
# --------------------------------------------------------------------------------------------------------


class TestFluxExclusiveHeadSelfDeadlockRegression:
    """24GB card, flux exclusive head: a plan admitted then reclaimed unmaterialised must not wedge the re-ask.

    The measured shape is committed 16090 + planned 8229 + candidate 8229 vs capacity 19270: the flux head's own
    earlier plan (8229) lingered after its target was reclaimed and, summed with the candidate, tipped the
    own-committed-shortfall branch over capacity so the head deferred indefinitely. Netting the head's own
    planned charge collapses the double count.
    """

    def _flux_state(self, *, planned_mb: float) -> DeviceVramState:
        """The incident's ledger: idle sibling contexts hold 16090 committed, with a lingering plan overlay."""
        total, baseline = 22000.0, 1218.0
        return DeviceVramState(
            total_vram_mb=total,
            baseline_mb=baseline,
            committed_vram_mb=16090.0,
            planned_unmaterialized_mb=planned_mb,
            committed_is_stale=False,
            noise_buffer_mb=admission_noise_buffer_mb(total),
            device_free_mb=max(0.0, total - baseline - 16090.0),
        )

    def _flux_head(self) -> VramRequest:
        """The flux head re-asking its preload, presenting its own lingering plan for the self-net."""
        return VramRequest(
            kind=VramRequestKind.PRELOAD,
            job_label="Flux.1-Schnell",
            baseline="flux_1",
            device_index=0,
            target_process_id=0,
            candidate_delta_mb=8229.0,
            own_planned_unmaterialized_mb=8229.0,
            is_head_of_queue=True,
            starved_seconds=_STARVATION_DIAGNOSTIC_SECONDS + 6.0,
            idle_contexts_teardownable=True,
        )

    def test_lingering_own_plan_no_longer_pins_the_own_shortfall_branch(self) -> None:
        """With the self-plan netted, the head is no longer blamed for 'own committed load holds the card'."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: self._flux_state(planned_mb=8229.0)}))
        verdict = arbiter.evaluate(self._flux_head())
        # The head still does not co-fit the 4 idle contexts, but it is now escalated to a verified context
        # teardown (the reclaim it is entitled to) rather than deferred forever on its own footprint.
        assert verdict.disposition is VramDisposition.DEFER
        assert "own committed load holds the card" not in verdict.reason
        assert ActuatorCommandKind.REDUCE_LIVE_CONTEXTS in [c.kind for c in verdict.required_actuations]

    def test_reask_admits_once_the_contexts_are_torn_down(self) -> None:
        """After the idle contexts exit, the committed floor drops and the head's re-ask admits (never forced)."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: self._flux_state(planned_mb=8229.0)}))
        assert arbiter.evaluate(self._flux_head()).disposition is VramDisposition.DEFER
        # The torn-down contexts drop committed to a single retained context; the lingering plan also decayed.
        total, baseline = 22000.0, 1218.0
        relieved = DeviceVramState(
            total_vram_mb=total,
            baseline_mb=baseline,
            committed_vram_mb=4000.0,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
            noise_buffer_mb=admission_noise_buffer_mb(total),
            device_free_mb=total - baseline - 4000.0,
        )
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: relieved}))
        assert arbiter.evaluate(self._flux_head()).disposition is VramDisposition.FITS


class TestResidentIdleDispatchRegression:
    """8GB card, SD1.5 resident and idle on the target: the dispatch materialises nothing and admits at once.

    The measured shape is committed 7714 + planned 3998 + candidate 4000 vs capacity 7256: the resident model's
    weights are in the committed floor, a stale plan persists, and the candidate re-charges the weights, a
    triple count that puts a no-op dispatch over capacity. A dispatch whose weights already occupy VRAM on the
    target adds no device demand and must release immediately, the whole-card analogue of a resident-lane stage.
    """

    def _resident_state(self) -> DeviceVramState:
        """The over-capacity 8GB card: the resident model plus its retained cache hold 7714 MB."""
        total, baseline = 8192.0, 512.0
        return DeviceVramState(
            total_vram_mb=total,
            baseline_mb=baseline,
            committed_vram_mb=7714.0,
            planned_unmaterialized_mb=3998.0,
            committed_is_stale=False,
            noise_buffer_mb=admission_noise_buffer_mb(total),
            device_free_mb=max(0.0, total - baseline - 7714.0),
        )

    def test_resident_idle_dispatch_admits_despite_over_capacity_committed(self) -> None:
        """The no-op dispatch admits even though committed alone exceeds the noise-adjusted admission ceiling."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: self._resident_state()}))
        request = VramRequest(
            kind=VramRequestKind.MONOLITHIC_DISPATCH,
            job_label="sd15_resident",
            baseline="stable_diffusion_1",
            device_index=0,
            target_process_id=2,
            candidate_delta_mb=4000.0,
            candidate_already_resident=True,
            own_planned_unmaterialized_mb=3998.0,
            is_head_of_queue=True,
        )
        verdict = arbiter.evaluate(request)
        assert verdict.disposition is VramDisposition.FITS
        assert verdict.measured.fits is False
        assert "already resident" in verdict.reason

    def test_a_staged_not_resident_dispatch_is_still_priced(self) -> None:
        """The admit is scoped to a genuinely-resident candidate: a staged one over this card still defers.

        Absent the resident flag the same over-capacity card prices the materialising dispatch through the
        identity, so the resident admit is scoped to a no-op and is not a blanket dispatch-admits rule.
        """
        arbiter = VramArbiter()
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: self._resident_state()}))
        staged = VramRequest(
            kind=VramRequestKind.MONOLITHIC_DISPATCH,
            job_label="sd15_staged",
            baseline="stable_diffusion_1",
            device_index=0,
            target_process_id=2,
            candidate_delta_mb=4000.0,
            candidate_already_resident=False,
            is_head_of_queue=True,
        )
        assert arbiter.evaluate(staged).disposition is not VramDisposition.FITS


class TestPlannedOverlayReleasedOnNonMaterialisation:
    """The planned overlay is empty once a target's admission is no longer in flight (the release path).

    The scheduler reconciles the overlay each cycle against the set of processes whose admitted load is still
    in flight; a target that finished, faulted, or died drops out of that set and its charge is released by
    omission. These pin that release at the ledger surface the scheduler drives, so a leaked charge cannot
    survive to double-count a re-ask.
    """

    def test_charge_released_when_target_leaves_the_live_set(self) -> None:
        """A planned charge whose target is absent from the live set reconciles to zero (dead-target release)."""
        ledger = CommittedReserveLedger()
        ledger.set_planned(_PRELOAD_FLOW, "0", vram_mb=8229.0, target_process_id=0, reserved_at_admit_mb=0.0)
        assert ledger.effective_planned_vram_mb({}) == pytest.approx(8229.0)
        # The target died: the scheduler's live set no longer contains it, so the reconcile drops the charge.
        ledger.reconcile_planned(_PRELOAD_FLOW, live_units=set())
        assert ledger.effective_planned_vram_mb({}) == pytest.approx(0.0)

    def test_own_charge_query_reads_the_outstanding_amount_without_ratcheting(self) -> None:
        """The self-net query reports a target's outstanding charge and never advances its materialisation."""
        ledger = CommittedReserveLedger()
        ledger.set_planned(_PRELOAD_FLOW, "0", vram_mb=8229.0, target_process_id=0, reserved_at_admit_mb=1000.0)
        # Read the outstanding charge twice against a reservation that has not grown: it stays full, unratcheted.
        first = ledger.planned_charge_for_unit(_PRELOAD_FLOW, "0", {0: 1000.0})
        second = ledger.planned_charge_for_unit(_PRELOAD_FLOW, "0", {0: 1000.0})
        assert first == pytest.approx(8229.0)
        assert second == pytest.approx(8229.0)
        # A genuinely-other unit carries no charge for this target.
        assert ledger.planned_charge_for_unit(_PRELOAD_FLOW, "9", {}) == pytest.approx(0.0)
