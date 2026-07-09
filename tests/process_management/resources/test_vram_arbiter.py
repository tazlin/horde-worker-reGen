"""Unit tests for the single VRAM arbiter's decision surface and observability counters.

The arbiter reasons from measured truth: a request fits iff its candidate outstanding cost fits the frozen
device-free reading net of the outstanding reservations that reading does not yet reflect and the one noise
buffer. There is no committed floor, no baseline term, and no foreign-pressure concept in the admission path:
those quantities are physically inside the device-free reading.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.resources.admission_identity import (
    admission_noise_buffer_mb,
)
from horde_worker_regen.process_management.resources.vram_arbiter import (
    _FIRST_PARTY_TEARDOWN_GRACE_SECONDS,
    _STARVATION_DIAGNOSTIC_SECONDS,
    ActuatorCommandKind,
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequest,
    VramRequestKind,
)


def _snapshot(state: DeviceVramState, *, device_index: int = 0) -> MeasuredVramSnapshot:
    """Wrap a single device state in a one-card snapshot."""
    return MeasuredVramSnapshot(devices={device_index: state})


def _roomy_state(**overrides: object) -> DeviceVramState:
    """A card with an ample device-free reading and no outstanding reservations, so an ordinary candidate fits.

    The floor noise buffer (512 MB) is left as the default so the admission arithmetic reads cleanly:
    available = device_free - reservations - 512. To simulate pressure a test lowers ``device_free_mb``; to
    stack admitted-but-unmaterialised demand it raises ``planned_unmaterialized_mb``.
    """
    defaults: dict[str, object] = {
        "total_vram_mb": 24000.0,
        "baseline_mb": 1000.0,
        "committed_vram_mb": 2000.0,
        "planned_unmaterialized_mb": 0.0,
        "committed_is_stale": False,
        "device_free_mb": 21000.0,
    }
    defaults.update(overrides)
    return DeviceVramState(**defaults)  # type: ignore[arg-type]


def _preload(**overrides: object) -> VramRequest:
    """A preload request with a moderate candidate delta, overridable per test."""
    defaults: dict[str, object] = {
        "kind": VramRequestKind.PRELOAD,
        "job_label": "model_a",
        "baseline": "stable_diffusion_xl",
        "device_index": 0,
        "candidate_delta_mb": 6000.0,
    }
    defaults.update(overrides)
    return VramRequest(**defaults)  # type: ignore[arg-type]


class TestAdmissionPath:
    """The measured-truth admission identity's dispositions and the escalation ladder."""

    def test_fits_when_candidate_fits_available_room(self) -> None:
        """A candidate whose cost fits the device-free reading net of the noise buffer admits."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state()))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=6000.0))
        assert verdict.disposition == VramDisposition.FITS
        assert verdict.admits is True
        assert verdict.measured.available_known is True

    def test_defer_emits_ladder_commands_in_order(self) -> None:
        """A candidate that overflows the device-free room defers with the ladder in escalation order."""
        arbiter = VramArbiter()
        # available = 5000 - 512 = 4488; a 5000 candidate does not fit, so the ladder is described.
        state = _roomy_state(device_free_mb=5000.0, idle_process_ids=frozenset({7, 3}))
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(
            _preload(
                candidate_delta_mb=5000.0,
                has_reclaimable_idle_model=True,
                can_reduce_live_contexts=True,
            ),
        )
        assert verdict.disposition == VramDisposition.DEFER
        kinds = [command.kind for command in verdict.required_actuations]
        assert kinds == [
            ActuatorCommandKind.RELEASE_CACHE,  # idle pid 3 (sorted first)
            ActuatorCommandKind.RELEASE_CACHE,  # idle pid 7
            ActuatorCommandKind.EVICT_IDLE_MODEL,
            ActuatorCommandKind.REDUCE_LIVE_CONTEXTS,
        ]
        release_targets = [
            c.target_process_id for c in verdict.required_actuations if c.kind == ActuatorCommandKind.RELEASE_CACHE
        ]
        assert release_targets == [3, 7]

    def test_ladder_omits_commands_that_could_free_nothing(self) -> None:
        """EVICT and REDUCE are emitted only when the request signals they could still free memory."""
        arbiter = VramArbiter()
        state = _roomy_state(device_free_mb=5000.0, idle_process_ids=frozenset({2}))
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(
            _preload(
                candidate_delta_mb=5000.0,
                has_reclaimable_idle_model=False,
                can_reduce_live_contexts=False,
            ),
        )
        assert verdict.disposition == VramDisposition.DEFER
        kinds = [command.kind for command in verdict.required_actuations]
        assert kinds == [ActuatorCommandKind.RELEASE_CACHE]

    def test_release_cache_never_targets_the_requesting_slot(self) -> None:
        """The request's own target slot is never asked to release the cache it is about to load into."""
        arbiter = VramArbiter()
        state = _roomy_state(device_free_mb=5000.0, idle_process_ids=frozenset({4, 9}))
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, target_process_id=4))
        release_targets = {
            c.target_process_id for c in verdict.required_actuations if c.kind == ActuatorCommandKind.RELEASE_CACHE
        }
        assert release_targets == {9}

    def test_release_cache_never_targets_a_busy_lane(self) -> None:
        """A busy lane is never a RELEASE_CACHE target even when listed idle by another key."""
        arbiter = VramArbiter()
        state = _roomy_state(
            device_free_mb=5000.0,
            idle_process_ids=frozenset({4, 5}),
            busy_process_ids=frozenset({5}),
        )
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=5000.0))
        release_targets = {
            c.target_process_id for c in verdict.required_actuations if c.kind == ActuatorCommandKind.RELEASE_CACHE
        }
        assert 5 not in release_targets
        assert release_targets == {4}

    def test_deny_when_candidate_cannot_fit_an_empty_card(self) -> None:
        """A candidate larger than an emptied card's room denies (no escalation could seat it)."""
        arbiter = VramArbiter()
        # An empty card offers total - noise = 24000 - 512 = 23488; a candidate above that can never seat.
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=1000.0)))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=30000.0))
        assert verdict.disposition == VramDisposition.DENY

    def test_candidate_already_resident_admits_as_a_no_op(self) -> None:
        """A dispatch onto already-resident weights admits even when the device-free reading has no room.

        The resident weights are physically inside the device-free reading already, so the dispatch
        materialises nothing; pricing its activation against the (now tight) reading would withhold a slot that
        is already open.
        """
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=100.0)))
        verdict = arbiter.evaluate(
            _preload(
                kind=VramRequestKind.MONOLITHIC_DISPATCH, candidate_delta_mb=6000.0, candidate_already_resident=True
            ),
        )
        assert verdict.disposition == VramDisposition.FITS
        assert "already resident" in verdict.reason
        assert verdict.measured.fits is False


class TestProportionalNoiseBuffer:
    """A device state's per-card noise buffer flows through the measured verdict it is priced against."""

    def test_verdict_uses_the_device_states_derived_buffer(self) -> None:
        """A 16375MB card assembled with the derived buffer prices room against 818.75, not the floor."""
        total = 16375.0
        state = _roomy_state(
            total_vram_mb=total,
            device_free_mb=12000.0,
            noise_buffer_mb=admission_noise_buffer_mb(total),
        )
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=1000.0))
        assert verdict.measured.noise_buffer_mb == pytest.approx(818.75)

    def test_derived_buffer_denies_a_demand_the_floor_would_admit(self) -> None:
        """The larger derived buffer flips an otherwise-fitting demand that the floor buffer would seat.

        With device-free 700 and a 100 MB candidate: the floor buffer leaves 188 MB of room (fits), the
        derived buffer (818.75) leaves negative room (does not fit).
        """
        total = 16375.0
        floor_state = _roomy_state(total_vram_mb=total, device_free_mb=700.0)
        derived_state = _roomy_state(
            total_vram_mb=total,
            device_free_mb=700.0,
            noise_buffer_mb=admission_noise_buffer_mb(total),
        )
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(floor_state))
        assert arbiter.evaluate(_preload(candidate_delta_mb=100.0)).disposition == VramDisposition.FITS
        arbiter.begin_cycle(_snapshot(derived_state))
        assert arbiter.evaluate(_preload(candidate_delta_mb=100.0)).disposition == VramDisposition.DEFER


class TestRelaxationAndMissingReading:
    """No cycle relaxes to the predictive path; a present card with no device-free reading defers, never denies."""

    def test_missing_snapshot_relaxes_to_fits(self) -> None:
        """Evaluating before a cycle is installed relaxes to admit on the predictive path."""
        arbiter = VramArbiter()
        verdict = arbiter.evaluate(_preload())
        assert verdict.disposition == VramDisposition.FITS

    def test_missing_device_relaxes_to_fits(self) -> None:
        """A request for a card absent from the snapshot relaxes to admit on the predictive path."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(), device_index=1))
        verdict = arbiter.evaluate(_preload(device_index=0))
        assert verdict.disposition == VramDisposition.FITS

    def test_present_card_without_device_free_defers_and_counts(self) -> None:
        """A present card with no device-free reading defers (never denies) and advances the missing counter."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=None)))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=6000.0, is_head_of_queue=True))
        assert verdict.disposition == VramDisposition.DEFER
        assert verdict.measured.available_known is False
        assert arbiter.device_free_missing_defers == 1

    def test_missing_device_free_diagnostic_throttled_per_cycle(self) -> None:
        """Two evaluations in one frozen cycle count each defer but warn at most once per card."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=None)))
        arbiter.evaluate(_preload())
        arbiter.evaluate(_preload())
        assert arbiter.device_free_missing_defers == 2


class TestReservationsReduceAvailable:
    """Outstanding reservations are subtracted from the device-free reading and can flip a verdict."""

    def test_reservations_reduce_the_available_room(self) -> None:
        """A candidate that fits the bare reading defers once outstanding reservations claim the room."""
        arbiter = VramArbiter()
        # available without reservations = 10000 - 512 = 9488 (fits 5000); with 6000 reserved = 3488 (does not).
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=10000.0)))
        assert arbiter.evaluate(_preload(candidate_delta_mb=5000.0)).disposition == VramDisposition.FITS
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=10000.0, planned_unmaterialized_mb=6000.0)))
        assert arbiter.evaluate(_preload(candidate_delta_mb=5000.0)).disposition == VramDisposition.DEFER

    def test_own_reservation_is_netted_so_a_reask_never_blocks_on_itself(self) -> None:
        """A re-ask nets its own outstanding reservation out, so its own admitted plan cannot deny it."""
        arbiter = VramArbiter()
        # 6000 reserved would defer a 5000 candidate, but if that reservation is the request's own it nets out.
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=10000.0, planned_unmaterialized_mb=6000.0)))
        blocked = arbiter.evaluate(_preload(candidate_delta_mb=5000.0))
        assert blocked.disposition == VramDisposition.DEFER
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=10000.0, planned_unmaterialized_mb=6000.0)))
        self_owned = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, own_planned_unmaterialized_mb=6000.0))
        assert self_owned.disposition == VramDisposition.FITS

    def test_stacked_reservations_deny_a_second_admission_in_one_window(self) -> None:
        """A reservation booked earlier in the window is seen, so the room is not admitted against twice."""
        arbiter = VramArbiter()
        device_free = 13000.0
        candidate = 6158.0
        dispositions: list[VramDisposition] = []
        planned_mb = 0.0
        for _cycle in range(3):
            arbiter.begin_cycle(
                _snapshot(_roomy_state(device_free_mb=device_free, planned_unmaterialized_mb=planned_mb))
            )
            verdict = arbiter.evaluate(_preload(candidate_delta_mb=candidate))
            dispositions.append(verdict.disposition)
            if verdict.disposition == VramDisposition.FITS:
                planned_mb += candidate
        # available: cycle0 12488 (fits), cycle1 6330 (fits), cycle2 172 (does not: the stacked reservations win).
        assert dispositions[0] == VramDisposition.FITS
        assert dispositions[1] == VramDisposition.FITS
        assert dispositions[2] != VramDisposition.FITS


class TestHeadProtection:
    """A fitting non-head request is withheld when admitting it would leave the head unable to fit.

    On the db0 4090 wedge a non-head job took physical room the head-of-queue job needed, starving the head
    while line-skippers consumed the card. The room the truthful device-free reading shows belongs to the head:
    a non-head request that would leave less than the head's priced demand defers, holding the room for the head.
    """

    def test_head_admits_into_the_room_the_reading_shows(self) -> None:
        """The true head admits when the device-free reading has physical room for it."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=8000.0)))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, is_head_of_queue=True))
        assert verdict.disposition == VramDisposition.FITS

    def test_non_head_is_held_when_admitting_it_would_starve_the_head(self) -> None:
        """A non-head request that fits defers when the head could not fit the room it would leave behind."""
        arbiter = VramArbiter()
        # available = 8000 - 512 = 7488; the non-head's 5000 leaves 2488, below the head's 5000 demand.
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=8000.0)))
        verdict = arbiter.evaluate(
            _preload(candidate_delta_mb=5000.0, is_head_of_queue=False, head_outstanding_mb=5000.0),
        )
        assert verdict.disposition == VramDisposition.DEFER
        assert "held for the head" in verdict.reason
        assert arbiter.admission_foreign_pressure_defers == 1

    def test_non_head_admits_when_room_remains_for_the_head(self) -> None:
        """A non-head request admits when the room it leaves still covers the head's demand."""
        arbiter = VramArbiter()
        # available = 20000 - 512 = 19488; the non-head's 5000 leaves 14488, above the head's 5000 demand.
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=20000.0)))
        verdict = arbiter.evaluate(
            _preload(candidate_delta_mb=5000.0, is_head_of_queue=False, head_outstanding_mb=5000.0),
        )
        assert verdict.disposition == VramDisposition.FITS
        assert arbiter.admission_foreign_pressure_defers == 0

    def test_non_head_without_a_known_head_demand_admits(self) -> None:
        """With the head's demand unknown at this seam, head protection is skipped and the non-head admits."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=8000.0)))
        verdict = arbiter.evaluate(
            _preload(candidate_delta_mb=5000.0, is_head_of_queue=False, head_outstanding_mb=None),
        )
        assert verdict.disposition == VramDisposition.FITS


class TestReserveDecoupling:
    """The operator's ``vram_reserve_mb`` never contributes to a load-feasibility denial (flux-wedge guard)."""

    def test_reserve_is_never_a_preload_denial_term(self) -> None:
        """A large operator reserve does not tip an otherwise-fitting preload into a denial."""
        arbiter = VramArbiter()
        # available = 21000 - 512 = 20488; the candidate exactly fits, and the 8192 reserve must not tip it over.
        state = _roomy_state(vram_reserve_mb=8192.0)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=20488.0, is_head_of_queue=True))
        assert verdict.disposition == VramDisposition.FITS


class TestUnpriceableCandidate:
    """An unpriceable candidate is charged nothing, so it never denies on an unknown cost."""

    def test_unpriceable_candidate_is_charged_nothing(self) -> None:
        """A None candidate delta is charged zero and admits within any non-negative room."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=5000.0)))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=None))
        assert verdict.disposition == VramDisposition.FITS
        assert verdict.measured.candidate_outstanding_mb == pytest.approx(0.0)


class TestStarvationDiagnostic:
    """A head deferred past the diagnostic horizon warns and counts, but still defers (never admits)."""

    def _wedged_head(self, starved_seconds: float) -> VramRequest:
        """A head whose candidate does not fit the device-free room, deferred for ``starved_seconds``."""
        return _preload(candidate_delta_mb=100.0, is_head_of_queue=True, starved_seconds=starved_seconds)

    def _wedged_state(self) -> DeviceVramState:
        """A card whose device-free room cannot hold even a 100 MB candidate net of the noise buffer."""
        return _roomy_state(device_free_mb=400.0)

    def test_diagnostic_emitted_past_the_horizon_without_admitting(self) -> None:
        """Past the horizon the diagnostic counter advances but the head still defers."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._wedged_state()))
        verdict = arbiter.evaluate(self._wedged_head(_STARVATION_DIAGNOSTIC_SECONDS + 5.0))
        assert verdict.disposition == VramDisposition.DEFER
        assert arbiter.starvation_diagnostics == 1

    def test_diagnostic_throttled_to_once_per_cycle_per_device(self) -> None:
        """Two evaluations within one frozen cycle advance the diagnostic counter at most once."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._wedged_state()))
        arbiter.evaluate(self._wedged_head(_STARVATION_DIAGNOSTIC_SECONDS + 5.0))
        arbiter.evaluate(self._wedged_head(_STARVATION_DIAGNOSTIC_SECONDS + 5.0))
        assert arbiter.starvation_diagnostics == 1

    def test_head_below_the_horizon_emits_no_diagnostic(self) -> None:
        """A head deferred below the diagnostic horizon defers silently (no counter movement)."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._wedged_state()))
        verdict = arbiter.evaluate(self._wedged_head(5.0))
        assert verdict.disposition == VramDisposition.DEFER
        assert arbiter.starvation_diagnostics == 0


class TestStarvationContextTeardown:
    """A head starved past the grace whose deficit is held by idle sibling contexts escalates to teardown.

    Idle sibling inference contexts hold a bare CUDA baseline that weight eviction cannot reclaim (a context is
    freed only when its process exits). On the db0 flux wedge the built ladder had no rung that could free that
    baseline, so an exclusive head starved indefinitely. Past the grace the arbiter escalates to a
    REDUCE_LIVE_CONTEXTS actuation that tears the idle contexts down, then admits once the room verifies.
    """

    def _starved_head(self, **overrides: object) -> VramRequest:
        """A head whose candidate does not fit, starved past the grace, with idle contexts to tear down."""
        defaults: dict[str, object] = {
            "candidate_delta_mb": 5000.0,
            "is_head_of_queue": True,
            "starved_seconds": _STARVATION_DIAGNOSTIC_SECONDS + 5.0,
            "idle_contexts_teardownable": True,
        }
        defaults.update(overrides)
        return _preload(**defaults)

    def test_starved_head_escalates_to_context_teardown(self) -> None:
        """The escalation defers with a single REDUCE_LIVE_CONTEXTS command and advances the teardown counter."""
        arbiter = VramArbiter()
        # available = 4000 - 512 = 3488; the 5000 candidate does not fit and the deficit is the idle contexts.
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=4000.0)))
        verdict = arbiter.evaluate(self._starved_head())
        assert verdict.disposition == VramDisposition.DEFER
        assert [c.kind for c in verdict.required_actuations] == [ActuatorCommandKind.REDUCE_LIVE_CONTEXTS]
        assert arbiter.starvation_context_teardowns == 1

    def test_teardown_then_admits_after_the_contexts_are_reduced(self) -> None:
        """Once the torn-down contexts free device room, the head's re-ask admits (never force-admitted)."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=4000.0)))
        assert arbiter.evaluate(self._starved_head()).disposition == VramDisposition.DEFER
        # The idle contexts exited, so the device-free reading recovered: the re-ask fits.
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=10000.0)))
        readmitted = arbiter.evaluate(self._starved_head())
        assert readmitted.disposition == VramDisposition.FITS

    def test_below_grace_defers_without_teardown(self) -> None:
        """A head starved below the escalation grace defers without a teardown command."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=4000.0)))
        verdict = arbiter.evaluate(self._starved_head(starved_seconds=5.0))
        assert verdict.disposition == VramDisposition.DEFER
        assert ActuatorCommandKind.REDUCE_LIVE_CONTEXTS not in [c.kind for c in verdict.required_actuations]
        assert arbiter.starvation_context_teardowns == 0

    def test_no_teardownable_contexts_no_escalation(self) -> None:
        """Without idle contexts to reclaim, a starved head defers via the ordinary shortfall path."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=4000.0)))
        verdict = arbiter.evaluate(self._starved_head(idle_contexts_teardownable=False))
        assert verdict.disposition == VramDisposition.DEFER
        assert verdict.required_actuations == ()
        assert arbiter.starvation_context_teardowns == 0

    def test_unstarved_dispatch_never_tears_a_context_down(self) -> None:
        """An ordinary (un-starved) MONOLITHIC_DISPATCH request never escalates to context teardown."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=4000.0)))
        dispatch = VramRequest(
            kind=VramRequestKind.MONOLITHIC_DISPATCH,
            job_label="model_a",
            baseline="stable_diffusion_xl",
            device_index=0,
            candidate_delta_mb=5000.0,
            is_head_of_queue=True,
            starved_seconds=1.0,
            idle_contexts_teardownable=True,
        )
        verdict = arbiter.evaluate(dispatch)
        assert verdict.disposition == VramDisposition.DEFER
        assert ActuatorCommandKind.REDUCE_LIVE_CONTEXTS not in [c.kind for c in verdict.required_actuations]
        assert arbiter.starvation_context_teardowns == 0

    def test_starved_dispatch_head_escalates_to_context_teardown(self) -> None:
        """A starved MONOLITHIC_DISPATCH head with no device-free room tears its own idle contexts down.

        Parity with the preload seam: once the candidate does not fit the device-free reading and the deficit is
        held by the head's own idle sibling contexts, the dispatch head escalates to the same teardown past the
        grace.
        """
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=4000.0)))
        dispatch = VramRequest(
            kind=VramRequestKind.MONOLITHIC_DISPATCH,
            job_label="model_a",
            baseline="stable_diffusion_xl",
            device_index=0,
            candidate_delta_mb=5000.0,
            is_head_of_queue=True,
            starved_seconds=_FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 5.0,
            idle_contexts_teardownable=True,
        )
        verdict = arbiter.evaluate(dispatch)
        assert verdict.disposition == VramDisposition.DEFER
        assert [c.kind for c in verdict.required_actuations] == [ActuatorCommandKind.REDUCE_LIVE_CONTEXTS]
        assert arbiter.starvation_context_teardowns == 1


class TestFirstPartyContextDefer:
    """A starved head's own idle sibling contexts are a reclaimable first-party deficit awaiting the grace.

    Before the teardown grace elapses, a head whose only unreclaimed deficit is its own teardownable idle
    contexts defers as a first-party reclaim (counted distinctly), not blamed on head-protection or diagnosed as
    a foreign wedge. Past the grace it escalates to the teardown.
    """

    def _first_party_head(self, **overrides: object) -> VramRequest:
        """A head whose only unreclaimed deficit is its own teardownable idle sibling contexts."""
        defaults: dict[str, object] = {
            "candidate_delta_mb": 5000.0,
            "is_head_of_queue": True,
            "idle_contexts_teardownable": True,
            "has_reclaimable_idle_model": False,
            "can_reduce_live_contexts": False,
        }
        defaults.update(overrides)
        return _preload(**defaults)

    def test_below_grace_defers_as_first_party(self) -> None:
        """Before the teardown grace the head defers on its own contexts, counted as a first-party defer."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=4000.0)))
        verdict = arbiter.evaluate(self._first_party_head(starved_seconds=5.0))
        assert verdict.disposition == VramDisposition.DEFER
        assert verdict.required_actuations == ()
        assert arbiter.first_party_context_defers == 1
        assert arbiter.admission_foreign_pressure_defers == 0
        assert arbiter.starvation_context_teardowns == 0

    def test_head_with_no_teardownable_contexts_is_not_a_first_party_defer(self) -> None:
        """A starved head with no teardownable contexts defers via the ordinary path, not as first-party."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=4000.0)))
        verdict = arbiter.evaluate(
            self._first_party_head(
                idle_contexts_teardownable=False,
                starved_seconds=_STARVATION_DIAGNOSTIC_SECONDS + 5.0,
            ),
        )
        assert verdict.disposition == VramDisposition.DEFER
        assert arbiter.first_party_context_defers == 0
        assert arbiter.starvation_diagnostics == 1


class TestFirstPartyTeardownGraceTiming:
    """The first-party context teardown fires after a short grace, distinct from the 60s diagnostic threshold.

    In the first-party geometry no alternative remedy can arrive (weight eviction cannot free a bare context; a
    busy sibling finishing does not surrender its context), so the escalation is evidence-based and quick: it
    waits only a short grace to ride out state churn, not the 60s diagnostic clock the genuinely-foreign
    starvation path keeps.
    """

    def _state(self) -> DeviceVramState:
        """A card whose device-free room cannot hold the candidate: the deficit is the idle contexts."""
        return _roomy_state(device_free_mb=4000.0)

    def _first_party_head(self, starved_seconds: float) -> VramRequest:
        """A head whose deficit is its own teardownable idle contexts, starved for the given duration."""
        return _preload(
            candidate_delta_mb=5000.0,
            is_head_of_queue=True,
            starved_seconds=starved_seconds,
            idle_contexts_teardownable=True,
        )

    def test_just_below_the_grace_defers_without_teardown(self) -> None:
        """A head starved just under the grace defers as first-party without a teardown actuation."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state()))
        verdict = arbiter.evaluate(self._first_party_head(_FIRST_PARTY_TEARDOWN_GRACE_SECONDS - 1.0))
        assert verdict.disposition == VramDisposition.DEFER
        assert ActuatorCommandKind.REDUCE_LIVE_CONTEXTS not in [c.kind for c in verdict.required_actuations]
        assert arbiter.starvation_context_teardowns == 0
        assert arbiter.first_party_context_defers == 1

    def test_just_past_the_grace_escalates_to_teardown(self) -> None:
        """A head starved just past the grace escalates to a REDUCE_LIVE_CONTEXTS teardown."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state()))
        verdict = arbiter.evaluate(self._first_party_head(_FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 1.0))
        assert verdict.disposition == VramDisposition.DEFER
        assert [c.kind for c in verdict.required_actuations] == [ActuatorCommandKind.REDUCE_LIVE_CONTEXTS]
        assert arbiter.starvation_context_teardowns == 1

    def test_teardown_fires_well_before_the_diagnostic_threshold(self) -> None:
        """Between the grace and the 60s diagnostic threshold the head tears down, not waits out the clock."""
        assert _FIRST_PARTY_TEARDOWN_GRACE_SECONDS < _STARVATION_DIAGNOSTIC_SECONDS
        midpoint = (_FIRST_PARTY_TEARDOWN_GRACE_SECONDS + _STARVATION_DIAGNOSTIC_SECONDS) / 2.0
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state()))
        verdict = arbiter.evaluate(self._first_party_head(midpoint))
        assert [c.kind for c in verdict.required_actuations] == [ActuatorCommandKind.REDUCE_LIVE_CONTEXTS]
        assert arbiter.starvation_context_teardowns == 1
        assert arbiter.starvation_diagnostics == 0

    def test_foreign_geometry_keeps_the_60s_diagnostic_timing(self) -> None:
        """A starved head with no first-party contexts still warns at 60s, not the short grace."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state()))
        past_grace = _preload(
            candidate_delta_mb=5000.0,
            is_head_of_queue=True,
            starved_seconds=_FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 1.0,
            idle_contexts_teardownable=False,
        )
        arbiter.evaluate(past_grace)
        assert arbiter.starvation_diagnostics == 0

        arbiter.begin_cycle(_snapshot(self._state()))
        past_diagnostic = _preload(
            candidate_delta_mb=5000.0,
            is_head_of_queue=True,
            starved_seconds=_STARVATION_DIAGNOSTIC_SECONDS + 1.0,
            idle_contexts_teardownable=False,
        )
        arbiter.evaluate(past_diagnostic)
        assert arbiter.starvation_diagnostics == 1


class TestFlagIndependentStarvationLivenessRegression:
    """The 4090 flag-off wedge: a weight-dominant head starved behind its own idle sibling contexts.

    A 24GB card with ``whole_card_exclusive_residency`` off held two idle sibling inference contexts (5036 MB,
    the worker's own) when a Flux fp8 head (16097 MB) reached the queue. The head fit the card once those
    contexts were torn down, but the deferral was misattributed and rerouted to the structural-wedge recovery
    supervisor, which destroyed the pool. The arbiter now escalates to a verified context teardown regardless of
    the config flag (the scheduler feeds ``idle_contexts_teardownable`` true on the emergency-liveness seam even
    with the flag off), and the head admits once the contexts exit.
    """

    _TOTAL_MB = 24074.0
    _BASELINE_MB = 3972.0
    _NOISE_MB = 1204.0
    _CANDIDATE_MB = 16097.0

    def _state(self, *, committed_mb: float) -> DeviceVramState:
        """The card with ``committed_mb`` of the worker's own load and a truthful free reading net of it."""
        return DeviceVramState(
            total_vram_mb=self._TOTAL_MB,
            baseline_mb=self._BASELINE_MB,
            committed_vram_mb=committed_mb,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
            noise_buffer_mb=self._NOISE_MB,
            device_free_mb=max(0.0, self._TOTAL_MB - self._BASELINE_MB - committed_mb),
        )

    def _head(self) -> VramRequest:
        """The Flux head re-asking its preload, starved past the threshold, its own idle contexts torn-down-able."""
        return _preload(
            job_label="Flux.1-Schnell",
            baseline="flux_1",
            candidate_delta_mb=self._CANDIDATE_MB,
            is_head_of_queue=True,
            starved_seconds=_STARVATION_DIAGNOSTIC_SECONDS + 51.0,
            idle_contexts_teardownable=True,
            has_reclaimable_idle_model=False,
            can_reduce_live_contexts=False,
        )

    def test_escalates_to_teardown_instead_of_rerouting_to_the_supervisor(self) -> None:
        """The head tears its own idle contexts down rather than being deferred and rerouted."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state(committed_mb=5036.0)))
        verdict = arbiter.evaluate(self._head())
        assert verdict.disposition == VramDisposition.DEFER
        assert [c.kind for c in verdict.required_actuations] == [ActuatorCommandKind.REDUCE_LIVE_CONTEXTS]
        assert arbiter.starvation_context_teardowns == 1
        # The teardown fires before the starvation diagnostic path is reached.
        assert arbiter.admission_foreign_pressure_defers == 0
        assert arbiter.starvation_diagnostics == 0

    def test_reask_admits_once_the_idle_contexts_exit(self) -> None:
        """After the torn-down contexts free device room, the head's re-ask fits the card."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state(committed_mb=5036.0)))
        assert arbiter.evaluate(self._head()).disposition == VramDisposition.DEFER
        # Both idle sibling contexts exited: the device-free reading recovered to a single retained context.
        arbiter.begin_cycle(_snapshot(self._state(committed_mb=2518.0)))
        assert arbiter.evaluate(self._head()).disposition == VramDisposition.FITS

    def test_reroute_remains_reachable_when_the_candidate_can_never_fit(self) -> None:
        """A candidate larger than an emptied card with no teardown target still DENIES: legitimate non-progress."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state(committed_mb=5036.0)))
        oversized = _preload(
            candidate_delta_mb=self._TOTAL_MB,  # cannot fit even a fully cleared card
            is_head_of_queue=True,
            starved_seconds=_STARVATION_DIAGNOSTIC_SECONDS + 51.0,
            idle_contexts_teardownable=False,
        )
        assert arbiter.evaluate(oversized).disposition == VramDisposition.DENY


class TestDisaggSampling:
    """The concurrent-sampling arithmetic, pinned with the honest measured co-residency figures."""

    _TOTAL_MB = 16375.0
    _SAMPLER_ONLY_MB = 6158.0
    _DECODE_SPIKE_MB = 2500.0
    _FULL_LANE_QUOTA_MB = 8192.0
    _OVERHEAD_MB = 1288.0
    _MARGINAL_MB = 300.0

    def _state(self, *, vae_lane_decode_spike_mb: float) -> DeviceVramState:
        """A 16375MB card with one sampler already in flight and the lane's decode spike charged."""
        return DeviceVramState(
            total_vram_mb=self._TOTAL_MB,
            baseline_mb=0.0,
            committed_vram_mb=0.0,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
            num_loaded_inference_processes=1,
            per_process_overhead_mb=self._OVERHEAD_MB,
            marginal_mb=self._MARGINAL_MB,
            vram_reserve_mb=0.0,
            vae_lane_decode_spike_mb=vae_lane_decode_spike_mb,
            active_sampling_peaks_total_mb=self._SAMPLER_ONLY_MB,
        )

    def _second_sample_request(self) -> VramRequest:
        """A second concurrent sampling of a same-size sampler (ledger already non-empty)."""
        return VramRequest(
            kind=VramRequestKind.DISAGG_SAMPLE,
            job_label="disagg_sample",
            baseline=None,
            device_index=0,
            sampling_peak_mb=self._SAMPLER_ONLY_MB,
            first_of_kind=False,
        )

    def test_first_of_kind_always_admits(self) -> None:
        """The first concurrent sampling admits on an empty ledger regardless of headroom."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state(vae_lane_decode_spike_mb=self._FULL_LANE_QUOTA_MB)))
        request = self._second_sample_request()
        first = VramRequest(**{**request.__dict__, "first_of_kind": True})
        verdict = arbiter.evaluate(first)
        assert verdict.disposition == VramDisposition.FITS

    def test_two_samplers_plus_bounded_decode_spike_admit(self) -> None:
        """Two 6158MB samplers plus the bounded 2500MB decode spike fit the 16375MB card."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state(vae_lane_decode_spike_mb=self._DECODE_SPIKE_MB)))
        verdict = arbiter.evaluate(self._second_sample_request())
        # headroom = 16375 - 1288 - 0 - 0 - 2500 = 12587; demand = 6158 + 6158 = 12316 <= 12587.
        assert verdict.disposition == VramDisposition.FITS

    def test_full_lane_quota_charge_denies_the_second_sampler(self) -> None:
        """Charging the full 8192MB lane quota instead denies the second sampler (the collapse tripwire)."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state(vae_lane_decode_spike_mb=self._FULL_LANE_QUOTA_MB)))
        verdict = arbiter.evaluate(self._second_sample_request())
        # headroom = 16375 - 1288 - 8192 = 6895; demand 12316 does not fit.
        assert verdict.disposition == VramDisposition.DEFER

    def test_missing_peak_admits(self) -> None:
        """An unsizable sampling peak admits rather than wedging on missing telemetry."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state(vae_lane_decode_spike_mb=self._FULL_LANE_QUOTA_MB)))
        request = VramRequest(**{**self._second_sample_request().__dict__, "sampling_peak_mb": None})
        verdict = arbiter.evaluate(request)
        assert verdict.disposition == VramDisposition.FITS


class TestStageDispatchNeverWithheld:
    """Encode and decode stage dispatches always proceed: the sampling gate is the pipeline's admission point.

    Gating the stages adds no admission control (an encode only leads to sampling if the concurrent-sampling
    gate admits the job) and every gating variant serialised the stage overlap the pipeline exists for.
    Decode in particular drains the pipeline: completing it releases the job's sampler hold, latents, and
    submit path, which is how memory pressure ends.
    """

    _TOTAL_MB = 16375.0
    _DECODE_SPIKE_MB = 2500.0

    def _state(self, *, device_free_mb: float | None) -> DeviceVramState:
        """A card whose device-free reading is tight enough that the admission identity would itself withhold."""
        return DeviceVramState(
            total_vram_mb=self._TOTAL_MB,
            baseline_mb=0.0,
            committed_vram_mb=0.0,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
            noise_buffer_mb=admission_noise_buffer_mb(self._TOTAL_MB),
            device_free_mb=device_free_mb,
        )

    def _stage_request(self, kind: VramRequestKind, *, candidate_delta_mb: float | None) -> VramRequest:
        """An encode or decode stage dispatch onto an already-resident process."""
        return VramRequest(
            kind=kind,
            job_label="disagg_stage",
            baseline=None,
            device_index=0,
            candidate_delta_mb=candidate_delta_mb,
        )

    def test_encode_fits_though_the_identity_would_withhold(self) -> None:
        """A tight device-free reading would deny under the identity, yet the encode stage still proceeds."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state(device_free_mb=100.0)))
        verdict = arbiter.evaluate(self._stage_request(VramRequestKind.DISAGG_ENCODE, candidate_delta_mb=None))
        assert verdict.disposition == VramDisposition.FITS
        assert verdict.admits is True
        # The admission identity is attached for observability and would itself have withheld here.
        assert verdict.measured.fits is False

    def test_decode_fits_even_with_no_device_free_room(self) -> None:
        """A decode proceeds even when the device-free reading has no room: draining ends the pressure."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state(device_free_mb=1.0)))
        verdict = arbiter.evaluate(
            self._stage_request(VramRequestKind.DISAGG_DECODE, candidate_delta_mb=self._DECODE_SPIKE_MB),
        )
        assert verdict.disposition == VramDisposition.FITS

    def test_unknown_reading_fits(self) -> None:
        """A card with no device-free reading admits the stage rather than wedging the pipeline."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state(device_free_mb=None)))
        verdict = arbiter.evaluate(self._stage_request(VramRequestKind.DISAGG_DECODE, candidate_delta_mb=99999.0))
        assert verdict.disposition == VramDisposition.FITS


class TestReclaimTripwire:
    """A preload over the device-free room defers with a reclaim command, then admits once the room frees.

    16375 MB card, 1878 baseline. With the first 6158 sampler resident and the lane holding its 2500 MB decode
    cache, the device-free reading is 5839 MB: a second 6158 sampler does not fit and must defer with a reclaim
    command, not perpetually defer while reclaimable memory exists. Once the lane releases its cache the
    reading recovers and the re-ask admits.
    """

    _TOTAL_MB = 16375.0
    _BASELINE_MB = 1878.0
    _SAMPLER_MB = 6158.0
    _LANE_CACHE_MB = 2500.0
    _LANE_PID = 9

    def _second_sampler_preload(self) -> VramRequest:
        return VramRequest(
            kind=VramRequestKind.PRELOAD,
            job_label="sampler_2",
            baseline="stable_diffusion_xl",
            device_index=0,
            target_process_id=1,
            candidate_delta_mb=self._SAMPLER_MB,
            is_head_of_queue=True,
            has_reclaimable_idle_model=True,
        )

    def _state(self, *, committed_mb: float, idle_process_ids: frozenset[int]) -> DeviceVramState:
        """A card whose device-free reading is the total net of baseline and the committed decomposition."""
        return DeviceVramState(
            total_vram_mb=self._TOTAL_MB,
            baseline_mb=self._BASELINE_MB,
            committed_vram_mb=committed_mb,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
            idle_process_ids=idle_process_ids,
            device_free_mb=max(0.0, self._TOTAL_MB - self._BASELINE_MB - committed_mb),
        )

    def test_second_sampler_defers_with_reclaim_then_admits_after_release(self) -> None:
        """The over-budget preload defers with a RELEASE_CACHE/EVICT command; releasing the cache admits the re-ask."""
        arbiter = VramArbiter()
        # First cycle: the first sampler is resident and the lane holds its decode cache (committed 8658).
        arbiter.begin_cycle(
            _snapshot(
                self._state(
                    committed_mb=self._SAMPLER_MB + self._LANE_CACHE_MB,
                    idle_process_ids=frozenset({self._LANE_PID}),
                ),
            ),
        )
        deferred = arbiter.evaluate(self._second_sampler_preload())
        assert deferred.disposition == VramDisposition.DEFER
        command_kinds = {command.kind for command in deferred.required_actuations}
        assert (
            ActuatorCommandKind.RELEASE_CACHE in command_kinds or ActuatorCommandKind.EVICT_IDLE_MODEL in command_kinds
        )
        assert any(
            command.kind == ActuatorCommandKind.RELEASE_CACHE and command.target_process_id == self._LANE_PID
            for command in deferred.required_actuations
        )

        # Next cycle: the lane released its 2500 MB cache, so the device-free reading recovered.
        arbiter.begin_cycle(_snapshot(self._state(committed_mb=self._SAMPLER_MB, idle_process_ids=frozenset())))
        readmitted = arbiter.evaluate(self._second_sampler_preload())
        assert readmitted.disposition == VramDisposition.FITS


def test_every_verdict_carries_a_populated_measured_verdict() -> None:
    """Every disposition attaches the measured admission identity the verdict was reasoned from."""
    arbiter = VramArbiter()
    arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=8000.0)))
    fits = arbiter.evaluate(_preload(candidate_delta_mb=1000.0))
    arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=4000.0)))
    deferred = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, is_head_of_queue=True))
    denied = arbiter.evaluate(_preload(candidate_delta_mb=30000.0))
    assert fits.disposition == VramDisposition.FITS
    assert deferred.disposition == VramDisposition.DEFER
    assert denied.disposition == VramDisposition.DENY
    for verdict in (fits, deferred, denied):
        assert verdict.measured is not None
        assert verdict.measured.available_known is True


def test_reason_line_renders_the_measured_arithmetic() -> None:
    """A verdict's reason renders the identity physically: candidate vs (device-free - reservations - noise)."""
    arbiter = VramArbiter()
    arbiter.begin_cycle(_snapshot(_roomy_state(device_free_mb=8000.0, planned_unmaterialized_mb=1000.0)))
    verdict = arbiter.evaluate(_preload(candidate_delta_mb=1000.0))
    assert "device-free" in verdict.reason
    assert "reservations" in verdict.reason
    assert "noise" in verdict.reason
