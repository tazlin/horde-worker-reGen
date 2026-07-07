"""Unit tests for the single VRAM arbiter's decision surface and observability counters."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.resources.admission_identity import (
    _ADMISSION_NOISE_BUFFER_MB,
    admission_noise_buffer_mb,
)
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
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


def _snapshot(state: DeviceVramState, *, device_index: int = 0) -> MeasuredVramSnapshot:
    """Wrap a single device state in a one-card snapshot."""
    return MeasuredVramSnapshot(devices={device_index: state})


def _roomy_state(**overrides: object) -> DeviceVramState:
    """A card with ample capacity and a small committed floor, so an ordinary candidate fits."""
    defaults: dict[str, object] = {
        "total_vram_mb": 24000.0,
        "baseline_mb": 1000.0,
        "committed_vram_mb": 2000.0,
        "planned_unmaterialized_mb": 0.0,
        "committed_is_stale": False,
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
    """The ledger-driven admission identity's dispositions and the escalation ladder."""

    def test_fits_when_demand_within_capacity(self) -> None:
        """A candidate whose demand is within capacity fits on the measured floor."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state()))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=6000.0))
        assert verdict.disposition == VramDisposition.FITS
        assert verdict.admits is True
        assert verdict.measured.used_measured_floor is True

    def test_defer_emits_ladder_commands_in_order(self) -> None:
        """A non-fitting demand defers with the ladder described in escalation order."""
        arbiter = VramArbiter()
        state = _roomy_state(
            committed_vram_mb=20000.0,
            idle_process_ids=frozenset({7, 3}),
        )
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
        state = _roomy_state(committed_vram_mb=20000.0, idle_process_ids=frozenset({2}))
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
        state = _roomy_state(committed_vram_mb=20000.0, idle_process_ids=frozenset({4, 9}))
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
            committed_vram_mb=20000.0,
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
        """A candidate larger than the whole capacity denies (no escalation could seat it)."""
        arbiter = VramArbiter()
        # Capacity is (24000 - 1000) - 512 = 22488; a candidate above that can never seat.
        arbiter.begin_cycle(_snapshot(_roomy_state()))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=30000.0))
        assert verdict.disposition == VramDisposition.DENY


class TestProportionalNoiseBuffer:
    """A device state's per-card noise buffer flows through the measured verdict it is priced against."""

    def test_verdict_uses_the_device_states_derived_buffer(self) -> None:
        """A 16375MB card assembled with the derived buffer prices capacity against 818.75, not the floor."""
        total = 16375.0
        state = _roomy_state(
            total_vram_mb=total,
            baseline_mb=1700.0,
            noise_buffer_mb=admission_noise_buffer_mb(total),
        )
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=1000.0))
        assert verdict.measured.noise_buffer_mb == pytest.approx(818.75)

    def test_derived_buffer_denies_a_demand_the_floor_would_admit(self) -> None:
        """The larger derived buffer flips an otherwise-fitting demand that the floor buffer would seat."""
        total, baseline = 16375.0, 1700.0
        floor_capacity = (total - baseline) - _ADMISSION_NOISE_BUFFER_MB
        committed = floor_capacity - 100.0
        floor_state = _roomy_state(total_vram_mb=total, baseline_mb=baseline, committed_vram_mb=committed)
        derived_state = _roomy_state(
            total_vram_mb=total,
            baseline_mb=baseline,
            committed_vram_mb=committed,
            noise_buffer_mb=admission_noise_buffer_mb(total),
        )
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(floor_state))
        assert arbiter.evaluate(_preload(candidate_delta_mb=0.0)).disposition == VramDisposition.FITS
        arbiter.begin_cycle(_snapshot(derived_state))
        assert arbiter.evaluate(_preload(candidate_delta_mb=0.0)).disposition == VramDisposition.DEFER


class TestStalenessAndColdStart:
    """Degraded modes relax to FITS and never deny."""

    def test_stale_ledger_relaxes_to_fits(self) -> None:
        """A stale committed ledger relaxes an over-demand to admit on the predictive path."""
        arbiter = VramArbiter()
        state = _roomy_state(committed_vram_mb=30000.0, committed_is_stale=True)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=6000.0))
        assert verdict.disposition == VramDisposition.FITS
        assert verdict.measured.used_measured_floor is False

    def test_cold_start_no_total_relaxes_to_fits(self) -> None:
        """An unknown device total (cold start) relaxes an over-demand to admit."""
        arbiter = VramArbiter()
        state = _roomy_state(total_vram_mb=None, committed_vram_mb=30000.0)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=6000.0))
        assert verdict.disposition == VramDisposition.FITS
        assert verdict.measured.used_measured_floor is False

    def test_missing_snapshot_relaxes_to_fits(self) -> None:
        """Evaluating before a cycle is installed relaxes to admit."""
        arbiter = VramArbiter()
        verdict = arbiter.evaluate(_preload())
        assert verdict.disposition == VramDisposition.FITS

    def test_missing_device_relaxes_to_fits(self) -> None:
        """A request for a card absent from the snapshot relaxes to admit."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(), device_index=1))
        verdict = arbiter.evaluate(_preload(device_index=0))
        assert verdict.disposition == VramDisposition.FITS


class TestReclaimStillPossibleAlwaysDefers:
    """While reclaim can still free space, a non-fitting demand never admits: it defers behind the ladder."""

    def test_ladder_in_progress_defers_an_oversized_candidate(self) -> None:
        """A non-fitting demand with a non-empty escalation ladder defers, whatever the device-free reading.

        Even a candidate that would physically fit the truthful device-free reading defers while any rung
        remains: the ladder can still relieve the over-commit, so the request re-asks next cycle after the
        caller runs the rung rather than admitting into a state reclaim could have cleared.
        """
        arbiter = VramArbiter()
        # device_free is roomy (would foreign-fit a 5000MB candidate) but an idle lane cache remains to release.
        state = _roomy_state(committed_vram_mb=20000.0, idle_process_ids=frozenset({6}), device_free_mb=8000.0)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, is_head_of_queue=True))
        assert verdict.disposition == VramDisposition.DEFER
        assert verdict.foreign_pressure_admit is False
        assert [c.kind for c in verdict.required_actuations] == [ActuatorCommandKind.RELEASE_CACHE]

    def test_saturated_verified_ladder_unfinished_defers_even_with_an_empty_arbiter_ladder(self) -> None:
        """A SATURATED card whose verified ladder is not yet proven unrelievable defers a non-fitting demand.

        The arbiter's own per-cycle ladder is empty (no idle lane, no reclaimable model), but the governor's
        verified reclaim ladder is still running (lane pauses, safety off-GPU are rungs the arbiter does not
        describe), so a candidate that would otherwise foreign-fit still defers until reclaim resolves.
        """
        arbiter = VramArbiter()
        state = _roomy_state(
            committed_vram_mb=20000.0,
            device_free_mb=8000.0,
            governor_state=GovernorState.SATURATED,
            reclaim_unresolved=False,
        )
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, is_head_of_queue=True))
        assert verdict.disposition == VramDisposition.DEFER
        assert verdict.foreign_pressure_admit is False


class TestExhaustedReclaimShortfall:
    """Once reclaim is exhausted, the verdict follows the shortfall's cause: own load, foreign fit, or deny."""

    def test_own_committed_load_over_capacity_defers(self) -> None:
        """When the worker's own committed load exceeds capacity after full reclaim, the head defers."""
        arbiter = VramArbiter()
        # capacity = (24000 - 1000) - 512 = 22488; committed 23000 alone already exceeds it.
        state = _roomy_state(committed_vram_mb=23000.0, device_free_mb=8000.0)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=100.0, is_head_of_queue=True))
        assert verdict.disposition == VramDisposition.DEFER
        assert verdict.foreign_pressure_admit is False
        assert "own committed load" in verdict.reason

    def test_foreign_shortfall_admits_into_reality_when_the_candidate_physically_fits(self) -> None:
        """The worker's own load fits capacity; the candidate physically fits device-free, so admit into reality.

        This is the one legitimate remnant of best-effort: the ledger says the card is over-committed only
        because foreign load consumes it, yet the truthful device-free reading has physical room for the
        candidate right now, so it admits (flagged for the heavy-head load grace) rather than deferring forever.
        """
        arbiter = VramArbiter()
        # own committed 20000 <= capacity 22488, candidate 5000 tips the ledger over; device-free has room.
        state = _roomy_state(committed_vram_mb=20000.0, device_free_mb=8000.0)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, is_head_of_queue=True))
        assert verdict.disposition == VramDisposition.FITS
        assert verdict.admits is True
        assert verdict.foreign_pressure_admit is True
        assert verdict.measured.fits is False

    def test_foreign_shortfall_defers_when_the_candidate_does_not_physically_fit(self) -> None:
        """Foreign pressure but the candidate does not fit device-free minus the buffer: defer, count it."""
        arbiter = VramArbiter()
        state = _roomy_state(committed_vram_mb=20000.0, device_free_mb=4000.0)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, is_head_of_queue=True))
        assert verdict.disposition == VramDisposition.DEFER
        assert verdict.foreign_pressure_admit is False
        assert arbiter.admission_foreign_pressure_defers == 1

    def test_foreign_shortfall_defers_when_device_free_unknown(self) -> None:
        """With no truthful device-free reading the candidate cannot be shown to physically fit, so it defers."""
        arbiter = VramArbiter()
        state = _roomy_state(committed_vram_mb=20000.0, device_free_mb=None)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, is_head_of_queue=True))
        assert verdict.disposition == VramDisposition.DEFER
        assert arbiter.admission_foreign_pressure_defers == 1

    def test_reserve_is_never_a_preload_denial_term(self) -> None:
        """The operator's ``vram_reserve_mb`` never contributes to a PRELOAD denial (flux-wedge guard)."""
        arbiter = VramArbiter()
        # A large reserve plus a candidate that exactly saturates capacity: the reserve must not tip it over.
        # capacity = (24000 - 1000) - 512 = 22488; committed 2000 + candidate 20488 == capacity, so it fits.
        state = _roomy_state(vram_reserve_mb=8192.0)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=20488.0, is_head_of_queue=True))
        assert verdict.disposition == VramDisposition.FITS


class TestStarvationDiagnostic:
    """A head deferred past the diagnostic horizon with reclaim exhausted warns and counts, never admits."""

    def _wedged_head(self, starved_seconds: float) -> VramRequest:
        """A head over budget on a card its own committed load holds, deferred for ``starved_seconds``."""
        return _preload(candidate_delta_mb=100.0, is_head_of_queue=True, starved_seconds=starved_seconds)

    def test_diagnostic_emitted_past_the_horizon_without_admitting(self) -> None:
        """Past the horizon with reclaim exhausted, the diagnostic counter advances but the head still defers."""
        arbiter = VramArbiter()
        state = _roomy_state(committed_vram_mb=23000.0)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(self._wedged_head(_STARVATION_DIAGNOSTIC_SECONDS + 5.0))
        assert verdict.disposition == VramDisposition.DEFER
        assert arbiter.starvation_diagnostics == 1

    def test_diagnostic_throttled_to_once_per_cycle_per_device(self) -> None:
        """Two evaluations within one frozen cycle advance the diagnostic counter at most once."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(committed_vram_mb=23000.0)))
        arbiter.evaluate(self._wedged_head(_STARVATION_DIAGNOSTIC_SECONDS + 5.0))
        arbiter.evaluate(self._wedged_head(_STARVATION_DIAGNOSTIC_SECONDS + 5.0))
        assert arbiter.starvation_diagnostics == 1

    def test_head_below_the_horizon_emits_no_diagnostic(self) -> None:
        """A head deferred below the diagnostic horizon defers silently (no counter movement)."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(committed_vram_mb=23000.0)))
        verdict = arbiter.evaluate(self._wedged_head(5.0))
        assert verdict.disposition == VramDisposition.DEFER
        assert arbiter.starvation_diagnostics == 0


class TestHeadOnlyOverBudgetAdmit:
    """The best-effort over-budget (foreign-pressure) admit is reserved for the true head of queue.

    On the db0 4090 wedge a non-head job took the over-budget admit and materialised the VRAM the head-of-queue
    job needed, starving the head while line-skippers consumed the card. The physical room the truthful
    device-free reading shows belongs to the head; a non-head request defers even when it would physically fit.
    """

    def test_head_admits_into_reality_when_candidate_physically_fits(self) -> None:
        """The true head still takes the over-budget admit when the card physically has room for it."""
        arbiter = VramArbiter()
        state = _roomy_state(committed_vram_mb=20000.0, device_free_mb=8000.0)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, is_head_of_queue=True))
        assert verdict.disposition == VramDisposition.FITS
        assert verdict.foreign_pressure_admit is True

    def test_non_head_is_denied_the_over_budget_admit_and_defers(self) -> None:
        """A non-head request that would physically fit defers instead of taking the head's room."""
        arbiter = VramArbiter()
        state = _roomy_state(committed_vram_mb=20000.0, device_free_mb=8000.0)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, is_head_of_queue=False))
        assert verdict.disposition == VramDisposition.DEFER
        assert verdict.foreign_pressure_admit is False
        assert "reserved for the head of queue" in verdict.reason
        assert arbiter.admission_foreign_pressure_defers == 1


class TestStarvationContextTeardown:
    """A head starved past the threshold whose deficit is held by idle sibling contexts escalates to teardown.

    Idle sibling inference contexts hold a bare CUDA baseline that weight eviction cannot reclaim (a context is
    freed only when its process exits). On the db0 flux wedge the built ladder had no rung that could free that
    baseline, so an exclusive head starved indefinitely. Past the starvation threshold the arbiter escalates to
    a REDUCE_LIVE_CONTEXTS actuation that tears the idle contexts down, then admits once the room verifies.
    """

    def _starved_head(self, **overrides: object) -> VramRequest:
        """An over-budget head starved past the diagnostic horizon with idle contexts to tear down."""
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
        # committed 20000 <= capacity 22488 (own load fits); candidate 5000 tips it over; no device-free room.
        state = _roomy_state(committed_vram_mb=20000.0, device_free_mb=4000.0)
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(self._starved_head())
        assert verdict.disposition == VramDisposition.DEFER
        assert [c.kind for c in verdict.required_actuations] == [ActuatorCommandKind.REDUCE_LIVE_CONTEXTS]
        assert verdict.foreign_pressure_admit is False
        assert arbiter.starvation_context_teardowns == 1

    def test_teardown_then_admits_after_the_contexts_are_reduced(self) -> None:
        """Once the torn-down contexts drop the committed floor, the head's re-ask admits (never force-admitted)."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(committed_vram_mb=20000.0, device_free_mb=4000.0)))
        assert arbiter.evaluate(self._starved_head()).disposition == VramDisposition.DEFER
        # The idle contexts exited, so the committed floor dropped well under capacity: the re-ask fits.
        arbiter.begin_cycle(_snapshot(_roomy_state(committed_vram_mb=12000.0, device_free_mb=10000.0)))
        readmitted = arbiter.evaluate(self._starved_head())
        assert readmitted.disposition == VramDisposition.FITS

    def test_below_threshold_does_not_tear_down_contexts(self) -> None:
        """A head starved below the escalation threshold defers without a teardown command."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(committed_vram_mb=20000.0, device_free_mb=4000.0)))
        verdict = arbiter.evaluate(self._starved_head(starved_seconds=5.0))
        assert verdict.disposition == VramDisposition.DEFER
        assert ActuatorCommandKind.REDUCE_LIVE_CONTEXTS not in [c.kind for c in verdict.required_actuations]
        assert arbiter.starvation_context_teardowns == 0

    def test_no_teardownable_contexts_no_escalation(self) -> None:
        """Without idle contexts to reclaim, a starved head defers via the ordinary shortfall path."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(_roomy_state(committed_vram_mb=20000.0, device_free_mb=4000.0)))
        verdict = arbiter.evaluate(self._starved_head(idle_contexts_teardownable=False))
        assert verdict.disposition == VramDisposition.DEFER
        assert verdict.required_actuations == ()
        assert arbiter.starvation_context_teardowns == 0

    def test_dispatch_gate_hold_never_tears_a_context_down(self) -> None:
        """A MONOLITHIC_DISPATCH request never escalates to context teardown, whatever its starvation age.

        The dispatch gate reconciles a staged job onto the card; it evicts idle residents but never collapses
        the context pool. Only the preload/whole-card path may tear contexts down.
        """
        arbiter = VramArbiter()
        state = _roomy_state(committed_vram_mb=20000.0, device_free_mb=4000.0)
        arbiter.begin_cycle(_snapshot(state))
        dispatch = VramRequest(
            kind=VramRequestKind.MONOLITHIC_DISPATCH,
            job_label="model_a",
            baseline="stable_diffusion_xl",
            device_index=0,
            candidate_delta_mb=5000.0,
            is_head_of_queue=True,
            starved_seconds=_STARVATION_DIAGNOSTIC_SECONDS + 5.0,
            idle_contexts_teardownable=True,
        )
        verdict = arbiter.evaluate(dispatch)
        assert verdict.disposition == VramDisposition.DEFER
        assert ActuatorCommandKind.REDUCE_LIVE_CONTEXTS not in [c.kind for c in verdict.required_actuations]
        assert arbiter.starvation_context_teardowns == 0


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

    def _state(self, *, committed_mb: float) -> DeviceVramState:
        """A card holding a committed floor, with the proportional noise buffer the scheduler assembles."""
        return DeviceVramState(
            total_vram_mb=self._TOTAL_MB,
            baseline_mb=0.0,
            committed_vram_mb=committed_mb,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
            noise_buffer_mb=admission_noise_buffer_mb(self._TOTAL_MB),
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

    def test_encode_fits_while_committed_exceeds_admission_capacity(self) -> None:
        """During sampling the committed floor overshoots the admission ceiling, yet the encode still fits.

        The overlap case: committed 15600 exceeds ``(total - baseline) - noise`` (roughly 15556 here) so the
        admission identity would deny, but the stage dispatch proceeds and the pipeline overlap is preserved.
        """
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state(committed_mb=15600.0)))
        verdict = arbiter.evaluate(self._stage_request(VramRequestKind.DISAGG_ENCODE, candidate_delta_mb=None))
        assert verdict.disposition == VramDisposition.FITS
        assert verdict.admits is True
        # The admission identity is attached for observability and would itself have denied here.
        assert verdict.measured.fits is False

    def test_decode_fits_even_beyond_the_physical_total(self) -> None:
        """A decode proceeds even when committed-plus-spike tops the physical total: draining ends pressure."""
        arbiter = VramArbiter()
        arbiter.begin_cycle(_snapshot(self._state(committed_mb=self._TOTAL_MB - 1.0)))
        verdict = arbiter.evaluate(
            self._stage_request(VramRequestKind.DISAGG_DECODE, candidate_delta_mb=self._DECODE_SPIKE_MB),
        )
        assert verdict.disposition == VramDisposition.FITS

    def test_unknown_total_fits(self) -> None:
        """A cold-start card with no known total admits the stage on the predictive path."""
        arbiter = VramArbiter()
        state = DeviceVramState(
            total_vram_mb=None,
            baseline_mb=0.0,
            committed_vram_mb=0.0,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
        )
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(self._stage_request(VramRequestKind.DISAGG_DECODE, candidate_delta_mb=99999.0))
        assert verdict.disposition == VramDisposition.FITS


class TestTwoSamplerReclaimTripwire:
    """The proven two-sampler figures: the second preload defers, reclaims, then the re-ask admits.

    16375 MB total, 1878 baseline, 512 noise -> capacity 13985. With the first 6158 sampler resident and
    the lane holding its 2500 MB decode cache (committed 8658), the second 6158 preload demands 14816: over
    capacity by 831 MB. It must defer with a reclaim command, not perpetually defer while reclaimable memory
    exists; once the lane releases its cache the committed floor drops and the re-ask admits.
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

    def test_second_sampler_defers_with_reclaim_then_admits_after_release(self) -> None:
        """The 831 MB gap defers with a RELEASE_CACHE/EVICT command; releasing the lane cache admits the re-ask."""
        arbiter = VramArbiter()
        # First cycle: the first sampler is resident and the lane holds its decode cache (committed 8658).
        pressured = DeviceVramState(
            total_vram_mb=self._TOTAL_MB,
            baseline_mb=self._BASELINE_MB,
            committed_vram_mb=self._SAMPLER_MB + self._LANE_CACHE_MB,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
            idle_process_ids=frozenset({self._LANE_PID}),
        )
        arbiter.begin_cycle(_snapshot(pressured))
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

        # Next cycle: the lane released its 2500 MB cache, so the committed floor drops to 6158.
        relieved = DeviceVramState(
            total_vram_mb=self._TOTAL_MB,
            baseline_mb=self._BASELINE_MB,
            committed_vram_mb=self._SAMPLER_MB,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
        )
        arbiter.begin_cycle(_snapshot(relieved))
        readmitted = arbiter.evaluate(self._second_sampler_preload())
        assert readmitted.disposition == VramDisposition.FITS


class TestStalenessNeverDeniesProperty:
    """Staleness drops the measured floor but never denies on its own: only a planned overlay can deny."""

    def test_stale_with_no_planned_demand_never_denies(self) -> None:
        """With no planned demand a stale ledger always admits a fitting candidate, whatever the stale floor.

        The measured committed floor is child telemetry and is dropped when stale, so however large it reads it
        can never itself flip the verdict; a candidate within capacity therefore always fits.
        """
        arbiter = VramArbiter()
        for committed in (0.0, 20000.0, 99999.0):
            for candidate in (0.0, 6000.0):
                state = DeviceVramState(
                    total_vram_mb=24000.0,
                    baseline_mb=1000.0,
                    committed_vram_mb=committed,
                    planned_unmaterialized_mb=0.0,
                    committed_is_stale=True,
                )
                arbiter.begin_cycle(_snapshot(state))
                verdict = arbiter.evaluate(_preload(candidate_delta_mb=candidate, is_head_of_queue=True))
                assert verdict.disposition == VramDisposition.FITS
                assert verdict.measured.used_measured_floor is False

    def test_cold_start_no_total_always_fits(self) -> None:
        """A cold start with no known total relaxes fully: even an oversized candidate admits, nothing knowable."""
        arbiter = VramArbiter()
        for committed in (0.0, 20000.0, 99999.0):
            for candidate in (0.0, 6000.0, 40000.0):
                state = DeviceVramState(
                    total_vram_mb=None,
                    baseline_mb=1000.0,
                    committed_vram_mb=committed,
                    planned_unmaterialized_mb=0.0,
                    committed_is_stale=False,
                )
                arbiter.begin_cycle(_snapshot(state))
                verdict = arbiter.evaluate(_preload(candidate_delta_mb=candidate, is_head_of_queue=True))
                assert verdict.disposition == VramDisposition.FITS
                assert verdict.measured.used_measured_floor is False

    def test_stale_planned_overlay_denies_stacked_admissions(self) -> None:
        """The startup storm at the arbiter: stale reports but stacked planned admissions still deny a preload."""
        arbiter = VramArbiter()
        state = DeviceVramState(
            total_vram_mb=16375.0,
            baseline_mb=1700.0,
            committed_vram_mb=99999.0,
            planned_unmaterialized_mb=12316.0,
            committed_is_stale=True,
            noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
        )
        arbiter.begin_cycle(_snapshot(state))
        verdict = arbiter.evaluate(_preload(candidate_delta_mb=6158.0))
        assert verdict.measured.used_measured_floor is False
        assert verdict.measured.fits is False
        assert verdict.disposition != VramDisposition.FITS

    def test_three_sequential_stale_preloads_admit_two_then_deny(self) -> None:
        """Startup storm: with stale reports the planned overlay accumulates, admitting two preloads then denying.

        Each cycle re-freezes the snapshot with the planned overlay grown by the prior admit's 6158 MB anchor.
        The measured floor is dropped (stale), the baseline is still 0 (no child has reported), yet the third
        stacked candidate pushes planned + candidate past the 16375 MB card's capacity and is denied.
        """
        arbiter = VramArbiter()
        candidate = 6158.0
        dispositions: list[VramDisposition] = []
        planned_mb = 0.0
        for _cycle in range(3):
            state = DeviceVramState(
                total_vram_mb=16375.0,
                baseline_mb=0.0,
                committed_vram_mb=0.0,
                planned_unmaterialized_mb=planned_mb,
                committed_is_stale=True,
                noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
            )
            arbiter.begin_cycle(_snapshot(state))
            verdict = arbiter.evaluate(_preload(candidate_delta_mb=candidate))
            dispositions.append(verdict.disposition)
            if verdict.disposition == VramDisposition.FITS:
                planned_mb += candidate
        assert dispositions[0] == VramDisposition.FITS
        assert dispositions[1] == VramDisposition.FITS
        assert dispositions[2] != VramDisposition.FITS


def test_every_verdict_carries_a_populated_measured_verdict() -> None:
    """Every disposition attaches the measured admission identity the verdict was reasoned from."""
    arbiter = VramArbiter()
    arbiter.begin_cycle(_snapshot(_roomy_state(committed_vram_mb=20000.0, device_free_mb=8000.0)))
    fits = arbiter.evaluate(_preload(candidate_delta_mb=1000.0))
    foreign_fit = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, is_head_of_queue=True))
    arbiter.begin_cycle(_snapshot(_roomy_state(committed_vram_mb=20000.0, device_free_mb=4000.0)))
    deferred = arbiter.evaluate(_preload(candidate_delta_mb=5000.0, is_head_of_queue=True))
    denied = arbiter.evaluate(_preload(candidate_delta_mb=30000.0))
    assert fits.disposition == VramDisposition.FITS
    assert foreign_fit.foreign_pressure_admit is True
    assert deferred.disposition == VramDisposition.DEFER
    assert denied.disposition == VramDisposition.DENY
    for verdict in (fits, foreign_fit, deferred, denied):
        assert verdict.measured is not None
        assert verdict.measured.used_measured_floor is True


def test_unpriceable_candidate_is_charged_nothing() -> None:
    """A None candidate delta is charged zero, so it never denies on an unpriceable cost."""
    arbiter = VramArbiter()
    arbiter.begin_cycle(_snapshot(_roomy_state(committed_vram_mb=20000.0)))
    verdict = arbiter.evaluate(_preload(candidate_delta_mb=None))
    assert verdict.disposition == VramDisposition.FITS
    assert verdict.measured.candidate_delta_mb == pytest.approx(0.0)
