"""Dispatch-time residency reconciliation: the gate that prices a staged job's VRAM against the card.

These pin the contracts the dispatch gate adds. Admission is consulted at preload and at the second
concurrent sampler, but the moment an already-RAM-staged job's weights actually commit to VRAM crosses no
gate on its own, so a job materialising beside an idle sibling's resident weights can over-commit the card
faster than tick-paced reclaim reacts. The gate prices that materialisation through the arbiter's single
MONOLITHIC_DISPATCH identity: a FITS releases the dispatch, a conflict holds it (the job keeps its queue
position, never faulted) and routes idle-resident eviction through the one reclaim owner, protecting the
head's own slot. The held job is not reaped by the watchdogs that time the preloaded-to-inference-started
transition, because it stays pending with its model resident and never enters in-progress.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.message_dispatcher import (
    _MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS,
    DeadlockSnapshot,
)
from horde_worker_regen.process_management.ipc.messages import (
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.admission_identity import evaluate_admission
from horde_worker_regen.process_management.resources.vram_arbiter import (
    _FIRST_PARTY_TEARDOWN_GRACE_SECONDS,
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequest,
    VramRequestKind,
    VramVerdict,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _fitting_state() -> DeviceVramState:
    """A device state with ample measured free room, so a staged dispatch fits without reclaim."""
    return DeviceVramState(
        total_vram_mb=24000.0,
        baseline_mb=1000.0,
        committed_vram_mb=2000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=21000.0,
    )


def _over_committed_state() -> DeviceVramState:
    """A device state whose measured free room is exhausted (idle residents physically hold the card)."""
    return DeviceVramState(
        total_vram_mb=16000.0,
        baseline_mb=0.0,
        committed_vram_mb=16000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=200.0,
    )


def _install_cycle(scheduler, state: DeviceVramState) -> None:  # noqa: ANN001
    """Freeze a crafted arbiter cycle on the scheduler's arbiter."""
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
    scheduler._vram_arbiter = arbiter


async def _scheduler_with_idle_sibling():  # noqa: ANN202
    """A scheduler whose head model is staged on its slot beside an evictable idle resident sibling."""
    target = make_mock_process_info(0, model_name="model_a", state=HordeProcessState.PRELOADED_MODEL)
    sibling = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.WAITING_FOR_JOB)
    process_map = ProcessMap({0: target, 1: sibling})
    job_tracker = JobTracker()
    job = make_job_pop_response("model_a")
    await track_popped_job_async(job_tracker, job)
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(image_models_to_load=["model_a", "model_b"]),
        max_concurrent=2,
        max_inference=2,
    )
    return scheduler, job, target, sibling


async def _scheduler_without_sibling():  # noqa: ANN202
    """A scheduler with only the head's slot: an over-commit here has no idle resident to evict."""
    target = make_mock_process_info(0, model_name="model_a", state=HordeProcessState.PRELOADED_MODEL)
    process_map = ProcessMap({0: target})
    job_tracker = JobTracker()
    job = make_job_pop_response("model_a")
    await track_popped_job_async(job_tracker, job)
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(image_models_to_load=["model_a"]),
        max_concurrent=2,
        max_inference=2,
    )
    return scheduler, job, target


class TestGatePredicate:
    """The gate releases a fitting dispatch and holds a conflicting one."""

    async def test_fitting_dispatch_is_released(self) -> None:
        """A staged job that fits the card is not held and records no conflict."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        _install_cycle(scheduler, _fitting_state())

        held = scheduler._dispatch_residency_reconciliation_holds(job, target)

        assert held is False
        assert scheduler.latest_dispatch_reconciliation_holds() == 0
        assert scheduler.latest_dispatch_reconciliation_conflicts() == 0

    async def test_conflict_holds_and_routes_eviction_protecting_the_head(self) -> None:
        """An over-committing dispatch holds and evicts the idle resident through the reclaim owner."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        _install_cycle(scheduler, _over_committed_state())
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]

        held = scheduler._dispatch_residency_reconciliation_holds(job, target)

        assert held is True
        # The eviction ran through unload_models_from_vram anchored on the head's own slot (which the sweep
        # protects), so the idle sibling is evicted and the head's staged model is spared.
        scheduler.unload_models_from_vram.assert_called_once()
        assert scheduler.unload_models_from_vram.call_args.args[0] is target
        # The job is never faulted: it keeps its queue position.
        assert job in scheduler._job_tracker.jobs_pending_inference
        assert scheduler.latest_dispatch_reconciliation_holds() == 1
        assert scheduler.latest_dispatch_reconciliation_conflicts() == 1

    async def test_hold_releases_by_reclaim_after_verified_free(self) -> None:
        """A held dispatch whose eviction ran is released, on a later fitting pass, as reclaim-attributed."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]

        _install_cycle(scheduler, _over_committed_state())
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

        # The reclaimed room drops the committed floor under capacity, so the re-ask releases the dispatch.
        _install_cycle(scheduler, _fitting_state())
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False

        assert scheduler.latest_dispatch_reconciliation_released_by_reclaim() == 1
        assert scheduler.latest_dispatch_reconciliation_released_by_natural_free() == 0
        assert scheduler.latest_dispatch_reconciliation_hold_seconds() >= 0.0

    async def test_hold_releases_by_natural_free_when_no_eviction_was_emitted(self) -> None:
        """A held dispatch with nothing to evict is released as natural-free when the card recovers on its own."""
        scheduler, job, target = await _scheduler_without_sibling()

        _install_cycle(scheduler, _over_committed_state())
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

        _install_cycle(scheduler, _fitting_state())
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False

        assert scheduler.latest_dispatch_reconciliation_released_by_reclaim() == 0
        assert scheduler.latest_dispatch_reconciliation_released_by_natural_free() == 1


class _CapturingArbiter:
    """A stand-in arbiter that records the request it is asked to evaluate and returns a fixed verdict."""

    def __init__(self, verdict: VramVerdict) -> None:
        self._verdict = verdict
        self.last_request: VramRequest | None = None

    @property
    def has_cycle(self) -> bool:
        """Always report a frozen cycle so the gate never tries to build its own snapshot."""
        return True

    def evaluate(self, request: VramRequest) -> VramVerdict:
        """Record the request and return the fixed verdict."""
        self.last_request = request
        return self._verdict


class TestLineSkipDispatchHeadTruth:
    """A line-skip dispatch is not the true head of queue, so the gate must present is_head_of_queue=False.

    The dispatch gate stamped is_head_of_queue=True unconditionally, which handed a line-skipper the head's
    best-effort over-budget admit. The truth of whether a dispatch is the genuine head (``line_skip is None``)
    is plumbed through so a line-skipper forfeits that admit and the head keeps first claim on the card.
    """

    def _fits_verdict(self) -> VramVerdict:
        """A FITS verdict on the predictive path, so the gate releases and only the request truth is asserted."""
        measured = evaluate_admission(
            candidate_outstanding_mb=0.0,
            device_free_mb=21000.0,
            outstanding_reservations_mb=0.0,
            total_vram_mb=24000.0,
        )
        return VramVerdict(
            disposition=VramDisposition.FITS,
            request_kind=VramRequestKind.MONOLITHIC_DISPATCH,
            device_index=None,
            reason="test-fits",
            measured=measured,
        )

    async def test_true_head_presents_head_and_line_skip_presents_non_head(self) -> None:
        """The gate stamps is_head_of_queue from the caller: True for the real head, False for a line-skip."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        capture = _CapturingArbiter(self._fits_verdict())
        scheduler._vram_arbiter = capture  # type: ignore[assignment]

        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False
        assert capture.last_request is not None
        assert capture.last_request.is_head_of_queue is True

        assert scheduler._dispatch_residency_reconciliation_holds(job, target, is_head_of_queue=False) is False
        assert capture.last_request is not None
        assert capture.last_request.is_head_of_queue is False


class TestStarvedDispatchHeadSignals:
    """The gate feeds the arbiter the starvation age and the teardownable-context signal for the dispatch head.

    The starved-head reality admit and the context-teardown escalation both live in the arbiter, but they can
    only fire if this seam reports how long the head has starved and whether idle sibling contexts exist. This
    pins that wiring: a genuine head beside an idle sibling presents a nonzero ``starved_seconds`` and
    ``idle_contexts_teardownable=True``.
    """

    def _fits_verdict(self) -> VramVerdict:
        """A cold-path FITS so the gate releases and only the request signals are asserted."""
        measured = evaluate_admission(
            candidate_outstanding_mb=0.0,
            device_free_mb=21000.0,
            outstanding_reservations_mb=0.0,
            total_vram_mb=24000.0,
        )
        return VramVerdict(
            disposition=VramDisposition.FITS,
            request_kind=VramRequestKind.MONOLITHIC_DISPATCH,
            device_index=None,
            reason="test-fits",
            measured=measured,
        )

    async def test_gate_reports_starvation_age_and_teardownable_contexts(self) -> None:
        """A starved head beside an idle sibling presents the age and the teardownable-context signal."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        scheduler._head_starved_seconds = Mock(  # type: ignore[method-assign]
            return_value=_FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 5.0,
        )
        capture = _CapturingArbiter(self._fits_verdict())
        scheduler._vram_arbiter = capture  # type: ignore[assignment]

        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False
        assert capture.last_request is not None
        assert capture.last_request.kind is VramRequestKind.MONOLITHIC_DISPATCH
        assert capture.last_request.starved_seconds == _FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 5.0
        assert capture.last_request.idle_contexts_teardownable is True

    async def test_line_skip_dispatch_reports_no_teardownable_context(self) -> None:
        """A line-skip dispatch (not the head) never reports a teardownable context, so it cannot tear one down."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        scheduler._head_starved_seconds = Mock(  # type: ignore[method-assign]
            return_value=_FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 5.0,
        )
        capture = _CapturingArbiter(self._fits_verdict())
        scheduler._vram_arbiter = capture  # type: ignore[assignment]

        assert scheduler._dispatch_residency_reconciliation_holds(job, target, is_head_of_queue=False) is False
        assert capture.last_request is not None
        assert capture.last_request.idle_contexts_teardownable is False


class TestHeldDispatchSurvivesWatchdogs:
    """A held dispatch is not reaped by the clocks that time the preloaded-to-inference-started transition."""

    async def test_stale_model_map_expiry_spares_the_resident_dispatch_target(self) -> None:
        """The stale-entry expiry never touches a resident (LOADED_IN_VRAM) dispatch target; it is not LOADING.

        With the model genuinely resident in VRAM on its target, the dispatch is a no-op: it materialises
        nothing, so the gate releases it even over an over-committed card rather than pricing a load that never
        happens. The subject here is the stale-entry expiry, which reclaims only LOADING entries and must leave
        the resident model in place so the released dispatch can run.
        """
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]
        scheduler._horde_model_map.update_entry(
            "model_a",
            load_state=ModelLoadState.LOADED_IN_VRAM,
            process_id=target.process_id,
        )
        _install_cycle(scheduler, _over_committed_state())

        # A dispatch to an already-VRAM-resident idle model is released, not held: it adds no device footprint.
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False
        scheduler.unload_models_from_vram.assert_not_called()

        expired = scheduler._expire_stale_model_map_entries()

        assert "model_a" not in expired
        assert "model_a" in scheduler._horde_model_map.root
        # The job is still queued (never faulted) and never entered in-progress, so the lost-result reap and the
        # orphaned-in-progress reconciler have nothing to act on.
        assert job in scheduler._job_tracker.jobs_pending_inference
        assert job not in scheduler._job_tracker.jobs_in_progress

    async def test_fresh_hold_is_not_a_structural_wedge(self) -> None:
        """A just-formed hold is far below the structural-wedge horizon, so the recovery supervisor stays out."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]
        _install_cycle(scheduler, _over_committed_state())

        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

        # The held dispatch presents to the deadlock detector as an all-idle queue whose head model is resident;
        # that is a queue deadlock, but only a queue deadlock that persists past the horizon is a structural
        # wedge. A fresh hold (reclaim resolves within a few ticks) never reaches it.
        snapshot = DeadlockSnapshot(
            in_deadlock=False,
            in_queue_deadlock=True,
            deadlock_started_at=0.0,
            queue_deadlock_started_at=100.0,
            queue_deadlock_model="model_a",
            queue_deadlock_process_id=target.process_id,
        )
        assert snapshot.indicates_structural_wedge(now=100.0) is False
        assert snapshot.indicates_structural_wedge(now=100.0 + _MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS + 1.0) is True
