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

from collections.abc import Callable
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.message_dispatcher import (
    _MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS,
    DeadlockSnapshot,
)
from horde_worker_regen.process_management.ipc.messages import (
    HeldComponentSnapshot,
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.admission_identity import evaluate_admission
from horde_worker_regen.process_management.resources.run_metrics import DecisionVerdict
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
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    _ESTABLISH_WINDOW_SECONDS,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.ledgers.head_admission import HEAD_PROTECTION_MAX_STARVE_SECONDS
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
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
        assert scheduler.dispatch_holds.holds == 0
        assert scheduler.dispatch_holds.conflicts == 0

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
        assert scheduler.dispatch_holds.holds == 1
        assert scheduler.dispatch_holds.conflicts == 1

    async def test_hold_releases_by_reclaim_after_verified_free(self) -> None:
        """A held dispatch whose eviction ran is released, on a later fitting pass, as reclaim-attributed."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]

        _install_cycle(scheduler, _over_committed_state())
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

        # The reclaimed room drops the committed floor under capacity, so the re-ask releases the dispatch.
        _install_cycle(scheduler, _fitting_state())
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False

        assert scheduler.dispatch_holds.released_by_reclaim == 1
        assert scheduler.dispatch_holds.released_by_natural_free == 0
        assert scheduler.dispatch_holds.hold_seconds >= 0.0

    async def test_hold_releases_by_natural_free_when_no_eviction_was_emitted(self) -> None:
        """A held dispatch with nothing to evict is released as natural-free when the card recovers on its own."""
        scheduler, job, target = await _scheduler_without_sibling()

        _install_cycle(scheduler, _over_committed_state())
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

        _install_cycle(scheduler, _fitting_state())
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False

        assert scheduler.dispatch_holds.released_by_reclaim == 0
        assert scheduler.dispatch_holds.released_by_natural_free == 1


_SMALL_CARD_INDEX = 1
"""The card holding the head's copy in the reroute tests."""


def _card_below_any_candidate() -> DeviceVramState:
    """A card whose foreign floor leaves an achievable ceiling below any candidate, so the arbiter DENIES there."""
    return DeviceVramState(
        total_vram_mb=16000.0,
        baseline_mb=0.0,
        committed_vram_mb=0.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=500.0,
        foreign_floor_mb=15500.0,
    )


def _install_two_card_cycle(scheduler, *, large: DeviceVramState, small: DeviceVramState) -> None:  # noqa: ANN001
    """Freeze an arbiter cycle with a large card 0 and the small card holding the head's copy."""
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: large, _SMALL_CARD_INDEX: small}))
    scheduler._vram_arbiter = arbiter


async def _scheduler_with_head_copy_on_the_small_card(  # noqa: ANN202
    *,
    queued_same_model: int = 0,
    idle_tenant_on_small_card: bool = False,
):
    """A two-card scheduler whose head model is held in RAM by name on a small-card lane.

    ``queued_same_model`` queues that many further jobs for the head's model behind it. With
    ``idle_tenant_on_small_card`` a second small-card lane idles with device-warm components, the tenancy the
    gate's standing-hold reclaim would ask for.
    """
    target = make_mock_process_info(
        0,
        model_name="model_a",
        state=HordeProcessState.PRELOADED_MODEL,
        device_index=_SMALL_CARD_INDEX,
    )
    large_card_lane = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    processes = {0: target, 1: large_card_lane}
    if idle_tenant_on_small_card:
        tenant = make_mock_process_info(
            2,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            device_index=_SMALL_CARD_INDEX,
        )
        tenant.held_components = [HeldComponentSnapshot(kind="unet", identity="idle-tenant", approx_ram_mb=1500.0)]
        processes[2] = tenant
    process_map = ProcessMap(processes)
    job_tracker = JobTracker()
    job = make_job_pop_response("model_a")
    await track_popped_job_async(job_tracker, job)
    for _ in range(queued_same_model):
        await track_popped_job_async(job_tracker, make_job_pop_response("model_a"))
    bridge_data = make_mock_bridge_data(image_models_to_load=["model_a"])
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        max_concurrent=1,
        max_inference=2,
        card_runtimes=make_test_card_runtimes(device_indices=(0, _SMALL_CARD_INDEX), config=bridge_data),
    )
    scheduler._horde_model_map.update_entry(
        "model_a",
        load_state=ModelLoadState.LOADED_IN_RAM,
        process_id=target.process_id,
    )
    return scheduler, job, target


def _sent_ram_unload(process_info: HordeProcessInfo) -> bool:
    """Whether the parent sent ``process_info`` an unload-from-RAM control message."""
    return any(
        isinstance(message := call.args[0], HordeControlMessage)
        and message.control_flag is HordeControlFlag.UNLOAD_MODELS_FROM_RAM
        for call in process_info.pipe_connection.send.call_args_list
    )


class TestCeilingDenyWithAnotherCardRetiresTheCopy:
    """A DENY on the card holding the head's copy, with another card able to seat it, retires that copy.

    The copy pins the head to its card. Dispatch places a job on the lane holding its model by name, and only
    the preload pass places a job across cards. A DENY is the same every pass, so a hold on that card cannot end
    in a fit.
    """

    async def test_a_deny_with_another_card_able_to_seat_retires_the_copy_and_releases_the_hold(self) -> None:
        """The copy is unloaded from RAM, the lane no longer records the model, and the hold releases as a reroute.

        The hold has stood past the stall horizon and an idle tenant shares the denied card, so the standing-hold
        reclaim would evict it on an ordinary hold. The reroute leaves the card, so nothing on it is reclaimed, and
        the pass records the release alone.
        """
        scheduler, job, target = await _scheduler_with_head_copy_on_the_small_card(idle_tenant_on_small_card=True)
        _install_two_card_cycle(scheduler, large=_fitting_state(), small=_card_below_any_candidate())
        scheduler._dispatch_hold_is_standing = Mock(return_value=True)  # type: ignore[method-assign]
        scheduler.unload_idle_model = Mock(return_value=True)  # type: ignore[method-assign]
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]
        execute_actuations = Mock(wraps=scheduler._executor.execute_actuations)
        scheduler._executor.execute_actuations = execute_actuations  # type: ignore[method-assign]
        decision_sink = Mock(return_value=None)
        scheduler._decision_sink = decision_sink

        held = scheduler._dispatch_residency_reconciliation_holds(job, target)

        assert held is True, "the head dispatched on the card where the arbiter denied it"
        assert _sent_ram_unload(target), "the copy on the denied card was not unloaded from RAM"
        assert target.loaded_horde_model_name is None, "the lane still records the head's model after the retire"
        holds = scheduler.dispatch_holds
        assert holds.released_by_reroute == 1
        assert holds.held_seconds(str(job.id_)) is None, "the hold still stands after the reroute"
        assert job in scheduler._job_tracker.jobs_pending_inference, "the rerouted head left the queue"
        scheduler.unload_idle_model.assert_not_called()
        scheduler.unload_models_from_vram.assert_not_called()
        issued = [command for call in execute_actuations.call_args_list for command in call.args[0]]
        assert not issued, f"the reroute tick issued reclaim commands {issued} on the card the head is leaving"
        records = [call.kwargs for call in decision_sink.call_args_list if call.kwargs["subject"] == str(job.id_)]
        assert [record["verdict"] for record in records] == [DecisionVerdict.NO_OP], (
            f"the reroute tick recorded {records}, not the reroute release alone"
        )
        assert records[0]["reason"] == "dispatch_rerouted"

    async def test_a_defer_keeps_the_hold_and_the_copy(self) -> None:
        """A card short of room right now is held as before, and the parent does not unload the copy from RAM."""
        scheduler, job, target = await _scheduler_with_head_copy_on_the_small_card()
        _install_two_card_cycle(scheduler, large=_fitting_state(), small=_over_committed_state())

        held = scheduler._dispatch_residency_reconciliation_holds(job, target)

        assert held is True
        assert not _sent_ram_unload(target)
        assert target.loaded_horde_model_name == "model_a"
        assert scheduler.dispatch_holds.held_seconds(str(job.id_)) is not None
        assert scheduler.dispatch_holds.released_by_reroute == 0

    async def test_a_deny_with_no_other_card_able_to_seat_keeps_the_hold_and_the_copy(self) -> None:
        """With every card below the candidate there is no card to reroute to, so the hold and the copy stand."""
        scheduler, job, target = await _scheduler_with_head_copy_on_the_small_card()
        _install_two_card_cycle(scheduler, large=_card_below_any_candidate(), small=_card_below_any_candidate())

        held = scheduler._dispatch_residency_reconciliation_holds(job, target)

        assert held is True
        assert not _sent_ram_unload(target)
        assert target.loaded_horde_model_name == "model_a"
        assert scheduler.dispatch_holds.held_seconds(str(job.id_)) is not None
        assert scheduler.dispatch_holds.released_by_reroute == 0

    async def test_a_further_queued_job_of_the_model_does_not_keep_the_copy(self) -> None:
        """With a second job for the model queued, the copy is still retired and the hold released as a reroute.

        The second job would be pinned to the same copy and denied on the same card, so keeping the copy for it
        would hold both jobs where neither can run.
        """
        scheduler, job, target = await _scheduler_with_head_copy_on_the_small_card(queued_same_model=1)
        _install_two_card_cycle(scheduler, large=_fitting_state(), small=_card_below_any_candidate())

        held = scheduler._dispatch_residency_reconciliation_holds(job, target)

        assert held is True
        assert _sent_ram_unload(target), "the copy was kept for the second job of the model"
        assert target.loaded_horde_model_name is None
        assert scheduler.dispatch_holds.released_by_reroute == 1
        assert len(scheduler._job_tracker.jobs_pending_inference) == 2, "a queued job of the model left the queue"


def _retire_through_the_reroute(scheduler: InferenceScheduler, job: ImageGenerateJobPopResponse) -> None:
    """Price ``job`` at the dispatch gate under a DENY on its copy's card that the large card can seat."""
    _install_two_card_cycle(scheduler, large=_fitting_state(), small=_card_below_any_candidate())
    target = scheduler.resident_process_for_job(job)
    assert target is not None
    scheduler._dispatch_residency_reconciliation_holds(job, target)


def _retire_through_the_give_up(scheduler: InferenceScheduler, job: ImageGenerateJobPopResponse) -> None:
    """Retire the copy that pins ``job`` as save-our-ship gives the job up."""
    scheduler.retire_copy_pinning_given_up_job(job)


def _lane_runs_a_job(process_info: HordeProcessInfo) -> None:
    """Report the lane as starting a job."""
    process_info.last_process_state = HordeProcessState.INFERENCE_STARTING


def _lane_is_loading(process_info: HordeProcessInfo) -> None:
    """Report the lane as loading a model."""
    process_info.last_process_state = HordeProcessState.PRELOADING_MODEL


def _ram_unload_sent(process_info: HordeProcessInfo) -> None:
    """Record an unload from RAM as the lane's last control message."""
    process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM


def _ram_unload_recent(process_info: HordeProcessInfo) -> None:
    """Mark the lane as recently unloaded from RAM."""
    process_info.recently_unloaded_from_ram = True


class TestPinningCopyKeepRules:
    """The dispatch gate's reroute and the save-our-ship give-up retire a pinning copy under one set of rules."""

    @pytest.mark.parametrize(
        "retire",
        [_retire_through_the_reroute, _retire_through_the_give_up],
        ids=["reroute", "give_up"],
    )
    @pytest.mark.parametrize(
        ("arrange_lane", "retired"),
        [
            (None, True),
            (_lane_runs_a_job, False),
            (_lane_is_loading, False),
            (_ram_unload_sent, False),
            (_ram_unload_recent, False),
        ],
        ids=["idle_copy", "lane_runs_a_job", "lane_is_loading", "ram_unload_sent", "ram_unload_recent"],
    )
    async def test_the_copy_is_unloaded_from_ram_unless_a_keep_rule_holds(
        self,
        retire: Callable[[InferenceScheduler, ImageGenerateJobPopResponse], None],
        arrange_lane: Callable[[HordeProcessInfo], None] | None,
        retired: bool,
    ) -> None:
        """An idle lane's copy is unloaded from RAM, and a lane that runs a job, loads or is unloading keeps it.

        A further queued job of the same model does not keep the copy for either caller, so every case queues one.
        """
        scheduler, job, target = await _scheduler_with_head_copy_on_the_small_card(queued_same_model=1)
        if arrange_lane is not None:
            arrange_lane(target)
        target.pipe_connection.send.reset_mock()

        retire(scheduler, job)

        assert _sent_ram_unload(target) is retired, (
            f"the parent {'did not send' if retired else 'sent'} the copy's lane an unload from RAM"
        )
        assert (target.loaded_horde_model_name is None) is retired


def _starve_head(scheduler: InferenceScheduler, job: ImageGenerateJobPopResponse, *, seconds: float) -> None:
    """Pin the clock and mark ``job`` the starved head for exactly ``seconds`` on the head-admission ledger."""
    scheduler._clock = lambda: 10_000.0  # type: ignore[method-assign]
    scheduler.head_admission.starvation_job_id = str(job.id_)
    scheduler.head_admission.starvation_since = 10_000.0 - seconds


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

    def measured_deficit_mb(self, request: VramRequest) -> float | None:
        """A fixed verdict prices nothing, so the measured-frame teardown widening has no deficit to read."""
        return None

    def device_state(self, device_index: int | None) -> None:
        """A fixed verdict carries no frozen card state for the snapshot to read."""
        return


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


class TestHeadProtectionIsBounded:
    """Reserving card room for the head is worth its cost only while the head converges on a dispatch.

    Head protection withholds room from the jobs behind the head so the head, not a line-skipper, takes the
    next opportunity. A head whose own admission keeps declining never takes it, and the reservation then
    holds an idle card against siblings that fit for as long as the queue lasts: the worker serves nothing at
    all. Past a bound the protection is released and the head keeps its queue position.
    """

    async def test_a_briefly_parked_head_still_reserves_its_room(self) -> None:
        """Within the window the head is priced as outstanding, so a line-skipper is measured against it."""
        scheduler, job, _target, _sibling = await _scheduler_with_idle_sibling()
        _install_cycle(scheduler, _fitting_state())
        scheduler._head_starved_seconds = Mock(return_value=5.0)  # type: ignore[method-assign]

        assert scheduler._displaced_head_outstanding_mb(job, device_index=None) is not None

    async def test_a_head_parked_past_the_bound_stops_reserving_room(self) -> None:
        """Past the window the head prices as no demand, so a fitting sibling is admitted rather than held."""
        scheduler, job, _target, _sibling = await _scheduler_with_idle_sibling()
        _install_cycle(scheduler, _fitting_state())
        scheduler._head_starved_seconds = Mock(  # type: ignore[method-assign]
            return_value=HEAD_PROTECTION_MAX_STARVE_SECONDS + 1.0,
        )

        assert scheduler._displaced_head_outstanding_mb(job, device_index=None) is None


class TestHeadOnABusyLaneStopsReservingRoom:
    """A head whose lanes are all running a job cannot use room before one frees.

    The lane it waits on returns its own sampler's memory when it frees, so holding room for that head only
    idles the line-skipper's lane. A head that could take the room now keeps it.
    """

    async def _parked_head(self):  # noqa: ANN202
        """A scheduler whose head is briefly parked, so only the lane's state can release head protection."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        _install_cycle(scheduler, _fitting_state())
        scheduler._head_starved_seconds = Mock(return_value=5.0)  # type: ignore[method-assign]
        return scheduler, job, target

    @staticmethod
    def _run_job_on(lane) -> None:  # noqa: ANN001
        """Give ``lane`` an in-flight job for its own model, sampling."""
        lane.record_inference_ownership(make_job_pop_response(lane.loaded_horde_model_name), attempt_ordinal=1)
        lane.last_process_state = HordeProcessState.INFERENCE_STARTING

    async def test_a_head_whose_lane_runs_a_job_prices_no_demand(self) -> None:
        """The head's only lane is sampling, so the line-skipper is not measured against the head."""
        scheduler, job, target = await self._parked_head()
        assert scheduler._displaced_head_outstanding_mb(job, device_index=None) is not None

        self._run_job_on(target)

        assert scheduler._displaced_head_outstanding_mb(job, device_index=None) is None

    async def test_a_lane_preloading_the_head_keeps_its_protection(self) -> None:
        """A lane mid-preload of the head's model runs no job; the head dispatches the moment it lands."""
        scheduler, job, target = await self._parked_head()
        target.last_process_state = HordeProcessState.PRELOADING_MODEL

        assert scheduler._displaced_head_outstanding_mb(job, device_index=None) is not None

    async def test_a_busy_lane_on_another_card_keeps_the_protection(self) -> None:
        """On a multi-card host a copy of the head may be funded on the skipper's card, so its room stays held."""
        scheduler, job, target = await self._parked_head()
        self._run_job_on(target)

        assert scheduler._displaced_head_outstanding_mb(job, device_index=1) is not None


class TestGovernorDeferredHeadStopsReservingRoom:
    """A head whose whole-card establishment a churn governor defers is not asking for the card.

    The governor brakes how fast the card may be rotated and says so: normal scheduling continues around the
    deferred head, which takes no establishment for the length of the deferral. Charging its whole-card demand
    against the ready work behind it therefore reserves the card for a claim nobody is making, and the card
    sits empty while smaller jobs that fit are turned away. Once the deferral clears, or its record outlives the
    rate limiter's window, the head's normal charge applies again.
    """

    async def _deferrable_head(self):  # noqa: ANN202
        """A scheduler whose head is briefly parked, so only the deferral can release head protection."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        _install_cycle(scheduler, _fitting_state())
        scheduler._head_starved_seconds = Mock(return_value=5.0)  # type: ignore[method-assign]
        return scheduler, job, target

    async def test_a_deferred_head_prices_no_demand(self) -> None:
        """With the deferral in force the head charges nothing, so a fitting line-skipper is not measured."""
        scheduler, job, _target = await self._deferrable_head()
        assert scheduler._displaced_head_outstanding_mb(job, device_index=None) is not None

        scheduler._whole_card_ledger.note_governor_defer(None, model=job.model, now=scheduler._clock())

        assert scheduler._displaced_head_outstanding_mb(job, device_index=None) is None

    async def test_a_deferral_of_another_model_leaves_this_head_protected(self) -> None:
        """The release is keyed on the deferred model, so an unrelated deferral changes nothing."""
        scheduler, job, _target = await self._deferrable_head()

        scheduler._whole_card_ledger.note_governor_defer(None, model="some_other_model", now=scheduler._clock())

        assert scheduler._displaced_head_outstanding_mb(job, device_index=None) is not None

    async def test_a_record_past_the_establish_window_restores_the_normal_charge(self) -> None:
        """A deferral record cannot outlast the limiter's window, so past it the head reserves room again."""
        now = [1_000.0]
        scheduler, job, _target = await self._deferrable_head()
        scheduler._clock = lambda: now[0]  # type: ignore[method-assign]
        scheduler._whole_card_ledger.note_governor_defer(None, model=job.model, now=now[0])
        assert scheduler._displaced_head_outstanding_mb(job, device_index=None) is None

        now[0] += _ESTABLISH_WINDOW_SECONDS

        assert scheduler._displaced_head_outstanding_mb(job, device_index=None) is not None

    async def test_a_fitting_line_skip_dispatches_while_the_head_is_deferred(self) -> None:
        """The end-to-end contract: the same fitting candidate that is held for the head runs once it stands down.

        The control half is the point of the pairing. Head protection must still hold a fitting line-skipper
        for a head that is genuinely waiting on the room; only the deferred head, which has stood down, stops
        reserving it.
        """
        scheduler, job, target = await self._deferrable_head()
        protected_demand_mb = scheduler._displaced_head_outstanding_mb(job, device_index=None)
        assert protected_demand_mb is not None

        # Priced so the candidate fits the card outright while the head's demand would not fit beside it,
        # which is exactly the arithmetic head protection exists to enforce.
        _install_cycle(
            scheduler,
            DeviceVramState(
                total_vram_mb=24000.0,
                baseline_mb=1000.0,
                committed_vram_mb=2000.0,
                planned_unmaterialized_mb=0.0,
                committed_is_stale=False,
                device_free_mb=protected_demand_mb * 1.5,
            ),
        )
        assert (
            scheduler._dispatch_residency_reconciliation_holds(
                job,
                target,
                is_head_of_queue=False,
                head_outstanding_mb=protected_demand_mb,
            )
            is True
        )

        scheduler._whole_card_ledger.note_governor_defer(None, model=job.model, now=scheduler._clock())

        assert (
            scheduler._dispatch_residency_reconciliation_holds(
                job,
                target,
                is_head_of_queue=False,
                head_outstanding_mb=scheduler._displaced_head_outstanding_mb(job, device_index=None),
            )
            is False
        )


class TestRateLimitedResidentHeadKeepsAsking:
    """The dispatch-time residency check defers a rate-limited head for as long as the limiter holds.

    A head whose weights are already resident reaches the card through this check rather than through preload
    admission. The limiter brakes the rotation it would start, so the head waits for the card. It never
    dispatches co-resident, since the forecast has already said its weights stream beside the siblings.
    """

    async def test_the_head_never_dispatches_co_resident_however_long_it_waits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every ask defers while the limiter holds, far past one window."""
        now = [10_000.0]
        scheduler, job, target = await _scheduler_without_sibling()
        scheduler._clock = lambda: now[0]  # type: ignore[method-assign]
        monkeypatch.setattr(scheduler, "_budget_active", lambda: True)
        monkeypatch.setattr(scheduler, "_whole_card_residency_enabled", lambda: True)
        monkeypatch.setattr(scheduler, "_forecast_streaming", lambda *_args, **_kwargs: Mock())
        monkeypatch.setattr(scheduler, "_whole_card_warranted", lambda _forecast: True)
        ledger = scheduler._whole_card_ledger
        monkeypatch.setattr(ledger, "residency_demanded", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(ledger, "establish_rate_exceeded", lambda _device_index, *, now: True)

        for _ask in range(10):
            assert scheduler._resident_whole_card_head_ready(job, target) is False
            assert ledger.governor_deferred_head(None, now=now[0]) == job.model
            now[0] += _ESTABLISH_WINDOW_SECONDS
        assert scheduler.is_whole_card_residency_active() is False
        assert scheduler._job_tracker.is_admitted_exclusive(job) is False


class TestStarvedDispatchHeadSignals:
    """The gate feeds the arbiter only evidence-backed starvation and context-reduction signals.

    The starved-head reality admit and the context-teardown escalation both live in the arbiter, but they can
    only fire if this seam reports how long the head has starved and whether reducing the live pool would
    actually make room. The mere existence of an idle sibling is insufficient: a computed target at or above
    the current pool means its context is not the deficit and must not trigger whole-card residency.
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

    async def test_gate_does_not_offer_teardown_when_target_already_exceeds_live_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A co-resident SDXL-class head does not claim whole-card residency merely because a sibling is idle."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        monkeypatch.setattr(pricing, "max_coresident_for_peak_mb", lambda *_args, **_kwargs: 4)
        _starve_head(scheduler, job, seconds=_FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 5.0)
        capture = _CapturingArbiter(self._fits_verdict())
        scheduler._vram_arbiter = capture  # type: ignore[assignment]

        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False
        assert capture.last_request is not None
        assert capture.last_request.kind is VramRequestKind.MONOLITHIC_DISPATCH
        assert capture.last_request.starved_seconds == _FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 5.0
        assert capture.last_request.idle_contexts_teardownable is False

    async def test_gate_offers_teardown_when_computed_target_is_below_live_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A head whose priced peak genuinely requires fewer contexts retains the bounded teardown remedy."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        monkeypatch.setattr(pricing, "max_coresident_for_peak_mb", lambda *_args, **_kwargs: 1)
        _starve_head(scheduler, job, seconds=_FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 5.0)
        capture = _CapturingArbiter(self._fits_verdict())
        scheduler._vram_arbiter = capture  # type: ignore[assignment]

        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False
        assert capture.last_request is not None
        assert capture.last_request.idle_contexts_teardownable is True

    async def test_line_skip_dispatch_reports_no_teardownable_context(self) -> None:
        """A line-skip dispatch (not the head) never reports a teardownable context, so it cannot tear one down."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        _starve_head(scheduler, job, seconds=_FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 5.0)
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
