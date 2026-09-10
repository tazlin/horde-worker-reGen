"""The preload gate ladder as a decision over the scheduling snapshot.

The gate rows are the ladder's own specification. The target-selection rows are differential while the scheduler
keeps its own ``_select_preload_process`` for the retention-affinity callers, which cannot take a snapshot mid
placement-order; when that copy goes, those rows become the selector's specification.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling.admission import preload
from horde_worker_regen.process_management.scheduling.admission.commands import FaultCause, FaultJob, ReplaceProcess
from horde_worker_regen.process_management.scheduling.admission.preload import (
    PreloadPassControl,
    RamChargeKind,
    decide_preload_gates,
    decide_ram_admission,
    every_pending_model_accounted_for,
    loaded_or_loading_models,
    preload_head,
    select_preload_target,
)
from horde_worker_regen.process_management.scheduling.governance.preload_admission import (
    AdmissionDecision,
    PreloadSlotSnapshot,
    select_follower_room_process_id,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_NOW = 10_000.0


def _slot(process_id: int, *, model: str | None, state: HordeProcessState = HordeProcessState.WAITING_FOR_JOB):  # noqa: ANN202
    return make_mock_process_info(process_id, model_name=model, state=state)


async def _worker(
    *,
    slots: dict[int, object],
    pending: list[str],
    in_progress: list[str] | None = None,
    bridge_data: Mock | None = None,
    available_ram_mb: float | None = 65536.0,
    quarantined: bool = False,
    queued_model_process_ids: list[int] | None = None,
    max_inference: int = 2,
    card_runtimes: dict | None = None,
) -> tuple[InferenceScheduler, list[ImageGenerateJobPopResponse]]:
    """A scheduler over the given slots and queue, with the lifecycle answers the gates read pinned."""
    tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        process_map=ProcessMap(slots),  # type: ignore[arg-type]
        job_tracker=tracker,
        bridge_data=bridge_data,
        available_ram_mb=available_ram_mb,
        max_inference=max_inference,
        max_concurrent=2,
        card_runtimes=card_runtimes,
        clock=lambda: _NOW,
    )
    scheduler._process_lifecycle.is_model_load_quarantined = Mock(return_value=quarantined)  # type: ignore[attr-defined]
    scheduler._process_lifecycle.get_processes_with_model_for_queued_job = Mock(  # type: ignore[attr-defined]
        return_value=queued_model_process_ids or [],
    )
    jobs: list[ImageGenerateJobPopResponse] = []
    for model in pending:
        job = make_job_pop_response(model=model)
        await track_popped_job_async(tracker, job, time_popped=_NOW - 1.0)
        jobs.append(job)
    for model in in_progress or []:
        job = make_job_pop_response(model=model)
        await track_popped_job_async(tracker, job, time_popped=_NOW - 1.0)
        await mark_job_in_progress_async(tracker, job)
        jobs.append(job)
    return scheduler, jobs


def _decide(scheduler: InferenceScheduler, job: ImageGenerateJobPopResponse, *, head: bool = True):  # noqa: ANN202
    snapshot = scheduler.snapshot()
    job_id = str(job.id_)
    head_id = preload_head(snapshot) if head else None
    return decide_preload_gates(
        snapshot, job_id, head_job_id=head_id, loaded_models=loaded_or_loading_models(snapshot)
    )


class TestPassShape:
    """The pass's fast exit and its head."""

    async def test_every_pending_model_resident_exits_the_pass(self) -> None:
        """Set equality with the loaded models, empty slots included, is the fast exit."""
        scheduler, _ = await _worker(slots={0: _slot(0, model="sd")}, pending=["sd"])
        snapshot = scheduler.snapshot()
        loaded = loaded_or_loading_models(snapshot)
        assert loaded == frozenset({"sd"})
        assert every_pending_model_accounted_for(snapshot, loaded) is True

    async def test_an_empty_slot_keeps_the_pass_open(self) -> None:
        """An empty slot's None stays in the loaded set, so the head's clocks still run this cycle."""
        scheduler, _ = await _worker(slots={0: _slot(0, model="sd"), 1: _slot(1, model=None)}, pending=["sd"])
        snapshot = scheduler.snapshot()
        loaded = loaded_or_loading_models(snapshot)
        assert None in loaded
        assert every_pending_model_accounted_for(snapshot, loaded) is False

    async def test_the_head_skips_in_progress_and_aux_gated_jobs(self) -> None:
        """The first pending job neither running nor waiting on auxiliary files anchors the pass."""
        scheduler, jobs = await _worker(slots={0: _slot(0, model=None)}, pending=["sd", "xl"])
        await mark_job_in_progress_async(scheduler._job_tracker, jobs[0])
        snapshot = scheduler.snapshot()
        assert preload_head(snapshot) == str(jobs[1].id_)


class TestGateLadder:
    """Each gate answers in order, naming the target and the commands the decision requires."""

    async def test_quarantined_model_faults_the_job(self) -> None:
        """A quarantined model is never preloaded again; the job is faulted for reissue."""
        scheduler, jobs = await _worker(slots={0: _slot(0, model=None)}, pending=["sd"], quarantined=True)

        plan = _decide(scheduler, jobs[0])

        assert plan.decision is AdmissionDecision.QUARANTINED
        assert plan.commands == (FaultJob(jobs[0], FaultCause.QUARANTINED, "model load quarantined"),)
        assert plan.pass_control is PreloadPassControl.NEXT_JOB

    async def test_a_quarantined_job_already_in_progress_is_not_faulted_again(self) -> None:
        """The in-progress copy keeps the decision but the fault belongs to the running attempt."""
        scheduler, jobs = await _worker(
            slots={0: _slot(0, model=None)}, pending=[], in_progress=["sd"], quarantined=True
        )

        plan = _decide(scheduler, jobs[0], head=False)

        assert plan.decision is AdmissionDecision.QUARANTINED and plan.commands == ()

    async def test_aux_gated_job_yields_to_a_sibling_without_recording(self) -> None:
        """A job waiting on its LoRA never competes while a sibling could sample; nothing is recorded."""
        from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry

        scheduler, jobs = await _worker(slots={0: _slot(0, model=None)}, pending=["xl"])
        gated = make_job_pop_response(model="sd", loras=[LorasPayloadEntry(name="a-lora", model=1, clip=1)])
        await track_popped_job_async(scheduler._job_tracker, gated)

        plan = _decide(scheduler, gated)

        assert plan.decision is AdmissionDecision.NEXT_JOB
        assert plan.records_admission is False

    async def test_aux_gated_job_alone_takes_only_an_unoccupied_slot(self) -> None:
        """With no sibling the gated job may stage, but never by displacing a resident model."""
        from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry

        # Two resident models plus the gated one exceed the two-process pool, so affinity pins nothing and
        # the occupied slot is the selector's answer.
        scheduler, _ = await _worker(slots={0: _slot(0, model="other"), 1: _slot(1, model="other2")}, pending=[])
        gated = make_job_pop_response(model="sd", loras=[LorasPayloadEntry(name="a-lora", model=1, clip=1)])
        await track_popped_job_async(scheduler._job_tracker, gated)

        plan = _decide(scheduler, gated)

        assert plan.decision is AdmissionDecision.NEXT_JOB
        assert plan.records_admission is True and plan.target_process_id == 0

    async def test_already_loaded_needs_nothing(self) -> None:
        """A resident copy the job can use means no preload."""
        scheduler, jobs = await _worker(slots={0: _slot(0, model="sd"), 1: _slot(1, model=None)}, pending=["sd"])

        plan = _decide(scheduler, jobs[0])

        assert plan.decision is AdmissionDecision.ALREADY_LOADED

    async def test_ram_danger_floor_defers_with_a_notice(self) -> None:
        """Below the host's danger floor a new load defers; the notice names the reading."""
        scheduler, jobs = await _worker(
            slots={0: _slot(0, model=None)},
            pending=["sd"],
            bridge_data=make_mock_bridge_data(enable_vram_budget=True, vram_reserve_mb=1024, ram_reserve_mb=1024),
            available_ram_mb=100.0,
        )

        plan = _decide(scheduler, jobs[0])

        assert plan.decision is AdmissionDecision.DEFER_RAM_PRESSURE
        assert plan.notice is not None and "RAM danger floor reached" in plan.notice
        assert plan.pass_control is PreloadPassControl.STOP_PASS

    async def test_no_target_when_every_slot_is_busy(self) -> None:
        """With every slot busy there is nothing to load onto."""
        scheduler, jobs = await _worker(
            slots={0: _slot(0, model="other", state=HordeProcessState.INFERENCE_STARTING)},
            pending=["sd"],
        )

        plan = _decide(scheduler, jobs[0])

        assert plan.decision is AdmissionDecision.NO_TARGET and plan.target_process_id is None

    async def test_the_head_overrides_the_queued_model_guard_for_a_slot(self) -> None:
        """A head whose every slot is protected still finds a displacement target; a follower does not."""
        # As many loaded slots as there is work, so the queued-model guard protects both resident models.
        scheduler, jobs = await _worker(
            slots={0: _slot(0, model="other-a"), 1: _slot(1, model="other-b")},
            pending=["sd", "xl"],
            queued_model_process_ids=[0, 1],
        )

        head_plan = _decide(scheduler, jobs[0])
        follower_plan = _decide(scheduler, jobs[1])

        assert head_plan.admits and head_plan.target_process_id == 0 and head_plan.is_head_blocker is True
        assert follower_plan.decision is AdmissionDecision.NO_TARGET

    async def test_growth_hold_defers_a_speculative_load(self) -> None:
        """The card's growth hold defers a load for a job not yet running."""
        scheduler, jobs = await _worker(slots={0: _slot(0, model=None)}, pending=["sd"])
        scheduler.set_vram_growth_hold(0, True)

        plan = _decide(scheduler, jobs[0])

        assert plan.decision is AdmissionDecision.DEFER_VRAM_GROWTH_HOLD and plan.target_process_id == 0

    async def test_growth_hold_exempts_a_job_already_in_progress(self) -> None:
        """A job the card is committed to keeps its preload under a growth hold."""
        scheduler, jobs = await _worker(slots={0: _slot(0, model=None)}, pending=[], in_progress=["sd"])
        scheduler.set_vram_growth_hold(0, True)

        plan = _decide(scheduler, jobs[0], head=False)

        assert plan.admits

    async def test_model_change_cycles_the_child(self) -> None:
        """A slot mid-state holding another model is cycled before it takes the new one."""
        scheduler, jobs = await _worker(
            slots={0: _slot(0, model="other", state=HordeProcessState.PRELOADED_MODEL)},
            pending=["sd"],
            bridge_data=make_mock_bridge_data(cycle_process_on_model_change=True),
        )

        plan = _decide(scheduler, jobs[0])

        assert plan.decision is AdmissionDecision.REPLACE_PROCESS
        assert plan.commands == (ReplaceProcess(0),)
        assert plan.pass_control is PreloadPassControl.STOP_PASS

    async def test_concurrency_gate_defers_behind_a_load_in_flight(self) -> None:
        """One load per card at a time; the notice names the count."""
        scheduler, jobs = await _worker(
            slots={0: _slot(0, model="other", state=HordeProcessState.PRELOADING_MODEL), 1: _slot(1, model=None)},
            pending=["sd"],
            bridge_data=make_mock_bridge_data(very_fast_disk_mode=False),
        )

        plan = _decide(scheduler, jobs[0])

        assert plan.decision is AdmissionDecision.DEFER_CONCURRENCY
        assert plan.notice == "Already preloading 1 models, waiting for one to finish before preloading sd"

    async def test_the_gates_admit_onto_an_empty_slot(self) -> None:
        """An empty slot is the cheapest target and the head admits onto it."""
        scheduler, jobs = await _worker(slots={0: _slot(0, model="other"), 1: _slot(1, model=None)}, pending=["sd"])

        plan = _decide(scheduler, jobs[0])

        assert plan.admits and plan.target_process_id == 1 and plan.commands == ()
        assert plan.is_head_blocker is True


class TestTargetSelectionAgreesWithTheScheduler:
    """Differential rows: the pure selector picks the slot the scheduler's own selector picks."""

    async def test_single_gpu_with_a_head_holder_and_a_spare(self) -> None:
        """The spare slot wins over the head's own holder on one card."""
        scheduler, jobs = await _worker(
            slots={1: _slot(1, model="head-model"), 2: _slot(2, model=None)},
            pending=["head-model", "next-model"],
        )
        follower = jobs[1]

        assert select_preload_target(scheduler.snapshot(), str(follower.id_), ()) == (
            scheduler._select_preload_process(follower, []).process_id  # type: ignore[union-attr]
        )

    async def test_multi_gpu_places_a_second_copy_off_the_saturated_serving_card(self) -> None:
        """The serving card is at its cap of one, so its spare slot loses to the idle card's."""
        model = "stable_diffusion"
        busy = make_mock_process_info(0, model_name=model, device_index=0, state=HordeProcessState.INFERENCE_STARTING)
        spare_0 = make_mock_process_info(1, model_name=None, device_index=0)
        spare_1 = make_mock_process_info(2, model_name=None, device_index=1)
        scheduler, jobs = await _worker(slots={0: busy, 1: spare_0, 2: spare_1}, pending=[model])
        scheduler._card_runtimes = make_test_card_runtimes(device_indices=(0, 1))

        chosen = select_preload_target(scheduler.snapshot(), str(jobs[0].id_), ())

        assert chosen == 2
        assert chosen == scheduler._select_preload_process(jobs[0], []).process_id  # type: ignore[union-attr]

    async def test_multi_gpu_sticky_holds_while_the_serving_card_has_spare_capacity(self) -> None:
        """With a cap of two the serving card can run the second copy at once, so it keeps its preference."""
        model = "stable_diffusion"
        busy = make_mock_process_info(0, model_name=model, device_index=0, state=HordeProcessState.INFERENCE_STARTING)
        spare_0 = make_mock_process_info(1, model_name=None, device_index=0)
        spare_1 = make_mock_process_info(2, model_name=None, device_index=1)
        scheduler, jobs = await _worker(slots={0: busy, 1: spare_0, 2: spare_1}, pending=[model])
        scheduler._card_runtimes = make_test_card_runtimes(device_indices=(0, 1), max_concurrent_inference=2)

        chosen = select_preload_target(scheduler.snapshot(), str(jobs[0].id_), ())

        assert chosen == 1
        assert chosen == scheduler._select_preload_process(jobs[0], []).process_id  # type: ignore[union-attr]

    async def test_starvation_agrees_with_the_scheduler(self) -> None:
        """The pure age gate and the scheduler's read the same clock and ttl."""
        scheduler, jobs = await _worker(slots={1: _slot(1, model=None)}, pending=["sd"])
        scheduler._state.recent_job_ttl = 100.0
        snapshot = scheduler.snapshot()
        head = snapshot.queue.jobs[str(jobs[0].id_)]

        assert preload.head_is_starving(snapshot, head) is scheduler._head_aged_past_anti_starvation(jobs[0])


class TestFollowerRoomSelection:
    """A follower whose only copies are busy may displace an idle lane the queue wants less."""

    @staticmethod
    async def _pool(
        *,
        pending: list[str],
        in_progress: list[str] | None = None,
        residents: dict[int, tuple[str | None, int, bool]],
        card_indices: tuple[int, ...] = (0, 1),
    ) -> tuple[InferenceScheduler, list[ImageGenerateJobPopResponse]]:
        """A multi-card pool whose lanes are ``process_id -> (model, device index, busy)``.

        Every card serves every class this pool names, so a job's eligibility never decides a row that is
        about which lane may make room.
        """
        served = sorted({*pending, *(in_progress or []), *(model for model, _, _ in residents.values() if model)})
        config = make_mock_bridge_data(image_models_to_load=served)
        slots = {
            process_id: make_mock_process_info(
                process_id,
                model_name=model,
                device_index=device_index,
                state=HordeProcessState.INFERENCE_STARTING if busy else HordeProcessState.WAITING_FOR_JOB,
            )
            for process_id, (model, device_index, busy) in residents.items()
        }
        scheduler, jobs = await _worker(
            slots=slots,
            pending=pending,
            in_progress=in_progress,
            bridge_data=config,
        )
        scheduler._card_runtimes = make_test_card_runtimes(device_indices=card_indices, config=config)
        return scheduler, jobs

    async def test_one_card_never_offers_follower_room(self) -> None:
        """On a single card a duplicate is pure waste, so the rule is a no-op there by construction."""
        busy = _slot(0, model="hot", state=HordeProcessState.INFERENCE_STARTING)
        idle = _slot(1, model="cold")
        scheduler, jobs = await _worker(slots={0: busy, 1: idle}, pending=["hot", "hot", "hot", "cold"])
        snapshot = scheduler.snapshot()

        assert snapshot.multi_gpu_routing_active is False
        assert preload.select_follower_room_target(snapshot, str(jobs[0].id_), head_job_id=None) is None

    async def test_the_least_wanted_idle_copy_makes_room(self) -> None:
        """Three jobs against one busy copy displace the idle lane whose class has one job outstanding."""
        scheduler, jobs = await self._pool(
            pending=["hot", "hot", "hot", "cold"],
            residents={0: ("hot", 0, True), 1: ("cold", 1, False)},
        )
        snapshot = scheduler.snapshot()

        assert preload.duplicate_copy_may_serve(snapshot, str(jobs[0].id_)) is True
        assert preload.select_follower_room_target(snapshot, str(jobs[1].id_), head_job_id=None) == 1

    async def test_an_equally_wanted_idle_copy_is_left_alone(self) -> None:
        """With as many jobs queued for the resident class as for the loading one, the follower waits."""
        scheduler, jobs = await self._pool(
            pending=["hot", "hot", "cold", "cold"],
            residents={0: ("hot", 0, True), 1: ("cold", 1, False)},
        )

        assert preload.select_follower_room_target(scheduler.snapshot(), str(jobs[1].id_), head_job_id=None) is None

    async def test_the_heads_own_copy_is_never_taken(self) -> None:
        """The queue head's warm copy is the one the horde is waiting on, so it is spared however cold it is."""
        scheduler, jobs = await self._pool(
            pending=["cold", "hot", "hot", "hot"],
            residents={0: ("hot", 0, True), 1: ("cold", 1, False)},
        )
        snapshot = scheduler.snapshot()

        assert snapshot.queue.jobs[snapshot.queue.pending_in_pop_order[0]].model == "cold"
        assert preload.select_follower_room_target(snapshot, str(jobs[1].id_), head_job_id=None) is None

    async def test_a_model_an_in_progress_job_is_using_is_never_taken(self) -> None:
        """A second copy of a running model is live work, so its idle lane is not room."""
        scheduler, jobs = await self._pool(
            pending=["hot", "hot", "hot"],
            in_progress=["warm"],
            residents={0: ("hot", 0, True), 1: ("warm", 1, False), 2: ("warm", 2, True)},
            card_indices=(0, 1, 2),
        )

        assert preload.select_follower_room_target(scheduler.snapshot(), str(jobs[0].id_), head_job_id=None) is None

    async def test_a_card_at_its_sampling_cap_offers_no_room(self) -> None:
        """A copy on a card that cannot start it would be weights the card paid for and cannot use."""
        scheduler, jobs = await self._pool(
            pending=["hot", "hot", "hot", "cold"],
            residents={0: ("hot", 0, True), 1: ("cold", 1, False), 2: ("other", 1, True)},
        )

        assert preload.select_follower_room_target(scheduler.snapshot(), str(jobs[1].id_), head_job_id=None) is None

    async def test_the_copy_bound_keeps_the_rule_from_being_reached(self) -> None:
        """One job outstanding earns one copy, so a resident-but-busy model is called loaded and waits."""
        scheduler, jobs = await self._pool(
            pending=["hot"],
            residents={0: ("hot", 0, True), 1: ("cold", 1, False)},
        )
        snapshot = scheduler.snapshot()

        assert preload.duplicate_copy_may_serve(snapshot, str(jobs[0].id_)) is False
        plan = _decide(scheduler, jobs[0])
        assert plan.decision is AdmissionDecision.ALREADY_LOADED

    def test_equal_demand_breaks_to_the_least_recently_dispatched(self) -> None:
        """Two equally wanted idle copies: the one whose model has gone longest without work gives way."""
        recent = PreloadSlotSnapshot(
            process_id=1,
            model_name="recent",
            can_accept_job=True,
            retained_resident_since=900.0,
        )
        stale = PreloadSlotSnapshot(
            process_id=2,
            model_name="stale",
            can_accept_job=True,
            retained_resident_since=100.0,
        )

        chosen = select_follower_room_process_id(
            (recent, stale),
            loading_model_demand=3,
            model_demand={"recent": 1, "stale": 1},
        )

        assert chosen == 2

    def test_an_unstamped_hold_is_the_cheapest_thing_to_take(self) -> None:
        """A lane holding nothing the scheduler promised to keep gives way before a stamped retention."""
        stamped = PreloadSlotSnapshot(
            process_id=1,
            model_name="stamped",
            can_accept_job=True,
            retained_resident_since=100.0,
        )
        unstamped = PreloadSlotSnapshot(process_id=2, model_name="unstamped", can_accept_job=True)

        chosen = select_follower_room_process_id(
            (stamped, unstamped),
            loading_model_demand=3,
            model_demand={"stamped": 1, "unstamped": 1},
        )

        assert chosen == 2

    def test_nothing_less_wanted_yields_no_room(self) -> None:
        """The rule is a comparison of two models' outstanding work, and it refuses a tie."""
        assert (
            select_follower_room_process_id(
                (PreloadSlotSnapshot(process_id=1, model_name="cold", can_accept_job=True),),
                loading_model_demand=2,
                model_demand={"cold": 2},
            )
            is None
        )


class TestRamAdmission:
    """The RAM verdict picks the marginal accounting the target and the job's class allow."""

    async def test_a_fresh_target_is_charged_the_whole_checkpoint(self) -> None:
        """No retained pages and no component charge: the whole checkpoint, at the budget's own verdict."""
        scheduler, jobs = await _worker(slots={0: _slot(0, model=None)}, pending=["sd"])

        admission = decide_ram_admission(scheduler.snapshot(), str(jobs[0].id_), 0)

        assert admission.kind is RamChargeKind.WHOLE
        assert admission.fits is admission.verdict.fits

    async def test_a_retaining_target_is_credited_its_pages(self) -> None:
        """An idle target that kept its unloaded model's pages prices the swap at its marginal growth."""
        retaining = _slot(0, model=None)
        retaining.ram_usage_bytes = 8000 * 1024 * 1024
        scheduler, jobs = await _worker(slots={0: retaining}, pending=["sd"])

        admission = decide_ram_admission(scheduler.snapshot(), str(jobs[0].id_), 0)

        assert admission.kind is RamChargeKind.PAGE_REUSE
        assert admission.verdict.reusable_credit_mb > 0.0

    async def test_a_disaggregation_class_job_with_a_sidecar_is_charged_its_component(self) -> None:
        """The UNet-only residual supersedes the page credit, and a checkpoint already staged charges nothing."""

        class _Sidecar:
            residual_tensor_bytes = 6000 * 1024 * 1024

        retaining = _slot(0, model=None)
        retaining.ram_usage_bytes = 8000 * 1024 * 1024
        scheduler, jobs = await _worker(slots={0: retaining}, pending=["sd"])
        scheduler._is_disaggregation_class_eligible = lambda _job: True  # type: ignore[method-assign]
        scheduler._read_component_sidecar = lambda _model: _Sidecar()  # type: ignore[assignment, method-assign, return-value]
        job_id = str(jobs[0].id_)

        admission = decide_ram_admission(scheduler.snapshot(), job_id, 0)
        assert admission.kind is RamChargeKind.COMPONENT
        assert admission.verdict.predicted_mb == 6000.0, "the residual, not the whole checkpoint net of pages"

        scheduler._checkpoint_models_held_on = lambda _pid: frozenset({"sd"})  # type: ignore[method-assign]
        assert preload.component_charge_mb(scheduler.snapshot(), job_id, 0) == 0.0
