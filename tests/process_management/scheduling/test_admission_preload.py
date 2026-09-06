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
    decide_preload_gates,
    every_pending_model_accounted_for,
    loaded_or_loading_models,
    preload_head,
    select_preload_target,
)
from horde_worker_regen.process_management.scheduling.governance.preload_admission import AdmissionDecision
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
) -> tuple[InferenceScheduler, list[ImageGenerateJobPopResponse]]:
    """A scheduler over the given slots and queue, with the lifecycle answers the gates read pinned."""
    tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        process_map=ProcessMap(slots),  # type: ignore[arg-type]
        job_tracker=tracker,
        bridge_data=bridge_data,
        available_ram_mb=available_ram_mb,
        max_inference=2,
        max_concurrent=2,
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

    async def test_multi_gpu_sticky_then_least_loaded(self) -> None:
        """A card already serving the model is preferred for its spare slot."""
        model = "stable_diffusion"
        busy = make_mock_process_info(0, model_name=model, device_index=0, state=HordeProcessState.INFERENCE_STARTING)
        spare_0 = make_mock_process_info(1, model_name=None, device_index=0)
        spare_1 = make_mock_process_info(2, model_name=None, device_index=1)
        scheduler, jobs = await _worker(slots={0: busy, 1: spare_0, 2: spare_1}, pending=[model])
        scheduler._card_runtimes = make_test_card_runtimes(device_indices=(0, 1))

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
