"""Regression guard for the whole-card residency convergence deadlock (the parked-head queue wedge).

The shape of the wedge, on a single 24GB card with ``whole_card_exclusive_residency`` enabled:

  * The head of the queue is a heavy Flux fp8 job whose forecast is ``needs_exclusive_residency``; it must
    take the card (evict sibling models, sample alone), so the residency machine pre-stages it into a spare
    process's RAM and then tries to *converge* to the forecast's budget target by stopping the excess idle
    siblings. On this 24GB box the target is two processes (the Flux holder plus one idle context the card has
    room for); the loop must stop the siblings beyond that.
  * Behind the head, the queue still holds ordinary SDXL jobs; and the excess idle sibling processes are
    still resident with exactly those models.

Originally ``_converge_whole_card_residency`` asked the *generic* ``scale_inference_processes`` to reduce the
live inference-process count to the forecast's target, and that path refuses to stop any process whose loaded
model is needed by a *queued* job (``get_processes_with_model_for_queued_job``). The excess idle siblings hold
models queued behind the head, so they were protected and never stopped. The count stayed above target,
``_whole_card_teardown_exhausted`` never returned True, and the pre-staged head was deferred every scheduling
tick (``_prestaged_whole_card_not_ready`` stays True). The head parked for
progressively longer with no dispatch until the recovery supervisor broke the wedge after the establish grace
lapsed, soft-resetting the pools, faulting the Flux job, and forcing process recoveries.

The fix: ``_converge_whole_card_residency`` now tells the scale-down it is a whole-card collapse by passing
``whole_card_model``, which narrows the teardown-exclusion set to spare only the head's holder (and other
cards), not every idle queued-model sibling. Whole-card residency means the heavy head owns the card's
sampling and the queued siblings beyond the budget target deliberately wait; their jobs reload once the head
drains. Busy processes are still never torn down.

These tests pin that fixed behaviour: the whole-card-aware shrink reaches the budget target even when the
excess siblings hold queued models, while the *default* (benchmark / pressure) shrink still protects them (so
the narrowing did not leak). The GREEN controls pin the cases that always worked (a model-free or
non-queued-model sibling is stoppable, a busy sibling is never the victim), so a regression cannot pass.
``TestWedgeDispatchDiagnostic``
guards the worker half of the logging<->detector seam (the stall is still attributed precisely if a future
change reintroduces an un-torn-down queued-model sibling).
"""

from __future__ import annotations

import multiprocessing
from unittest.mock import Mock

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_runtime_config,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

# Representative of the 24GB 4090 the wedge was observed on. Kept as ints because the process VRAM attributes
# (total_vram_mb / vram_usage_mb) are typed int.
_DEVICE_TOTAL_VRAM_MB = 24074
_PER_PROCESS_OVERHEAD_MB = 4213  # the forecast's overhead/proc on this card
_VRAM_RESERVE_MB = 2048.0
_RAM_RESERVE_MB = 4096.0

_FLUX_MODEL = "Flux.1-Schnell fp8 (Compact)"
_FLUX_WEIGHTS_MB = 11500.0
_RESIDENT_SDXL = "CyberRealistic Pony"
_OTHER_SDXL = "Juggernaut XL"
_THIRD_SDXL = "AlbedoBase XL (SDXL)"


def _wedge_bridge_data(**overrides: object) -> Mock:
    """Budget-on, whole-card-on config for the observed scenario (safety-off-GPU disabled to isolate scale-down).

    The safety pause is a separate convergence step; the deadlock is purely the inference-process scale-down,
    so these tests disable ``whole_card_residency_safety_off_gpu`` to keep the failure to one moving part.
    """
    base: dict[str, object] = {
        "enable_vram_budget": True,
        "whole_card_exclusive_residency": True,
        "whole_card_residency_safety_off_gpu": False,
        "safety_on_gpu": False,
        "vram_reserve_mb": _VRAM_RESERVE_MB,
        "ram_reserve_mb": _RAM_RESERVE_MB,
        "vram_per_process_overhead_mb": _PER_PROCESS_OVERHEAD_MB,
        "overbudget_exclusive_mode": True,
        "whole_card_residency_cooldown_seconds": 0,
        "image_models_to_load": [_RESIDENT_SDXL, _OTHER_SDXL, _THIRD_SDXL, _FLUX_MODEL],
        "max_threads": 1,
    }
    base.update(overrides)
    return make_mock_bridge_data(**base)


def _make_real_plm(
    *,
    process_map: ProcessMap,
    job_tracker: JobTracker,
    horde_model_map: HordeModelMap,
    bridge_data: Mock,
    target_process_count: int = 4,
) -> ProcessLifecycleManager:
    """A real ProcessLifecycleManager sharing the given map/tracker, so the real scale-down guard runs.

    The mocked mp pipes make ``_end_inference_process`` / ``retire_process`` safe to drive without real OS
    processes, so ``scale_inference_processes`` exercises ``get_processes_with_model_for_queued_job`` and the
    victim-selection exactly as in production. ``target_process_count`` sets the launched-process ceiling.
    """
    return ProcessLifecycleManager(
        ctx=multiprocessing.get_context("spawn"),  # type: ignore[arg-type]
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        process_message_queue=Mock(),
        card_runtimes=make_test_card_runtimes(target_process_count=target_process_count, config=bridge_data),
        disk_lock=Mock(),
        aux_model_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        max_safety_processes=1,
        amd_gpu=False,
        directml=None,
        abort_callback=Mock(),
        state=WorkerState(),
    )


def _flux_whole_card_forecast(*, free_now_mb: float) -> StreamForecast:
    """A whole-card Flux forecast on the 24GB box; its budget-relative target keeps one sibling context.

    ``wants_whole_card`` no longer collapses ``max_resident_processes()`` straight to 1: the depth is the same
    budget arithmetic an ordinary model uses (total - weights - reserve, against the per-context overhead),
    which on this 24GB card with the ~11.5GB fp8 weights leaves room for one idle sibling context: a target
    of two. ``free_now_mb`` controls whether the card reads drained (so ``fits_weights_now``, the final
    teardown-exhausted gate, can pass once the count is at target).
    """
    return StreamForecast(
        weights_mb=_FLUX_WEIGHTS_MB,
        reserve_mb=2975.0,
        base_reserve_mb=_VRAM_RESERVE_MB,
        free_now_mb=free_now_mb,
        free_if_alone_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
        free_after_model_evict_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
        total_vram_mb=_DEVICE_TOTAL_VRAM_MB,
        per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        wants_whole_card=True,
    )


# --------------------------------------------------------------------------------------------------------- #
#  The core mechanism: the real scale-down guard refuses to stop a queued-model sibling.                     #
# --------------------------------------------------------------------------------------------------------- #


class TestScaleDownProtectsQueuedSiblingModel:
    """``scale_inference_processes`` toward the whole-card target, at the lifecycle seam (no scheduler/forecast).

    Two live processes, one holding the pre-staged whole-card head, one idle holding a model that is still
    queued behind the head. The whole-card-aware shrink (``whole_card_model=...``) must reach the target of
    one process by stopping that idle sibling; the *default* shrink must still protect it (the narrowing is
    convergence-only). Busy siblings are never the victim either way.
    """

    def _wedge_map(self) -> tuple[ProcessMap, HordeModelMap]:
        """Process 4 holds the pre-staged Flux head; process 3 is idle holding the queued SDXL model."""
        flux_holder = make_mock_process_info(4, model_name=_FLUX_MODEL, state=HordeProcessState.PRELOADED_MODEL)
        sdxl_idle = make_mock_process_info(3, model_name=_RESIDENT_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
        for proc in (flux_holder, sdxl_idle):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map = ProcessMap({3: sdxl_idle, 4: flux_holder})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL,
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=4,
        )
        return process_map, horde_model_map

    async def test_converges_to_sole_residency_with_queued_sibling_model(self) -> None:
        """The faithful repro, now fixed: reduce to one process while an SDXL job sits behind the Flux head.

        The whole-card-aware shrink must stop the idle SDXL sibling so the head gets sole residency, even
        though ``CyberRealistic Pony`` is queued behind it. Before the fix the queued-model guard protected
        the sibling and the count never fell below 2 (the wedge).
        """
        process_map, horde_model_map = self._wedge_map()
        job_tracker = JobTracker()
        plm = _make_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_wedge_bridge_data(),
        )

        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))

        remaining = plm.scale_inference_processes(1, device_index=None, whole_card_model=_FLUX_MODEL)

        assert remaining == 1, (
            "the whole-card head must reach sole residency; the idle SDXL sibling holding a queued model "
            f"must be stoppable under the whole-card-aware shrink, but {remaining} live processes remain"
        )

    async def test_default_shrink_still_protects_queued_sibling_model(self) -> None:
        """Companion: the *default* (non-whole-card) shrink must still protect a queued-model sibling.

        The convergence narrowing is scoped to ``whole_card_model``; the benchmark / RAM-pressure shrink must
        keep its queued-model protection so it never tears down a process whose model a queued job needs. This
        pins that the narrowing did not leak into the default path.
        """
        process_map, horde_model_map = self._wedge_map()
        job_tracker = JobTracker()
        plm = _make_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_wedge_bridge_data(),
        )

        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))

        remaining = plm.scale_inference_processes(1, device_index=None)

        assert remaining == 2, (
            "the default shrink must spare the idle sibling holding a queued model (only the whole-card "
            f"convergence shrink may stop it), but it fell to {remaining}"
        )

    async def test_model_free_idle_sibling_is_stoppable(self) -> None:
        """GREEN control: a model-free idle sibling is not protected, so convergence reaches the target."""
        flux_holder = make_mock_process_info(4, model_name=_FLUX_MODEL, state=HordeProcessState.PRELOADED_MODEL)
        bare_idle = make_mock_process_info(3, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        for proc in (flux_holder, bare_idle):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map = ProcessMap({3: bare_idle, 4: flux_holder})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=4
        )

        job_tracker = JobTracker()
        plm = _make_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_wedge_bridge_data(),
        )
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))

        remaining = plm.scale_inference_processes(1, device_index=None)

        assert remaining == 1, "a model-free idle sibling must be stoppable so the head reaches sole residency"

    async def test_idle_sibling_with_non_queued_model_is_stoppable(self) -> None:
        """GREEN control: an idle sibling whose model is NOT queued is stoppable (the guard does not protect it).

        Same geometry as the wedge but the resident sibling model (``AlbedoBase XL``) is absent from the queue,
        so the protection that wedges the real case does not engage and convergence reaches the target.
        """
        flux_holder = make_mock_process_info(4, model_name=_FLUX_MODEL, state=HordeProcessState.PRELOADED_MODEL)
        sdxl_idle = make_mock_process_info(3, model_name=_THIRD_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
        for proc in (flux_holder, sdxl_idle):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map = ProcessMap({3: sdxl_idle, 4: flux_holder})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=4
        )

        job_tracker = JobTracker()
        plm = _make_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_wedge_bridge_data(),
        )
        # Queue holds the Flux head and an *unrelated* SDXL model, not the one the idle sibling holds.
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        await track_popped_job_async(job_tracker, make_job_pop_response(_OTHER_SDXL))

        remaining = plm.scale_inference_processes(1, device_index=None)

        assert remaining == 1

    async def test_in_progress_sibling_model_also_converges(self) -> None:
        """Variation: the shared model is in-progress (on another process) rather than pending.

        ``get_processes_with_model_for_queued_job`` protects in-progress models too, so the default guard would
        also pin this idle sibling. The whole-card-aware shrink spares only the head's holder, so the idle
        sibling (its in-progress job runs elsewhere, this process is idle) is stoppable and convergence reaches
        the target.
        """
        process_map, horde_model_map = self._wedge_map()
        job_tracker = JobTracker()
        plm = _make_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_wedge_bridge_data(),
        )
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        sdxl_job = await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))
        await job_tracker.mark_inference_started(sdxl_job)

        remaining = plm.scale_inference_processes(1, device_index=None, whole_card_model=_FLUX_MODEL)

        assert remaining == 1

    async def test_multiple_queued_model_siblings_all_torn_down(self) -> None:
        """Corner case: every idle sibling holds a different queued model; all must still be stoppable.

        On a wider worker the head can be behind several distinct SDXL jobs, each resident on its own idle
        sibling. The teardown to sole residency needs all of them stopped; the whole-card-aware shrink spares
        only the head's holder, so it collapses to one even though every sibling holds a queued model.
        """
        flux_holder = make_mock_process_info(4, model_name=_FLUX_MODEL, state=HordeProcessState.PRELOADED_MODEL)
        sib_a = make_mock_process_info(3, model_name=_RESIDENT_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
        sib_b = make_mock_process_info(2, model_name=_OTHER_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
        sib_c = make_mock_process_info(1, model_name=_THIRD_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
        for proc in (flux_holder, sib_a, sib_b, sib_c):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map = ProcessMap({1: sib_c, 2: sib_b, 3: sib_a, 4: flux_holder})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=4
        )

        job_tracker = JobTracker()
        plm = _make_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_wedge_bridge_data(),
        )
        for model in (_FLUX_MODEL, _RESIDENT_SDXL, _OTHER_SDXL, _THIRD_SDXL):
            await track_popped_job_async(job_tracker, make_job_pop_response(model, width=1216, height=1216))

        remaining = plm.scale_inference_processes(1, device_index=None, whole_card_model=_FLUX_MODEL)

        assert remaining == 1

    async def test_busy_sibling_is_never_the_victim(self) -> None:
        """GREEN guard: convergence must never stop a *busy* process even when it would help reach target.

        Not a wedge to fix; a sanity check that the fix for the queued-model case must keep: a process
        mid-inference is never a teardown victim, so a reduction request that can only be satisfied by killing
        a busy process correctly makes no progress (the in-flight job drains first).
        """
        flux_holder = make_mock_process_info(4, model_name=_FLUX_MODEL, state=HordeProcessState.PRELOADED_MODEL)
        busy = make_mock_process_info(3, model_name=_RESIDENT_SDXL, state=HordeProcessState.INFERENCE_STARTING)
        for proc in (flux_holder, busy):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map = ProcessMap({3: busy, 4: flux_holder})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=4
        )

        job_tracker = JobTracker()
        plm = _make_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_wedge_bridge_data(),
        )
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))

        remaining = plm.scale_inference_processes(1, device_index=None)

        assert remaining == 2, "a busy process must not be torn down; convergence waits for it to drain"


# --------------------------------------------------------------------------------------------------------- #
#  End to end: the scheduler's convergence loop wedges, the head never becomes dispatchable.                 #
# --------------------------------------------------------------------------------------------------------- #


def _wire_scheduler_with_real_plm(
    *,
    process_map: ProcessMap,
    job_tracker: JobTracker,
    horde_model_map: HordeModelMap,
    bridge_data: Mock,
) -> InferenceScheduler:
    """An InferenceScheduler whose ``_process_lifecycle`` is a real PLM sharing its map/tracker/model-map.

    The default scheduler test harness mocks the lifecycle, so the real scale-down guard never runs; this
    swaps in a real PLM so ``_converge_whole_card_residency`` drives the genuine teardown path.
    """
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        horde_model_map=horde_model_map,
        bridge_data=bridge_data,
        max_concurrent=1,
        max_inference=4,
    )
    scheduler._process_lifecycle = _make_real_plm(
        process_map=process_map,
        job_tracker=job_tracker,
        horde_model_map=horde_model_map,
        bridge_data=bridge_data,
    )
    return scheduler


class TestConvergenceLoopWedge:
    """Driving ``_converge_whole_card_residency`` with the real PLM must collapse to sole residency."""

    def _staged_wedge(
        self,
        *,
        sibling_models: tuple[str | None, ...],
        holder_state: HordeProcessState = HordeProcessState.PRELOADED_MODEL,
    ) -> tuple[InferenceScheduler, ProcessMap, JobTracker, StreamForecast]:
        """Post-pre-stage state: Flux held on one process, idle siblings holding ``sibling_models``.

        The budget-relative target on this 24GB box is two processes (Flux plus one idle context), so to leave
        a genuine teardown for the convergence loop the pool starts with *more* siblings than that: the loop
        must stop one to reach the target. Each entry of ``sibling_models`` is one idle sibling process (a model
        name pins it to a queued job; None leaves it model-free). The wedge fires when every excess sibling
        holds a *queued* model; the generic scale-down would protect them all and pin the count above target.
        """
        flux_holder = make_mock_process_info(4, model_name=_FLUX_MODEL, state=holder_state)
        siblings = [
            make_mock_process_info(3 - offset, model_name=model, state=HordeProcessState.WAITING_FOR_JOB)
            for offset, model in enumerate(sibling_models)
        ]
        for proc in (flux_holder, *siblings):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map = ProcessMap({proc.process_id: proc for proc in (flux_holder, *siblings)})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=4
        )

        job_tracker = JobTracker()
        scheduler = _wire_scheduler_with_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_wedge_bridge_data(),
        )
        # Record the held whole-card residency the pre-stage leaves (the None-keyed single-GPU residency).
        forecast = _flux_whole_card_forecast(free_now_mb=15007.0)
        scheduler._whole_card_forecast = forecast
        scheduler._sibling_teardown_for_model = _FLUX_MODEL
        return scheduler, process_map, job_tracker, forecast

    async def test_residency_converges_despite_queued_sibling(self) -> None:
        """End to end: many convergence ticks drive the pool down to the budget target and ready the head.

        Three processes (the Flux holder plus two idle siblings, each holding a model still queued behind the
        head) against a budget target of two: the loop must stop one queued-model sibling. The pre-fix generic
        shrink protected *every* queued-model sibling, so the count stayed at 3 > target and
        ``_whole_card_teardown_exhausted`` never returned True: the exact state that wedged the queue. The
        whole-card-aware shrink now spares only the head's holder, so it reaches the target.
        """
        scheduler, process_map, job_tracker, forecast = self._staged_wedge(
            sibling_models=(_RESIDENT_SDXL, _OTHER_SDXL),
        )
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))
        await track_popped_job_async(job_tracker, make_job_pop_response(_OTHER_SDXL))

        for _ in range(30):
            scheduler._converge_whole_card_residency()

        target = forecast.max_resident_processes()
        assert process_map.num_loaded_inference_processes() == target, (
            "the queued-model siblings must not pin the count above the residency's budget target"
        )
        # The same gate the pre-staged head dispatches behind: exhausted only once at (or below) the target.
        assert scheduler._whole_card_teardown_exhausted(forecast) is True, (
            "with the pool at the target the head's teardown must read exhausted so it can dispatch"
        )

    async def test_residency_converges_after_holder_becomes_waiting_for_job(self) -> None:
        """Incident shape: a fully resident idle holder must still drive the convergence shrink.

        Once the pre-stage has finished, the Flux holder may be ``WAITING_FOR_JOB`` rather than
        ``PRELOADED_MODEL``. The residency is still valid and the queued sibling must still be stoppable;
        otherwise the head parks until save-our-ship soft-resets the pool.
        """
        scheduler, process_map, job_tracker, forecast = self._staged_wedge(
            sibling_models=(_RESIDENT_SDXL, _OTHER_SDXL),
            holder_state=HordeProcessState.WAITING_FOR_JOB,
        )
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))
        await track_popped_job_async(job_tracker, make_job_pop_response(_OTHER_SDXL))

        for _ in range(30):
            scheduler._converge_whole_card_residency()

        assert process_map.num_loaded_inference_processes() == forecast.max_resident_processes()
        assert scheduler._whole_card_teardown_exhausted(forecast) is True

    async def test_residency_converges_when_sibling_is_model_free(self) -> None:
        """GREEN control: model-free idle siblings let the convergence loop reach the target and ready the head.

        The same end-to-end path with nothing protecting the excess siblings, proving the wiring is sound: the
        loop stops one to reach the budget target and the head clears its dispatch gate.
        """
        scheduler, process_map, job_tracker, forecast = self._staged_wedge(sibling_models=(None, None))
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))

        for _ in range(30):
            scheduler._converge_whole_card_residency()

        assert process_map.num_loaded_inference_processes() == forecast.max_resident_processes()
        assert scheduler._whole_card_teardown_exhausted(forecast) is True


class TestWedgeDispatchDiagnostic:
    """The worker half of the logging<->detector seam: the stall is *attributed* to the convergence wedge.

    The detector contract pins that ``detect_whole_card_convergence_wedge`` fires on the golden log line; this
    pins that the scheduler actually produces that line from the real wedge state, so the two halves cannot
    drift apart. Before this attribution the same state logged the misleading "no matching gate" message and
    was filed as a generic scheduler bug.
    """

    def _wedge_scheduler(
        self,
        *,
        extra_sibling: bool = False,
    ) -> tuple[InferenceScheduler, JobTracker]:
        """A staged wedge with the Flux head and SDXL jobs queued, residency held for Flux.

        With ``extra_sibling=True``, three processes are present: the Flux holder plus two idle SDXL
        siblings. The budget target of 2 on the 24 GB card leaves an excess sibling that must be torn
        down; while it remains, the pre-staged head is not ready and the diagnostic fires.
        """
        flux_holder = make_mock_process_info(4, model_name=_FLUX_MODEL, state=HordeProcessState.PRELOADED_MODEL)
        procs: dict[int, object] = {4: flux_holder}
        if extra_sibling:
            sdxl_idle_2 = make_mock_process_info(2, model_name=_OTHER_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
            for proc in (sdxl_idle_2,):
                proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
                proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
            procs[2] = sdxl_idle_2
        sdxl_idle_3 = make_mock_process_info(3, model_name=_RESIDENT_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
        for proc in (flux_holder, sdxl_idle_3):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        procs[3] = sdxl_idle_3
        process_map = ProcessMap(dict(procs.items()))  # type: ignore[arg-type]
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=4
        )
        job_tracker = JobTracker()
        scheduler = _wire_scheduler_with_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_wedge_bridge_data(),
        )
        scheduler._whole_card_forecast = _flux_whole_card_forecast(free_now_mb=15007.0)
        scheduler._sibling_teardown_for_model = _FLUX_MODEL
        return scheduler, job_tracker

    async def test_dispatch_stall_attributes_the_convergence_wedge(self) -> None:
        """The stall reason must name the wedge and the pinned sibling, matching the detector's regex.

        The phrase ``whole-card residency stuck: cannot reach sole residency`` is the seam the
        ``detect_whole_card_convergence_wedge`` detector keys off; the reason must also identify the
        protected sibling holding a queued model so the post-mortem points straight at the cause.

        Uses three processes (one excess beyond the 2-process budget target on a 24 GB card) so the
        convergence is genuinely stuck: the stored forecast reports the teardown as not exhausted
        because the process count exceeds the target.
        """
        scheduler, job_tracker = self._wedge_scheduler(extra_sibling=True)
        flux_head = await track_popped_job_async(
            job_tracker,
            make_job_pop_response(_FLUX_MODEL, width=1216, height=1216),
        )
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))

        reason = scheduler._diagnose_dispatch_stall(flux_head, {})

        assert "whole-card residency stuck: cannot reach sole residency" in reason
        assert "process 3" in reason
        assert _RESIDENT_SDXL in reason
        assert "no matching gate" not in reason

    async def test_no_wedge_attribution_when_sibling_model_not_queued(self) -> None:
        """Control: with the idle sibling's model absent from the queue, the wedge attribution must not fire.

        The teardown can stop a non-queued-model sibling, so this is not the convergence wedge; the diagnostic
        must not mislabel it (it falls through to the generic gate-less reason instead).
        """
        scheduler, job_tracker = self._wedge_scheduler()
        flux_head = await track_popped_job_async(
            job_tracker,
            make_job_pop_response(_FLUX_MODEL, width=1216, height=1216),
        )
        # The queue does NOT contain the idle sibling's model, so nothing pins the teardown.
        await track_popped_job_async(job_tracker, make_job_pop_response(_OTHER_SDXL))

        reason = scheduler._diagnose_dispatch_stall(flux_head, {})

        assert "whole-card residency stuck: cannot reach sole residency" not in reason
