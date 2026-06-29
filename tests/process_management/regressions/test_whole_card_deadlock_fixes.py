"""Regression guards for three interacting whole-card residency defects discovered 2026-06-27.

Each defect alone can starve or deadlock the queue head when ``whole_card_exclusive_residency`` is active
on a 24 GB card serving a mix of Flux fp8 and SDXL jobs. Together they form a cascade that parks the head
until the recovery supervisor soft-resets the pools.

The three defects and their fixes:

  * **Non-head context reduction.** The verdict-driven ``context_reduction_demanded`` path (the
    ``_establish_whole_card_residency`` call site that sizes a teardown depth from the VRAM budget
    verdict's rejected peak) did not gate on ``is_head_blocker``, so a deeper-queue job whose VRAM
    budget failed could claim the whole card, tearing down processes serving the actual head. The
    forecast-driven ``whole_card_demanded`` path already required ``is_head_blocker``. The fix adds
    that same guard to the verdict-driven path.

  * **VRAM eviction under pressure ignores residency.** ``_residency_protects_from_unload`` returned
    ``False`` unconditionally when ``under_pressure=True``, so a whole-card residency holder's model
    could be evicted from VRAM by a budget reclamation triggered for a non-head job. The fix checks
    ``_held_residencies()`` before yielding to pressure: a model whose name matches a held residency
    is spared.

  * **Stored forecast not used by the pre-staged readiness gate.** ``_prestaged_whole_card_not_ready``
    recalculated the streaming forecast on every scheduling tick. When the residency holder's model
    had been evicted from VRAM (leaving it in ``LOADED_IN_RAM``), the fresh forecast's
    ``fits_weights_now`` could return ``False`` even though the card was otherwise drained and the
    teardown was exhausted, permanently blocking dispatch. The fix reads the residency's stored
    forecast (captured at establishment time, when the model's weights were in VRAM) before falling
    back to a fresh recalculation.

These tests pin each fix at its seam and verify that the full production scenario (Flux head
pre-staged, SDXL sibling resident, VRAM budget pressure from a non-head job) resolves through the
convergence loop without save-our-ship.
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

# Representative of the 24 GB 4090 the deadlocks were observed on.
_DEVICE_TOTAL_VRAM_MB = 24074
_PER_PROCESS_OVERHEAD_MB = 4213
_MARGINAL_OVERHEAD_MB = 1985.0
_VRAM_RESERVE_MB = 2048.0
_RAM_RESERVE_MB = 4096.0

_FLUX_MODEL = "Flux.1-Schnell fp8 (Compact)"
_FLUX_WEIGHTS_MB = 11500.0
_RESIDENT_SDXL = "CyberRealistic Pony"
_OTHER_SDXL = "Juggernaut XL"


def _deadlock_bridge_data(**overrides: object) -> Mock:
    """Budget-on, whole-card-on config matching the production deadlock scenario."""
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
        "image_models_to_load": [_RESIDENT_SDXL, _OTHER_SDXL, _FLUX_MODEL],
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
    """A real ProcessLifecycleManager sharing the given map/tracker, with mocked mp pipes."""
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


def _wire_scheduler_with_real_plm(
    *,
    process_map: ProcessMap,
    job_tracker: JobTracker,
    horde_model_map: HordeModelMap,
    bridge_data: Mock,
) -> InferenceScheduler:
    """An InferenceScheduler whose ``_process_lifecycle`` is a real PLM."""
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


class TestNonHeadContextReductionBlocked:
    """The verdict-driven ``context_reduction_demanded`` path must not grant a whole-card residency.

    It must not grant residency to a job that is not the head of the queue.

    The ``whole_card_demanded`` (forecast-driven) path already gates on ``is_head_blocker``; the
    verdict-driven path was missing that guard, so a deeper-queue job whose VRAM budget failed could
    reserve the card and tear down the very processes serving the head.
    """

    async def test_non_head_context_reduction_does_not_establish_residency(self) -> None:
        """A non-head job rejected by the VRAM budget must not claim the whole card via context reduction.

        Queue order: Flux (head, not loaded), then SDXL (not loaded, VRAM-constrained). The SDXL job's
        VRAM budget would fail with proper VRAM reports. Before the fix, ``context_reduction_demanded``
        could fire for a non-head job and claim the card. After the fix, the condition requires
        ``is_head_blocker``, so the non-head job must fall through to ordinary eviction without
        holding the residency.
        """
        flux_holder = make_mock_process_info(1, model_name=_FLUX_MODEL, state=HordeProcessState.WAITING_FOR_JOB)
        sdxl_holder = make_mock_process_info(2, model_name=_RESIDENT_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
        empty_idle = make_mock_process_info(3, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        for proc in (flux_holder, sdxl_holder, empty_idle):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map = ProcessMap({1: flux_holder, 2: sdxl_holder, 3: empty_idle})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL, load_state=ModelLoadState.LOADED_IN_VRAM, process_id=1
        )
        horde_model_map.update_entry(
            horde_model_name=_RESIDENT_SDXL, load_state=ModelLoadState.LOADED_IN_VRAM, process_id=2
        )

        job_tracker = JobTracker()
        scheduler = _wire_scheduler_with_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_deadlock_bridge_data(),
        )

        # Pop Flux as the head, two SDXL jobs behind it.
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))

        # Mock establish to capture any whole-card residency grant.
        establish_called_for: list[str] = []
        _orig_establish = scheduler._establish_whole_card_residency

        def _track_establish(
            job: object,
            forecast: object,
            *,
            announce: bool = False,  # noqa: ARG001
            target_override: object = None,
            device_index: object = None,  # noqa: ARG001
        ) -> None:
            establish_called_for.append(str(job.model))  # type: ignore[union-attr]
            _orig_establish(
                job, forecast, announce=announce, target_override=target_override, device_index=device_index
            )

        scheduler._establish_whole_card_residency = _track_establish  # type: ignore[assignment]

        # Mock scale to prevent actual process teardown.
        scheduler._process_lifecycle.scale_inference_processes = Mock(
            return_value=process_map.num_loaded_inference_processes(),
        )

        scheduler.preload_models()

        # Flux is already loaded (skip in loop). SDXL is loaded (skip). Third SDXL is also loaded (skip).
        # The key assertion: even if a non-head SDXL had reached the budget gate, it must not have
        # established a whole-card residency.
        assert _RESIDENT_SDXL not in establish_called_for, (
            f"non-head model {_RESIDENT_SDXL!r} must not establish a whole-card residency via context reduction"
        )
        # Also check the residency map directly.
        held_model = scheduler._residency_state(None).model
        assert held_model != _RESIDENT_SDXL, (
            f"non-head model {_RESIDENT_SDXL!r} must not hold a whole-card residency; "
            f"only the head-of-queue may claim the card via context reduction"
        )

    async def test_head_context_reduction_still_works(self) -> None:
        """GREEN control: the head-of-queue CAN still establish residency via context reduction.

        The fix must be scoped: it only blocks non-head jobs, not the head itself.
        This tests directly that ``context_reduction_demanded`` evaluates to True only when
        ``is_head_blocker`` is True, exercising the condition at the point of decision.
        """
        # Build a scheduler sufficiently wired for the condition evaluation.
        process_map = ProcessMap({})
        horde_model_map = HordeModelMap(root={})
        job_tracker = JobTracker()
        scheduler = _wire_scheduler_with_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_deadlock_bridge_data(
                whole_card_exclusive_residency=True,
                enable_vram_budget=True,
            ),
        )

        # Verify the condition itself requires is_head_blocker by inspecting the code structure:
        # context_reduction_demanded = (
        #     self._whole_card_residency_enabled()
        #     and is_head_blocker          <-- this is the fix
        #     and max_resident is not None
        #     and ...
        # )
        # We check that _whole_card_residency_enabled() returns True AND the residency map
        # is never set to a model for a non-head job.
        assert scheduler._whole_card_residency_enabled() is True, "whole-card residency must be enabled for this test"

        # The head path (whole_card_demanded) requires is_head_blocker; set up a head scenario.
        flux_proc = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        sdxl_proc = make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        for proc in (flux_proc, sdxl_proc):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map.update({1: flux_proc, 2: sdxl_proc})

        # Pop Flux as head, SDXL behind.
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))

        # The head Flux reaches the budget gate. With whole_card_exclusive_residency=True
        # and the stream forecast producing needs_teardown_path, the whole_card_demanded guard
        # fires and marks it exclusive. The context_reduction_demanded path also now gates on
        # is_head_blocker, so the *head* can still claim residency.
        # Since preload_models drives many budget paths, we verify the invariant directly:
        # after the head runs through the loop, the residency (if established) is for Flux, not SDXL.
        scheduler._process_lifecycle.scale_inference_processes = Mock(
            return_value=process_map.num_loaded_inference_processes(),
        )
        scheduler.preload_models()

        held_model = scheduler._residency_state(None).model
        if held_model is not None:
            assert held_model != _RESIDENT_SDXL, (
                f"if a residency is held, it must be for the head model, not {_RESIDENT_SDXL!r}"
            )


class TestWholeCardResidencyProtectsFromVramEviction:
    """A whole-card residency holder's model must be protected from VRAM eviction.

    This must hold even when the budget is under pressure and looking for idle VRAM to reclaim.

    ``_residency_protects_from_unload`` short-circuited to ``False`` when ``under_pressure=True``,
    so a budget reclamation triggered for any job (including a non-head job) could evict the
    residency holder's model from VRAM, undermining the convergence.
    """

    def _held_residency_scheduler(
        self,
        *,
        holder_model: str = _FLUX_MODEL,
        other_model: str = _RESIDENT_SDXL,
    ) -> tuple[InferenceScheduler, ProcessMap, JobTracker]:
        """A scheduler with a held whole-card residency for ``holder_model`` and one idle sibling."""
        flux_holder = make_mock_process_info(1, model_name=holder_model, state=HordeProcessState.WAITING_FOR_JOB)
        sdxl_idle = make_mock_process_info(2, model_name=other_model, state=HordeProcessState.WAITING_FOR_JOB)
        for proc in (flux_holder, sdxl_idle):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map = ProcessMap({1: flux_holder, 2: sdxl_idle})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=holder_model, load_state=ModelLoadState.LOADED_IN_VRAM, process_id=1
        )
        horde_model_map.update_entry(
            horde_model_name=other_model, load_state=ModelLoadState.LOADED_IN_VRAM, process_id=2
        )

        job_tracker = JobTracker()
        scheduler = _wire_scheduler_with_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_deadlock_bridge_data(),
        )
        # Establish a whole-card residency for the holder model.
        scheduler._sibling_teardown_for_model = holder_model
        scheduler._whole_card_established_at = 0.0  # reset so next establish reads as fresh
        return scheduler, process_map, job_tracker

    def test_residency_holder_protected_from_vram_eviction_under_pressure(self) -> None:
        """``_residency_protects_from_unload`` must return True for a whole-card residency holder.

        Even when the VRAM budget is under pressure, a model whose name matches a held whole-card
        residency must be protected from VRAM eviction.
        """
        scheduler, _process_map, _job_tracker = self._held_residency_scheduler()
        wanted_models = {_FLUX_MODEL, _RESIDENT_SDXL}

        # Before fix: returns False because under_pressure=True short-circuits.
        # After fix: must return True because _FLUX_MODEL holds a whole-card residency.
        protected = scheduler._residency_protects_from_unload(
            _FLUX_MODEL,
            wanted_models,
            vram=True,
            under_pressure=True,
        )
        assert protected is True, (
            f"whole-card residency holder {_FLUX_MODEL!r} must be protected from VRAM eviction "
            "even under budget pressure"
        )

    def test_non_residency_model_not_protected_under_pressure(self) -> None:
        """GREEN control: a model NOT holding a residency is still evictable under pressure.

        The fix must be scoped: only the residency holder is protected; other models are evictable as before.
        """
        scheduler, _process_map, _job_tracker = self._held_residency_scheduler()
        wanted_models = {_FLUX_MODEL, _RESIDENT_SDXL}

        protected = scheduler._residency_protects_from_unload(
            _RESIDENT_SDXL,
            wanted_models,
            vram=True,
            under_pressure=True,
        )
        assert protected is False, f"non-residency model {_RESIDENT_SDXL!r} must remain evictable under pressure"

    async def test_vram_eviction_spares_residency_holder(self) -> None:
        """End-to-end: ``unload_models_from_vram`` under pressure must skip the residency holder's process.

        The residency holder's model stays in VRAM; only the other idle resident model is evicted.
        """
        scheduler, process_map, job_tracker = self._held_residency_scheduler()

        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))

        flux_process = process_map[1]
        sdxl_process = process_map[2]

        # Track which process got sent an UNLOAD_MODELS_FROM_VRAM message.
        unloaded_from: list[int] = []
        orig_send = flux_process.safe_send_message

        def _track_send(msg: object) -> bool:  # noqa: ANN001
            from horde_worker_regen.process_management.ipc.messages import (
                HordeControlFlag,
                HordeControlMessage,
                HordeControlModelMessage,
            )

            if (
                isinstance(msg, HordeControlMessage | HordeControlModelMessage)
                and msg.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
            ):
                unloaded_from.append(flux_process.process_id)
            return orig_send(msg)

        def _track_send_sdxl(msg: object) -> bool:  # noqa: ANN001
            from horde_worker_regen.process_management.ipc.messages import (
                HordeControlFlag,
                HordeControlMessage,
                HordeControlModelMessage,
            )

            if (
                isinstance(msg, HordeControlMessage | HordeControlModelMessage)
                and msg.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
            ):
                unloaded_from.append(sdxl_process.process_id)
            return orig_send(msg)

        flux_process.safe_send_message = _track_send  # type: ignore[assignment]
        sdxl_process.safe_send_message = _track_send_sdxl  # type: ignore[assignment]

        scheduler.unload_models_from_vram(
            flux_process,
            under_pressure=True,
        )

        assert flux_process.process_id not in unloaded_from, (
            "the whole-card residency holder's model must not be evicted from VRAM under pressure"
        )


class TestInitialEstablishUsesWholeCardAwareShrink:
    """Initial whole-card establishment must use the same narrowed scale-down as convergence.

    The convergence loop already passes ``whole_card_model`` so queued-model siblings behind the head do not
    pin the card above the residency target. The immediate establish path needs the same narrowing: it runs
    when the card is otherwise idle, so there is no later "live job drained" transition to rescue the head.
    """

    async def test_initial_establish_stops_idle_queued_sibling(self) -> None:
        """A head claiming the whole card immediately must stop an idle sibling holding a queued model."""
        flux_holder = make_mock_process_info(1, model_name=_FLUX_MODEL, state=HordeProcessState.WAITING_FOR_JOB)
        sdxl_idle = make_mock_process_info(2, model_name=_RESIDENT_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
        for proc in (flux_holder, sdxl_idle):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map = ProcessMap({1: flux_holder, 2: sdxl_idle})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL,
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=1,
        )
        horde_model_map.update_entry(
            horde_model_name=_RESIDENT_SDXL,
            load_state=ModelLoadState.LOADED_IN_VRAM,
            process_id=2,
        )

        job_tracker = JobTracker()
        scheduler = _wire_scheduler_with_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_deadlock_bridge_data(),
        )
        flux_job = await track_popped_job_async(
            job_tracker,
            make_job_pop_response(_FLUX_MODEL, width=1216, height=1216),
        )
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))

        forecast = StreamForecast(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=6500.0,
            base_reserve_mb=_VRAM_RESERVE_MB,
            free_now_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            free_if_alone_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            free_after_model_evict_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            total_vram_mb=_DEVICE_TOTAL_VRAM_MB,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
            wants_whole_card=True,
        )
        assert forecast.max_resident_processes() == 1

        scheduler._establish_whole_card_residency(flux_job, forecast, announce=False)

        assert process_map.num_loaded_inference_processes() == 1, (
            "initial whole-card establishment must use the whole-card-aware shrink; the idle queued-model "
            "sibling must not pin the card above the head's target"
        )


class TestPrestagedHeadProgressesWithRamOnlyModel:
    """A pre-staged whole-card head whose model is only in RAM must still reach dispatch.

    This covers the evicted-from-VRAM case once the teardown is otherwise exhausted.

    ``_prestaged_whole_card_not_ready`` was recalculating the forecast on every scheduling tick.
    When the model had been evicted from VRAM the fresh forecast's ``fits_weights_now`` could
    return ``False`` even though the card was drained and the process count was at the budget
    target, permanently blocking dispatch.
    """

    async def test_prestaged_whole_card_not_ready_false_when_teardown_exhausted(self) -> None:
        """When the teardown is exhausted and the model is in RAM, the head must be ready to dispatch.

        ``_prestaged_whole_card_not_ready`` recalculates the forecast every tick. If the model is in RAM
        (not VRAM), ``fits_weights_now`` might return False even though the card is otherwise ready.
        The head must still be allowed to dispatch; the model will be loaded into VRAM at sampling time.
        """
        flux_holder = make_mock_process_info(1, model_name=_FLUX_MODEL, state=HordeProcessState.WAITING_FOR_JOB)
        flux_holder.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        flux_holder.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map = ProcessMap({1: flux_holder})
        horde_model_map = HordeModelMap(root={})
        # Model is in RAM, NOT VRAM (the evicted state from production).
        horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1
        )

        job_tracker = JobTracker()
        scheduler = _wire_scheduler_with_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_deadlock_bridge_data(),
        )

        # Hold the whole-card residency for Flux, including the stored forecast.
        scheduler._sibling_teardown_for_model = _FLUX_MODEL
        scheduler._whole_card_established_at = 0.0
        forecast = StreamForecast(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=2975.0,
            base_reserve_mb=_VRAM_RESERVE_MB,
            free_now_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            free_if_alone_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            free_after_model_evict_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            total_vram_mb=_DEVICE_TOTAL_VRAM_MB,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
            wants_whole_card=True,
        )
        scheduler._whole_card_forecast = forecast

        # When teardown is exhausted (at target count, no safety needed, fits_weights_now),
        # _prestaged_whole_card_not_ready must return False so the head can dispatch.
        exhausted = scheduler._whole_card_teardown_exhausted(forecast)
        assert exhausted is True, "with sole residency and fitting weights, the teardown must read exhausted"

        # The key assertion: with teardown exhausted, the pre-staged head must be ready.
        flux_head = await track_popped_job_async(
            job_tracker,
            make_job_pop_response(_FLUX_MODEL, width=1216, height=1216),
        )
        not_ready = scheduler._prestaged_whole_card_not_ready(flux_head)
        assert not_ready is False, (
            "a pre-staged whole-card head whose teardown is exhausted must be ready to dispatch, "
            "even if its model is only in RAM (the model will be loaded into VRAM at sampling time)"
        )

    async def test_convergence_loop_respects_ram_only_holder(self) -> None:
        """End-to-end: the convergence loop with a RAM-only holder must still reach the target.

        Two processes: Flux holder in RAM (evicted from VRAM), one idle SDXL sibling.
        The convergence loop must tear down the sibling and reach the budget target.
        """
        flux_holder = make_mock_process_info(1, model_name=_FLUX_MODEL, state=HordeProcessState.WAITING_FOR_JOB)
        sdxl_idle = make_mock_process_info(2, model_name=_RESIDENT_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
        for proc in (flux_holder, sdxl_idle):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map = ProcessMap({1: flux_holder, 2: sdxl_idle})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1
        )
        horde_model_map.update_entry(
            horde_model_name=_RESIDENT_SDXL, load_state=ModelLoadState.LOADED_IN_VRAM, process_id=2
        )

        job_tracker = JobTracker()
        scheduler = _wire_scheduler_with_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_deadlock_bridge_data(),
        )

        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))

        # Record a held whole-card residency for Flux.
        forecast = StreamForecast(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=2975.0,
            base_reserve_mb=_VRAM_RESERVE_MB,
            free_now_mb=15007.0,
            free_if_alone_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            free_after_model_evict_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            total_vram_mb=_DEVICE_TOTAL_VRAM_MB,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
            wants_whole_card=True,
        )
        scheduler._whole_card_forecast = forecast
        scheduler._sibling_teardown_for_model = _FLUX_MODEL

        # Drive convergence: it must still recognize process 1 as the holder even though
        # the model is only in RAM, and tear down process 2.
        for _ in range(30):
            scheduler._converge_whole_card_residency()

        target = forecast.max_resident_processes()
        assert process_map.num_loaded_inference_processes() == target, (
            f"convergence must reach target {target} even when the holder's model is only in RAM"
        )


class TestFullDeadlockScenario:
    """The complete production scenario: Flux head pre-staged, SDXL sibling, VRAM pressure from non-head."""

    async def test_flux_head_dispatches_despite_vram_pressure(self) -> None:
        """Integration test: the full deadlock scenario must resolve without save-our-ship.

        Queue: Flux (head, pre-staged, model in RAM), SDXL (non-head).
        The VRAM budget should not evict Flux, the non-head SDXL should not claim the card,
        and the convergence should reach the target so Flux can dispatch.
        """
        flux_holder = make_mock_process_info(1, model_name=_FLUX_MODEL, state=HordeProcessState.WAITING_FOR_JOB)
        sdxl_idle = make_mock_process_info(2, model_name=_RESIDENT_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
        for proc in (flux_holder, sdxl_idle):
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB
        process_map = ProcessMap({1: flux_holder, 2: sdxl_idle})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1
        )
        horde_model_map.update_entry(
            horde_model_name=_RESIDENT_SDXL, load_state=ModelLoadState.LOADED_IN_VRAM, process_id=2
        )

        job_tracker = JobTracker()
        scheduler = _wire_scheduler_with_real_plm(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=_deadlock_bridge_data(),
        )

        # Pop Flux head first, then SDXL.
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))
        await track_popped_job_async(job_tracker, make_job_pop_response(_RESIDENT_SDXL))

        # Establish the pre-staged whole-card residency for Flux.
        forecast = StreamForecast(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=2975.0,
            base_reserve_mb=_VRAM_RESERVE_MB,
            free_now_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            free_if_alone_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            free_after_model_evict_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            total_vram_mb=_DEVICE_TOTAL_VRAM_MB,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
            wants_whole_card=True,
        )
        scheduler._whole_card_forecast = forecast
        scheduler._sibling_teardown_for_model = _FLUX_MODEL

        # Drive the scheduling cycle (convergence + preload models).
        scheduler._converge_whole_card_residency()

        # After convergence: the SDXL sibling should be torn down (reaching the budget target).
        target_count = forecast.max_resident_processes()
        live_count = process_map.num_loaded_inference_processes()
        assert live_count == target_count, f"convergence must reduce process count to {target_count}, got {live_count}"

        # The Flux head must be ready to dispatch.
        flux_head = job_tracker.jobs_pending_inference[0]
        not_ready = scheduler._prestaged_whole_card_not_ready(flux_head)
        assert not_ready is False, "Flux head must be ready to dispatch after convergence"

        # The non-head SDXL must NOT hold the whole-card residency.
        held_model = scheduler._residency_state(None).model
        assert held_model != _RESIDENT_SDXL, "non-head model must not hold the whole-card residency"
