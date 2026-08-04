"""Wiring tests: the dispatcher observes attributable VRAM footprints into the learned-footprint store.

Shadow-only. These exercise the observation seam (``MessageDispatcher._handle_memory_report``) that the
parent's message pump uses, confirming only cleanly-attributable figures produce a store entry and that
ambiguous reports do not. Three footprints are attributable there: a running monolithic inference job's
sampling peak, an idle slot's resident weights, and the safety process's at-rest residency.
"""

from __future__ import annotations

import queue
import sys
import time
from unittest.mock import Mock

from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger
from horde_worker_regen.process_management.ipc.message_dispatcher import (
    _RESIDENT_OBSERVATION_SETTLE_SECONDS,
    MessageDispatcher,
)
from horde_worker_regen.process_management.ipc.messages import (
    HordeModelStateChangeMessage,
    HordeProcessMemoryMessage,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.resource_budget import (
    CommittedReserveLedger,
    platform_context_constant_mb,
)
from horde_worker_regen.process_management.resources.vram_footprints import (
    SAFETY_PROCESS_BASELINE,
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
    ResolutionBucket,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
    make_test_runtime_config,
    mark_job_in_progress_async,
)

_MODEL = "stable_diffusion"
_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl


def _dispatcher_with_store(
    *,
    process_map: ProcessMap,
    job_tracker: JobTracker,
    store: LearnedFootprintStore,
) -> MessageDispatcher:
    """A dispatcher whose model reference knows ``_MODEL``'s baseline, with the store registered."""
    reference = {_MODEL: make_mock_model_reference_record(_MODEL, baseline=_BASELINE)}
    dispatcher = MessageDispatcher(
        process_map=process_map,
        horde_model_map=HordeModelMap(root={}),
        job_tracker=job_tracker,
        process_message_queue=Mock(spec=queue.Queue),
        runtime_config=make_test_runtime_config(bridge_data=make_mock_bridge_data()),
        model_metadata=make_test_model_metadata(reference),
        action_ledger=ActionLedger(),
        reserve_ledger=CommittedReserveLedger(),
        on_unload_vram=Mock(),
        state=WorkerState(),
    )
    dispatcher.set_footprint_store(store)
    return dispatcher


def _memory_message(
    process_id: int,
    *,
    peak_mb: int | None = None,
    reserved_mb: int | None = None,
    allocated_mb: int | None = None,
    vram_usage_mb: int = 0,
    sampled_at: float | None = None,
) -> HordeProcessMemoryMessage:
    return HordeProcessMemoryMessage(
        process_id=process_id,
        process_launch_identifier=0,
        info="Memory report",
        ram_usage_bytes=1024,
        vram_usage_mb=vram_usage_mb,
        process_peak_reserved_mb=peak_mb,
        process_reserved_mb=reserved_mb,
        process_allocated_mb=allocated_mb,
        sampled_at=sampled_at,
    )


def _resident_key(checkpoint: str = _MODEL) -> FootprintKey:
    return FootprintKey(
        model_baseline=str(_BASELINE),
        resolution_bucket=None,
        platform=sys.platform,
        stage=FootprintStage.RESIDENT,
        checkpoint=checkpoint,
    )


def _safety_key() -> FootprintKey:
    return FootprintKey(
        model_baseline=SAFETY_PROCESS_BASELINE,
        resolution_bucket=None,
        platform=sys.platform,
        stage=FootprintStage.SAFETY,
    )


def _confirm_model_loaded(dispatcher: MessageDispatcher, process_id: int, *, model: str = _MODEL) -> None:
    """Drive the child-confirmed model load that anchors the resident settle window."""
    dispatcher._handle_model_state_change(
        HordeModelStateChangeMessage(
            process_id=process_id,
            process_launch_identifier=0,
            info="Model loaded",
            process_state=HordeProcessState.PRELOADED_MODEL,
            horde_model_name=model,
            horde_model_state=ModelLoadState.LOADED_IN_VRAM,
        ),
    )


async def test_running_inference_peak_is_recorded_with_the_right_key() -> None:
    """A monolithic inference slot's peak, with a running job and known baseline, lands under its key."""
    process_info = make_mock_process_info(1, model_name=_MODEL)
    job = make_job_pop_response(model=_MODEL, width=512, height=512)
    process_info.last_job_referenced = job
    process_map = ProcessMap({1: process_info})
    job_tracker = JobTracker()
    await mark_job_in_progress_async(job_tracker, job)
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=job_tracker, store=store)

    dispatcher._handle_memory_report(_memory_message(1, peak_mb=11000))

    expected_key = FootprintKey(
        model_baseline=str(_BASELINE),
        resolution_bucket=ResolutionBucket.LE_512,
        platform=sys.platform,
        stage=FootprintStage.SAMPLE,
    )
    observation = store.get_observation(expected_key)
    assert observation is not None
    assert observation.watermark_mb == 11000.0
    assert len(store) == 1


async def test_idle_slot_without_a_running_job_is_not_attributed() -> None:
    """A report whose referenced job is not in progress is left unattributed (no guess)."""
    process_info = make_mock_process_info(1, model_name=_MODEL)
    process_info.last_job_referenced = make_job_pop_response(model=_MODEL)
    process_map = ProcessMap({1: process_info})
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=JobTracker(), store=store)

    dispatcher._handle_memory_report(_memory_message(1, peak_mb=11000))

    assert len(store) == 0


async def test_non_inference_lane_peak_is_not_attributed() -> None:
    """A VAE lane's peak is not bound to a stage/job at this seam, so nothing is recorded."""
    process_info = make_mock_process_info(1, model_name=_MODEL, process_type=HordeProcessType.VAE_LANE)
    job = make_job_pop_response(model=_MODEL)
    process_info.last_job_referenced = job
    process_map = ProcessMap({1: process_info})
    job_tracker = JobTracker()
    await mark_job_in_progress_async(job_tracker, job)
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=job_tracker, store=store)

    dispatcher._handle_memory_report(_memory_message(1, peak_mb=11000))

    assert len(store) == 0


async def test_missing_peak_is_not_attributed() -> None:
    """A report with no peak reading (off-GPU child) records nothing."""
    process_info = make_mock_process_info(1, model_name=_MODEL)
    job = make_job_pop_response(model=_MODEL)
    process_info.last_job_referenced = job
    process_map = ProcessMap({1: process_info})
    job_tracker = JobTracker()
    await mark_job_in_progress_async(job_tracker, job)
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=job_tracker, store=store)

    dispatcher._handle_memory_report(_memory_message(1, peak_mb=None))

    assert len(store) == 0


def test_settled_idle_slot_records_its_resident_weights() -> None:
    """An idle slot whose residency has settled records weights plus its context under the resident key."""
    process_map = ProcessMap({1: make_mock_process_info(1, model_name=_MODEL)})
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=JobTracker(), store=store)

    loaded_at = time.time()
    _confirm_model_loaded(dispatcher, 1)
    dispatcher._handle_memory_report(
        _memory_message(
            1,
            allocated_mb=4900,
            reserved_mb=5200,
            sampled_at=loaded_at + _RESIDENT_OBSERVATION_SETTLE_SECONDS + 1.0,
        ),
    )

    observation = store.get_observation(_resident_key())
    assert observation is not None
    assert observation.watermark_mb == 4900.0 + platform_context_constant_mb()


def test_resident_observation_ignores_the_allocator_cache() -> None:
    """A slot whose reservation still caches a past job's freed blocks is priced by its live allocation.

    The caching allocator keeps blocks it has freed, so an idle slot's reservation is the high-water mark of
    the work it last ran. Folding that into the raise-only resident watermark would price residency at a
    running job's cost forever.
    """
    process_map = ProcessMap({1: make_mock_process_info(1, model_name=_MODEL)})
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=JobTracker(), store=store)

    loaded_at = time.time()
    _confirm_model_loaded(dispatcher, 1)
    dispatcher._handle_memory_report(
        _memory_message(
            1,
            allocated_mb=4900,
            reserved_mb=12000,
            sampled_at=loaded_at + _RESIDENT_OBSERVATION_SETTLE_SECONDS + 1.0,
        ),
    )

    observation = store.get_observation(_resident_key())
    assert observation is not None
    assert observation.watermark_mb == 4900.0 + platform_context_constant_mb()


def test_a_slot_reporting_no_live_allocation_is_not_attributed() -> None:
    """A report carrying only a reservation gives no at-rest figure, so nothing is recorded."""
    process_map = ProcessMap({1: make_mock_process_info(1, model_name=_MODEL)})
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=JobTracker(), store=store)

    loaded_at = time.time()
    _confirm_model_loaded(dispatcher, 1)
    dispatcher._handle_memory_report(
        _memory_message(
            1,
            reserved_mb=4900,
            allocated_mb=None,
            sampled_at=loaded_at + _RESIDENT_OBSERVATION_SETTLE_SECONDS + 1.0,
        ),
    )

    assert store.get_observation(_resident_key()) is None


def test_a_slot_inside_the_settle_window_is_not_attributed() -> None:
    """A reading taken just after a load describes an allocation still in motion, so it is discarded."""
    process_map = ProcessMap({1: make_mock_process_info(1, model_name=_MODEL)})
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=JobTracker(), store=store)

    loaded_at = time.time()
    _confirm_model_loaded(dispatcher, 1)
    dispatcher._handle_memory_report(
        _memory_message(1, allocated_mb=4900, sampled_at=loaded_at + _RESIDENT_OBSERVATION_SETTLE_SECONDS - 1.0),
    )

    assert store.get_observation(_resident_key()) is None


async def test_a_busy_slot_records_no_resident_footprint() -> None:
    """A slot running its job holds live activation on top of the weights, which is not a resident figure."""
    process_info = make_mock_process_info(1, model_name=_MODEL, state=HordeProcessState.INFERENCE_STARTING)
    job = make_job_pop_response(model=_MODEL, width=512, height=512)
    process_info.last_job_referenced = job
    process_map = ProcessMap({1: process_info})
    job_tracker = JobTracker()
    await mark_job_in_progress_async(job_tracker, job)
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=job_tracker, store=store)

    loaded_at = time.time()
    _confirm_model_loaded(dispatcher, 1)
    dispatcher._handle_memory_report(
        _memory_message(
            1,
            allocated_mb=12000,
            sampled_at=loaded_at + _RESIDENT_OBSERVATION_SETTLE_SECONDS + 1.0,
        ),
    )

    assert store.get_observation(_resident_key()) is None


async def test_an_idle_looking_slot_still_holding_a_tracked_job_is_not_attributed() -> None:
    """A slot back in an idle state while its job is still tracked in progress has not released it yet."""
    process_info = make_mock_process_info(1, model_name=_MODEL, state=HordeProcessState.INFERENCE_COMPLETE)
    job = make_job_pop_response(model=_MODEL, width=512, height=512)
    process_info.last_job_referenced = job
    process_map = ProcessMap({1: process_info})
    job_tracker = JobTracker()
    await mark_job_in_progress_async(job_tracker, job)
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=job_tracker, store=store)

    loaded_at = time.time()
    _confirm_model_loaded(dispatcher, 1)
    dispatcher._handle_memory_report(
        _memory_message(
            1,
            allocated_mb=12000,
            sampled_at=loaded_at + _RESIDENT_OBSERVATION_SETTLE_SECONDS + 1.0,
        ),
    )

    assert store.get_observation(_resident_key()) is None


def test_a_slot_with_no_confirmed_residency_is_not_attributed() -> None:
    """Without a confirmed model-load transition there is no evidence the reservation has come to rest."""
    process_map = ProcessMap({1: make_mock_process_info(1, model_name=_MODEL)})
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=JobTracker(), store=store)

    dispatcher._handle_memory_report(_memory_message(1, allocated_mb=4900, sampled_at=time.time()))

    assert len(store) == 0


def test_a_device_view_reading_is_never_folded_into_a_footprint() -> None:
    """Device-view VRAM is not a per-process charge, so a report carrying only that records nothing.

    ``vram_usage_mb`` reports device-wide occupancy on one platform and a per-process view on another, so
    treating it as this process's footprint would learn a watermark that includes every other tenant.
    """
    process_map = ProcessMap(
        {
            1: make_mock_process_info(1, model_name=_MODEL),
            2: make_mock_process_info(2, model_name=None, process_type=HordeProcessType.SAFETY),
        },
    )
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=JobTracker(), store=store)

    loaded_at = time.time()
    _confirm_model_loaded(dispatcher, 1)
    settled = loaded_at + _RESIDENT_OBSERVATION_SETTLE_SECONDS + 1.0
    dispatcher._handle_memory_report(_memory_message(1, allocated_mb=None, vram_usage_mb=15000, sampled_at=settled))
    dispatcher._handle_memory_report(_memory_message(2, allocated_mb=None, vram_usage_mb=15000, sampled_at=settled))

    assert len(store) == 0


def test_idle_safety_process_records_its_at_rest_residency() -> None:
    """A GPU-resident safety process's live allocation plus its context lands under the safety key."""
    process_info = make_mock_process_info(2, model_name=None, process_type=HordeProcessType.SAFETY)
    process_map = ProcessMap({2: process_info})
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=JobTracker(), store=store)

    dispatcher._handle_memory_report(_memory_message(2, allocated_mb=2800, reserved_mb=5100, peak_mb=6000))

    observation = store.get_observation(_safety_key())
    assert observation is not None
    # Neither the evaluation peak nor the reservation that still caches it: both are reclaimable and are not
    # what safety costs the card while it waits.
    assert observation.watermark_mb == 2800.0 + platform_context_constant_mb()


def test_an_evaluating_safety_process_is_not_attributed() -> None:
    """A safety process mid-evaluation still holds that evaluation, so its allocation is not at-rest."""
    process_info = make_mock_process_info(
        2,
        model_name=None,
        state=HordeProcessState.EVALUATING_SAFETY,
        process_type=HordeProcessType.SAFETY,
    )
    process_map = ProcessMap({2: process_info})
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=JobTracker(), store=store)

    dispatcher._handle_memory_report(_memory_message(2, allocated_mb=2800))

    assert len(store) == 0


def test_a_cpu_only_safety_process_records_nothing() -> None:
    """A safety process that never initialised CUDA reports no allocator fields, so nothing is learned."""
    process_info = make_mock_process_info(2, model_name=None, process_type=HordeProcessType.SAFETY)
    process_map = ProcessMap({2: process_info})
    store = LearnedFootprintStore()
    dispatcher = _dispatcher_with_store(process_map=process_map, job_tracker=JobTracker(), store=store)

    dispatcher._handle_memory_report(_memory_message(2, allocated_mb=None))

    assert len(store) == 0
