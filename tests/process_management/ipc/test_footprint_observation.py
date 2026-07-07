"""Wiring tests: the dispatcher observes attributable VRAM peaks into the learned-footprint store.

Shadow-only. These exercise the observation seam (``MessageDispatcher._handle_memory_report``) that the
parent's message pump uses, confirming only cleanly-attributable peaks (a running monolithic inference
job with a known baseline and a positive peak) produce a store entry, and that ambiguous reports do not.
"""

from __future__ import annotations

import queue
import sys
from unittest.mock import Mock

from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger
from horde_worker_regen.process_management.ipc.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.ipc.messages import HordeProcessMemoryMessage
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from horde_worker_regen.process_management.resources.vram_footprints import (
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


def _memory_message(process_id: int, *, peak_mb: int | None) -> HordeProcessMemoryMessage:
    return HordeProcessMemoryMessage(
        process_id=process_id,
        process_launch_identifier=0,
        info="Memory report",
        ram_usage_bytes=1024,
        process_peak_reserved_mb=peak_mb,
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
