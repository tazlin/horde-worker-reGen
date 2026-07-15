"""The post-processing/sampling co-residency measured-truth second admission path.

When the static reported-total co-residency gate withholds a dispatch, the parent's measured device-free
reading gets a second say: during an active chain that reading already reflects the chain's real
allocations, so it admits overlaps the ledger's worst-case reserve would needlessly hold. The path is
fenced off under WDDM demand-paging (the driver's free figure is a lie there) and requires a fixed margin
over the reserve, the sampling peak, and any not-yet-allocated pending chain's predicted reserve.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling import inference_scheduler as _sched_mod
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.workload_flow import POST_PROCESS_RESERVE_FLOW
from tests.process_management.conftest import make_job_pop_response, make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

# The scheduler's default VRAM reserve until a live config value is read; the measured path subtracts it.
_RESERVE_MB = 2048.0
_SAMPLING_PEAK_MB = 8000.0
_MEASURED_MARGIN_MB = _sched_mod._PP_OVERLAP_MEASURED_MARGIN_MB


def _make_scheduler_with_active_chain(
    *,
    device_free_mb: float | None,
) -> tuple[InferenceScheduler, HordeProcessInfo]:
    """Build a scheduler whose active post-processing chain has committed VRAM and a busy lane on card 0."""
    target_process = make_mock_process_info(0, state=HordeProcessState.PRELOADED_MODEL, device_index=0)
    pp_process = make_mock_process_info(
        7,
        model_name=None,
        state=HordeProcessState.POST_PROCESSING,
        process_type=HordeProcessType.POST_PROCESS,
        device_index=0,
    )
    process_map = ProcessMap({0: target_process, 7: pp_process})
    scheduler = _make_inference_scheduler(process_map=process_map, device_free_mb=device_free_mb)
    # A committed post-processing reserve makes the active-chain branch live (~6GB worst-case hold).
    scheduler._reserve_ledger.set(POST_PROCESS_RESERVE_FLOW, "job-pp", vram_mb=6000.0)
    # Static reported-total accounting withholds: this test exercises the measured second path given a miss.
    scheduler.pp_sampling_coresidency_affordable = Mock(return_value=False)  # type: ignore[method-assign]
    # Pin the sampling peak so the measured arithmetic is deterministic, independent of the peak estimator.
    scheduler._sampling_peak_mb = lambda _job: _SAMPLING_PEAK_MB  # type: ignore[method-assign]
    return scheduler, target_process


def _make_scheduler_with_pending_chain(
    *,
    device_free_mb: float | None,
    pending_reserve_mb: float,
) -> tuple[InferenceScheduler, HordeProcessInfo]:
    """Build a scheduler with no active chain but a feasible pending chain of the given predicted reserve."""
    target_process = make_mock_process_info(0, state=HordeProcessState.PRELOADED_MODEL, device_index=0)
    process_map = ProcessMap({0: target_process})
    scheduler = _make_inference_scheduler(process_map=process_map, device_free_mb=device_free_mb)
    scheduler.pp_sampling_coresidency_affordable = Mock(return_value=False)  # type: ignore[method-assign]
    scheduler._sampling_peak_mb = lambda _job: _SAMPLING_PEAK_MB  # type: ignore[method-assign]
    # No active reserve: force the pending branch with a fixed predicted reserve.
    scheduler._pending_post_processing_reserve_mb = lambda *, device_index: pending_reserve_mb  # type: ignore[method-assign]
    return scheduler, target_process


class TestActiveChainMeasuredAdmission:
    """During an active chain, generous measured free admits an overlap the static gate withheld."""

    def test_generous_free_paging_quiet_admits(self) -> None:
        """Static-unaffordable plus ample measured free with paging quiet dispatches (not deferred)."""
        scheduler, target = _make_scheduler_with_active_chain(device_free_mb=20000.0)
        next_job = make_job_pop_response("stable_diffusion")

        assert scheduler._should_defer_dispatch_for_post_processing(next_job, process_with_model=target) is False

    def test_paging_active_holds(self) -> None:
        """Under active WDDM paging the driver's free figure is untrustworthy, so the static fence stands."""
        scheduler, target = _make_scheduler_with_active_chain(device_free_mb=20000.0)
        scheduler.note_wddm_paging({100001: 512.0}, active=True)
        next_job = make_job_pop_response("stable_diffusion")

        assert scheduler._should_defer_dispatch_for_post_processing(next_job, process_with_model=target) is True

    def test_missing_measured_reading_holds(self) -> None:
        """With no measured reading the measured path is unavailable, preserving today's static-hold behavior."""
        scheduler, target = _make_scheduler_with_active_chain(device_free_mb=None)
        next_job = make_job_pop_response("stable_diffusion")

        assert scheduler._should_defer_dispatch_for_post_processing(next_job, process_with_model=target) is True

    def test_margin_boundary_one_short_holds(self) -> None:
        """Measured free one MB short of clearing the margin stays deferred."""
        one_short = _RESERVE_MB + _SAMPLING_PEAK_MB + _MEASURED_MARGIN_MB - 1.0
        scheduler, target = _make_scheduler_with_active_chain(device_free_mb=one_short)
        next_job = make_job_pop_response("stable_diffusion")

        assert scheduler._should_defer_dispatch_for_post_processing(next_job, process_with_model=target) is True

    def test_margin_boundary_exact_admits(self) -> None:
        """Measured free exactly at the margin admits (the boundary is inclusive)."""
        exact = _RESERVE_MB + _SAMPLING_PEAK_MB + _MEASURED_MARGIN_MB
        scheduler, target = _make_scheduler_with_active_chain(device_free_mb=exact)
        next_job = make_job_pop_response("stable_diffusion")

        assert scheduler._should_defer_dispatch_for_post_processing(next_job, process_with_model=target) is False


class TestPendingChainMeasuredAdmission:
    """A not-yet-allocated pending chain's predicted reserve is still charged against measured free."""

    def test_pending_reserve_is_charged(self) -> None:
        """Free that would admit an active chain (charge 0) defers a pending chain once its reserve is charged."""
        pending_reserve_mb = 4000.0
        # Enough for the active-branch arithmetic (reserve + peak + margin) but not once the pending reserve
        # is also subtracted.
        free_mb = _RESERVE_MB + _SAMPLING_PEAK_MB + _MEASURED_MARGIN_MB + 1000.0
        scheduler, target = _make_scheduler_with_pending_chain(
            device_free_mb=free_mb,
            pending_reserve_mb=pending_reserve_mb,
        )
        next_job = make_job_pop_response("stable_diffusion")

        assert scheduler._should_defer_dispatch_for_post_processing(next_job, process_with_model=target) is True

    def test_pending_admits_with_room_for_its_reserve(self) -> None:
        """With free covering the reserve, peak, pending reserve, and margin, the pending overlap admits."""
        pending_reserve_mb = 4000.0
        free_mb = _RESERVE_MB + _SAMPLING_PEAK_MB + pending_reserve_mb + _MEASURED_MARGIN_MB
        scheduler, target = _make_scheduler_with_pending_chain(
            device_free_mb=free_mb,
            pending_reserve_mb=pending_reserve_mb,
        )
        next_job = make_job_pop_response("stable_diffusion")

        assert scheduler._should_defer_dispatch_for_post_processing(next_job, process_with_model=target) is False


class TestEdgeTriggeredAdmissionLog:
    """A dispatch admitted via the measured path logs once, not once per scheduler tick."""

    def test_measured_admit_logs_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repeated admit-via-measured passes emit a single INFO, matching the edge-log latch."""
        scheduler, target = _make_scheduler_with_active_chain(device_free_mb=20000.0)
        next_job = make_job_pop_response("stable_diffusion")
        fake_logger = Mock()
        monkeypatch.setattr(_sched_mod, "logger", fake_logger)

        for _ in range(3):
            assert scheduler._should_defer_dispatch_for_post_processing(next_job, process_with_model=target) is False

        measured_admit_logs = [
            call for call in fake_logger.info.call_args_list if "via measured device truth" in call.args[0]
        ]
        assert len(measured_admit_logs) == 1
        assert scheduler._pp_mutex_measured_admit_logged is True


class TestHoldLivenessEscape:
    """A genuine mutex hold (static and measured both refuse) is bounded: it clears when the chain completes."""

    def test_hold_dispatches_once_chain_completes(self) -> None:
        """A head held by insufficient measured free dispatches once the active chain releases the card."""
        # Measured free too low for either the static or measured path: a genuine hold.
        scheduler, target = _make_scheduler_with_active_chain(device_free_mb=4000.0)
        next_job = make_job_pop_response("stable_diffusion")
        assert scheduler._should_defer_dispatch_for_post_processing(next_job, process_with_model=target) is True

        # The chain completes: its committed reserve is released and its lane is no longer busy.
        scheduler._reserve_ledger.release(POST_PROCESS_RESERVE_FLOW, "job-pp")
        for process_info in scheduler._process_map.values():
            if process_info.process_type == HordeProcessType.POST_PROCESS:
                process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB

        assert scheduler._should_defer_dispatch_for_post_processing(next_job, process_with_model=target) is False
