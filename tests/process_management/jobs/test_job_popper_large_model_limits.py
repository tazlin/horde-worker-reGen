"""Integration of the large-model pop limiters into JobPopper's offer filtering.

These exercise ``_apply_large_model_pop_limits`` on a real JobPopper: the offered model set is filtered by
the governor using the popper's live process map (loaded models), job tracker (queue + idle escape), and the
injected whole-card residency-lease accessor. The named VRAM-heavy checkpoints classify as large without any
model metadata, so these use those names directly.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_popper import JobPopper
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_api_sessions,
    make_test_runtime_config,
    track_popped_job_async,
)

_FLUX = "Flux.1-Schnell fp8 (Compact)"
_CASCADE = "Stable Cascade 1.0"
_SMALL = "stable_diffusion"


def _make_popper(
    *,
    process_map: ProcessMap | None = None,
    job_tracker: JobTracker | None = None,
    bridge_data: Mock | None = None,
    whole_card_residency_active: Callable[[], bool] | None = None,
) -> JobPopper:
    """A JobPopper with mostly-mocked dependencies, for exercising the large-model offer filter."""
    return JobPopper(
        state=WorkerState(),
        process_map=process_map if process_map is not None else ProcessMap({}),
        job_tracker=job_tracker if job_tracker is not None else JobTracker(),
        shutdown_manager=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data or make_mock_bridge_data()),
        api_sessions=make_test_api_sessions(horde_client_session=Mock(), aiohttp_session=Mock()),
        max_inference_processes=2,
        max_concurrent_inference_processes=1,
        whole_card_residency_active=whole_card_residency_active,
    )


def _process_map_with_loaded(model_name: str) -> ProcessMap:
    proc = make_mock_process_info(0, model_name=model_name, state=HordeProcessState.WAITING_FOR_JOB)
    return ProcessMap({0: proc})


class TestSwitchThrottleFiltersOffer:
    """A different large model is dropped from the offer while one is already loaded."""

    async def test_different_large_model_withheld_small_models_kept(self) -> None:
        """With Flux in play and the switch throttle on, Cascade is dropped from the offer; Flux/small stay."""
        bridge_data = make_mock_bridge_data(large_model_switch_min_seconds=30)
        # A queued Flux job keeps the worker non-idle (so the idle escape does not fire) and marks Flux as the
        # large model in play.
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX))
        popper = _make_popper(
            process_map=_process_map_with_loaded(_FLUX),
            job_tracker=job_tracker,
            bridge_data=bridge_data,
        )

        result = popper._apply_large_model_pop_limits({_SMALL, _CASCADE, _FLUX}, bridge_data)

        assert result == {_SMALL, _FLUX}, "the different large model (Cascade) must be withheld; Flux/small kept"

    async def test_disabled_by_default_offers_everything(self) -> None:
        """With the default config (both limiters off) every model is offered."""
        bridge_data = make_mock_bridge_data()  # switch=0, reentry=-1 inheriting cooldown 0 -> both disabled
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX))
        popper = _make_popper(
            process_map=_process_map_with_loaded(_FLUX),
            job_tracker=job_tracker,
            bridge_data=bridge_data,
        )

        result = popper._apply_large_model_pop_limits({_SMALL, _CASCADE, _FLUX}, bridge_data)

        assert result == {_SMALL, _CASCADE, _FLUX}


class TestIdleEscapeAtPopper:
    """A worker holding no local work offers large models even mid-window."""

    def test_idle_worker_offers_all_models(self) -> None:
        """A resident large model with an empty local queue does not suppress the offer (idle escape)."""
        bridge_data = make_mock_bridge_data(large_model_switch_min_seconds=30)
        # Flux is resident but the local queue is empty (num_jobs_total == 0): the idle escape fires.
        popper = _make_popper(
            process_map=_process_map_with_loaded(_FLUX),
            job_tracker=JobTracker(),
            bridge_data=bridge_data,
        )

        result = popper._apply_large_model_pop_limits({_SMALL, _CASCADE, _FLUX}, bridge_data)

        assert result == {_SMALL, _CASCADE, _FLUX}


class TestReentryDurationResolution:
    """The -1 sentinel inherits the whole-card residency cooldown; explicit values win."""

    def test_negative_inherits_residency_cooldown(self) -> None:
        """A -1 re-entry value resolves to the configured whole-card residency cooldown."""
        bridge_data = make_mock_bridge_data(
            large_model_switch_min_seconds=5,
            large_model_reentry_cooldown_seconds=-1,
            whole_card_residency_cooldown_seconds=45,
        )
        popper = _make_popper(bridge_data=bridge_data)
        assert popper._resolve_large_model_pop_durations(bridge_data) == (5.0, 45.0)

    def test_explicit_value_overrides(self) -> None:
        """A non-negative re-entry value is used as-is, not the residency cooldown."""
        bridge_data = make_mock_bridge_data(
            large_model_switch_min_seconds=0,
            large_model_reentry_cooldown_seconds=20,
            whole_card_residency_cooldown_seconds=45,
        )
        popper = _make_popper(bridge_data=bridge_data)
        assert popper._resolve_large_model_pop_durations(bridge_data) == (0.0, 20.0)
