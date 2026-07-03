"""Tests for budget-gated VRAM retention between same-model jobs.

hordelib evicts a job's model from VRAM after every run so sibling GPU instances never collectively
over-commit. That eviction forces a RAM->VRAM reload on the next job, the dominant non-sampling cost on
small jobs. :meth:`InferenceScheduler._should_keep_model_resident` decides when to suppress that eviction
for one dispatch: only when the next queued inference job reuses the model *and* the measured VRAM budget
confirms the card could still admit this job from scratch. The measured free figure is taken while the
job's own weights occupy the card, so the check credits them back; without that credit, retention is
unreachable on exactly the contended cards where the reload skip pays. Retention is granted on evidence
and a wrong call degrades to a reload (never an OOM, thanks to hordelib's force-load overflow backstop).
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeInferenceControlMessage
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.scheduling import inference_scheduler as inference_scheduler_module
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MODEL = "WAI-NSFW-illustrious-SDXL"
_OTHER_MODEL = "CyberRealistic Pony"
_AMPLE_FREE_VRAM_MB = 12000.0


def _budget_on_scheduler(
    job_tracker: JobTracker,
    *,
    free_vram_mb: float | None = _AMPLE_FREE_VRAM_MB,
) -> InferenceScheduler:
    """A scheduler with the VRAM budget active and a controllable measured-free-VRAM reading."""
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=2048,
        ram_reserve_mb=4096,
    )
    scheduler = _make_inference_scheduler(job_tracker=job_tracker, bridge_data=bridge_data)
    scheduler._measured_free_vram_mb = Mock(return_value=free_vram_mb)  # type: ignore[method-assign]
    return scheduler


async def test_retains_when_same_model_queued_and_budget_fits() -> None:
    """Another queued job reuses the model and VRAM fits, so the model stays resident for the reload skip."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await track_popped_job_async(job_tracker, make_job_pop_response(model=_MODEL))

    scheduler = _budget_on_scheduler(job_tracker)

    assert scheduler._should_keep_model_resident(dispatched, device_index=None) is True


async def test_no_retain_when_next_queued_model_differs() -> None:
    """The only other queued job needs a different model, so retaining this one would idly pin VRAM."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await track_popped_job_async(job_tracker, make_job_pop_response(model=_OTHER_MODEL))

    scheduler = _budget_on_scheduler(job_tracker)

    assert scheduler._should_keep_model_resident(dispatched, device_index=None) is False


async def test_no_retain_when_no_other_job_queued() -> None:
    """With nothing else queued there is no imminent reuse to justify holding the weights."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)

    assert scheduler._should_keep_model_resident(dispatched, device_index=None) is False


async def test_no_retain_when_budget_inactive() -> None:
    """Without the VRAM budget the worker cannot vouch for the headroom, so it evicts as before."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await track_popped_job_async(job_tracker, make_job_pop_response(model=_MODEL))

    # Default mock bridge_data leaves enable_vram_budget unset (non-bool), so _budget_active() is False.
    scheduler = _make_inference_scheduler(job_tracker=job_tracker)
    scheduler._measured_free_vram_mb = Mock(return_value=_AMPLE_FREE_VRAM_MB)  # type: ignore[method-assign]

    assert scheduler._should_keep_model_resident(dispatched, device_index=None) is False


async def test_no_retain_when_free_vram_unmeasured() -> None:
    """A cold start with no VRAM telemetry must not assume headroom; retention is evidence-gated."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await track_popped_job_async(job_tracker, make_job_pop_response(model=_MODEL))

    scheduler = _budget_on_scheduler(job_tracker, free_vram_mb=None)

    assert scheduler._should_keep_model_resident(dispatched, device_index=None) is False


async def test_budget_check_credits_resident_weights(monkeypatch) -> None:  # noqa: ANN001
    """The fit check runs against free VRAM plus the job's own resident weights.

    The free reading is taken while the dispatched model occupies the card; asking the footprint to
    fit inside the remainder would charge the weights twice and deny retention on any busy card.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await track_popped_job_async(job_tracker, make_job_pop_response(model=_MODEL))

    scheduler = _budget_on_scheduler(job_tracker, free_vram_mb=3000.0)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_weight_mb", lambda job, baseline: 5000.0)
    seen_free: list[float] = []

    def record_check(job, baseline, free_vram_mb, committed_reserve_mb=0.0):  # noqa: ANN001, ANN202
        seen_free.append(free_vram_mb)
        return Mock(fits=True)

    scheduler._vram_budget.check_job = record_check  # type: ignore[method-assign]

    assert scheduler._should_keep_model_resident(dispatched, device_index=None) is True
    assert seen_free == [8000.0]


async def test_no_retain_when_budget_rejects_footprint() -> None:
    """Under VRAM pressure the budget says the model does not fit, so retention is refused (would starve a swap)."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await track_popped_job_async(job_tracker, make_job_pop_response(model=_MODEL))

    scheduler = _budget_on_scheduler(job_tracker)
    scheduler._vram_budget.check_job = Mock(return_value=Mock(fits=False))  # type: ignore[method-assign]

    assert scheduler._should_keep_model_resident(dispatched, device_index=None) is False


def test_inference_control_message_defaults_to_eviction() -> None:
    """The dispatch message preserves today's aggressive eviction unless the scheduler opts in."""
    message = HordeInferenceControlMessage(
        control_flag=HordeControlFlag.START_INFERENCE,
        horde_model_name=_MODEL,
        sdk_api_job_info=make_job_pop_response(model=_MODEL),
    )

    assert message.keep_model_resident_after is False
