"""Reachability of the measured-load attempt: a servable head must not take a terminal or unbounded exit.

The arbiter judges a candidate against a conservative static prediction and an instantaneous device-free
reading, and it holds one escape hatch for the regime where that arithmetic is the least trustworthy: a head
at a card the worker has nothing left to reclaim on gets one real load, so measured reality decides rather
than the prediction. Two preconditions on that hatch used to make it unreachable for heads the card
demonstrably serves, and both exits were the wrong kind: one terminal, one unbounded.

- A candidate over the achievable ceiling by a within-allowance margin was refused the hatch while the card
  still carried an idle sibling context, because the card was not yet converged-empty. The terminal ceiling
  DENY is evaluated ahead of the idle-context teardown rung, so the head could never reach the state that
  would make its own attempt eligible: it was faulted for reissue and its model put on the ceiling hold. The
  identical head on a pool the residency forecast happened to collapse first was served, which made
  servability a side effect of the pool's shape rather than a fact about the card.
- A candidate under the ceiling but missing available by more than the uncertainty band was refused the hatch
  on a converged-empty card. There, available cannot improve: nothing is left to reclaim, so the deferral has
  nothing to wait for and the head parks until the recovery supervisor takes it.

Both are held here against the modelled card the bounded-dispatch matrix drives, so the assertions are about a
scheduler actuating a real teardown and a real load rather than about arbiter arithmetic in isolation. The
control row holds the other side: a candidate past the allowance on a converged card still takes the terminal
exit, so the hatch is not a licence to bang an impossible demand into the card forever.
"""

from __future__ import annotations

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from horde_worker_regen.process_management.resources.vram_arbiter import _MEASURED_ATTEMPT_BAND_MB
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness.test_bounded_dispatch_matrix import (
    _CARD_8GB,
    _FLUX,
    _SDXL,
    _DispatchWorld,
    _ModelClass,
)

_TICK_SECONDS = 10.0
"""Short enough to sample the ten-second teardown grace and the sixty-second starvation horizon separately.

Both preconditions on the attempt are stated in seconds, and a row that stepped over either in one advance
could not distinguish "served after waiting out the grace" from "served immediately"."""

_SERVICE_BOUND_TICKS = 15
"""Ticks a servable head is allowed before it must have reached sampling.

Covers the longest legitimate path to the attempt: the starvation horizon, then the teardown the deficit
warrants, then the load itself. Generous against that path so the row fails on a head that never arrives
rather than on a head that arrived a tick later than the arithmetic predicted."""

_TERMINAL_BOUND_TICKS = 15
"""Ticks the control row drives, matching the service rows so the two outcomes are compared over one span."""


def _world(*, lane_count: int) -> _DispatchWorld:
    """Build the modelled 8 GB card with a single sampling slot and ``lane_count`` inference lanes."""
    return _DispatchWorld(
        card=_CARD_8GB,
        lane_count=lane_count,
        max_threads=1,
        queue_depth=3,
        tick_seconds=_TICK_SECONDS,
    )


def _candidate_mb(world: _DispatchWorld, job: ImageGenerateJobPopResponse) -> float:
    """Return what admission prices this job's load at, through the scheduler's own pricing seam."""
    assert job.model is not None
    baseline = world.scheduler._model_metadata.get_baseline(job.model)
    candidate_mb = world.scheduler._measured_admission_candidate_delta_mb(
        job,
        baseline,
        process_id=None,
        disaggregated=False,
    )
    assert candidate_mb is not None, "the row needs a priceable candidate to state its regime"
    return candidate_mb


def _ceiling_mb(world: _DispatchWorld) -> float:
    """Return the card's achievable ceiling, read from the same frozen snapshot the arbiter judges against."""
    snapshot = world.scheduler.build_vram_arbiter_snapshot(device_free_mb_by_device={0: world.device_free_mb()})
    state = snapshot.device(None)
    assert state is not None
    ceiling_mb = state.achievable_ceiling_mb()
    assert ceiling_mb is not None, "the row needs a sized card to state its regime"
    return ceiling_mb


async def _drive(world: _DispatchWorld, job: ImageGenerateJobPopResponse, *, ticks: int) -> None:
    """Pop the job and advance the world, leaving its outcome on the world for the row to read."""
    await world.pop(job)
    for _ in range(ticks):
        await world.step()


def _served_message(world: _DispatchWorld, complaint: str) -> str:
    """Return a failure message carrying the complaint and the card's end state."""
    return f"{complaint}\n  world: {world.state_dump()}"


class TestOverCeilingHeadOnAPopulatedCard:
    """A head marginally over the ceiling is driven to a converged card and served, not faulted for reissue."""

    async def test_the_regime_is_a_within_allowance_overshoot_on_a_populated_card(self) -> None:
        """The candidate clears the ceiling by a small margin while a second lane still holds a context."""
        world = _world(lane_count=2)
        job = make_job_pop_response(_SDXL.name, width=768, height=768, ddim_steps=12)
        candidate_mb = _candidate_mb(world, job)
        ceiling_mb = _ceiling_mb(world)
        overshoot_mb = candidate_mb - ceiling_mb
        # Over the ceiling, so the terminal DENY is the verdict in play, but by far less than the allowance
        # that decides whether one real load is worth trying.
        assert overshoot_mb > 0.0
        assert overshoot_mb <= min(ceiling_mb * 0.10, _MEASURED_ATTEMPT_BAND_MB)

    async def test_the_head_reaches_sampling_rather_than_the_ceiling_hold(self) -> None:
        """The idle sibling context is torn down and the head is served; nothing is faulted or held."""
        world = _world(lane_count=2)
        job = make_job_pop_response(_SDXL.name, width=768, height=768, ddim_steps=12)
        await _drive(world, job, ticks=_SERVICE_BOUND_TICKS)

        assert job.id_ is not None
        assert world.dispatch_tick(job) is not None, _served_message(
            world,
            f"the head ({job.model}) never reached sampling on a card that serves it once the pool converges",
        )
        assert world.job_tracker.was_faulted_by_scheduling_recovery(job.id_) is False, _served_message(
            world,
            f"the head ({job.model}) was handed back for reissue rather than served",
        )
        assert world.job_tracker.is_model_held_by_ceiling(_SDXL.name) is False, _served_message(
            world,
            f"{_SDXL.name} was put on the ceiling hold despite the card serving it",
        )


class TestConvergedEmptyHeadOutsideTheBand:
    """On a converged-empty card the shortfall's size cannot justify waiting: the head attempts or holds."""

    async def test_the_regime_is_a_beyond_band_shortfall_under_the_ceiling(self) -> None:
        """The candidate sits under the ceiling yet misses available by more than the uncertainty band."""
        world = _world(lane_count=1)
        job = make_job_pop_response(_SDXL.name, width=512, height=768, ddim_steps=12)
        candidate_mb = _candidate_mb(world, job)
        ceiling_mb = _ceiling_mb(world)
        snapshot = world.scheduler.build_vram_arbiter_snapshot(device_free_mb_by_device={0: world.device_free_mb()})
        state = snapshot.device(None)
        assert state is not None
        assert state.device_free_mb is not None
        available_mb = state.device_free_mb - state.noise_buffer_mb
        # Under the ceiling, so the demand is possible in principle and never structurally denied; past the
        # band, so the flat uncertainty test is what refuses the attempt.
        assert candidate_mb < ceiling_mb
        assert candidate_mb - available_mb > _MEASURED_ATTEMPT_BAND_MB

    async def test_the_head_reaches_sampling_rather_than_parking(self) -> None:
        """The head takes its one real load instead of deferring against a reading that cannot improve."""
        world = _world(lane_count=1)
        job = make_job_pop_response(_SDXL.name, width=512, height=768, ddim_steps=12)
        await _drive(world, job, ticks=_SERVICE_BOUND_TICKS)

        assert job.id_ is not None
        assert world.dispatch_tick(job) is not None, _served_message(
            world,
            f"the head ({job.model}) never reached sampling on a converged-empty card with nothing left to wait for",
        )
        assert world.stage(job) is not JobStage.PENDING_INFERENCE, _served_message(
            world,
            f"the head ({job.model}) was still waiting for inference at the end of the run",
        )


class TestImpossibleCandidateStillTakesTheTerminalExit:
    """The other side of the hatch: a demand past the allowance is refused, so the exit stays bounded."""

    async def test_the_regime_is_an_overshoot_past_the_allowance(self) -> None:
        """The candidate clears the ceiling by more than any real load could be expected to recover."""
        world = _world(lane_count=1)
        job = make_job_pop_response(_FLUX.name, width=512, height=512, ddim_steps=8)
        overshoot_mb = _candidate_mb(world, job) - _ceiling_mb(world)
        assert overshoot_mb > _MEASURED_ATTEMPT_BAND_MB

    @pytest.mark.parametrize("model", [_FLUX], ids=lambda model: model.label)
    async def test_the_model_is_held_and_its_job_handed_back(self, model: _ModelClass) -> None:
        """The model goes on the ceiling hold and its job is returned for reissue rather than banging the card."""
        world = _world(lane_count=1)
        job = make_job_pop_response(model.name, width=512, height=512, ddim_steps=8)
        await _drive(world, job, ticks=_TERMINAL_BOUND_TICKS)

        assert job.id_ is not None
        assert world.dispatch_tick(job) is None, _served_message(
            world,
            f"a demand past the attempt allowance was loaded onto the card anyway ({model.name})",
        )
        assert world.job_tracker.is_model_held_by_ceiling(model.name) is True, _served_message(
            world,
            f"{model.name} was not put on the ceiling hold despite exceeding what the card can ever offer",
        )
        assert world.stage(job) is not JobStage.PENDING_INFERENCE, _served_message(
            world,
            f"the impossible head ({model.name}) was left queued rather than handed back for reissue",
        )
