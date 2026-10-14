"""The dispatch world's workload inputs: served-class weights, stated footprints, per-job seconds, a seeded store.

Each knob is off by default, so every other row runs on the classes, predictors and rates the world carries. A
row here sets one knob on a one-lane card and reads back from the world's own records that the card was charged,
or the job was timed, as the knob stated. The lane start latency is read on a two-lane pool, since it applies to a
lane the pool regrows.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.resources.vram_footprints import (
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
)
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_24GB,
    _KNOWN_MODEL_CLASSES,
    _SD15,
    _DispatchWorld,
    _JobWindow,
    _ModelClass,
)

pytestmark = pytest.mark.closed_loop

_TICK_SECONDS = 1.0
"""Fine enough that a phase of a few seconds spans several ticks, so a charge is observed inside its phase."""
_MAX_TICKS = 60
"""Ample for one or two jobs on an idle card at one-second ticks."""
_CARD_INDEX = 0

_UNLISTED_SDXL = _ModelClass(
    "unlisted_sdxl",
    "world-inputs-unlisted-sdxl",
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
    6100.0,
)
"""A served class absent from the module's known classes, at a weight no known class carries."""


@dataclass(frozen=True)
class _Served:
    """Represents what the card carried while one job sampled and once its lane had let it go."""

    window: _JobWindow
    committed_while_sampling_mb: float
    resident_while_sampling_mb: dict[int, float]
    committed_after_release_mb: float
    resident_after_release_mb: dict[int, float]
    released_at: float

    @property
    def released_activation_mb(self) -> float:
        """Return the job's sampling transient, what the card let go at release beyond weights.

        A lane without a retention grant returns its weights at the job's end, so the weights on either side are
        taken out of the committed figures before they are compared.
        """
        while_sampling = self.committed_while_sampling_mb - sum(self.resident_while_sampling_mb.values())
        after_release = self.committed_after_release_mb - sum(self.resident_after_release_mb.values())
        return while_sampling - after_release


def _one_lane_world(
    *,
    extra_model_classes: tuple[_ModelClass, ...] = (),
    true_footprint_mb: Callable[[ImageGenerateJobPopResponse], tuple[float, float]] | None = None,
    footprint_undershoot: float = 1.0,
    sample_seconds_by_job: Mapping[str, float] | None = None,
    decode_seconds_by_job: Mapping[str, float] | None = None,
    footprint_store: LearnedFootprintStore | None = None,
    learned_footprints: bool = False,
) -> _DispatchWorld:
    """One closed-loop lane on a 24 GB card, serial, with the knob under test."""
    return _DispatchWorld(
        card=_CARD_24GB,
        lane_count=1,
        max_threads=1,
        queue_depth=2,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        extra_model_classes=extra_model_classes,
        true_footprint_mb=true_footprint_mb,
        footprint_undershoot=footprint_undershoot,
        sample_seconds_by_job=sample_seconds_by_job,
        decode_seconds_by_job=decode_seconds_by_job,
        footprint_store=footprint_store,
        learned_footprints=learned_footprints,
    )


async def _serve(world: _DispatchWorld, job: ImageGenerateJobPopResponse) -> _Served:
    """Pop ``job``, step until its lane releases it, and return what the card carried along the way."""
    await world.pop(job)
    job_id = str(job.id_)
    committed_while_sampling_mb: float | None = None
    resident_while_sampling_mb: dict[int, float] = {}
    for _ in range(_MAX_TICKS):
        await world.step()
        window = world.job_windows.get(job_id)
        if window is None or window.sample_until is None:
            continue
        if committed_while_sampling_mb is None and world.now < window.sample_until:
            committed_while_sampling_mb = world.card_committed_mb(_CARD_INDEX)
            resident_while_sampling_mb = world.card_resident_mb(_CARD_INDEX)
        if world.lane_serving(job) is None:
            assert committed_while_sampling_mb is not None, (
                f"job {job_id} finished sampling before any tick observed it. {world.state_dump()}"
            )
            return _Served(
                window=window,
                committed_while_sampling_mb=committed_while_sampling_mb,
                resident_while_sampling_mb=resident_while_sampling_mb,
                committed_after_release_mb=world.card_committed_mb(_CARD_INDEX),
                resident_after_release_mb=world.card_resident_mb(_CARD_INDEX),
                released_at=world.now,
            )
    raise AssertionError(f"job {job_id} was not released within {_MAX_TICKS} ticks. {world.state_dump()}")


async def test_a_served_class_outside_the_known_classes_is_charged_its_weights() -> None:
    """A class passed in ``extra_model_classes`` loads and samples resident at its own ``weights_mb``."""
    assert _UNLISTED_SDXL not in _KNOWN_MODEL_CLASSES
    world = _one_lane_world(extra_model_classes=(_UNLISTED_SDXL,))
    job = make_job_pop_response(_UNLISTED_SDXL.name, width=1024, height=1024, ddim_steps=30)

    served = await _serve(world, job)

    assert served.window.loaded_now, "the job's weights were never loaded onto the card"
    assert list(served.resident_while_sampling_mb.values()) == [pytest.approx(_UNLISTED_SDXL.weights_mb)], (
        f"the card carries {served.resident_while_sampling_mb} for a class weighing {_UNLISTED_SDXL.weights_mb} MB"
    )


async def test_a_stated_footprint_is_charged_with_the_undershoot_on_top() -> None:
    """The truth hook sets the resident weights and the sampling transient, both scaled by the undershoot."""
    true_weights_mb, true_transient_mb, undershoot = 7000.0, 2500.0, 1.2
    asked: list[str] = []

    def true_footprint_mb(job: ImageGenerateJobPopResponse) -> tuple[float, float]:
        asked.append(str(job.id_))
        return true_weights_mb, true_transient_mb

    world = _one_lane_world(true_footprint_mb=true_footprint_mb, footprint_undershoot=undershoot)
    job = make_job_pop_response(_SD15.name, width=512, height=512, ddim_steps=30)

    served = await _serve(world, job)

    assert str(job.id_) in asked, "the world never consulted the truth hook for the job"
    assert list(served.resident_while_sampling_mb.values()) == [pytest.approx(true_weights_mb * undershoot)], (
        f"the card carries {served.resident_while_sampling_mb} as weights, not the stated {true_weights_mb} MB "
        f"scaled by {undershoot}"
    )
    released_transient_mb = served.released_activation_mb
    assert released_transient_mb == pytest.approx(true_transient_mb * undershoot), (
        f"sampling held {released_transient_mb} MB beyond the weights, not the stated {true_transient_mb} MB "
        f"scaled by {undershoot}"
    )


async def test_per_job_seconds_replace_the_rates_for_the_jobs_they_name() -> None:
    """A named job samples and decodes for its stated seconds, and an unnamed one keeps the per-baseline rates."""
    named = make_job_pop_response(_SD15.name, width=512, height=512, ddim_steps=30)
    unnamed = make_job_pop_response(_SD15.name, width=512, height=512, ddim_steps=30)
    sample_seconds, decode_seconds = 7.5, 4.25
    world = _one_lane_world(
        sample_seconds_by_job={str(named.id_): sample_seconds},
        decode_seconds_by_job={str(named.id_): decode_seconds},
    )

    served_named = await _serve(world, named)
    served_unnamed = await _serve(world, unnamed)

    named_window = served_named.window
    assert named_window.sample_from is not None and named_window.sample_until is not None
    assert named_window.sample_until - named_window.sample_from == pytest.approx(sample_seconds)
    decode_until = named_window.sample_until + decode_seconds
    assert decode_until <= served_named.released_at < decode_until + _TICK_SECONDS, (
        f"the named job was released at {served_named.released_at}, not on the first tick at or after its "
        f"stated decode end {decode_until}"
    )
    unnamed_window = served_unnamed.window
    assert unnamed_window.sample_from is not None and unnamed_window.sample_until is not None
    # stable_diffusion_1 samples at 0.06 s per step per megapixel: 30 steps over a quarter megapixel.
    assert unnamed_window.sample_until - unnamed_window.sample_from == pytest.approx(30 * 0.06 * 0.25)


async def test_a_seeded_store_is_the_parents_prior_and_keeps_learning() -> None:
    """The store a row hands in is the one the scheduler prices from, and the children's reports feed it."""
    seed_key = FootprintKey(
        model_baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl.value,
        resolution_bucket=None,
        platform=sys.platform,
        stage=FootprintStage.RESIDENT,
        checkpoint="world-inputs-unserved-checkpoint",
    )
    store = LearnedFootprintStore()
    store.observe_peak(seed_key, 6500.0)
    world = _one_lane_world(footprint_store=store, learned_footprints=True)
    job = make_job_pop_response(_SD15.name, width=512, height=512, ddim_steps=30)

    assert world.footprint_store is store
    assert world.scheduler._footprint_store is store, "the scheduler prices from a store other than the seeded one"
    await _serve(world, job)

    assert store.observation_count(seed_key) == 1, "the seeded observation did not survive the run"
    assert len(store) > 1, "the children's reports never reached the seeded store"


def _lane_state(world: _DispatchWorld, lane_id: int) -> HordeProcessState:
    """Return the state the world's process map holds for ``lane_id``."""
    return world._process_map[lane_id].last_process_state


@pytest.mark.parametrize("lane_start_seconds", [0.0, 3.0], ids=["instant", "three_seconds"])
async def test_a_regrown_lane_charges_its_context_once_started(lane_start_seconds: float) -> None:
    """A regrown lane is starting, with no context on the card, until its start latency has passed.

    A live process passes ``PROCESS_STARTING`` before its CUDA context exists, so the card carries the lane only
    once the start completes. Zero, the default, adds the lane waiting for jobs with its context charged at once.
    """
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=2,
        max_threads=1,
        queue_depth=2,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        lane_start_seconds=lane_start_seconds,
    )
    two_lane_committed_mb = world.card_committed_mb(_CARD_INDEX)
    assert world.scale_inference_processes(1) == 1
    one_lane_committed_mb = world.card_committed_mb(_CARD_INDEX)
    assert one_lane_committed_mb < two_lane_committed_mb
    kept = set(world.inference_lane_ids())
    assert world.scale_inference_processes(2) == 2
    (regrown,) = set(world.inference_lane_ids()) - kept

    if lane_start_seconds == 0.0:
        assert _lane_state(world, regrown) == HordeProcessState.WAITING_FOR_JOB
        assert world.card_committed_mb(_CARD_INDEX) == pytest.approx(two_lane_committed_mb)
        return
    assert _lane_state(world, regrown) == HordeProcessState.PROCESS_STARTING
    assert world.card_committed_mb(_CARD_INDEX) == pytest.approx(one_lane_committed_mb), (
        "a lane still starting was charged its context"
    )
    started_at = world.now + lane_start_seconds
    while world.now < started_at - _TICK_SECONDS / 2:
        await world.step()
        if world.now < started_at - _TICK_SECONDS / 2:
            assert _lane_state(world, regrown) == HordeProcessState.PROCESS_STARTING
            assert world.card_committed_mb(_CARD_INDEX) == pytest.approx(one_lane_committed_mb)
    assert _lane_state(world, regrown) == HordeProcessState.WAITING_FOR_JOB, f"{world.state_dump()}"
    assert world.card_committed_mb(_CARD_INDEX) == pytest.approx(two_lane_committed_mb)
