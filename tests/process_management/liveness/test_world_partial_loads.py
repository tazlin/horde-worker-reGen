"""The dispatch world's partial loads: offloads that outlive their job, decode unloads, and skip-listed loads.

Each row runs one closed-loop lane on a 24 GB card and states its jobs' true footprints, so the child's shortfall
arithmetic, and not the parent's forecast, decides what reaches the card. The rows read the world's own records:
each job's window, the lane's host-RAM remainder between jobs, and the support components the lane holds. Further
rows cover what funds an activation and a decode spike, the encoder released before the clearance wait, and the
activation reserve a grant carries into the load.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import override

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.resources.resource_budget import predict_job_decode_spike_mb
from horde_worker_regen.process_management.scheduling.admission.pricing import staging_charge_mb
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_24GB,
    _CHILD_FREE_MARGIN_MB,
    _FIRST_CONTEXT_MB,
    _FLUX,
    _LOAD_SECONDS_PER_GB,
    _SDXL,
    _DispatchWorld,
    _JobWindow,
    _ModelClass,
    _support_component_mb,
)

pytestmark = pytest.mark.closed_loop

_TICK_SECONDS = 1.0
"""Fine enough that a phase of a few seconds spans several ticks, so a lane is observed inside each phase."""
_MAX_TICKS = 200
"""Ample for a two-job streak of multi-gigabyte loads on an idle card at one-second ticks."""
_LANE_ROOM_MB = _CARD_24GB.total_mb - _FIRST_CONTEXT_MB
"""What one lane's process can hold on the card beside its own context."""


@dataclass(frozen=True)
class _ServedJob:
    """Represents one finished job of a streak, with what its lane held while it sampled and once it had finished."""

    window: _JobWindow
    support_while_sampling_mb: float
    offloaded_after_mb: float
    """The lane's host-RAM remainder of its model at the close of the tick the job left the lane.

    A successor dispatched on that same tick has already returned what it could, so only the streak's last job
    reads its own remainder here."""


def _one_lane_world(
    *,
    true_footprint_mb: Callable[[ImageGenerateJobPopResponse], tuple[float, float]],
    partial_loads: bool = True,
    phase_resident_components: bool = False,
    clearance_lease: bool = False,
) -> _DispatchWorld:
    """One closed-loop lane on a 24 GB card, serial, its jobs charged their stated footprints."""
    return _DispatchWorld(
        card=_CARD_24GB,
        lane_count=1,
        max_threads=1,
        queue_depth=2,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        true_footprint_mb=true_footprint_mb,
        phase_resident_components=phase_resident_components,
        clearance_lease=clearance_lease,
        partial_loads=partial_loads,
    )


def _stated(weights_mb: float, transient_mb: float) -> Callable[[ImageGenerateJobPopResponse], tuple[float, float]]:
    """Return a truth hook stating the same footprint for every job."""

    def true_footprint_mb(_job: ImageGenerateJobPopResponse) -> tuple[float, float]:
        return weights_mb, transient_mb

    return true_footprint_mb


def _job(model: _ModelClass) -> ImageGenerateJobPopResponse:
    """A one-megapixel, thirty-step job of ``model``."""
    return make_job_pop_response(model.name, width=1024, height=1024, ddim_steps=30)


async def _serve_streak(world: _DispatchWorld, jobs: list[ImageGenerateJobPopResponse]) -> list[_ServedJob]:
    """Pop every job of a streak at once, step until each has left its lane, and return what each left behind."""
    for job in jobs:
        await world.pop(job)
    served: dict[str, _ServedJob] = {}
    support_while_sampling: dict[str, float] = {}
    for _ in range(_MAX_TICKS):
        await world.step()
        for job in jobs:
            job_id = str(job.id_)
            window = world.job_windows.get(job_id)
            if window is None or job_id in served:
                continue
            sampling = window.sample_from is not None and window.sample_from <= world.now
            if sampling and job_id not in support_while_sampling:
                support_while_sampling[job_id] = world._support_resident_mb.get(window.lane_id, 0.0)
            if window.retention_granted is None:
                continue
            served[job_id] = _ServedJob(
                window=window,
                support_while_sampling_mb=support_while_sampling.get(job_id, 0.0),
                offloaded_after_mb=world._offloaded_mb.get(window.lane_id, 0.0),
            )
        if len(served) == len(jobs):
            return [served[str(job.id_)] for job in jobs]
    raise AssertionError(f"the streak did not finish within {_MAX_TICKS} ticks. {world.state_dump()}")


def _return_seconds(returned_mb: float) -> float:
    """The load seconds the world charges for bringing ``returned_mb`` back onto the card."""
    return (returned_mb / 1024.0) * _LOAD_SECONDS_PER_GB


# A checkpoint that fits the lane with the margin to spare, and an activation that does not fit beside it, so the
# child funds the activation by leaving part of its weights in host RAM.
_STREAK_WEIGHTS_MB = 20_000.0
_STREAK_TRANSIENT_MB = 4_000.0
_STREAK_OFFLOAD_MB = _STREAK_TRANSIENT_MB + _CHILD_FREE_MARGIN_MB - (_LANE_ROOM_MB - _STREAK_WEIGHTS_MB)


async def test_an_offload_outlives_its_job_and_the_next_job_pays_to_return_it() -> None:
    """A same-model streak under a sampling shortfall carries the offload across jobs and pays for its return."""
    world = _one_lane_world(true_footprint_mb=_stated(_STREAK_WEIGHTS_MB, _STREAK_TRANSIENT_MB))

    first, second = await _serve_streak(world, [_job(_SDXL), _job(_SDXL)])

    assert first.window.retention_granted, "the streak's first job was not retained, so nothing can carry over"
    assert first.window.loaded_now and first.window.load_moved_mb == pytest.approx(_STREAK_WEIGHTS_MB)
    assert not second.window.loaded_now, "the second job reloaded the checkpoint its lane was retaining"
    assert second.window.load_returned_mb == pytest.approx(_STREAK_OFFLOAD_MB)
    assert second.window.load_moved_mb == pytest.approx(_STREAK_OFFLOAD_MB)
    assert second.window.load_seconds == pytest.approx(_return_seconds(_STREAK_OFFLOAD_MB)), (
        f"the return of {_STREAK_OFFLOAD_MB} MB took {second.window.load_seconds} s"
    )
    assert second.offloaded_after_mb == pytest.approx(_STREAK_OFFLOAD_MB), (
        "the second job's activation did not push the same weights off the card again"
    )


async def test_without_partial_loads_a_job_forgets_its_offload() -> None:
    """Off, the same streak drops the offload at the first job's end and the second job loads nothing."""
    world = _one_lane_world(
        true_footprint_mb=_stated(_STREAK_WEIGHTS_MB, _STREAK_TRANSIENT_MB),
        partial_loads=False,
    )

    first, second = await _serve_streak(world, [_job(_SDXL), _job(_SDXL)])

    assert first.window.retention_granted
    assert second.offloaded_after_mb == 0.0
    assert not second.window.loaded_now
    assert second.window.load_seconds == 0.0
    assert second.window.load_returned_mb == 0.0
    assert second.window.load_moved_mb == 0.0


async def test_a_decode_unloads_the_lanes_own_weights_and_the_next_job_returns_them() -> None:
    """A decode spike the believed free does not cover is funded from the lane's weights, then paid back on reuse."""
    probe = _job(_SDXL)
    decode_spike_mb = predict_job_decode_spike_mb(probe, str(_SDXL.baseline))
    assert decode_spike_mb is not None and decode_spike_mb > 0.0
    support_mb = _support_component_mb(_SDXL)
    transient_mb = 1000.0
    # The weights leave exactly the activation and the margin free beside the support components, so sampling
    # fits and the larger decode spike does not.
    weights_mb = _LANE_ROOM_MB - support_mb - _CHILD_FREE_MARGIN_MB - transient_mb
    decode_unload_mb = decode_spike_mb - transient_mb
    world = _one_lane_world(
        true_footprint_mb=_stated(weights_mb, transient_mb),
        phase_resident_components=True,
    )

    first, second = await _serve_streak(world, [probe, _job(_SDXL)])

    assert first.window.retention_granted, "the streak's first job was not retained, so nothing can carry over"
    assert first.window.load_offloaded_mb == 0.0, "the first job's load was already short of room"
    assert first.window.decode_unloads == 1
    assert first.window.decode_unloaded_mb == pytest.approx(decode_unload_mb)
    assert first.support_while_sampling_mb == pytest.approx(support_mb)
    assert second.window.load_returned_mb == pytest.approx(decode_unload_mb), (
        f"the second job returned {second.window.load_returned_mb} MB of the {decode_unload_mb} MB its "
        "predecessor's decode unloaded"
    )
    assert second.window.load_seconds == pytest.approx(_return_seconds(decode_unload_mb))
    assert second.window.decode_unloads == 1
    # A retained model keeps its remainder for the next job, and an unretained one goes back to the RAM cache whole.
    kept_mb = decode_unload_mb if second.window.retention_granted else 0.0
    assert second.offloaded_after_mb == pytest.approx(kept_mb), (
        f"the lane kept {second.offloaded_after_mb} MB in host RAM after the streak's last job "
        f"(retained: {second.window.retention_granted}), not {kept_mb} MB"
    )


async def test_a_skip_listed_model_loads_partially_into_the_room_its_released_encoder_left() -> None:
    """Flux, its weights above what its child believes is free, offloads the rest and samples with no encoder.

    The encoder left the card before the clearance wait, so the child believes the whole lane's room is free. The
    load keeps clear the larger of the child's margin and the activation reserve its grant carried.
    """
    weights_mb = 23_000.0
    believed_free_mb = _LANE_ROOM_MB
    world = _one_lane_world(
        true_footprint_mb=_stated(weights_mb, 500.0),
        phase_resident_components=True,
        clearance_lease=True,
    )

    (served,) = await _serve_streak(world, [_job(_FLUX)])

    margin_mb = served.window.load_margin_mb
    assert margin_mb is not None and margin_mb >= _CHILD_FREE_MARGIN_MB
    offload_mb = weights_mb + margin_mb - believed_free_mb
    assert offload_mb > 0.0, "the stated weights fit the lane, so the row would show no partial load"
    assert world.clearance_grants, "the job was never cleared, so the row says nothing about a clear"
    assert served.window.load_believed_free_mb == pytest.approx(believed_free_mb)
    assert served.window.load_offloaded_mb == pytest.approx(offload_mb), (
        f"the child left {served.window.load_offloaded_mb} MB in host RAM, not the {offload_mb} MB its believed "
        "free less the margin could not take"
    )
    assert served.window.load_moved_mb == pytest.approx(believed_free_mb - margin_mb)
    assert served.support_while_sampling_mb == 0.0, (
        f"the lane held {served.support_while_sampling_mb} MB of support components while Flux sampled, after its "
        "encoder left the card before the wait"
    )


async def test_a_forced_load_frees_the_support_components_before_its_own_weights() -> None:
    """A model the worker force-loads takes its shortfall from the encoder first and loads its weights whole."""
    support_mb = _support_component_mb(_SDXL)
    weights_mb = 22_000.0
    support_freed_mb = weights_mb + _CHILD_FREE_MARGIN_MB - (_LANE_ROOM_MB - support_mb)
    assert 0.0 < support_freed_mb < support_mb
    world = _one_lane_world(true_footprint_mb=_stated(weights_mb, 0.0), phase_resident_components=True)

    (served,) = await _serve_streak(world, [_job(_SDXL)])

    assert served.window.load_offloaded_mb == 0.0
    assert served.window.load_moved_mb == pytest.approx(weights_mb)
    assert served.support_while_sampling_mb == pytest.approx(support_mb - support_freed_mb)


_SCRIPTED_FOREIGN_MB = 3300.0
"""Device memory no lane holds, scripted from the first tick on."""


async def _foreign_samples(*, lane_reports_carry_context: bool, service_contexts: bool) -> list[float]:
    """Serve a two-lane SDXL pair under scripted foreign VRAM and return the parent's foreign samples after it lands.

    The parent's floor clock is the world's, so each sample is stamped with the tick that took it.
    """
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=2,
        max_threads=2,
        queue_depth=2,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        service_contexts=service_contexts,
        foreign_vram_mb_by_tick={1: _SCRIPTED_FOREIGN_MB},
        lane_reports_carry_context=lane_reports_carry_context,
    )
    world.scheduler._foreign_floor_clock = lambda: world.now
    await world.step()
    landed_at = world.now
    await _serve_streak(world, [_job(_SDXL), _job(_SDXL)])
    samples = world.scheduler._foreign_vram_floor._samples.get(0, ())
    return [foreign_mb for sampled_at, foreign_mb in samples if sampled_at > landed_at]


@pytest.mark.parametrize("service_contexts", [False, True], ids=["inference_only", "service_contexts"])
async def test_reports_carrying_context_read_the_scripted_foreign_vram(service_contexts: bool) -> None:
    """With each process reporting its context share, the parent's foreign sample is the scripted figure."""
    samples = await _foreign_samples(lane_reports_carry_context=True, service_contexts=service_contexts)

    assert samples, "the parent took no foreign sample once the scripted figure landed"
    off_by = [foreign_mb for foreign_mb in samples if foreign_mb != pytest.approx(_SCRIPTED_FOREIGN_MB)]
    assert not off_by, f"the parent read foreign VRAM of {sorted(set(off_by))} MB against {_SCRIPTED_FOREIGN_MB} MB"


async def test_tenant_only_reports_read_foreign_above_the_scripted_figure() -> None:
    """Off, the world's context charge the parent cannot attribute reads as foreign on top of the scripted figure."""
    samples = await _foreign_samples(lane_reports_carry_context=False, service_contexts=False)

    assert samples
    assert min(samples) > _SCRIPTED_FOREIGN_MB + 1000.0


async def test_a_retained_lane_stages_its_whole_encode_and_releases_it_before_the_wait() -> None:
    """A retained lane keeps its weights and no encoder, so its next job encodes in full and waits without it.

    The encoder of the retained job left the card before that job's wait, so nothing of it is on the lane when
    the next job stages. That job's encode is the whole staging charge, and it leaves the card before the next
    clearance wait as well.
    """
    staging_mb = staging_charge_mb(str(_SDXL.baseline))
    world = _one_lane_world(
        true_footprint_mb=_stated(6000.0, 1000.0),
        phase_resident_components=True,
        clearance_lease=True,
    )
    first_job, second_job = _job(_SDXL), _job(_SDXL)
    for job in (first_job, second_job):
        await world.pop(job)
    second_id = str(second_job.id_)
    tenant_mb: float | None = None
    for _ in range(_MAX_TICKS):
        tenant_mb = world._support_resident_mb.get(0, 0.0)
        await world.step()
        if second_id in world.job_windows:
            break
    assert second_id in world.job_windows, f"the second job was never staged. {world.state_dump()}"
    first_window = world.job_windows[str(first_job.id_)]
    assert first_window.retention_granted, "the first job was not retained, so the lane was not carried over"
    assert tenant_mb == 0.0, f"the retained lane held a {tenant_mb:.0f} MB support tenant into the next stage"
    second_window = world.job_windows[second_id]
    assert second_window.support_released_before_wait_mb == pytest.approx(staging_mb), (
        f"the second job released {second_window.support_released_before_wait_mb:.0f} MB before its wait, not its "
        f"whole {staging_mb:.0f} MB encode"
    )
    assert not world._encode_staging_mb, f"a staged lane still holds an encode: {world._encode_staging_mb}"


_Z_IMAGE_WEIGHTS_MB = 18_000.0
"""A Z-Image-sized checkpoint whose forced load frees part of its support tenant, so a later stage of the same
model needs more encoder than the tenant still holds and more room than the card has beside the weights."""
_Z_IMAGE = _ModelClass(
    "z_image", "z-image-checkpoint", KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo, _Z_IMAGE_WEIGHTS_MB
)


@dataclass(frozen=True)
class _StageReading:
    """Represents one staged lane as its encode settled, before the tick's clearance pass could grant it."""

    resident_mb: float
    offloaded_mb: float
    card_free_mb: float


class _StageReadingWorld(_DispatchWorld):
    """A world that reads the staged lane once each dispatch's encode has settled.

    The clearance pass that closes a dispatch tick grants the staged job and its load returns what the stage
    unloaded, so a reading taken when the tick ends no longer shows the stage.
    """

    @cached_property
    def stage_readings(self) -> dict[str, _StageReading]:
        """Return each staged job's reading, by job id."""
        return {}

    @override
    def _begin_staged_prefetch(self, admitted: ImageGenerateJobPopResponse, *, lane_id: int) -> None:
        self.stage_readings[str(admitted.id_)] = _StageReading(
            resident_mb=self._resident_mb.get(lane_id, 0.0),
            offloaded_mb=self._offloaded_mb.get(lane_id, 0.0),
            card_free_mb=self.card_free_mb(self._card_of(lane_id)),
        )
        super()._begin_staged_prefetch(admitted, lane_id=lane_id)


async def test_a_stage_beside_retained_weights_unloads_them_and_the_next_load_returns_them() -> None:
    """A retained lane staged for its next job funds the encoder from its own weights, then gets them back.

    ComfyUI's free-memory pass makes room for the text encoder by unloading the process's own diffusion weights,
    so the stage commits no more than the child believes is free less its margin. The next load of the same model
    returns everything the lane left in host RAM, the stage's unload included. The stage is read as its encode
    settles, since the same tick's clearance pass grants the job and its load returns the weights.
    """
    world = _StageReadingWorld(
        card=_CARD_24GB,
        lane_count=1,
        max_threads=1,
        queue_depth=2,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        true_footprint_mb=_stated(_Z_IMAGE_WEIGHTS_MB, 1000.0),
        phase_resident_components=True,
        clearance_lease=True,
        partial_loads=True,
        extra_model_classes=(_Z_IMAGE,),
    )
    first_job, second_job = _job(_Z_IMAGE), _job(_Z_IMAGE)
    for job in (first_job, second_job):
        await world.pop(job)
    second_id = str(second_job.id_)
    resident_before_stage_mb: float | None = None
    for _ in range(_MAX_TICKS):
        resident_before_mb = world._resident_mb.get(0, 0.0)
        await world.step()
        if resident_before_stage_mb is None and second_id in world.job_windows:
            resident_before_stage_mb = resident_before_mb
        if world.completed_jobs == 2:
            break

    assert resident_before_stage_mb is not None, f"the second job was never staged. {world.state_dump()}"
    assert world.job_windows[str(first_job.id_)].retention_granted, "the first job was not retained"
    stage = world.stage_readings[second_id]
    stage_unloaded_mb = resident_before_stage_mb - stage.resident_mb
    offloaded_after_stage_mb = stage.offloaded_mb
    free_after_stage_mb = stage.card_free_mb
    assert stage_unloaded_mb > 0.0, "the stage found room beside the retained weights, so the row proves nothing"
    assert free_after_stage_mb >= _CHILD_FREE_MARGIN_MB - 1e-6, (
        f"the stage left {free_after_stage_mb:.0f} MB free, under the child's {_CHILD_FREE_MARGIN_MB:.0f} MB margin"
    )
    second_window = world.job_windows[second_id]
    assert second_window.load_returned_mb == pytest.approx(offloaded_after_stage_mb), (
        f"the next load returned {second_window.load_returned_mb:.0f} MB of the {offloaded_after_stage_mb:.0f} MB "
        "the lane held in host RAM after the stage"
    )
    assert not world.card_overcommits, f"the card was over-committed: {world.card_overcommits[:3]}"


_ACTIVATION_SPARE_MB = 200.0
"""Free VRAM beyond the child's margin that the weight load leaves, so the activation finds a shortfall."""


async def test_an_activation_is_funded_from_the_support_components_before_the_weights() -> None:
    """An activation the believed free does not cover releases the encoder and VAE before any diffusion weights.

    ComfyUI's sampling load frees the process's other models before it partially unloads the diffusion model it
    samples with, so the activation's shortfall comes out of the support tenant while the tenant can cover it.
    """
    support_mb = _support_component_mb(_SDXL)
    transient_mb = 1000.0
    weights_mb = _LANE_ROOM_MB - support_mb - _CHILD_FREE_MARGIN_MB - _ACTIVATION_SPARE_MB
    activation_shortfall_mb = transient_mb - _ACTIVATION_SPARE_MB
    assert 0.0 < activation_shortfall_mb < support_mb
    world = _one_lane_world(true_footprint_mb=_stated(weights_mb, transient_mb), phase_resident_components=True)

    (served,) = await _serve_streak(world, [_job(_SDXL)])

    assert served.window.load_offloaded_mb == 0.0, "the weight load was already short of room"
    assert served.support_while_sampling_mb == pytest.approx(support_mb - activation_shortfall_mb), (
        f"the lane held {served.support_while_sampling_mb} MB of support components while sampling, not the "
        f"{support_mb - activation_shortfall_mb} MB left after funding a {activation_shortfall_mb} MB shortfall"
    )
    assert not any("own weights to fund" in line for line in world.child_shortfall_frees), (
        f"the activation returned diffusion weights while the support tenant could fund it: "
        f"{world.child_shortfall_frees}"
    )
    assert served.window.activation_support_released_mb == pytest.approx(activation_shortfall_mb)


async def test_an_encoder_released_before_the_wait_leaves_no_support_tenant() -> None:
    """With the release modelled, a staged lane waits holding no encoder, and its commit charges no tenant.

    hordelib's sampling lease hook returns the text encoder to its CPU origins after the encode and before the
    first acquire, so neither the encode's allocation nor a support tenant is on the card while the job waits
    or samples.
    """
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=1,
        max_threads=1,
        queue_depth=2,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        true_footprint_mb=_stated(6000.0, 1000.0),
        phase_resident_components=True,
        clearance_lease=True,
        partial_loads=True,
        text_encoder_released_before_wait=True,
    )

    (served,) = await _serve_streak(world, [_job(_SDXL)])

    assert world.clearance_grants, "the job was never cleared, so the row says nothing about a clear"
    assert served.window.support_released_before_wait_mb == pytest.approx(staging_charge_mb(str(_SDXL.baseline)))
    assert served.support_while_sampling_mb == 0.0, (
        f"the lane held {served.support_while_sampling_mb} MB of support components while sampling, after its "
        "encoder left the card before the wait"
    )
    assert not world._encode_staging_mb, f"a staged lane still holds an encode: {world._encode_staging_mb}"


def test_the_encoder_release_needs_the_lease_and_the_support_tenant() -> None:
    """The release runs in the sampling lease hook and frees the support tenant, so a world without them refuses it."""
    with pytest.raises(ValueError, match="encoder release"):
        _DispatchWorld(
            card=_CARD_24GB,
            lane_count=1,
            max_threads=1,
            queue_depth=2,
            closed_loop=True,
            partial_loads=True,
            phase_resident_components=True,
            text_encoder_released_before_wait=True,
        )


async def test_a_partial_load_leaves_the_grants_activation_reserve() -> None:
    """A skip-listed load cleared under the lease leaves the reserve its grant carried in place of the child's margin.

    The grant carries the activation reserve the parent priced the window with, and hordelib raises ComfyUI's
    partial sizing to it, so the load leaves that room for the activation in place of the child's smaller estimate.
    """
    weights_mb = 25_000.0
    world = _one_lane_world(true_footprint_mb=_stated(weights_mb, 500.0), clearance_lease=True)
    job = _job(_FLUX)
    job_id = str(job.id_)
    await world.pop(job)
    granted_reserve_mb: float | None = None
    for _ in range(_MAX_TICKS):
        await world.step()
        granted_reserve_mb = world._granted_sampling_reserve_mb.get(job_id, granted_reserve_mb)
        if world.completed_jobs == 1:
            break

    window = world.job_windows[job_id]
    assert world.clearance_grants, "the job was never cleared, so the row says nothing about a grant"
    assert granted_reserve_mb is not None, "the grant carried no activation reserve"
    assert granted_reserve_mb > _CHILD_FREE_MARGIN_MB, (
        f"the grant's {granted_reserve_mb:.0f} MB reserve is within the child's margin, so the row proves nothing"
    )
    assert window.load_margin_mb == pytest.approx(granted_reserve_mb)
    assert window.load_believed_free_mb is not None
    assert window.load_moved_mb == pytest.approx(window.load_believed_free_mb - granted_reserve_mb), (
        f"the load moved {window.load_moved_mb:.0f} MB into {window.load_believed_free_mb:.0f} MB believed free, "
        f"not leaving the granted {granted_reserve_mb:.0f} MB reserve"
    )


_DECODE_FOREIGN_TICK = 3
"""The tick a foreign tenant takes the card, after the job's load committed and before its decode."""
_DECODE_STATED_WEIGHTS_MB = 2000.0
"""Weights small enough that the decode spike's shortfall outlasts them."""


async def test_a_decode_with_too_few_weights_left_takes_the_support_components() -> None:
    """A decode spike the lane's own weights cannot fund releases its encoder instead of over-committing the card.

    A foreign tenant arrives while the job samples and leaves the lane exactly its margin free. The decode then
    unloads all of the lane's weights and funds the rest from the support tenant, as ComfyUI's free-memory pass
    unloads a resident text encoder to fit the VAE's working set.
    """
    probe = _job(_FLUX)
    decode_spike_mb = predict_job_decode_spike_mb(probe, str(_FLUX.baseline))
    assert decode_spike_mb is not None
    support_mb = _support_component_mb(_FLUX)
    support_needed_mb = decode_spike_mb - _DECODE_STATED_WEIGHTS_MB
    assert 0.0 < support_needed_mb < support_mb
    foreign_mb = _LANE_ROOM_MB - support_mb - _DECODE_STATED_WEIGHTS_MB - _CHILD_FREE_MARGIN_MB
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=1,
        max_threads=1,
        queue_depth=2,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        true_footprint_mb=_stated(_DECODE_STATED_WEIGHTS_MB, 0.0),
        phase_resident_components=True,
        partial_loads=True,
        foreign_vram_mb_by_tick={_DECODE_FOREIGN_TICK: foreign_mb},
    )

    (served,) = await _serve_streak(world, [probe])

    assert served.window.sample_until is not None
    assert served.window.sample_until > _DECODE_FOREIGN_TICK * _TICK_SECONDS, "the decode began before the tenant"
    assert served.window.load_offloaded_mb == 0.0, "the load was already short of room"
    assert served.window.decode_unloaded_mb == pytest.approx(_DECODE_STATED_WEIGHTS_MB)
    assert not world.card_overcommits, f"the card was over-committed: {world.card_overcommits[:3]}"
    assert served.window.decode_support_released_mb == pytest.approx(support_needed_mb)
