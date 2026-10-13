"""The dispatch world's partial loads: offloads that outlive their job, decode unloads, and skip-listed loads.

Each row runs one closed-loop lane on a 24 GB card and states its jobs' true footprints, so the child's shortfall
arithmetic, and not the parent's forecast, decides what reaches the card. The rows read the world's own records:
each job's window, the lane's host-RAM remainder between jobs, and the support components the lane holds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


async def test_a_skip_listed_model_loads_partially_beside_its_resident_encoder() -> None:
    """Flux, cleared at a price above what its child believes is free, offloads the rest and keeps its encoder."""
    support_mb = _support_component_mb(_FLUX)
    weights_mb = 19_000.0
    believed_free_mb = _LANE_ROOM_MB - support_mb
    offload_mb = weights_mb + _CHILD_FREE_MARGIN_MB - believed_free_mb
    assert offload_mb > 0.0, "the stated weights fit beside the encoder, so the row would show no partial load"
    world = _one_lane_world(
        true_footprint_mb=_stated(weights_mb, 500.0),
        phase_resident_components=True,
        clearance_lease=True,
    )

    (served,) = await _serve_streak(world, [_job(_FLUX)])

    assert world.clearance_grants, "the job was never cleared, so the row says nothing about a clear"
    assert served.window.load_believed_free_mb == pytest.approx(believed_free_mb)
    assert served.window.load_offloaded_mb == pytest.approx(offload_mb), (
        f"the child left {served.window.load_offloaded_mb} MB in host RAM, not the {offload_mb} MB its believed "
        "free less the margin could not take"
    )
    assert served.window.load_moved_mb == pytest.approx(believed_free_mb - _CHILD_FREE_MARGIN_MB)
    assert served.support_while_sampling_mb == pytest.approx(support_mb), (
        f"the lane held {served.support_while_sampling_mb} MB of support components while Flux sampled, not its "
        f"whole {support_mb} MB encoder and VAE"
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


async def test_a_staged_encode_is_netted_against_the_support_components_its_lane_holds() -> None:
    """A retained lane staged for its next job commits only the encode its support tenant does not already hold.

    The staging charge prices the text encoder, which the retained tenant already holds on the card, and
    ComfyUI's component cache allocates nothing for a resident encoder. The parent's price is not netted.
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
    staged_mb: float | None = None
    tenant_mb = 0.0
    for _ in range(_MAX_TICKS):
        tenant_mb = world._support_resident_mb.get(0, 0.0)
        await world.step()
        if second_id in world.job_windows:
            # Read on the staging tick itself. The same tick releases the first job's decode tail, so the card's
            # total moves by more than the stage alone, and the stage's own charge is what the card took for it.
            staged_mb = world._encode_staging_mb.get(world.job_windows[second_id].lane_id, 0.0)
            break
    assert staged_mb is not None, f"the second job was never staged. {world.state_dump()}"
    first_window = world.job_windows[str(first_job.id_)]
    assert first_window.retention_granted, "the first job was not retained, so no tenant stood on the lane"
    assert tenant_mb == pytest.approx(_support_component_mb(_SDXL)), "the retained lane held no support tenant"
    assert staged_mb <= max(0.0, staging_mb - tenant_mb) + 1e-6, (
        f"staging the second job charged the card {staged_mb:.0f} MB, above the {staging_mb:.0f} MB encode less "
        f"the {tenant_mb:.0f} MB tenant the lane already held"
    )


_Z_IMAGE_WEIGHTS_MB = 18_000.0
"""A Z-Image-sized checkpoint whose forced load frees part of its support tenant, so a later stage of the same
model needs more encoder than the tenant still holds and more room than the card has beside the weights."""
_Z_IMAGE = _ModelClass(
    "z_image", "z-image-checkpoint", KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo, _Z_IMAGE_WEIGHTS_MB
)


async def test_a_stage_beside_retained_weights_unloads_them_and_the_next_load_returns_them() -> None:
    """A retained lane staged for its next job funds the encoder from its own weights, then gets them back.

    ComfyUI's free-memory pass makes room for the text encoder by unloading the process's own diffusion weights,
    so the stage commits no more than the child believes is free less its margin. The next load of the same model
    returns everything the lane left in host RAM, the stage's unload included.
    """
    world = _DispatchWorld(
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
    stage_unloaded_mb: float | None = None
    offloaded_after_stage_mb = 0.0
    free_after_stage_mb = 0.0
    for _ in range(_MAX_TICKS):
        resident_before_mb = world._resident_mb.get(0, 0.0)
        await world.step()
        if stage_unloaded_mb is None and second_id in world.job_windows:
            stage_unloaded_mb = resident_before_mb - world._resident_mb.get(0, 0.0)
            offloaded_after_stage_mb = world._offloaded_mb.get(0, 0.0)
            free_after_stage_mb = world.card_free_mb(0)
        if world.completed_jobs == 2:
            break

    assert stage_unloaded_mb is not None, f"the second job was never staged. {world.state_dump()}"
    assert world.job_windows[str(first_job.id_)].retention_granted, "the first job was not retained"
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
