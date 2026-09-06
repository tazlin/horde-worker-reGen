"""Golden per-tick traces of the inference scheduler, driven over the closed-loop dispatch world.

Each row builds a dispatch world, runs a fixed number of scheduling ticks, and records everything the
scheduler decided and actuated on each of them (see :mod:`tests.process_management.liveness.golden_trace`).
The recorded trace is compared byte for byte against a committed file under ``golden/``. A refactor that only
moves code produces an identical trace; anything else is a behavior change that has to be named and its
golden regenerated as a reviewed diff.

Regeneration is a separate, opt-in test (``-m golden_regen``) that runs every row twice and refuses to write
a row whose two runs disagree, so a trace that is not reproducible never reaches a golden file.

The catalog is chosen for coverage of the paths the admission redesign cuts through rather than for breadth:
steady dispatch, retention grant and reuse, whole-card residency, a RAM-pressure defer, the clearance lease,
two-card routing, the post-processing co-residency defer, a disaggregated head, a safety placement
demotion, and a card packed with idle held components.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_16GB,
    _CARD_24GB,
    _FLUX,
    _SD15,
    _SD15_OTHER,
    _SDXL,
    _SDXL_OTHER,
    _DispatchWorld,
    _ModelClass,
)
from tests.process_management.liveness.golden_trace import TraceRecorder, diff, serialise

GOLDEN_DIR = Path(__file__).parent / "golden"
"""Where the committed traces live, one JSON file per row."""

_TICK_SECONDS = 2.0
"""Seconds of world clock per tick, matching the incident scenarios' cadence.

Short enough that a job's load, sampling and decode phases are each sampled several times, so the trace
carries the intermediate states a coarser tick would step over."""

_PINNED_TOTAL_RAM_MB = 32_768.0
"""Host RAM total every row is priced against.

The scheduler reads the real host's total through psutil, and the RAM danger floor is a percentage of it, so
an unpinned row would record a different floor on every machine and the golden would be a property of the
box that generated it rather than of the scheduler."""

_AMPLE_RAM_MB = 65_536.0
"""Available host RAM for every row that does not vary it: far above any floor, so RAM gates stay out."""

_HELD_COMPONENT_MB = 6_600.0
"""What one idle lane's device-warm component cache holds: an SDXL checkpoint's worth of components.

A tenant of the card that belongs to no job, so no dispatch's footprint includes it and only an unload the
parent actuates returns it."""


def _pin_host_ram(world: _DispatchWorld, *, available_mb: float = _AMPLE_RAM_MB) -> None:
    """Pin both host-RAM readings the scheduler prices against, so a row's trace is not the host's.

    The world already pins the available reading; the total is read from psutil and reaches the danger floor
    and every hold record derived from it.
    """
    world.scheduler.set_available_ram_mb_provider(lambda: available_mb)
    world.scheduler._measured_total_ram_mb = lambda: _PINNED_TOTAL_RAM_MB  # type: ignore[method-assign]


async def _pop_queue(
    world: _DispatchWorld,
    models: list[_ModelClass],
    *,
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    post_processing: list[str] | None = None,
) -> list[ImageGenerateJobPopResponse]:
    """Pop one job per entry in ``models``, in order, and return them."""
    jobs: list[ImageGenerateJobPopResponse] = []
    for model in models:
        job = make_job_pop_response(
            model.name,
            width=width,
            height=height,
            ddim_steps=steps,
            post_processing=post_processing,
        )
        await world.pop(job)
        jobs.append(job)
    return jobs


# --------------------------------------------------------------------------------------------------------
# The catalog
# --------------------------------------------------------------------------------------------------------


async def _build_steady_dispatch() -> _DispatchWorld:
    """One card, two lanes, a small-model queue: dispatch with nothing contending for the device."""
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=2,
        max_threads=2,
        queue_depth=4,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
    )
    _pin_host_ram(world)
    await _pop_queue(world, [_SD15, _SD15, _SD15, _SD15], width=512, height=512, steps=8)
    return world


async def _build_retention_reuse() -> _DispatchWorld:
    """A same-model streak on a card that can hold one copy: retention is granted and then reused."""
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=2,
        queue_depth=4,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
    )
    _pin_host_ram(world)
    await _pop_queue(world, [_SDXL] * 5)
    return world


async def _build_whole_card_residency() -> _DispatchWorld:
    """An extra-large head on a 16 GB card: the whole-card residency establishes and is then restored."""
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=1,
        queue_depth=4,
        whole_card_enabled=True,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
    )
    _pin_host_ram(world)
    await _pop_queue(world, [_FLUX, _FLUX, _SD15], width=1024, height=1024, steps=8)
    return world


async def _build_ram_pressure_defer() -> _DispatchWorld:
    """Available host RAM below the danger floor: every preload defers on RAM rather than on the card."""
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=2,
        max_threads=2,
        queue_depth=4,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
    )
    # Below 15% of the pinned total, which is the floor the danger verdict is taken against.
    _pin_host_ram(world, available_mb=2_048.0)
    await _pop_queue(world, [_SDXL, _SD15, _SDXL], steps=8)
    return world


async def _build_clearance_lease() -> _DispatchWorld:
    """Dispatch as a staging step: the weights land at clearance, priced by the real clearance controller."""
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=2,
        queue_depth=4,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        clearance_lease=True,
    )
    _pin_host_ram(world)
    await _pop_queue(world, [_SDXL, _SDXL, _SDXL_OTHER])
    return world


async def _build_two_card_routing() -> _DispatchWorld:
    """Two cards whose resolution ceilings differ, so a job's size alone decides which card may serve it."""
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=2,
        queue_depth=4,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        card_max_pixels={0: 1024 * 1024, 1: 512 * 512},
    )
    _pin_host_ram(world)
    await _pop_queue(world, [_SDXL], width=1024, height=1024, steps=8)
    await _pop_queue(world, [_SD15, _SD15_OTHER], width=512, height=512, steps=8)
    return world


async def _build_post_processing_defer() -> _DispatchWorld:
    """Jobs whose post-processing shares the card, so the co-residency gate defers a dispatch behind one."""
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=2,
        queue_depth=4,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        service_contexts=True,
    )
    _pin_host_ram(world)
    await _pop_queue(world, [_SDXL, _SDXL, _SDXL], post_processing=["RealESRGAN_x4plus"])
    return world


async def _build_disaggregated_head() -> _DispatchWorld:
    """Disaggregation-class jobs: each sampler is priced for the UNet it holds rather than a whole job."""
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=2,
        queue_depth=4,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        service_contexts=True,
        disaggregated=True,
    )
    _pin_host_ram(world)
    await _pop_queue(world, [_SDXL, _SDXL, _SD15])
    return world


async def _build_safety_placement() -> _DispatchWorld:
    """Safety on a card an extra-large head takes over: the demotion and its restore are both traced.

    The heaviest head the card can seat leaves no room for the safety context beside it, so establishing its
    residency moves safety off the card and draining the queue behind it brings safety back. The readiness
    window is modelled, so the flip is the process cycle it really is rather than an instant.
    """
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=1,
        queue_depth=4,
        whole_card_enabled=True,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        service_contexts=True,
        safety_off_gpu_allowed=True,
        safety_readiness_seconds=8.0,
        safety_load_transient_mb=1_000.0,
        unload_release_delay_seconds=4.0,
    )
    _pin_host_ram(world)
    await _pop_queue(world, [_FLUX, _FLUX], width=1024, height=1024, steps=8)
    await _pop_queue(world, [_SD15, _SD15], width=512, height=512, steps=8)
    return world


async def _build_held_components() -> _DispatchWorld:
    """Idle lanes holding device-warm components a job does not own, so the head must reclaim to fit.

    The head's weights are staged rather than popped cold, which puts the dispatch residency-reconciliation
    hold in charge of the fit instead of the preload admission path that runs before it.
    """
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=3,
        max_threads=1,
        queue_depth=4,
        whole_card_enabled=True,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
    )
    _pin_host_ram(world)
    world.seed_held_components(0, _HELD_COMPONENT_MB)
    world.seed_held_components(1, _HELD_COMPONENT_MB)
    world.seed_resident(2, _SDXL, in_vram=False)
    await _pop_queue(world, [_SDXL, _SDXL])
    return world


@dataclass(frozen=True)
class _Scenario:
    """One traced row: how to build its world and how many ticks to drive it for.

    Attributes:
        label: The golden file's stem and the parametrized id.
        build: Builds the world with its queue already popped.
        ticks: Scheduling ticks driven, fixed so the trace has a defined end.
    """

    label: str
    build: Callable[[], Awaitable[_DispatchWorld]]
    ticks: int


_SCENARIOS: tuple[_Scenario, ...] = (
    _Scenario("steady_dispatch", _build_steady_dispatch, 24),
    _Scenario("retention_reuse", _build_retention_reuse, 40),
    _Scenario("whole_card_residency", _build_whole_card_residency, 40),
    _Scenario("ram_pressure_defer", _build_ram_pressure_defer, 16),
    _Scenario("clearance_lease", _build_clearance_lease, 40),
    _Scenario("two_card_routing", _build_two_card_routing, 24),
    _Scenario("post_processing_defer", _build_post_processing_defer, 32),
    _Scenario("disaggregated_head", _build_disaggregated_head, 32),
    _Scenario("safety_placement", _build_safety_placement, 40),
    _Scenario("held_components", _build_held_components, 32),
)


# --------------------------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------------------------


class _SequentialJobIds:
    """A stand-in for :func:`uuid.uuid4` that hands out ids in a fixed order.

    Job ids are the one random input a row takes, and they reach the tracker's keys and the scheduler's
    subjects. The trace aliases them by first mention, so the identifiers themselves never reach a golden
    file, but their ordering can still reach a tie-break; pinning them removes the question.
    """

    def __init__(self) -> None:
        """Start the sequence at one, so no job carries the nil uuid."""
        self._issued = 0

    def __call__(self) -> uuid.UUID:
        """Return the next id in the sequence."""
        self._issued += 1
        return uuid.UUID(int=self._issued)


@pytest.fixture
def _deterministic_job_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace random job ids with a fixed sequence for the duration of one test."""
    monkeypatch.setattr(uuid, "uuid4", _SequentialJobIds())


async def _record_scenario(scenario: _Scenario) -> dict[str, Any]:
    """Build, drive and trace one row."""
    world = await scenario.build()
    recorder = TraceRecorder(world)
    for _ in range(scenario.ticks):
        await world.step()
        recorder.capture_tick()
    return recorder.trace(scenario=scenario.label, ticks=scenario.ticks)


def _golden_path(scenario: _Scenario) -> Path:
    return GOLDEN_DIR / f"{scenario.label}.json"


# --------------------------------------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("_deterministic_job_ids")
@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[scenario.label for scenario in _SCENARIOS])
async def test_scheduler_trace_matches_its_golden(scenario: _Scenario) -> None:
    """The scheduler decides and actuates exactly what the committed trace for this row says it does."""
    golden_path = _golden_path(scenario)
    assert golden_path.is_file(), (
        f"{scenario.label}: no golden trace at {golden_path}. Generate it with "
        f"`pytest -m golden_regen tests/process_management/liveness/test_golden_traces.py` and review the diff."
    )
    expected = json.loads(golden_path.read_text(encoding="utf-8"))

    actual = await _record_scenario(scenario)

    divergence = diff(expected, actual)
    assert not divergence, f"{scenario.label}: the scheduler's trace diverged from its golden.\n{divergence}"


@pytest.mark.golden_regen
@pytest.mark.usefixtures("_deterministic_job_ids")
@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[scenario.label for scenario in _SCENARIOS])
async def test_regenerate_golden_trace(scenario: _Scenario) -> None:
    """Rewrite one row's golden trace, refusing to write one the row does not reproduce.

    Two runs of the same row in the same process must agree exactly. A row that cannot manage that is
    recording something that is not the scheduler's decision, and a golden taken from it would fail for
    whoever ran it next.
    """
    first = await _record_scenario(scenario)
    second = await _record_scenario(scenario)

    divergence = diff(first, second)
    assert not divergence, (
        f"{scenario.label}: two runs of the same row disagree, so its trace is not reproducible and no "
        f"golden was written.\n{divergence}"
    )

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    _golden_path(scenario).write_text(serialise(first), encoding="utf-8")


def test_the_catalog_covers_the_paths_the_redesign_cuts() -> None:
    """Every row has a distinct label, a golden file, and drives a positive number of ticks."""
    labels = [scenario.label for scenario in _SCENARIOS]
    assert len(labels) == len(set(labels)), f"duplicate row labels in the catalog: {labels}"
    assert len(labels) >= 8, "the catalog is meant to cover the admission paths broadly, not a sample of them"
    for scenario in _SCENARIOS:
        assert scenario.ticks > 0, f"{scenario.label}: a row with no ticks records nothing"
        assert _golden_path(scenario).is_file(), f"{scenario.label}: no committed golden trace"
